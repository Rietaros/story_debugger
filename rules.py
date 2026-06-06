# rules.py
from __future__ import annotations

from collections import defaultdict
import re
from schemas import ChapterExtraction, Issue
from lore_graph import LoreGraph


WEAK_TITLE_RE = re.compile(
    r"^(?:sc|scene|event|ev)[\s_:#-]*\d+(?:[\s_:#-]*\d+)?$",
    re.IGNORECASE,
)


def _compact_title(value: str | None, max_chars: int = 72) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rsplit(" ", 1)[0].strip()
    return f"{clipped}..." if clipped else text[:max_chars]


def _useful_title(title: str | None, fallback: str | None, identifier: str) -> str:
    text = _compact_title(title)
    lowered = text.lower()
    is_placeholder = (
        not text
        or text == identifier
        or bool(WEAK_TITLE_RE.fullmatch(text))
        or "scene title" in lowered
        or "event title" in lowered
        or lowered.startswith("untitled")
    )
    if not is_placeholder:
        return text
    return _compact_title(fallback) or identifier


class PlotHoleDetector:
    def __init__(self) -> None:
        self.inventory: dict[str, set[str]] = defaultdict(set)
        self.item_owner: dict[str, str] = {}
        self.knowledge: dict[str, set[str]] = defaultdict(set)
        self.position_by_time: dict[tuple[str, str, int], dict] = {}
        self.seen_events: set[str] = set()
        self.event_summaries: dict[str, str] = {}

    def _issue(self, severity: str, rule: str, message: str, evidence: dict) -> Issue:
        return Issue(
            severity=severity,
            rule=rule,
            message=message,
            evidence=evidence,
        )

    def check(self, chapter: ChapterExtraction, lore: LoreGraph) -> list[Issue]:
        issues: list[Issue] = []

        for scene in chapter.scenes:
            for event in sorted(scene.events, key=lambda e: e.seq):
                current_location = event.location or scene.location or "Unknown"
                event_title = _useful_title(event.title, event.summary, event.event_id)
                scene_title = _useful_title(scene.title, scene.summary, scene.scene_id)
                current_position = {
                    "chapter_id": chapter.chapter_id,
                    "scene_id": scene.scene_id,
                    "scene_title": scene_title,
                    "event_id": event.event_id,
                    "event_title": event_title,
                    "seq": event.seq,
                    "location": current_location,
                    "summary": event.summary,
                }
                self.seen_events.add(event.event_id)

                # 1. Duplicate event ID with different meaning
                if event.event_id in self.event_summaries:
                    if self.event_summaries[event.event_id] != event.summary:
                        issues.append(
                            self._issue(
                                "high",
                                "duplicate_event_id_conflict",
                                f"Event ID {event.event_id} appears with different summaries.",
                                {
                                    "event_id": event.event_id,
                                    "old_summary": self.event_summaries[event.event_id],
                                    "new_summary": event.summary,
                                },
                            )
                        )
                else:
                    self.event_summaries[event.event_id] = event.summary

                # 2. Unknown causal parent
                for parent in event.causal_parents:
                    if parent not in self.seen_events:
                        issues.append(
                            self._issue(
                                "medium",
                                "unknown_causal_parent",
                                f"{event.event_id} depends on unseen parent {parent}.",
                                {"event_id": event.event_id, "parent": parent},
                            )
                        )

                # 3. Double location in same chapter + seq
                for participant in event.participants:
                    key = (participant, chapter.chapter_id, event.seq)
                    previous_position = self.position_by_time.get(key)
                    previous_location = (
                        previous_position.get("location")
                        if previous_position
                        else None
                    )

                    if previous_location and previous_location != current_location:
                        issues.append(
                            self._issue(
                                "high",
                                "double_location_same_step",
                                f"{participant} appears in two different locations at the same story step: '{previous_location}' and '{current_location}'.",
                                {
                                    "character": participant,
                                    "chapter_id": chapter.chapter_id,
                                    "seq": event.seq,
                                    "previous_location": previous_location,
                                    "current_location": current_location,
                                    "previous_scene_id": previous_position.get("scene_id"),
                                    "previous_scene_title": previous_position.get("scene_title"),
                                    "previous_event_id": previous_position.get("event_id"),
                                    "previous_event_title": previous_position.get("event_title"),
                                    "previous_event_summary": previous_position.get("summary"),
                                    "current_scene_id": scene.scene_id,
                                    "current_scene_title": scene_title,
                                    "current_event_id": event.event_id,
                                    "current_event_title": event_title,
                                    "current_event_summary": event.summary,
                                },
                            )
                        )

                    self.position_by_time[key] = current_position

                # 4. Item used before acquired
                for participant in event.participants:
                    for item in event.used_items:
                        if item not in self.inventory[participant]:
                            issues.append(
                                self._issue(
                                    "high",
                                    "item_used_before_acquired",
                                    f"{participant} uses {item} before acquiring it.",
                                    {
                                        "character": participant,
                                        "item": item,
                                        "event_id": event.event_id,
                                    },
                                )
                            )

                # 5. Item owned by someone else but used here
                for participant in event.participants:
                    for item in event.used_items:
                        owner = self.item_owner.get(item)
                        if owner and owner != participant:
                            issues.append(
                                self._issue(
                                    "high",
                                    "item_used_by_non_owner",
                                    f"{participant} uses {item}, but current owner is {owner}.",
                                    {
                                        "character": participant,
                                        "item": item,
                                        "current_owner": owner,
                                        "event_id": event.event_id,
                                    },
                                )
                            )

                # 6. Multiple acquisition / suspicious ownership transfer
                for item in event.acquired_items:
                    previous_owner = self.item_owner.get(item)

                    for participant in event.participants:
                        if previous_owner and previous_owner != participant:
                            issues.append(
                                self._issue(
                                    "medium",
                                    "item_ownership_transfer_without_loss",
                                    f"{item} moves from {previous_owner} to {participant} without explicit loss/give event.",
                                    {
                                        "item": item,
                                        "previous_owner": previous_owner,
                                        "new_owner": participant,
                                        "event_id": event.event_id,
                                    },
                                )
                            )

                        self.inventory[participant].add(item)
                        self.item_owner[item] = participant

                # 7. Revelation not learned by POV
                for fact in event.revelations:
                    if scene.pov and fact not in self.knowledge[scene.pov]:
                        issues.append(
                            self._issue(
                                "low",
                                "pov_knows_untracked_fact",
                                f"POV character {scene.pov} narrates or witnesses {fact}, but knowledge is not tracked.",
                                {
                                    "fact": fact,
                                    "pov": scene.pov,
                                    "event_id": event.event_id,
                                },
                            )
                        )

                # 8. Contradiction keywords in summary
                lowered_summary = event.summary.lower()

                contradiction_markers = [
                    "tidak pernah",
                    "padahal",
                    "mustahil",
                    "seharusnya",
                    "aneh",
                    "tapi tadi",
                    "bukankah",
                    "wait",
                    "impossible",
                    "never happened",
                    "i thought",
                ]

                if any(marker in lowered_summary for marker in contradiction_markers):
                    issues.append(
                        self._issue(
                            "medium",
                            "possible_explicit_contradiction",
                            "Event summary contains contradiction markers.",
                            {
                                "event_id": event.event_id,
                                "summary": event.summary,
                            },
                        )
                    )

                # 9. Knowledge gains
                for character, facts in event.knowledge_gains.items():
                    self.knowledge[character].update(facts)

                # 10. If a character says they remember/know something but no fact is recorded
                memory_markers = [
                    "ingat",
                    "mengingat",
                    "aku tahu",
                    "sudah terjadi",
                    "pernah",
                    "remember",
                    "i know",
                    "happened before",
                ]

                if any(marker in lowered_summary for marker in memory_markers):
                    has_any_knowledge_event = bool(event.knowledge_gains or event.revelations)
                    if not has_any_knowledge_event:
                        issues.append(
                            self._issue(
                                "medium",
                                "memory_claim_without_knowledge_event",
                                "Event implies memory/knowledge, but no knowledge_gains or revelations were extracted.",
                                {
                                    "event_id": event.event_id,
                                    "summary": event.summary,
                                },
                            )
                        )

        # 11. Causality cycle
        for cycle in lore.cycle_report():
            issues.append(
                self._issue(
                    "high",
                    "causality_cycle",
                    "A cycle was found in the event causality graph.",
                    {"cycle": cycle},
                )
            )

        return issues
