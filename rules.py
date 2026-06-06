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
        self.item_owners: dict[str, set[str]] = defaultdict(set)
        self.knowledge: dict[str, set[str]] = defaultdict(set)
        self.position_by_step: dict[tuple, dict] = {}
        self.seen_events: set[str] = set()
        self.event_summaries: dict[tuple[str, str], str] = {}

    def _issue(self, severity: str, rule: str, message: str, evidence: dict) -> Issue:
        return Issue(
            severity=severity,
            rule=rule,
            message=message,
            evidence=evidence,
        )

    def _known_location(self, location: str | None) -> bool:
        return bool(location and location.strip() and location.strip().lower() != "unknown")

    def _position_keys(
        self,
        participant: str,
        chapter_id: str,
        scene_id: str,
        event_seq: int,
        start_time: str | None,
        end_time: str | None,
    ) -> list[tuple[str, tuple]]:
        keys = [
            (
                "scene_step",
                (participant, chapter_id, scene_id, event_seq),
            )
        ]

        for label, value in (("start_time", start_time), ("end_time", end_time)):
            if value and str(value).strip().lower() not in {"unknown", "null", "none"}:
                keys.append(
                    (
                        label,
                        (participant, chapter_id, label, str(value).strip().lower()),
                    )
                )

        return keys

    def _shared_item_context(self, summary: str) -> bool:
        text = (summary or "").lower()
        markers = [
            "borrow",
            "borrowed",
            "lend",
            "lent",
            "loan",
            "share",
            "shared",
            "together",
            "as a group",
            "party",
            "hands",
            "gives",
            "receives",
            "passes",
            "pinjam",
            "meminjam",
            "dipinjam",
            "meminjamkan",
            "berbagi",
            "bersama",
            "memberikan",
            "diberikan",
            "menyerahkan",
            "menerima",
            "dipakai bersama",
        ]
        return any(marker in text for marker in markers)

    def _transfer_item_context(self, summary: str) -> bool:
        text = (summary or "").lower()
        markers = [
            "give",
            "gives",
            "gave",
            "given",
            "hand",
            "hands",
            "passes",
            "transfer",
            "receives",
            "lost",
            "stolen",
            "dropped",
            "memberikan",
            "diberikan",
            "menyerahkan",
            "menerima",
            "hilang",
            "dicuri",
            "menjatuhkan",
        ]
        return any(marker in text for marker in markers)

    def check(self, chapter: ChapterExtraction, lore: LoreGraph) -> list[Issue]:
        issues: list[Issue] = []

        for scene in chapter.scenes:
            for event in sorted(scene.events, key=lambda e: e.seq):
                current_location = event.location or scene.location or "Unknown"
                event_title = _useful_title(event.title, event.summary, event.event_id)
                scene_title = _useful_title(scene.title, scene.summary, scene.scene_id)
                current_record = (
                    lore.event_record(
                        chapter.chapter_id,
                        scene.scene_id,
                        event.event_id,
                        event.seq,
                    )
                    if lore
                    else None
                )
                current_order = current_record.get("order") if current_record else None
                current_position = {
                    "chapter_id": chapter.chapter_id,
                    "scene_id": scene.scene_id,
                    "scene_title": scene_title,
                    "event_id": event.event_id,
                    "event_title": event_title,
                    "seq": event.seq,
                    "story_order": current_order,
                    "start_time": event.start_time,
                    "end_time": event.end_time,
                    "location": current_location,
                    "summary": event.summary,
                }

                # 1. Duplicate event ID with different meaning
                event_summary_key = (chapter.chapter_id, event.event_id)
                if event_summary_key in self.event_summaries:
                    if self.event_summaries[event_summary_key] != event.summary:
                        issues.append(
                            self._issue(
                                "high",
                                "duplicate_event_id_conflict",
                                f"Event ID {event.event_id} appears with different summaries.",
                                {
                                    "event_id": event.event_id,
                                    "chapter_id": chapter.chapter_id,
                                    "old_summary": self.event_summaries[event_summary_key],
                                    "new_summary": event.summary,
                                },
                            )
                        )
                else:
                    self.event_summaries[event_summary_key] = event.summary

                # 2. Unknown causal parent. Resolve against the full lore graph
                # before falling back to local in-process state.
                for parent in event.causal_parents:
                    parent_record = (
                        lore.resolve_event_reference(
                            parent,
                            before_order=current_order,
                            current_chapter_id=chapter.chapter_id,
                        )
                        if lore and current_order is not None
                        else None
                    )
                    if parent_record or parent in self.seen_events:
                        continue

                    issues.append(
                        self._issue(
                            "medium",
                            "unknown_causal_parent",
                            f"{event.event_id} depends on a causal parent that has not occurred earlier: {parent}.",
                            {
                                "event_id": event.event_id,
                                "event_title": event_title,
                                "chapter_id": chapter.chapter_id,
                                "scene_id": scene.scene_id,
                                "seq": event.seq,
                                "story_order": current_order,
                                "parent": parent,
                            },
                        )
                    )

                # 3. Double location in same story step. Local event.seq is scoped
                # to a scene, so scene_id is part of the default key. Explicit
                # start/end times are still checked across scenes.
                for participant in event.participants:
                    for step_type, key in self._position_keys(
                        participant,
                        chapter.chapter_id,
                        scene.scene_id,
                        event.seq,
                        event.start_time,
                        event.end_time,
                    ):
                        previous_position = self.position_by_step.get(key)
                        previous_location = (
                            previous_position.get("location")
                            if previous_position
                            else None
                        )

                        if (
                            previous_location
                            and previous_location != current_location
                            and self._known_location(previous_location)
                            and self._known_location(current_location)
                        ):
                            issues.append(
                                self._issue(
                                    "high",
                                    "double_location_same_step",
                                    f"{participant} appears in two different locations at the same story step: '{previous_location}' and '{current_location}'.",
                                    {
                                        "character": participant,
                                        "chapter_id": chapter.chapter_id,
                                        "seq": event.seq,
                                        "step_type": step_type,
                                        "story_order": current_order,
                                        "previous_location": previous_location,
                                        "current_location": current_location,
                                        "previous_scene_id": previous_position.get("scene_id"),
                                        "previous_scene_title": previous_position.get("scene_title"),
                                        "previous_event_id": previous_position.get("event_id"),
                                        "previous_event_title": previous_position.get("event_title"),
                                        "previous_event_summary": previous_position.get("summary"),
                                        "previous_story_order": previous_position.get("story_order"),
                                        "current_scene_id": scene.scene_id,
                                        "current_scene_title": scene_title,
                                        "current_event_id": event.event_id,
                                        "current_event_title": event_title,
                                        "current_event_summary": event.summary,
                                    },
                                )
                            )

                        self.position_by_step[key] = current_position

                shared_context = self._shared_item_context(event.summary)
                transfer_context = self._transfer_item_context(event.summary)

                # 4. Item used before acquired. Shared or borrowed usage is allowed
                # when the context explicitly supports it.
                for participant in event.participants:
                    for item in event.used_items:
                        known_owners = set(self.item_owners.get(item, set()))
                        available_in_event = item in event.acquired_items
                        owner_in_event = bool(known_owners.intersection(event.participants))
                        participant_has_item = item in self.inventory[participant]

                        if participant_has_item or available_in_event:
                            continue

                        if known_owners or shared_context:
                            continue

                        issues.append(
                            self._issue(
                                "high",
                                "item_used_before_acquired",
                                f"{participant} uses {item} before acquiring it.",
                                {
                                    "character": participant,
                                    "item": item,
                                    "known_owners": sorted(known_owners),
                                    "event_id": event.event_id,
                                },
                            )
                        )

                # 5. Item owned by someone else but used here. Borrowed/shared
                # context and party co-presence are allowed.
                for participant in event.participants:
                    for item in event.used_items:
                        known_owners = set(self.item_owners.get(item, set()))
                        available_in_event = item in event.acquired_items
                        owner_in_event = bool(known_owners.intersection(event.participants))
                        if (
                            known_owners
                            and participant not in known_owners
                            and not available_in_event
                            and not owner_in_event
                            and not shared_context
                        ):
                            issues.append(
                                self._issue(
                                    "medium",
                                    "item_used_by_non_owner",
                                    f"{participant} uses {item}, but known owners are {', '.join(sorted(known_owners))}.",
                                    {
                                        "character": participant,
                                        "item": item,
                                        "current_owners": sorted(known_owners),
                                        "event_id": event.event_id,
                                    },
                                )
                            )

                # 6. Multiple acquisition / suspicious ownership transfer
                for item in event.acquired_items:
                    previous_owners = set(self.item_owners.get(item, set()))
                    new_owners = set(event.participants)

                    for participant in event.participants:
                        if (
                            previous_owners
                            and participant not in previous_owners
                            and not previous_owners.intersection(new_owners)
                            and not shared_context
                            and not transfer_context
                        ):
                            issues.append(
                                self._issue(
                                    "medium",
                                    "item_ownership_transfer_without_loss",
                                    f"{item} moves from {', '.join(sorted(previous_owners))} to {participant} without explicit loss/give/share event.",
                                    {
                                        "item": item,
                                        "previous_owners": sorted(previous_owners),
                                        "new_owner": participant,
                                        "event_id": event.event_id,
                                    },
                                )
                            )

                        self.inventory[participant].add(item)
                        self.item_owners[item].add(participant)
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

                self.seen_events.add(event.event_id)

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
