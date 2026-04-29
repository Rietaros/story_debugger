# rules.py
from __future__ import annotations

from collections import defaultdict
from schemas import ChapterExtraction, Issue
from lore_graph import LoreGraph


class PlotHoleDetector:
    def __init__(self) -> None:
        self.inventory: dict[str, set[str]] = defaultdict(set)
        self.item_owner: dict[str, str] = {}
        self.knowledge: dict[str, set[str]] = defaultdict(set)
        self.position_by_time: dict[tuple[str, str, int], str] = {}
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
                    previous_location = self.position_by_time.get(key)

                    if previous_location and previous_location != current_location:
                        issues.append(
                            self._issue(
                                "high",
                                "double_location_same_step",
                                f"{participant} appears in two locations at the same story step.",
                                {
                                    "character": participant,
                                    "chapter_id": chapter.chapter_id,
                                    "seq": event.seq,
                                    "previous_location": previous_location,
                                    "current_location": current_location,
                                },
                            )
                        )

                    self.position_by_time[key] = current_location

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