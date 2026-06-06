# lore_graph.py
from __future__ import annotations

from collections import defaultdict
import re
import networkx as nx
from schemas import ChapterExtraction


class LoreGraph:
    def __init__(self) -> None:
        self.relations = nx.MultiDiGraph()
        self.causality = nx.DiGraph()
        self._order = 0

        self.item_owner: dict[str, str] = {}
        self.item_owners: dict[str, set[str]] = defaultdict(set)
        self.item_history: dict[str, list[dict]] = defaultdict(list)
        self.character_locations: dict[str, list[dict]] = defaultdict(list)
        self.character_knowledge: dict[str, set[str]] = defaultdict(set)
        self.event_index: dict[str, dict] = {}
        self.event_history: dict[str, list[dict]] = defaultdict(list)
        self.event_lookup: dict[tuple[str, str, str, int], dict] = {}
        self._pending_causal_edges: list[dict] = []

        # More sensitive audit memory
        self.action_log: list[dict] = []
        self.location_transitions: dict[str, list[dict]] = defaultdict(list)
        self.fact_history: dict[str, list[dict]] = defaultdict(list)
        self.character_mentions: dict[str, list[dict]] = defaultdict(list)
        self.suspicious_events: list[dict] = []

    def _log_action(self, kind: str, **data) -> None:
        row = {"order": self._order, "kind": kind}
        row.update(data)
        self.action_log.append(row)

    def _event_key(
        self,
        chapter_id: str,
        scene_id: str,
        event_id: str,
        seq: int,
    ) -> str:
        return f"{chapter_id}:{scene_id}:{event_id}:{seq}"

    def _event_label(self, record: dict) -> str:
        return (
            f"{record.get('chapter_id')}:{record.get('scene_id')}:"
            f"{record.get('event_id')}"
        )

    def resolve_event_reference(
        self,
        reference: str,
        *,
        before_order: int | None = None,
        current_chapter_id: str | None = None,
    ) -> dict | None:
        candidates: list[dict] = []

        if reference in self.event_index:
            candidates.append(self.event_index[reference])

        candidates.extend(self.event_history.get(reference, []))

        if before_order is not None:
            candidates = [
                candidate
                for candidate in candidates
                if candidate.get("order", 0) < before_order
            ]

        if not candidates:
            return None

        if current_chapter_id:
            chapter_candidates = [
                candidate
                for candidate in candidates
                if candidate.get("chapter_id") == current_chapter_id
            ]
            if chapter_candidates:
                candidates = chapter_candidates

        return max(candidates, key=lambda candidate: candidate.get("order", 0))

    def event_record(
        self,
        chapter_id: str,
        scene_id: str,
        event_id: str,
        seq: int,
    ) -> dict | None:
        return self.event_lookup.get((chapter_id, scene_id, event_id, seq))

    def _resolve_pending_causal_edges(self) -> None:
        still_pending = []

        for pending in self._pending_causal_edges:
            parent = self.resolve_event_reference(
                pending["parent_ref"],
                current_chapter_id=pending.get("current_chapter_id"),
            )
            if not parent:
                still_pending.append(pending)
                continue

            self.causality.add_edge(
                parent["event_key"],
                pending["child_key"],
                relation="EXPLICIT_CAUSE",
                parent_ref=pending["parent_ref"],
            )

            placeholder = pending.get("placeholder_key")
            if placeholder and self.causality.has_edge(placeholder, pending["child_key"]):
                self.causality.remove_edge(placeholder, pending["child_key"])

        self._pending_causal_edges = still_pending

    def _extract_sensitive_markers(self, summary: str) -> list[str]:
        text = (summary or "").lower()

        markers = {
            "possible_contradiction": [
                "tidak mungkin", "mustahil", "aneh", "padahal", "tapi tadi",
                "bukankah", "seharusnya", "tidak pernah", "impossible",
                "but earlier", "i thought", "never happened",
            ],
            "memory_claim": [
                "ingat", "mengingat", "pernah", "sudah terjadi", "berkali-kali",
                "remember", "memory", "happened before",
            ],
            "knowledge_claim": [
                "tahu", "mengetahui", "rahasia", "mengerti", "menyadari",
                "know", "secret", "realize", "understand",
            ],
            "movement_claim": [
                "pergi", "berjalan", "berlari", "tiba", "muncul", "menghilang",
                "teleport", "arrived", "appeared", "vanished",
            ],
            "item_claim": [
                "mengambil", "membawa", "memegang", "menggunakan", "memberikan",
                "hilang", "dicuri", "finds", "takes", "uses", "gives", "lost", "stolen",
            ],
        }

        found = []
        for label, words in markers.items():
            if any(word in text for word in words):
                found.append(label)

        return found

    def ingest(self, chapter: ChapterExtraction) -> None:
        self.relations.add_node(chapter.chapter_id, type="chapter", title=chapter.title)

        for char in chapter.characters:
            self.relations.add_node(
                char.canonical,
                type="character",
                aliases=char.aliases,
            )

        for scene in chapter.scenes:
            scene_node = f"scene:{scene.scene_id}"
            self.relations.add_node(
                scene_node,
                type="scene",
                chapter_id=chapter.chapter_id,
                title=scene.title,
                location=scene.location,
                pov=scene.pov,
                mood=scene.mood,
                summary=getattr(scene, "summary", ""),
            )
            self.relations.add_edge(chapter.chapter_id, scene_node, relation="HAS_SCENE")

            if scene.location:
                self.relations.add_node(scene.location, type="location")
                self.relations.add_edge(scene_node, scene.location, relation="SET_IN")

            previous_event_id = None

            for event in sorted(scene.events, key=lambda e: e.seq):
                self._order += 1

                event_key = self._event_key(
                    chapter.chapter_id,
                    scene.scene_id,
                    event.event_id,
                    event.seq,
                )
                event_node = f"event:{event_key}"
                location = event.location or scene.location or "Unknown"
                markers = self._extract_sensitive_markers(event.summary)

                event_record = {
                    "order": self._order,
                    "event_key": event_key,
                    "chapter_id": chapter.chapter_id,
                    "scene_id": scene.scene_id,
                    "scene_title": scene.title,
                    "seq": event.seq,
                    "event_id": event.event_id,
                    "title": event.title,
                    "location": location,
                    "summary": event.summary,
                    "participants": list(event.participants),
                    "acquired_items": list(event.acquired_items),
                    "used_items": list(event.used_items),
                    "revelations": list(event.revelations),
                    "knowledge_gains": dict(event.knowledge_gains),
                    "causal_parents": list(event.causal_parents),
                    "markers": markers,
                }
                self.event_index[event_key] = event_record
                self.event_index[event.event_id] = event_record
                self.event_history[event.event_id].append(event_record)
                self.event_lookup[
                    (chapter.chapter_id, scene.scene_id, event.event_id, event.seq)
                ] = event_record

                self.relations.add_node(
                    event_node,
                    type="event",
                    order=self._order,
                    event_key=event_key,
                    event_id=event.event_id,
                    chapter_id=chapter.chapter_id,
                    scene_id=scene.scene_id,
                    scene_title=scene.title,
                    seq=event.seq,
                    title=event.title,
                    summary=event.summary,
                    location=location,
                    start_time=event.start_time,
                    end_time=event.end_time,
                    markers=markers,
                )
                self.relations.add_edge(scene_node, event_node, relation="HAS_EVENT")

                self._log_action(
                    "EVENT",
                    event_id=event.event_id,
                    event_title=event.title,
                    chapter_id=chapter.chapter_id,
                    scene_id=scene.scene_id,
                    scene_title=scene.title,
                    seq=event.seq,
                    location=location,
                    summary=event.summary,
                    markers=markers,
                )

                if markers:
                    self.suspicious_events.append(
                        {
                            "order": self._order,
                            "event_id": event.event_id,
                            "event_title": event.title,
                            "chapter_id": chapter.chapter_id,
                            "scene_id": scene.scene_id,
                            "scene_title": scene.title,
                            "markers": markers,
                            "summary": event.summary,
                        }
                )

                # Causality graph
                self.causality.add_node(
                    event_key,
                    order=self._order,
                    event_id=event.event_id,
                    chapter_id=chapter.chapter_id,
                    scene_id=scene.scene_id,
                    title=event.title,
                )

                # Add explicit causal parents
                for parent in event.causal_parents:
                    parent_record = self.resolve_event_reference(
                        parent,
                        before_order=self._order,
                        current_chapter_id=chapter.chapter_id,
                    )
                    if parent_record:
                        self.causality.add_edge(
                            parent_record["event_key"],
                            event_key,
                            relation="EXPLICIT_CAUSE",
                            parent_ref=parent,
                        )
                    else:
                        placeholder_key = f"unresolved:{parent}"
                        self.causality.add_node(
                            placeholder_key,
                            order=0,
                            event_id=parent,
                            unresolved=True,
                        )
                        self.causality.add_edge(
                            placeholder_key,
                            event_key,
                            relation="UNRESOLVED_EXPLICIT_CAUSE",
                            parent_ref=parent,
                        )
                        self._pending_causal_edges.append(
                            {
                                "parent_ref": parent,
                                "child_key": event_key,
                                "placeholder_key": placeholder_key,
                                "current_chapter_id": chapter.chapter_id,
                            }
                        )

                # Add soft sequence causality inside same scene
                if previous_event_id:
                    self.causality.add_edge(previous_event_id, event_key, relation="SCENE_SEQUENCE")

                previous_event_id = event_key

                # Character presence and movement
                for participant in event.participants:
                    self.relations.add_node(participant, type="character")

                    self.relations.add_edge(
                        participant,
                        event_node,
                        relation="PARTICIPATES_IN",
                        order=self._order,
                    )

                    self.relations.add_edge(
                        participant,
                        location,
                        relation="AT_ORDER",
                        order=self._order,
                        chapter_id=chapter.chapter_id,
                        scene_id=scene.scene_id,
                        event_id=event.event_id,
                        seq=event.seq,
                    )

                    previous_location = (
                        self.character_locations[participant][-1]["location"]
                        if self.character_locations[participant]
                        else None
                    )

                    movement_record = {
                        "order": self._order,
                        "chapter_id": chapter.chapter_id,
                        "scene_id": scene.scene_id,
                        "scene_title": scene.title,
                        "event_id": event.event_id,
                        "event_title": event.title,
                        "seq": event.seq,
                        "from_location": previous_location,
                        "to_location": location,
                    }

                    self.location_transitions[participant].append(movement_record)

                    self.character_locations[participant].append(
                        {
                            "order": self._order,
                            "chapter_id": chapter.chapter_id,
                            "scene_id": scene.scene_id,
                            "scene_title": scene.title,
                            "event_id": event.event_id,
                            "event_title": event.title,
                            "seq": event.seq,
                            "location": location,
                        }
                    )

                    self.character_mentions[participant].append(
                        {
                            "order": self._order,
                            "chapter_id": chapter.chapter_id,
                            "scene_id": scene.scene_id,
                            "scene_title": scene.title,
                            "event_id": event.event_id,
                            "event_title": event.title,
                            "summary": event.summary,
                        }
                    )

                    self._log_action(
                        "CHARACTER_PRESENT",
                        character=participant,
                        location=location,
                        event_id=event.event_id,
                    )

                # Item acquisition
                for item in event.acquired_items:
                    self.relations.add_node(item, type="item")

                    for participant in event.participants:
                        previous_owner = self.item_owner.get(item)
                        previous_owners = sorted(self.item_owners[item])
                        self.item_owners[item].add(participant)
                        self.item_owner[item] = participant

                        record = {
                            "order": self._order,
                            "event_id": event.event_id,
                            "event_title": event.title,
                            "chapter_id": chapter.chapter_id,
                            "scene_id": scene.scene_id,
                            "scene_title": scene.title,
                            "seq": event.seq,
                            "action": "ACQUIRED",
                            "item": item,
                            "owner": participant,
                            "previous_owner": previous_owner,
                            "previous_owners": previous_owners,
                            "owners_after": sorted(self.item_owners[item]),
                        }

                        self.item_history[item].append(record)
                        self._log_action("ITEM_ACQUIRED", **record)

                        self.relations.add_edge(
                            participant,
                            item,
                            relation="OWNS",
                            from_order=self._order,
                            event_id=event.event_id,
                        )

                # Item usage
                for item in event.used_items:
                    self.relations.add_node(item, type="item")

                    owner = self.item_owner.get(item)
                    owners = sorted(self.item_owners[item])

                    record = {
                        "order": self._order,
                        "event_id": event.event_id,
                        "event_title": event.title,
                        "chapter_id": chapter.chapter_id,
                        "scene_id": scene.scene_id,
                        "scene_title": scene.title,
                        "seq": event.seq,
                        "action": "USED",
                        "item": item,
                        "users": list(event.participants),
                        "known_owner": owner,
                        "known_owners": owners,
                    }

                    self.item_history[item].append(record)
                    self._log_action("ITEM_USED", **record)

                    self.relations.add_edge(
                        event_node,
                        item,
                        relation="USES",
                        order=self._order,
                        known_owner=owner,
                    )

                    for participant in event.participants:
                        self.relations.add_edge(
                            participant,
                            item,
                            relation="USES_ITEM",
                            order=self._order,
                            event_id=event.event_id,
                        )

                # Knowledge gains
                for character, facts in event.knowledge_gains.items():
                    self.relations.add_node(character, type="character")

                    for fact in facts:
                        self.character_knowledge[character].add(fact)

                        record = {
                            "order": self._order,
                            "event_id": event.event_id,
                            "event_title": event.title,
                            "chapter_id": chapter.chapter_id,
                            "scene_id": scene.scene_id,
                            "scene_title": scene.title,
                            "seq": event.seq,
                            "character": character,
                            "fact": fact,
                            "action": "LEARNS",
                        }

                        self.fact_history[fact].append(record)
                        self._log_action("KNOWLEDGE_GAINED", **record)

                        self.relations.add_node(fact, type="fact")
                        self.relations.add_edge(
                            character,
                            fact,
                            relation="KNOWS",
                            known_from_order=self._order,
                            event_id=event.event_id,
                        )

                # Revelations
                for fact in event.revelations:
                    self.relations.add_node(fact, type="fact")
                    self.fact_history[fact].append(
                        {
                            "order": self._order,
                            "event_id": event.event_id,
                            "event_title": event.title,
                            "chapter_id": chapter.chapter_id,
                            "scene_id": scene.scene_id,
                            "scene_title": scene.title,
                            "seq": event.seq,
                            "action": "REVEALED",
                        }
                    )

                    self._log_action(
                        "FACT_REVEALED",
                        fact=fact,
                        event_id=event.event_id,
                        chapter_id=chapter.chapter_id,
                    )

                    self.relations.add_edge(
                        event_node,
                        fact,
                        relation="REVEALS",
                        order=self._order,
                    )

        self._resolve_pending_causal_edges()

    def owner_of(self, item: str) -> str | None:
        return self.item_owner.get(item)

    def owners_of(self, item: str) -> list[str]:
        return sorted(self.item_owners.get(item, set()))

    def item_events(self, item: str) -> list[dict]:
        return self.item_history.get(item, [])

    def locations_of(self, character: str) -> list[dict]:
        return self.character_locations.get(character, [])

    def transitions_of(self, character: str) -> list[dict]:
        return self.location_transitions.get(character, [])

    def facts_known_by(self, character: str, until_order: int | None = None) -> list[str]:
        facts = set()

        for _, target, _, data in self.relations.out_edges(character, keys=True, data=True):
            if data.get("relation") != "KNOWS":
                continue

            known_from = int(data.get("known_from_order", 0))
            if until_order is None or known_from <= until_order:
                facts.add(target)

        return sorted(facts)

    def suspicious_event_report(self) -> list[dict]:
        return self.suspicious_events

    def action_report(self) -> list[dict]:
        return self.action_log

    def cycle_report(self) -> list[list[str]]:
        cycles = []
        for cycle in nx.simple_cycles(self.causality):
            if any(self.causality.nodes[node].get("unresolved") for node in cycle):
                continue
            cycles.append(
                [
                    self._event_label(self.event_index.get(node, {"event_id": node}))
                    for node in cycle
                ]
            )
        return cycles

    def narrative_order(self) -> list[str]:
        return list(nx.topological_sort(self.causality))
