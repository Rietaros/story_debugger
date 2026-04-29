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
        self.item_history: dict[str, list[dict]] = defaultdict(list)
        self.character_locations: dict[str, list[dict]] = defaultdict(list)
        self.character_knowledge: dict[str, set[str]] = defaultdict(set)
        self.event_index: dict[str, dict] = {}

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

                event_node = f"event:{event.event_id}"
                location = event.location or scene.location or "Unknown"
                markers = self._extract_sensitive_markers(event.summary)

                self.event_index[event.event_id] = {
                    "order": self._order,
                    "chapter_id": chapter.chapter_id,
                    "scene_id": scene.scene_id,
                    "seq": event.seq,
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

                self.relations.add_node(
                    event_node,
                    type="event",
                    order=self._order,
                    chapter_id=chapter.chapter_id,
                    scene_id=scene.scene_id,
                    seq=event.seq,
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
                    chapter_id=chapter.chapter_id,
                    scene_id=scene.scene_id,
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
                            "chapter_id": chapter.chapter_id,
                            "markers": markers,
                            "summary": event.summary,
                        }
                    )

                # Causality graph
                self.causality.add_node(event.event_id, order=self._order)

                # Add explicit causal parents
                for parent in event.causal_parents:
                    self.causality.add_edge(parent, event.event_id, relation="EXPLICIT_CAUSE")

                # Add soft sequence causality inside same scene
                if previous_event_id:
                    self.causality.add_edge(previous_event_id, event.event_id, relation="SCENE_SEQUENCE")

                previous_event_id = event.event_id

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
                        "event_id": event.event_id,
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
                            "event_id": event.event_id,
                            "seq": event.seq,
                            "location": location,
                        }
                    )

                    self.character_mentions[participant].append(
                        {
                            "order": self._order,
                            "chapter_id": chapter.chapter_id,
                            "scene_id": scene.scene_id,
                            "event_id": event.event_id,
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
                        self.item_owner[item] = participant

                        record = {
                            "order": self._order,
                            "event_id": event.event_id,
                            "chapter_id": chapter.chapter_id,
                            "scene_id": scene.scene_id,
                            "action": "ACQUIRED",
                            "item": item,
                            "owner": participant,
                            "previous_owner": previous_owner,
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

                    record = {
                        "order": self._order,
                        "event_id": event.event_id,
                        "chapter_id": chapter.chapter_id,
                        "scene_id": scene.scene_id,
                        "action": "USED",
                        "item": item,
                        "users": list(event.participants),
                        "known_owner": owner,
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
                            "chapter_id": chapter.chapter_id,
                            "scene_id": scene.scene_id,
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
                            "chapter_id": chapter.chapter_id,
                            "scene_id": scene.scene_id,
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

    def owner_of(self, item: str) -> str | None:
        return self.item_owner.get(item)

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
        return list(nx.simple_cycles(self.causality))

    def narrative_order(self) -> list[str]:
        return list(nx.topological_sort(self.causality))