# lore_graph.py
from __future__ import annotations

from collections import defaultdict
import networkx as nx
from schemas import ChapterExtraction


class LoreGraph:
    def __init__(self) -> None:
        self.relations = nx.MultiDiGraph()
        self.causality = nx.DiGraph()
        self._order = 0

        # Strong state memory
        self.item_owner: dict[str, str] = {}
        self.item_history: dict[str, list[dict]] = defaultdict(list)
        self.character_locations: dict[str, list[dict]] = defaultdict(list)
        self.character_knowledge: dict[str, set[str]] = defaultdict(set)
        self.event_index: dict[str, dict] = {}

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
            )
            self.relations.add_edge(chapter.chapter_id, scene_node, relation="HAS_SCENE")

            if scene.location:
                self.relations.add_node(scene.location, type="location")
                self.relations.add_edge(scene_node, scene.location, relation="SET_IN")

            for event in sorted(scene.events, key=lambda e: e.seq):
                self._order += 1

                event_node = f"event:{event.event_id}"
                location = event.location or scene.location or "Unknown"

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
                )
                self.relations.add_edge(scene_node, event_node, relation="HAS_EVENT")

                self.causality.add_node(event.event_id, order=self._order)

                for parent in event.causal_parents:
                    self.causality.add_edge(parent, event.event_id)

                for participant in event.participants:
                    self.relations.add_edge(participant, event_node, relation="PARTICIPATES_IN")
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

                for item in event.acquired_items:
                    self.relations.add_node(item, type="item")

                    for participant in event.participants:
                        previous_owner = self.item_owner.get(item)
                        self.item_owner[item] = participant

                        self.item_history[item].append(
                            {
                                "order": self._order,
                                "event_id": event.event_id,
                                "chapter_id": chapter.chapter_id,
                                "action": "ACQUIRED",
                                "owner": participant,
                                "previous_owner": previous_owner,
                            }
                        )

                        self.relations.add_edge(
                            participant,
                            item,
                            relation="OWNS",
                            from_order=self._order,
                            event_id=event.event_id,
                        )

                for item in event.used_items:
                    self.relations.add_node(item, type="item")
                    self.item_history[item].append(
                        {
                            "order": self._order,
                            "event_id": event.event_id,
                            "chapter_id": chapter.chapter_id,
                            "action": "USED",
                            "users": list(event.participants),
                        }
                    )
                    self.relations.add_edge(
                        event_node,
                        item,
                        relation="USES",
                        order=self._order,
                    )

                for character, facts in event.knowledge_gains.items():
                    for fact in facts:
                        self.character_knowledge[character].add(fact)

                        self.relations.add_node(fact, type="fact")
                        self.relations.add_edge(
                            character,
                            fact,
                            relation="KNOWS",
                            known_from_order=self._order,
                            event_id=event.event_id,
                        )

                for fact in event.revelations:
                    self.relations.add_node(fact, type="fact")
                    self.relations.add_edge(
                        event_node,
                        fact,
                        relation="REVEALS",
                        order=self._order,
                    )

    def owner_of(self, item: str) -> str | None:
        return self.item_owner.get(item)

    def facts_known_by(self, character: str, until_order: int | None = None) -> list[str]:
        facts = set()

        for _, target, _, data in self.relations.out_edges(character, keys=True, data=True):
            if data.get("relation") != "KNOWS":
                continue

            known_from = int(data.get("known_from_order", 0))
            if until_order is None or known_from <= until_order:
                facts.add(target)

        return sorted(facts)

    def cycle_report(self) -> list[list[str]]:
        return list(nx.simple_cycles(self.causality))

    def narrative_order(self) -> list[str]:
        return list(nx.topological_sort(self.causality))