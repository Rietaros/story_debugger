# extract_llm.py
from __future__ import annotations

import json
import os

from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_MODEL
from schemas import ChapterExtraction


os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


class LLMExtractor:
    def __init__(
        self,
        model_name: str | None = None,
        max_input_chars: int = 12000,
    ):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model_name or OPENAI_MODEL
        self.max_input_chars = max_input_chars

    def extract(
        self,
        chapter_id: str,
        title: str,
        chapter_text: str,
    ) -> ChapterExtraction:
        system_prompt = """
You are a strict literary information extraction engine.

Return ONLY valid JSON.
No markdown.
No explanation.
No extra text.

Your highest priority:
Extract characters accurately.

A CHARACTER is:
- a named person
- a named being
- a god/demon/monster/dragon/spirit with agency
- an AI/persona/manas that can speak, decide, advise, or act
- the first-person narrator if identity is inferable

A CHARACTER is NOT:
- a skill name
- weapon name
- magic name
- place name
- chapter title word
- generic role/title alone
- sentence starter
- abstract concept
- organization unless acting as a person
""".strip()

        user_prompt = f"""
Return ONLY valid JSON with exactly this structure:

{{
  "chapter_id": "{chapter_id}",
  "title": "{title}",
  "synopsis": "short summary of the chapter",
  "characters": [
    {{
      "canonical": "character canonical name",
      "aliases": ["alias 1", "alias 2"]
    }}
  ],
  "scenes": [
    {{
      "scene_id": "sc_001",
      "chapter_id": "{chapter_id}",
      "title": "scene title",
      "location": "location name or Unknown",
      "pov": "canonical character name or null",
      "mood": "scene mood",
      "summary": "what happens in this scene",
      "text": "short compressed scene text",
      "events": [
        {{
          "event_id": "ev_001_001",
          "chapter_id": "{chapter_id}",
          "scene_id": "sc_001",
          "seq": 1,
          "location": "location name or Unknown",
          "start_time": null,
          "end_time": null,
          "participants": ["canonical character names involved in this event"],
          "acquired_items": [],
          "used_items": [],
          "revelations": [],
          "knowledge_gains": {{}},
          "causal_parents": [],
          "summary": "event summary"
        }}
      ]
    }}
  ]
}}

Character extraction instructions:
1. Extract ALL real characters mentioned in the chapter.
2. Include characters from narration, dialogue, POV, memory, combat, and internal conversation.
3. Include characters even if they only appear once.
4. Include non-human agents if they speak, think, decide, advise, or act.
5. Normalize honorifics:
   - "Ciel-san" -> canonical "Ciel", alias "Ciel-san"
   - "Veldora-san" -> canonical "Veldora", alias "Veldora-san"
   - "Tuan Rimuru" -> canonical "Rimuru", alias "Tuan Rimuru"
6. If first-person "Aku/aku/Saya/saya" clearly refers to a named narrator, add those as aliases.
7. Do NOT include generic capitalized words.
8. Do NOT include skills, weapons, abilities, techniques, places, titles, or concepts as characters.
9. Every event participant must use canonical character names from characters list.
10. If a character acts in an event, include them in participants.
11. Use "summary", not "description".
12. Events must be objects, not strings.
13. Do not add extra fields.

Before returning JSON, silently verify:
- characters contains only true agents
- participants are characters, not items or concepts
- aliases are not duplicated as separate canonical characters

Chapter text:
{chapter_text[: self.max_input_chars]}
""".strip()

        response = self.client.responses.create(
            model=self.model_name,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        text = response.output_text
        text = text.replace("```json", "").replace("```", "").strip()

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:
            raise ValueError(f"No JSON found in model output:\n{text}")

        data = json.loads(text[start : end + 1])

        data["chapter_id"] = chapter_id
        data["title"] = title
        data.pop("text", None)
        data.setdefault("synopsis", "")
        data.setdefault("characters", [])
        data.setdefault("scenes", [])

        fixed_characters = []

        for character in data.get("characters", []):
            if isinstance(character, str):
                fixed_characters.append(
                    {
                        "canonical": character,
                        "aliases": [],
                    }
                )

            elif isinstance(character, dict):
                canonical = (
                    character.get("canonical")
                    or character.get("name")
                    or character.get("character")
                )

                if canonical:
                    fixed_characters.append(
                        {
                            "canonical": canonical,
                            "aliases": character.get("aliases", []),
                        }
                    )

        data["characters"] = fixed_characters

        for scene_idx, scene in enumerate(data.get("scenes", []), start=1):
            scene.setdefault("scene_id", f"sc_{scene_idx:03d}")
            scene.setdefault("chapter_id", chapter_id)
            scene.setdefault("title", f"Scene {scene_idx}")
            scene.setdefault("location", "Unknown")
            scene.setdefault("pov", None)
            scene.setdefault("mood", "")
            scene.setdefault("summary", "")
            scene.setdefault("text", "")
            scene.setdefault("events", [])

            fixed_events = []

            for event_idx, event in enumerate(scene.get("events", []), start=1):
                if isinstance(event, str):
                    event = {
                        "event_id": f"ev_{scene_idx:03d}_{event_idx:03d}",
                        "chapter_id": chapter_id,
                        "scene_id": scene["scene_id"],
                        "seq": event_idx,
                        "location": scene.get("location", "Unknown"),
                        "start_time": None,
                        "end_time": None,
                        "participants": [],
                        "acquired_items": [],
                        "used_items": [],
                        "revelations": [],
                        "knowledge_gains": {},
                        "causal_parents": [],
                        "summary": event,
                    }

                elif isinstance(event, dict):
                    event.setdefault("event_id", f"ev_{scene_idx:03d}_{event_idx:03d}")
                    event.setdefault("chapter_id", chapter_id)
                    event.setdefault("scene_id", scene["scene_id"])
                    event.setdefault("seq", event_idx)
                    event.setdefault("location", scene.get("location", "Unknown"))

                else:
                    continue

                allowed_event_keys = {
                    "event_id",
                    "chapter_id",
                    "scene_id",
                    "seq",
                    "location",
                    "start_time",
                    "end_time",
                    "participants",
                    "acquired_items",
                    "used_items",
                    "revelations",
                    "knowledge_gains",
                    "causal_parents",
                    "summary",
                }

                if "description" in event and "summary" not in event:
                    event["summary"] = event.pop("description")

                if "details" in event and "summary" not in event:
                    details = event.pop("details")
                    event["summary"] = (
                        details.get("info", "")
                        if isinstance(details, dict)
                        else str(details)
                    )

                event = {k: v for k, v in event.items() if k in allowed_event_keys}

                event.setdefault("summary", "")
                event.setdefault("participants", [])
                event.setdefault("acquired_items", [])
                event.setdefault("used_items", [])
                event.setdefault("revelations", [])
                event.setdefault("knowledge_gains", {})
                event.setdefault("causal_parents", [])
                event.setdefault("start_time", None)
                event.setdefault("end_time", None)

                fixed_events.append(event)

            scene["events"] = fixed_events

        return ChapterExtraction.model_validate(data)