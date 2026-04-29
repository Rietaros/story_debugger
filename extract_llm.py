# extract_llm.py
from __future__ import annotations

from openai import OpenAI
import os
from config import OPENAI_API_KEY, OPENAI_MODEL
import json

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

client = OpenAI()
model_name = OPENAI_MODEL

from schemas import ChapterExtraction



class LLMExtractor:
    def __init__(
        self,
        model_name: str | None = None,
        max_input_chars: int = 12000,
    ):
        self.client = OpenAI()
        self.model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.max_input_chars = max_input_chars

    def extract(self, chapter_id: str, title: str, chapter_text: str) -> ChapterExtraction:
        system_prompt = """
    Return ONLY valid JSON.
    No explanation.
    No markdown.
    """.strip()

        user_prompt = f"""
Return ONLY valid JSON with exactly this structure:

{{
  "chapter_id": "{chapter_id}",
  "title": "{title}",
  "synopsis": "short summary of the chapter",
  "characters": [],
  "scenes": [
    {{
      "scene_id": "sc_001",
      "chapter_id": "{chapter_id}",
      "title": "scene title",
      "location": "location name",
      "pov": "character name or null",
      "mood": "scene mood",
      "summary": "what happens in this scene",
      "text": "short compressed scene text",
      "events": []
    }}
  ]
}}

Use "summary", not "description", for events.
Events must be objects, not strings.
Do not add extra fields.
Do not include markdown.
Do not include explanation.

Chapter text:
{chapter_text[:8000]}
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

        json_text = text[start:end + 1]
        data = json.loads(json_text)

        data["chapter_id"] = chapter_id
        data["title"] = title

        data.pop("text", None)
        data.setdefault("synopsis", "")
        data.setdefault("characters", [])
        data.setdefault("scenes", [])

        fixed_characters = []
        
        for character in data.get("characters", []):
            if isinstance(character, str):
                fixed_characters.append({
                    "canonical": character,
                    "aliases": []
                })
        
            elif isinstance(character, dict):
                canonical = (
                    character.get("canonical")
                    or character.get("name")
                    or character.get("character")
                )
        
                if canonical:
                    fixed_characters.append({
                        "canonical": canonical,
                        "aliases": character.get("aliases", [])
                    })
        
        data["characters"] = fixed_characters
        
        for scene_idx, scene in enumerate(data.get("scenes", []), start=1):
            scene.setdefault("scene_id", f"sc_{scene_idx:03d}")
            scene.setdefault("chapter_id", chapter_id)
            scene.setdefault("location", "Unknown")
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
        
                # ADD IT HERE
                ALLOWED_EVENT_KEYS = {
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
                    if isinstance(details, dict):
                        event["summary"] = details.get("info", "")
                    else:
                        event["summary"] = str(details)
        
                event = {k: v for k, v in event.items() if k in ALLOWED_EVENT_KEYS}
        
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