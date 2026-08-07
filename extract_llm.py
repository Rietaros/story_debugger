# extract_llm.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_MODEL
from schemas import ChapterExtraction, Issue, CharacterSheetItem, LLMAnalysisResult


EXTRACTION_CACHE_VERSION = "extract-v2"
ANALYSIS_CACHE_VERSION = "analysis-v2-compact"
DEFAULT_CACHE_ROOT = Path(".cache/story_debugger")
DEFAULT_ANALYSIS_EXCERPT_CHARS = 3500


GENERIC_TITLE_RE = re.compile(
    r"^(?:sc|scene|event|ev)[\s_:#-]*\d+(?:[\s_:#-]*\d+)?$",
    re.IGNORECASE,
)


def _compact_title(value: str | None, max_chars: int = 72) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" -:;,.\"'")
    if not text:
        return ""

    sentence_match = re.search(r"(.+?[.!?])(?:\s|$)", text)
    if sentence_match:
        sentence = sentence_match.group(1).strip()
        if 12 <= len(sentence) <= max_chars:
            return sentence

    if len(text) <= max_chars:
        return text

    clipped = text[:max_chars].rsplit(" ", 1)[0].strip()
    return f"{clipped}..." if clipped else text[:max_chars]


def _is_weak_title(value: str | None) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    lowered = text.lower()
    return (
        bool(GENERIC_TITLE_RE.fullmatch(text))
        or "scene title" in lowered
        or "event title" in lowered
        or lowered.startswith("untitled")
    )


def _event_title_from_data(event: dict, event_idx: int, scene_location: str) -> str:
    if not _is_weak_title(event.get("title")):
        return _compact_title(event.get("title"))

    summary = event.get("summary") or event.get("description") or ""
    if not summary and isinstance(event.get("details"), dict):
        summary = event["details"].get("summary") or event["details"].get("info") or ""

    title = _compact_title(summary)
    if title:
        return title

    participants = ", ".join(event.get("participants") or [])
    location = event.get("location") or scene_location or "Unknown"
    if participants and location != "Unknown":
        return _compact_title(f"{participants} at {location}")
    if participants:
        return _compact_title(f"{participants} story beat")
    return f"Event {event_idx}"


def _scene_title_from_data(scene: dict, scene_idx: int) -> str:
    if not _is_weak_title(scene.get("title")):
        return _compact_title(scene.get("title"))

    title = _compact_title(scene.get("summary") or scene.get("text"))
    if title:
        return title

    location = scene.get("location")
    if location and location != "Unknown":
        return _compact_title(f"Scene at {location}")
    return f"Scene {scene_idx}"


def _chapter_text_hash(chapter_text: str) -> str:
    return hashlib.sha256(chapter_text.encode("utf-8")).hexdigest()


def _cache_key(*parts: object) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_cache(path: Path) -> dict | None:
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache(path: Path, data: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        # Cache writes should never block the analysis pipeline.
        return


def _require_openai_api_key() -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Add it to .env or export it before "
            "running LLM extraction/analysis."
        )
    return OPENAI_API_KEY


def _truncate_text(value: object, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rsplit(" ", 1)[0].strip()
    return f"{clipped}..." if clipped else text[:max_chars]


def _list_preview(values: list[str], limit: int = 12, item_chars: int = 90) -> list[str]:
    return [_truncate_text(value, item_chars) for value in (values or [])[:limit]]


def _compact_extraction_for_analysis(extraction: ChapterExtraction) -> dict:
    """Summarize extraction for downstream LLM calls without carrying scene text."""
    return {
        "chapter_id": extraction.chapter_id,
        "title": extraction.title,
        "synopsis": _truncate_text(extraction.synopsis, 900),
        "characters": [character.canonical for character in extraction.characters],
        "scenes": [
            {
                "scene_id": scene.scene_id,
                "title": _truncate_text(scene.title, 120),
                "location": scene.location,
                "pov": scene.pov,
                "mood": scene.mood,
                "summary": _truncate_text(scene.summary, 500),
                "events": [
                    {
                        "event_id": event.event_id,
                        "title": _truncate_text(event.title, 120),
                        "seq": event.seq,
                        "location": event.location or scene.location or "Unknown",
                        "participants": _list_preview(event.participants),
                        "acquired_items": _list_preview(event.acquired_items),
                        "used_items": _list_preview(event.used_items),
                        "revelations": _list_preview(event.revelations, limit=8, item_chars=120),
                        "knowledge_gains": {
                            character: _list_preview(facts, limit=8, item_chars=120)
                            for character, facts in event.knowledge_gains.items()
                        },
                        "causal_parents": _list_preview(event.causal_parents, limit=8),
                        "summary": _truncate_text(event.summary, 500),
                    }
                    for event in sorted(scene.events, key=lambda item: item.seq)
                ],
            }
            for scene in extraction.scenes
        ],
    }


def _normalize_extraction_data(data: dict, chapter_id: str, title: str) -> dict:
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
        scene.setdefault("location", "Unknown")
        scene.setdefault("pov", None)
        scene.setdefault("mood", "")
        scene.setdefault("summary", "")
        scene.setdefault("text", "")
        scene.setdefault("events", [])
        scene["title"] = _scene_title_from_data(scene, scene_idx)

        fixed_events = []

        for event_idx, event in enumerate(scene.get("events", []), start=1):
            if isinstance(event, str):
                event = {
                    "event_id": f"ev_{scene_idx:03d}_{event_idx:03d}",
                    "chapter_id": chapter_id,
                    "scene_id": scene["scene_id"],
                    "seq": event_idx,
                    "title": _compact_title(event),
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
                event["title"] = _event_title_from_data(
                    event,
                    event_idx,
                    scene.get("location", "Unknown"),
                )

            else:
                continue

            allowed_event_keys = {
                "event_id",
                "chapter_id",
                "scene_id",
                "seq",
                "title",
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

    return data


class LLMExtractor:
    def __init__(
        self,
        model_name: str | None = None,
        max_input_chars: int = 12000,
        cache_dir: str | Path | None = DEFAULT_CACHE_ROOT / "extractions",
        use_cache: bool = True,
    ):
        self.client = OpenAI(api_key=_require_openai_api_key())
        self.model_name = model_name or OPENAI_MODEL
        self.max_input_chars = max_input_chars
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache

    def _cache_path(self, chapter_id: str, title: str, chapter_text: str) -> Path | None:
        if not self.cache_dir or not self.use_cache:
            return None

        key = _cache_key(
            EXTRACTION_CACHE_VERSION,
            self.model_name,
            self.max_input_chars,
            chapter_id,
            title,
            _chapter_text_hash(chapter_text),
        )
        return self.cache_dir / f"{key}.json"

    def extract(
        self,
        chapter_id: str,
        title: str,
        chapter_text: str,
    ) -> ChapterExtraction:
        cache_path = self._cache_path(chapter_id, title, chapter_text)
        cached_data = _load_cache(cache_path) if cache_path else None
        if cached_data:
            try:
                data = _normalize_extraction_data(cached_data, chapter_id, title)
                return ChapterExtraction.model_validate(data)
            except Exception:
                pass

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
      "title": "scene title (a short descriptive sentence summarizing this scene)",
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
          "title": "event title (a short descriptive sentence summarizing this event)",
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
14. Scene "title" must be a short descriptive name based on scene content, never only the scene_id.
15. Event "title" must be a short descriptive name based on event content, never only the event_id.

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
            max_output_tokens=6000,
        )

        text = response.output_text
        text = text.replace("```json", "").replace("```", "").strip()

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:
            raise ValueError(f"No JSON found in model output:\n{text}")

        data = _normalize_extraction_data(json.loads(text[start : end + 1]), chapter_id, title)
        extraction = ChapterExtraction.model_validate(data)

        if cache_path:
            _write_cache(cache_path, extraction.model_dump(mode="json"))

        return extraction


class LLMStoryAnalyzer:
    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: str | Path | None = DEFAULT_CACHE_ROOT / "analysis",
        use_cache: bool = True,
        max_chapter_excerpt_chars: int = DEFAULT_ANALYSIS_EXCERPT_CHARS,
    ):
        self.client = OpenAI(api_key=_require_openai_api_key())
        self.model_name = model_name or OPENAI_MODEL
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache
        self.max_chapter_excerpt_chars = max_chapter_excerpt_chars

    def _cache_path(
        self,
        chapter_id: str,
        title: str,
        chapter_text: str,
        current_extraction: ChapterExtraction,
        history: list[dict],
    ) -> Path | None:
        if not self.cache_dir or not self.use_cache:
            return None

        key = _cache_key(
            ANALYSIS_CACHE_VERSION,
            self.model_name,
            self.max_chapter_excerpt_chars,
            chapter_id,
            title,
            _chapter_text_hash(chapter_text),
            _compact_extraction_for_analysis(current_extraction),
            history,
        )
        return self.cache_dir / f"{key}.json"

    def analyze(
        self,
        chapter_id: str,
        title: str,
        chapter_text: str,
        current_extraction: ChapterExtraction,
        history: list[dict],
    ) -> LLMAnalysisResult:
        cache_path = self._cache_path(
            chapter_id,
            title,
            chapter_text,
            current_extraction,
            history,
        )
        cached_data = _load_cache(cache_path) if cache_path else None
        if cached_data:
            try:
                return LLMAnalysisResult.model_validate(cached_data)
            except Exception:
                pass

        # Format history of previous chapters
        history_summary = ""
        if history:
            history_summary = "\n".join([
                (
                    f"- Chapter {h['chapter_id']}: "
                    f"{_truncate_text(h.get('title', ''), 80)}\n"
                    f"  Synopsis: {_truncate_text(h.get('synopsis', ''), 450)}"
                )
                for h in history[-8:]
            ])
        else:
            history_summary = "No previous chapters (this is Chapter 1)."

        # Format characters list
        characters_list = ", ".join([c.canonical for c in current_extraction.characters])
        compact_extraction = _compact_extraction_for_analysis(current_extraction)
        chapter_excerpt = _truncate_text(chapter_text, self.max_chapter_excerpt_chars)

        system_prompt = """
You are an expert narrative analysis engine for a story debugging system.
Your job is to identify plot holes and construct a character sheet for the current chapter.

You must return ONLY valid JSON matching the specified structure. No extra text, explanation, or markdown.
""".strip()

        user_prompt = f"""
Analyze this chapter of a story. The story might be written in English, Bahasa Indonesia, or a mix. Write summaries, messages, and analysis in the same language as the story.

Chapter ID: {chapter_id}
Title: {title}

--- STORY CONTEXT SO FAR (PREVIOUS CHAPTERS) ---
{history_summary}

--- CURRENT CHAPTER STRUCTURE ---
{json.dumps(compact_extraction, ensure_ascii=False)}

--- SHORT SOURCE EXCERPT FOR TONE/EVIDENCE ---
{chapter_excerpt}

--- CHARACTERS PRESENT IN THIS CHAPTER ---
{characters_list}

--- ANALYSIS INSTRUCTIONS ---
1. Detect and describe only clear plot holes in this chapter, using the structured extraction first and the source excerpt only for tone/evidence. If any plot holes are found, categorize them into one of the following rules:
   - "contradiction": A character acts completely out of line with their established personality traits, or previously known facts are suddenly altered to fit a new scene.
   - "missing_details": A vital piece of information, a key item, or a character’s injury is forgotten, conveniently disappears, or magically reappears between chapters or scenes.
   - "forgotten_subplot": A secondary character is introduced with a heavy, specific conflict (like being cursed or having a missing family member), but this thread is completely abandoned before the story ends.
   - "out_of_character": This occurs when a character acts outside of their established nature, typically for the sake of moving the plot forward.
   Each plot hole should be returned as an issue with:
   - "severity": "high" | "medium" | "low"
   - "rule": one of "contradiction", "missing_details", "forgotten_subplot", "out_of_character"
   - "message": A concise description of what is wrong and why it is a plot hole.
   - "evidence": {{"character": "name of character involved or 'General'", "description": "concise quote or summary of context"}}

2. Create a Character Sheet for EVERY character listed above in the "CHARACTERS PRESENT IN THIS CHAPTER" section. Each character sheet entry must contain:
   - "character": The canonical name of the character.
   - "narrative_summary": A concise summary of their role and presence in this chapter.
   - "current_actions": A concise description of what they do in this chapter.
   - "risk_of_plot_hole": Any action, setup, unresolved event, or decision in this chapter that could lead to a future plot hole if mishandled. If no risk, write "None".
   - "acquired_items_or_spells": A list of items, equipment, or spells they acquired in this chapter. If none, return an empty list.

Return ONLY a JSON object with this exact structure:
{{
  "plot_holes": [
    {{
      "severity": "high" | "medium" | "low",
      "rule": "contradiction" | "missing_details" | "forgotten_subplot" | "out_of_character",
      "message": "...",
      "evidence": {{"character": "...", "description": "..."}}
    }}
  ],
  "character_sheet": [
    {{
      "character": "...",
      "narrative_summary": "...",
      "current_actions": "...",
      "risk_of_plot_hole": "...",
      "acquired_items_or_spells": ["item/spell 1", "item/spell 2"]
    }}
  ]
}}
""".strip()

        response = self.client.responses.create(
            model=self.model_name,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_output_tokens=3500,
        )

        text = response.output_text
        text = text.replace("```json", "").replace("```", "").strip()

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:
            raise ValueError(f"No JSON found in model output:\n{text}")

        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as e:
            print("--- MODEL OUTPUT START ---")
            print(text[start : end + 1])
            print("--- MODEL OUTPUT END ---")
            raise e

        # Ensure correct formats and fields
        plot_holes = []
        for ph in data.get("plot_holes", []):
            plot_holes.append(
                Issue(
                    severity=ph.get("severity", "medium"),
                    rule=ph.get("rule", "contradiction"),
                    message=ph.get("message", ""),
                    evidence=ph.get("evidence", {}),
                )
            )

        character_sheet = []
        for cs in data.get("character_sheet", []):
            character_sheet.append(
                CharacterSheetItem(
                    character=cs.get("character", ""),
                    narrative_summary=cs.get("narrative_summary", ""),
                    current_actions=cs.get("current_actions") or cs.get("actions", ""),
                    risk_of_plot_hole=cs.get("risk_of_plot_hole") or cs.get("future_plot_hole_risk", "None"),
                    acquired_items_or_spells=cs.get("acquired_items_or_spells", []),
                )
            )

        result = LLMAnalysisResult(plot_holes=plot_holes, character_sheet=character_sheet)

        if cache_path:
            _write_cache(cache_path, result.model_dump(mode="json"))

        return result
