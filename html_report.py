# html_report.py
from __future__ import annotations

from pathlib import Path
from html import escape
from html.parser import HTMLParser
import hashlib
import json
import re

from config import OPENAI_API_KEY, OPENAI_MODEL


ARC_ANALYSIS_CACHE_VERSION = "arc-analysis-v1"
DEFAULT_CACHE_ROOT = Path(".cache/story_debugger")


WEAK_TITLE_RE = re.compile(
    r"^(?:sc|scene|event|ev)[\s_:#-]*\d+(?:[\s_:#-]*\d+)?$",
    re.IGNORECASE,
)


def safe_text(value) -> str:
    if value is None:
        return ""
    return escape(str(value))


def _cache_key(*parts: object) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_cache(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _write_cache(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError:
        return


def safe_json_for_script(data: dict) -> str:
    text = json.dumps(data, ensure_ascii=False)
    return (
        text.replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


class _ArcHTMLSanitizer(HTMLParser):
    allowed_tags = {"div", "h4", "p", "strong", "em", "br", "ul", "ol", "li"}
    allowed_classes = {"arc-analysis"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in self.allowed_tags:
            return

        safe_attrs = []
        if tag == "div":
            for name, value in attrs:
                if name == "class" and value in self.allowed_classes:
                    safe_attrs.append(f'class="{safe_text(value)}"')

        attr_text = f" {' '.join(safe_attrs)}" if safe_attrs else ""
        self.parts.append(f"<{tag}{attr_text}>")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.allowed_tags and tag != "br":
            self.parts.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        self.parts.append(safe_text(data))

    def handle_entityref(self, name: str) -> None:
        self.parts.append(safe_text(f"&{name};"))

    def handle_charref(self, name: str) -> None:
        self.parts.append(safe_text(f"&#{name};"))

    def get_html(self) -> str:
        return "".join(self.parts).strip()


def sanitize_arc_html(html: str, character: str) -> str:
    parser = _ArcHTMLSanitizer()
    parser.feed(html or "")
    sanitized = parser.get_html()
    return sanitized or fallback_character_arc_analysis(character)


def compact_label(value, max_chars: int = 78) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rsplit(" ", 1)[0].strip()
    return f"{clipped}..." if clipped else text[:max_chars]


def label_with_title(identifier: str, title: str | None, max_title_chars: int = 58) -> str:
    title = compact_label(title, max_title_chars)
    if title and title != identifier:
        return f"{identifier} · {title}"
    return identifier


def useful_title(title: str | None, fallback: str | None, identifier: str) -> str:
    text = compact_label(title)
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
    return compact_label(fallback) or identifier


def analyze_drift(drift_score: float) -> str:
    if drift_score < 0.20:
        return "Low drift. The chapter remains semantically close to the early story baseline."
    if drift_score < 0.40:
        return "Moderate drift. The chapter introduces noticeable development while still staying connected."
    return "High drift. The chapter may strongly shift tone, theme, mood, or narrative direction."


def analyze_bug(issue, bug_id: str | None = None, timeline_locations: list[dict] | None = None) -> str:
    rule = getattr(issue, "rule", "")
    severity = getattr(issue, "severity", "")
    message = getattr(issue, "message", "")
    evidence = getattr(issue, "evidence", {}) or {}
    timeline_locations = timeline_locations or []

    if rule == "duplicate_event_id_conflict":
        event_id = evidence.get("event_id", "")
        old_summary = evidence.get("old_summary", "")
        new_summary = evidence.get("new_summary", "")

        meaning = f"""
        <p><strong>Meaning:</strong> The same event ID <code>{safe_text(event_id)}</code>
        is used for different narrative moments.</p>
        <p><strong>Earlier version:</strong> {safe_text(old_summary)}</p>
        <p><strong>New version:</strong> {safe_text(new_summary)}</p>
        <p><strong>Why it matters:</strong> duplicated event IDs can corrupt causality,
        item ownership, memory tracking, and location tracking.</p>
        """

    elif rule == "contradiction":
        meaning = """
        <p><strong>Type:</strong> The Contradiction</p>
        <p><strong>Definition:</strong> A character acts completely out of line with their established personality traits, or previously known facts are suddenly altered to fit a new scene.</p>
        """

    elif rule == "missing_details":
        meaning = """
        <p><strong>Type:</strong> Missing Details</p>
        <p><strong>Definition:</strong> A vital piece of information, a key item, or a character’s injury is forgotten, conveniently disappears, or magically reappears between chapters or scenes.</p>
        """

    elif rule == "forgotten_subplot":
        meaning = """
        <p><strong>Type:</strong> Forgotten Subplots</p>
        <p><strong>Definition:</strong> A secondary character is introduced with a heavy, specific conflict (like being cursed or having a missing family member), but this thread is completely abandoned before the story ends.</p>
        """

    elif rule == "out_of_character":
        meaning = """
        <p><strong>Type:</strong> Out-of-Character Moments</p>
        <p><strong>Definition:</strong> This occurs when a character acts outside of their established nature, typically for the sake of moving the plot forward.</p>
        """

    elif rule == "double_location_same_step":
        prev_loc = evidence.get("previous_location", "Unknown")
        curr_loc = evidence.get("current_location", "Unknown")
        prev_scene_id = evidence.get("previous_scene_id", "-")
        prev_scene_title = evidence.get("previous_scene_title", "-")
        prev_event_id = evidence.get("previous_event_id", "-")
        prev_event_title = evidence.get("previous_event_title", "-")
        prev_summary = evidence.get("previous_event_summary", "-")
        curr_scene_id = evidence.get("current_scene_id", "-")
        curr_scene_title = evidence.get("current_scene_title", "-")
        curr_event_id = evidence.get("current_event_id", "-")
        curr_event_title = evidence.get("current_event_title", "-")
        curr_summary = evidence.get("current_event_summary", "-")
        meaning = f"""
        <p><strong>Type:</strong> Spatial Continuity Error</p>
        <p><strong>Conflicting Locations:</strong> <code>{safe_text(prev_loc)}</code> and <code>{safe_text(curr_loc)}</code></p>
        <p><strong>Where it appears first:</strong>
        <code>{safe_text(prev_scene_id)}</code> · {safe_text(prev_scene_title)}
        / <code>{safe_text(prev_event_id)}</code> · {safe_text(prev_event_title)}
        at <code>{safe_text(prev_loc)}</code></p>
        <p><strong>First event summary:</strong> {safe_text(prev_summary)}</p>
        <p><strong>Where it conflicts:</strong>
        <code>{safe_text(curr_scene_id)}</code> · {safe_text(curr_scene_title)}
        / <code>{safe_text(curr_event_id)}</code> · {safe_text(curr_event_title)}
        at <code>{safe_text(curr_loc)}</code></p>
        <p><strong>Conflicting event summary:</strong> {safe_text(curr_summary)}</p>
        <p><strong>Why it matters:</strong> A character cannot physically exist in two different locations at the exact same sequence point in a chapter.</p>
        """

    elif "item" in rule:
        meaning = """
        <p><strong>Meaning:</strong> This indicates item continuity risk.
        A character may be using, owning, or transferring an item without proper setup.</p>
        """

    elif "location" in rule:
        meaning = """
        <p><strong>Meaning:</strong> This indicates spatial continuity risk.
        A character may appear in conflicting places at the same story step.</p>
        """

    elif "knowledge" in rule or "memory" in rule:
        meaning = """
        <p><strong>Meaning:</strong> This indicates memory or knowledge inconsistency.
        A character may know, remember, or reference something that has not been established.</p>
        """

    elif "causal" in rule:
        meaning = """
        <p><strong>Meaning:</strong> This indicates causality or timeline risk.
        An event may depend on something that did not happen yet, or a causal loop exists.</p>
        """

    else:
        meaning = """
        <p><strong>Meaning:</strong> This may indicate a narrative consistency issue
        that needs human review.</p>
        """

    timeline_html = ""
    if timeline_locations:
        location_rows = []
        for location in timeline_locations:
            anchor = location.get("anchor")
            event_label = location.get("event_label", "-")
            scene_label = location.get("scene_label", "-")
            role = location.get("role", "event")
            event_location = location.get("location", "-")
            if anchor:
                event_html = f'<a href="#{safe_text(anchor)}">{safe_text(event_label)}</a>'
            else:
                event_html = safe_text(event_label)

            location_rows.append(
                "<li>"
                f"<strong>{safe_text(role)}:</strong> {event_html} "
                f"in {safe_text(scene_label)} at <code>{safe_text(event_location)}</code>"
                "</li>"
            )

        timeline_html = f"""
        <div class="bug-timeline-links">
            <p><strong>Appears on timeline:</strong></p>
            <ul>{"".join(location_rows)}</ul>
        </div>
        """

    return f"""
    <div class="bug-item severity-{safe_text(severity)}" data-rule="{safe_text(rule)}">
        <div class="bug-head">
            <span class="badge badge-{safe_text(severity)}">{safe_text(severity)}</span>
            <strong>{safe_text(rule)}</strong>
        </div>
        <p>{safe_text(message)}</p>
        {timeline_html}
        <div class="bug-detail">
            {meaning}
        </div>
    </div>
    """


def llm_character_arc_analysis(
    character: str,
    chapter_id: str,
    arc_rows: list[dict],
    chapter_summary: str = "",
    character_events: list[dict] | None = None,
    model_name: str | None = None,
    cache_dir: str | Path | None = DEFAULT_CACHE_ROOT / "arc_analysis",
    use_cache: bool = True,
) -> str:
    if not OPENAI_API_KEY or not arc_rows:
        return fallback_character_arc_analysis(character)

    try:
        from openai import OpenAI
    except ImportError:
        return fallback_character_arc_analysis(character) + "<br><em>LLM analysis unavailable: OpenAI package is not installed.</em>"

    client = OpenAI(api_key=OPENAI_API_KEY)
    model_name = model_name or OPENAI_MODEL
    character_events = character_events or []

    compact_arc_rows = []
    for row in arc_rows[:35]:
        compact_arc_rows.append(
            {
                "sentence_index": row.get("sentence_index"),
                "emotion": row.get("emotion"),
                "score": row.get("score"),
                "sentence": str(row.get("sentence", ""))[:220],
            }
        )

    compact_events = []
    for event in character_events[:20]:
        compact_events.append(
            {
                "event_id": event.get("event_id"),
                "scene_id": event.get("scene_id"),
                "seq": event.get("seq"),
                "location": event.get("location"),
                "summary": str(event.get("summary", ""))[:260],
                "acquired_items": event.get("acquired_items", []),
                "used_items": event.get("used_items", []),
                "revelations": event.get("revelations", []),
            }
        )

    cache_path = None
    if cache_dir and use_cache:
        cache_key = _cache_key(
            ARC_ANALYSIS_CACHE_VERSION,
            model_name,
            character,
            chapter_id,
            chapter_summary,
            compact_arc_rows,
            compact_events,
        )
        cache_path = Path(cache_dir) / f"{cache_key}.html"
        cached = _load_cache(cache_path)
        if cached:
            return sanitize_arc_html(cached, character)

    prompt = f"""
You are a narrative analyst for a story debugging system.

Analyze the character emotional arc using BOTH:
1. emotion timeline
2. what the character actually does in the chapter

Chapter ID: {chapter_id}
Character: {character}

Chapter summary:
{chapter_summary}

Character actions/events in this chapter:
{compact_events}

Emotion timeline:
{compact_arc_rows}

Return the analysis as clean HTML only.

Use this exact structure:

<div class="arc-analysis">
  <h4>Emotional Arc Summary</h4>
  <p>...</p>

  <h4>Action-Based Interpretation</h4>
  <p>...</p>

  <h4>Key Turning Point</h4>
  <p>...</p>

  <h4>Narrative Meaning</h4>
  <p>...</p>

  <h4>Potential Writing Issue</h4>
  <p>...</p>
</div>

Rules:
- Do not use markdown.
- Do not use ###.
- Do not wrap in ```html.
- Keep it concise and readable.
- Connect emotion changes to concrete story actions.

Important:
- Do not analyze emotion alone.
- Connect emotion changes to concrete story actions.
- If emotional spikes/drops happen without matching event support, mention it as possible inconsistency.
- Keep it under 220 words.
""".strip()

    try:
        response = client.responses.create(
            model=model_name,
            input=prompt,
            max_output_tokens=900,
        )
        sanitized = sanitize_arc_html(response.output_text.strip(), character)
        if cache_path:
            _write_cache(cache_path, sanitized)
        return sanitized
    except Exception as exc:
        return fallback_character_arc_analysis(character) + f"<br><em>LLM analysis unavailable: {safe_text(exc)}</em>"


def fallback_character_arc_analysis(character: str) -> str:
    return (
        f"The emotion arc for <strong>{safe_text(character)}</strong> shows how emotional "
        "intensity changes through the chapter. Spikes may indicate conflict, realization, "
        "danger, or turning points. Sudden drops may indicate trauma, contradiction, or unstable characterization."
    )


def _timeline_key(chapter_id: str, scene_id: str, event_id: str, seq: int | None) -> tuple:
    return (chapter_id or "", scene_id or "", event_id or "", seq or 0)


def _graph_timeline_maps(lore, chapter_id: str) -> dict[str, dict]:
    maps = {
        "events": {},
        "movements": {},
        "item_events": {},
        "fact_events": {},
    }

    if lore is None:
        return maps

    action_report = getattr(lore, "action_report", None)
    actions = action_report() if callable(action_report) else getattr(lore, "action_log", [])

    for action in actions:
        if action.get("chapter_id") != chapter_id:
            continue

        key = _timeline_key(
            action.get("chapter_id"),
            action.get("scene_id"),
            action.get("event_id"),
            action.get("seq"),
        )

        if action.get("kind") == "EVENT":
            maps["events"][key] = action

    for character, transitions in getattr(lore, "location_transitions", {}).items():
        for transition in transitions:
            if transition.get("chapter_id") != chapter_id:
                continue

            key = _timeline_key(
                transition.get("chapter_id"),
                transition.get("scene_id"),
                transition.get("event_id"),
                transition.get("seq"),
            )
            maps["movements"].setdefault(key, []).append(
                {
                    "character": character,
                    "from_location": transition.get("from_location"),
                    "to_location": transition.get("to_location"),
                }
            )

    for item, item_events in getattr(lore, "item_history", {}).items():
        for item_event in item_events:
            if item_event.get("chapter_id") != chapter_id:
                continue

            key = _timeline_key(
                item_event.get("chapter_id"),
                item_event.get("scene_id"),
                item_event.get("event_id"),
                item_event.get("seq"),
            )
            maps["item_events"].setdefault(key, []).append(
                {
                    "item": item,
                    "action": item_event.get("action"),
                    "owner": item_event.get("owner"),
                    "previous_owner": item_event.get("previous_owner"),
                    "users": item_event.get("users"),
                    "known_owner": item_event.get("known_owner"),
                }
            )

    for fact, fact_events in getattr(lore, "fact_history", {}).items():
        for fact_event in fact_events:
            if fact_event.get("chapter_id") != chapter_id:
                continue

            key = _timeline_key(
                fact_event.get("chapter_id"),
                fact_event.get("scene_id"),
                fact_event.get("event_id"),
                fact_event.get("seq"),
            )
            maps["fact_events"].setdefault(key, []).append(
                {
                    "fact": fact,
                    "action": fact_event.get("action"),
                    "character": fact_event.get("character"),
                }
            )

    return maps


def _issue_event_ids(issue) -> list[dict]:
    evidence = getattr(issue, "evidence", {}) or {}
    event_fields = [
        ("event_id", "event"),
        ("current_event_id", "conflict"),
        ("previous_event_id", "first appearance"),
    ]

    event_refs = []
    for field, role in event_fields:
        value = evidence.get(field)
        if value:
            event_refs.append({"event_id": value, "role": role})

    cycle = evidence.get("cycle")
    if isinstance(cycle, list):
        for event_id in cycle:
            event_refs.append({"event_id": event_id, "role": "causality cycle"})

    seen = set()
    deduped = []
    for ref in event_refs:
        key = (ref["event_id"], ref["role"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)

    return deduped


def _issue_payload(issue, idx: int, role: str = "direct") -> dict:
    return {
        "id": f"bug_{idx:03d}",
        "severity": getattr(issue, "severity", ""),
        "rule": getattr(issue, "rule", ""),
        "message": getattr(issue, "message", ""),
        "evidence": getattr(issue, "evidence", {}) or {},
        "timeline_role": role,
    }


def _issue_match_score(issue, row: dict) -> int:
    evidence = getattr(issue, "evidence", {}) or {}
    message = str(getattr(issue, "message", "") or "").lower()
    description = str(evidence.get("description") or "").lower()
    haystack = " ".join(
        str(row.get(key) or "")
        for key in (
            "event_title",
            "summary",
            "scene_title",
            "scene_summary",
            "event_location",
            "scene_location",
        )
    ).lower()

    score = 0
    character = str(evidence.get("character") or "").lower()
    if character and character != "general":
        participants = [str(p).lower() for p in row.get("participants", [])]
        if character in participants or character in haystack:
            score += 4

    item = str(evidence.get("item") or "").lower()
    if item:
        items = [str(i).lower() for i in row.get("acquired_items", []) + row.get("used_items", [])]
        if item in items or item in haystack:
            score += 4

    for text in (description, message):
        words = {
            word
            for word in re.findall(r"[A-Za-zÀ-ÿ0-9_]{4,}", text)
            if word not in {"that", "this", "with", "from", "chapter", "story", "plot", "hole", "character"}
        }
        score += min(5, sum(1 for word in words if word in haystack))

    return score


def attach_issues_to_timeline(rows: list[dict], issues: list) -> tuple[list[dict], list[dict]]:
    for row in rows:
        row["bugs"] = []

    unresolved = []
    if not issues:
        return rows, unresolved

    by_event_id = {}
    for row in rows:
        by_event_id.setdefault(row["event_id"], []).append(row)

    for idx, issue in enumerate(issues, start=1):
        attached = False
        for ref in _issue_event_ids(issue):
            for row in by_event_id.get(ref["event_id"], []):
                row["bugs"].append(_issue_payload(issue, idx, ref["role"]))
                attached = True

        if attached:
            continue

        best_row = None
        best_score = 0
        for row in rows:
            score = _issue_match_score(issue, row)
            if score > best_score:
                best_row = row
                best_score = score

        if best_row and best_score >= 4:
            best_row["bugs"].append(_issue_payload(issue, idx, "matched by evidence"))
        else:
            unresolved.append(_issue_payload(issue, idx, "chapter-level"))

    return rows, unresolved


def bug_timeline_locations(rows: list[dict], unresolved_bugs: list[dict]) -> dict[str, list[dict]]:
    locations: dict[str, list[dict]] = {}

    for row in rows:
        anchor = f"timeline-{row.get('event_id')}"
        for bug in row.get("bugs", []):
            locations.setdefault(bug["id"], []).append(
                {
                    "anchor": anchor,
                    "role": bug.get("timeline_role", "event"),
                    "event_label": row.get("event_label") or row.get("event_id"),
                    "scene_label": row.get("scene_label") or row.get("scene_id"),
                    "location": row.get("event_location") or row.get("scene_location") or "Unknown",
                }
            )

    for bug in unresolved_bugs:
        locations.setdefault(bug["id"], []).append(
            {
                "anchor": "",
                "role": bug.get("timeline_role", "chapter-level"),
                "event_label": "No exact event match",
                "scene_label": "Chapter-level issue",
                "location": "Unknown",
            }
        )

    return locations


def build_timeline_data(extraction, lore=None, issues: list | None = None) -> tuple[list[dict], list[dict]]:
    rows = []
    graph_maps = _graph_timeline_maps(lore, extraction.chapter_id)

    for scene in extraction.scenes:
        scene_title = useful_title(scene.title, scene.summary, scene.scene_id)
        for event in scene.events:
            event_title = useful_title(
                getattr(event, "title", None),
                getattr(event, "summary", ""),
                event.event_id,
            )
            key = _timeline_key(extraction.chapter_id, scene.scene_id, event.event_id, event.seq)
            graph_event = graph_maps["events"].get(key, {})
            rows.append(
                {
                    "chapter_id": extraction.chapter_id,
                    "chapter_title": extraction.title,
                    "chapter_synopsis": extraction.synopsis,

                    "scene_id": scene.scene_id,
                    "scene_title": scene_title,
                    "scene_label": label_with_title(scene.scene_id, scene_title),
                    "scene_location": scene.location,
                    "scene_pov": scene.pov,
                    "scene_mood": scene.mood,
                    "scene_summary": scene.summary,
                    "scene_text": scene.text,

                    "event_id": event.event_id,
                    "seq": event.seq,
                    "event_title": event_title,
                    "event_label": label_with_title(event.event_id, event_title),
                    "event_location": event.location or scene.location or "Unknown",
                    "graph_order": graph_event.get("order"),
                    "graph_markers": graph_event.get("markers", []),
                    "graph_movements": graph_maps["movements"].get(key, []),
                    "graph_item_events": graph_maps["item_events"].get(key, []),
                    "graph_fact_events": graph_maps["fact_events"].get(key, []),
                    "start_time": event.start_time,
                    "end_time": event.end_time,
                    "participants": list(event.participants),
                    "acquired_items": list(event.acquired_items),
                    "used_items": list(event.used_items),
                    "revelations": list(event.revelations),
                    "knowledge_gains": event.knowledge_gains,
                    "causal_parents": list(event.causal_parents),
                    "summary": event.summary,
                }
            )

    rows = sorted(rows, key=lambda x: (x.get("graph_order") or 10**9, x["scene_id"], x["seq"]))
    return attach_issues_to_timeline(rows, issues or [])


def build_graph_data(extraction) -> dict:
    nodes = {}
    links = []

    def add_node(node_id: str, label: str, group: str, title: str | None = None):
        if node_id not in nodes:
            nodes[node_id] = {
                "id": node_id,
                "label": label,
                "group": group,
                "title": title or label
            }

    add_node(extraction.chapter_id, extraction.chapter_id, "chapter")

    for character in extraction.characters:
        add_node(character.canonical, character.canonical, "character")
        links.append(
            {
                "source": extraction.chapter_id,
                "target": character.canonical,
                "label": "HAS_CHARACTER",
            }
        )

    for scene in extraction.scenes:
        scene_id = f"scene:{scene.scene_id}"
        scene_title = useful_title(scene.title, scene.summary, scene.scene_id)
        scene_label = label_with_title(scene.scene_id, scene_title)
        add_node(scene_id, scene_label, "scene", f"{scene.scene_id}: {scene_title}")
        links.append({"source": extraction.chapter_id, "target": scene_id, "label": "HAS_SCENE"})

        if scene.location:
            add_node(scene.location, scene.location, "location")
            links.append({"source": scene_id, "target": scene.location, "label": "SET_IN"})

        for event in scene.events:
            event_id = f"event:{event.event_id}"
            event_title = useful_title(
                getattr(event, "title", None),
                getattr(event, "summary", ""),
                event.event_id,
            )
            event_label = label_with_title(event.event_id, event_title)
            add_node(event_id, event_label, "event", f"{event.event_id}: {event_title}")
            links.append({"source": scene_id, "target": event_id, "label": "HAS_EVENT"})

            for participant in event.participants:
                add_node(participant, participant, "character")
                links.append({"source": participant, "target": event_id, "label": "PARTICIPATES"})

            for item in event.acquired_items:
                add_node(item, item, "item")
                links.append({"source": event_id, "target": item, "label": "ACQUIRES"})

            for item in event.used_items:
                add_node(item, item, "item")
                links.append({"source": event_id, "target": item, "label": "USES"})

            for fact in event.revelations:
                add_node(fact, fact, "fact")
                links.append({"source": event_id, "target": fact, "label": "REVEALS"})

    return {"nodes": list(nodes.values()), "links": links}


def character_events_from_extraction(extraction, character: str) -> list[dict]:
    if extraction is None:
        return []

    events = []

    for scene in extraction.scenes:
        for event in scene.events:
            participants = list(event.participants or [])

            # Include direct participant events
            direct_match = character in participants

            # Include soft mention in summary
            summary = event.summary or ""
            summary_match = character.lower() in summary.lower()

            if direct_match or summary_match:
                events.append(
                    {
                        "event_id": event.event_id,
                        "scene_id": scene.scene_id,
                        "seq": event.seq,
                        "location": event.location or scene.location or "Unknown",
                        "summary": event.summary,
                        "participants": participants,
                        "acquired_items": list(event.acquired_items),
                        "used_items": list(event.used_items),
                        "revelations": list(event.revelations),
                        "knowledge_gains": event.knowledge_gains,
                        "causal_parents": list(event.causal_parents),
                    }
                )

    return events


def make_chapter_html(
    chapter_id: str,
    title: str,
    synopsis: str,
    drift_score: float,
    similarity_score: float,
    issues: list,
    arc_image_paths: list[str],
    output_path: str,
    arc_df=None,
    use_llm_arc_analysis: bool = True,
    extraction=None,
    lore=None,
    character_sheet: list[dict] | None = None,
):
    issue_count = len(issues)
    high_count = sum(1 for i in issues if getattr(i, "severity", "") == "high")
    medium_count = sum(1 for i in issues if getattr(i, "severity", "") == "medium")
    low_count = sum(1 for i in issues if getattr(i, "severity", "") == "low")

    arc_html = ""

    if arc_image_paths:
        for img_path in arc_image_paths:
            character = Path(img_path).stem.replace("arc_", "")

            arc_rows = []
            if arc_df is not None and not arc_df.empty and "character" in arc_df.columns:
                char_df = arc_df[arc_df["character"] == character].copy()
                arc_rows = char_df.to_dict(orient="records")

            character_events = character_events_from_extraction(extraction, character)

            analysis = (
                llm_character_arc_analysis(
                    character=character,
                    chapter_id=chapter_id,
                    arc_rows=arc_rows,
                    chapter_summary=synopsis,
                    character_events=character_events,
                )
                if use_llm_arc_analysis
                else fallback_character_arc_analysis(character)
            )

            arc_html += f"""
            <div class="arc-card">
                <h3>{safe_text(character)}</h3>
                <img src="{safe_text(img_path)}" alt="Emotion arc for {safe_text(character)}">
                <div class="analysis-box">
                {analysis}
                </div>
            </div>
            """
    else:
        arc_html = "<p>No character arc visualization found for this chapter.</p>"

    if extraction:
        timeline_data, timeline_unresolved_bugs = build_timeline_data(
            extraction,
            lore=lore,
            issues=issues,
        )
    else:
        timeline_data, timeline_unresolved_bugs = [], []

    timeline_locations = bug_timeline_locations(timeline_data, timeline_unresolved_bugs)
    bug_html = (
        "\n".join(
            analyze_bug(
                issue,
                bug_id=f"bug_{idx:03d}",
                timeline_locations=timeline_locations.get(f"bug_{idx:03d}", []),
            )
            for idx, issue in enumerate(issues, start=1)
        )
        if issues
        else "<p class='good'>No story bugs detected by current rules.</p>"
    )

    graph_data = build_graph_data(extraction) if extraction else {"nodes": [], "links": []}

    data_json = safe_json_for_script(
        {
            "arcData": [],  # We plot utilizing direct CSV logic or matplotlib image
            "timelineData": timeline_data,
            "timelineUnresolvedBugs": timeline_unresolved_bugs,
            "graphData": graph_data,
            "issueStats": {
                "high": high_count,
                "medium": medium_count,
                "low": low_count,
            },
            "characterSheet": character_sheet or [],
        },
    )

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'none'; object-src 'none'; base-uri 'none'; frame-ancestors 'none';">
<meta name="referrer" content="no-referrer">
<title>{safe_text(chapter_id)} Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>

<style>
:root {{
    --primary: #4f46e5;
    --primary-hover: #4338ca;
    --primary-light: #e0e7ff;
    --dark: #0f172a;
    --bg: #f8fafc;
    --card-bg: #ffffff;
    --text-main: #1e293b;
    --text-muted: #64748b;
    --border: #e2e8f0;
    --accent-red: #ef4444;
    --accent-red-bg: #fee2e2;
    --accent-amber: #f59e0b;
    --accent-amber-bg: #fef3c7;
    --accent-blue: #3b82f6;
    --accent-blue-bg: #dbeafe;
    --accent-green: #10b981;
    --accent-green-bg: #d1fae5;
}}

body {{
    margin: 0;
    font-family: 'Inter', Arial, sans-serif;
    background: var(--bg);
    color: var(--text-main);
}}

header {{
    padding: 36px 48px;
    background: linear-gradient(135deg, #1e1b4b, #312e81, #4f46e5);
    color: white;
    box-shadow: 0 4px 20px rgba(0,0,0,0.12);
    border-bottom: 4px solid var(--primary);
}}

header h1 {{
    margin: 0;
    font-family: 'Outfit', sans-serif;
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(to right, #ffffff, #e0e7ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

header p {{
    margin: 8px 0 0 0;
    font-size: 15px;
    opacity: 0.95;
    font-weight: 500;
}}

.container {{
    padding: 40px;
    max-width: 1400px;
    margin: 0 auto;
}}

.grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 20px;
    margin-bottom: 32px;
}}

.metric-card {{
    background: var(--card-bg);
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.04);
    border: 1px solid var(--border);
    transition: transform 0.2s, box-shadow 0.2s;
}}

.metric-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 20px 40px rgba(15, 23, 42, 0.08);
}}

.metric-card .label {{
    color: var(--text-muted);
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

.metric-card .value {{
    font-family: 'Outfit', sans-serif;
    font-size: 32px;
    font-weight: 800;
    color: var(--dark);
    margin-top: 10px;
}}

.card {{
    background: var(--card-bg);
    border-radius: 24px;
    padding: 30px;
    margin-bottom: 32px;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.04);
    border: 1px solid var(--border);
}}

.card h2 {{
    font-family: 'Outfit', sans-serif;
    font-size: 22px;
    font-weight: 800;
    margin-top: 0;
    margin-bottom: 20px;
    color: var(--dark);
}}

.tabs {{
    display: flex;
    gap: 8px;
    margin-bottom: 32px;
    flex-wrap: wrap;
    background: #f1f5f9;
    padding: 6px;
    border-radius: 16px;
    border: 1px solid var(--border);
    width: fit-content;
}}

.tab-btn {{
    border: none;
    padding: 10px 20px;
    border-radius: 12px;
    background: transparent;
    cursor: pointer;
    font-weight: 700;
    font-size: 14px;
    color: var(--text-muted);
    transition: all 0.2s ease;
}}

.tab-btn:hover {{
    color: var(--dark);
    background: rgba(255, 255, 255, 0.6);
}}

.tab-btn.active {{
    background: var(--card-bg);
    color: var(--primary);
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
}}

.tab {{
    display: none;
    animation: fadeIn 0.3s ease-in-out;
}}

.tab.active {{
    display: block;
}}

@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

.good {{
    color: var(--accent-green);
    font-weight: 700;
    background: var(--accent-green-bg);
    padding: 16px 20px;
    border-radius: 16px;
    border: 1px solid rgba(16, 185, 129, 0.2);
}}

/* Bug list and items styling */
.bug-item {{
    border-left: 6px solid var(--accent-red);
    padding: 20px;
    margin-bottom: 20px;
    background: #fff5f5;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(239, 68, 68, 0.02);
    transition: transform 0.2s, box-shadow 0.2s;
    border: 1px solid var(--border);
    border-left-width: 6px;
}}

.bug-item:hover {{
    transform: translateX(4px);
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
}}

.bug-item.severity-high {{
    border-left-color: var(--accent-red);
    background: #fff8f8;
}}

.bug-item.severity-medium {{
    border-left-color: var(--accent-amber);
    background: #fffdf5;
}}

.bug-item.severity-low {{
    border-left-color: var(--accent-blue);
    background: #f8fafc;
}}

.bug-head {{
    display: flex;
    gap: 12px;
    align-items: center;
    margin-bottom: 8px;
}}

.badge {{
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

.badge-high {{
    background: var(--accent-red-bg);
    color: #991b1b;
}}

.badge-medium {{
    background: var(--accent-amber-bg);
    color: #92400e;
}}

.badge-low {{
    background: var(--accent-blue-bg);
    color: #1e40af;
}}

.bug-detail {{
    margin-top: 12px;
    background: rgba(255, 255, 255, 0.6);
    padding: 12px 16px;
    border-radius: 12px;
    border: 1px solid rgba(226, 232, 240, 0.8);
    font-size: 14px;
}}

.bug-detail p {{
    margin: 6px 0;
}}

.bug-timeline-links {{
    margin: 12px 0;
    padding: 12px 14px;
    border-radius: 12px;
    background: #f8fafc;
    border: 1px solid var(--border);
}}

.bug-timeline-links p {{
    margin: 0 0 6px 0;
    font-size: 13px;
}}

.bug-timeline-links ul {{
    margin: 0;
    padding-left: 18px;
}}

.bug-timeline-links li {{
    margin: 4px 0;
    font-size: 13px;
}}

.bug-timeline-links a {{
    color: var(--primary);
    font-weight: 700;
    text-decoration: none;
}}

/* Character Sheet styles */
.character-sheet-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
    gap: 24px;
    margin-top: 20px;
}}

.character-card {{
    background: var(--card-bg);
    border-radius: 20px;
    border: 1px solid var(--border);
    box-shadow: 0 4px 15px rgba(15, 23, 42, 0.02);
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
    display: flex;
    flex-direction: column;
}}

.character-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 16px 30px rgba(15, 23, 42, 0.08);
}}

.char-header {{
    background: linear-gradient(135deg, #4f46e5, #6366f1);
    padding: 20px 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    color: white;
}}

.char-avatar {{
    width: 44px;
    height: 44px;
    background: rgba(255, 255, 255, 0.2);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Outfit', sans-serif;
    font-size: 20px;
    font-weight: 800;
    text-transform: uppercase;
}}

.char-name-container h3 {{
    margin: 0;
    font-family: 'Outfit', sans-serif;
    font-size: 18px;
    font-weight: 800;
    letter-spacing: -0.01em;
}}

.char-body {{
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    flex-grow: 1;
}}

.char-section h4 {{
    margin: 0 0 6px 0;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    font-weight: 700;
}}

.char-section p {{
    margin: 0;
    font-size: 14px;
    line-height: 1.5;
    color: var(--text-main);
}}

.risk-none .risk-message {{
    background: var(--accent-green-bg);
    color: #065f46;
    padding: 10px 14px;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 600;
}}

.risk-active .risk-message {{
    background: var(--accent-red-bg);
    color: #991b1b;
    padding: 10px 14px;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 600;
    border-left: 4px solid var(--accent-red);
    line-height: 1.4;
}}

.items-container {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}}

.item-badge {{
    background: #f1f5f9;
    color: #475569;
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid #e2e8f0;
}}

.no-items {{
    font-size: 13px;
    color: var(--text-muted);
    font-style: italic;
}}

.analysis-box {{
    margin-top: 16px;
    background: #f8fafc;
    border-left: 5px solid var(--primary);
    padding: 16px 20px;
    border-radius: 12px;
    border: 1px solid var(--border);
    border-left-width: 5px;
}}

.arc-analysis h4 {{
    margin: 12px 0 4px 0;
    color: var(--primary);
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 15px;
}}

.arc-analysis p {{
    margin: 0 0 8px 0;
    line-height: 1.5;
    font-size: 14px;
}}

.arc-card {{
    margin-bottom: 32px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 24px;
}}

.arc-card img {{
    max-width: 100%;
    border-radius: 16px;
    border: 1px solid var(--border);
    margin-top: 12px;
}}

.timeline {{
    border-left: 4px solid var(--primary);
    margin-left: 12px;
    padding-left: 24px;
}}

.timeline-item {{
    margin-bottom: 24px;
    position: relative;
    padding: 18px 20px;
    border: 1px solid var(--border);
    border-radius: 16px;
    background: #ffffff;
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.03);
}}

.timeline-item::before {{
    content: "";
    position: absolute;
    left: -34px;
    top: 6px;
    width: 16px;
    height: 16px;
    background: var(--primary);
    border: 4px solid var(--bg);
    border-radius: 50%;
    box-shadow: 0 0 0 2px var(--primary);
}}

.timeline-item h4 {{
    margin: 0 0 8px 0;
    font-family: 'Outfit', sans-serif;
    color: var(--dark);
}}

.timeline-item.has-bugs {{
    border-color: rgba(239, 68, 68, 0.35);
    box-shadow: 0 8px 24px rgba(239, 68, 68, 0.08);
}}

.timeline-item.has-bugs::before {{
    background: var(--accent-red);
    box-shadow: 0 0 0 2px var(--accent-red);
}}

.timeline-bugs {{
    margin: 14px 0;
    display: grid;
    gap: 10px;
}}

.timeline-bug {{
    border: 1px solid rgba(239, 68, 68, 0.22);
    border-left: 5px solid var(--accent-red);
    background: #fff8f8;
    border-radius: 12px;
    padding: 12px 14px;
}}

.timeline-bug.severity-medium {{
    border-left-color: var(--accent-amber);
    background: #fffdf5;
}}

.timeline-bug.severity-low {{
    border-left-color: var(--accent-blue);
    background: #f8fafc;
}}

.timeline-bug-head {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    margin-bottom: 6px;
}}

.timeline-bug p {{
    margin: 0;
    font-size: 13px;
    line-height: 1.45;
}}

.timeline-empty-bugs {{
    margin-bottom: 18px;
    padding: 14px 16px;
    border-radius: 12px;
    border: 1px solid var(--border);
    background: #f8fafc;
}}

.graph-detail-grid {{
    display: grid;
    gap: 8px;
    margin-top: 12px;
}}

.graph-detail-row {{
    background: #f8fafc;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 12px;
    font-size: 13px;
    line-height: 1.45;
}}

.graph-box {{
    height: 540px;
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
    background: #f8fafc;
}}

.node-label {{
    font-size: 10px;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    pointer-events: none;
}}

.search-box {{
    padding: 12px 16px;
    width: 100%;
    max-width: 420px;
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 20px;
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.02);
}}

.search-box:focus {{
    outline: none;
    border-color: var(--primary);
    box-shadow: 0 0 0 3px var(--primary-light);
}}

.timeline-meta {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 8px 0 12px 0;
}}

.timeline-meta span {{
    background: #eef2ff;
    color: var(--primary);
    padding: 6px 10px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 600;
}}

details {{
    background: #f8fafc;
    padding: 14px 18px;
    border-radius: 12px;
    margin-top: 12px;
    border: 1px solid var(--border);
}}

summary {{
    cursor: pointer;
    font-weight: 700;
    font-size: 14px;
    color: var(--dark);
}}

</style>
</head>

<body>
<header>
    <h1>{safe_text(title)}</h1>
    <p>Interactive Narrative Debugging Dashboard · Chapter ID: <strong>{safe_text(chapter_id)}</strong></p>
</header>

<div class="container">

    <div class="grid">
        <div class="metric-card">
            <div class="label">Similarity to Baseline</div>
            <div class="value">{similarity_score:.4f}</div>
        </div>
        <div class="metric-card">
            <div class="label">Drift Score</div>
            <div class="value">{drift_score:.4f}</div>
        </div>
        <div class="metric-card">
            <div class="label">Bugs Found</div>
            <div class="value">{issue_count}</div>
        </div>
        <div class="metric-card">
            <div class="label">High Severity</div>
            <div class="value">{high_count}</div>
        </div>
    </div>

    <div class="card">
        <h2>Overall Chapter Summary</h2>
        <p>{safe_text(synopsis)}</p>
    </div>

    <div class="tabs">
        <button id="tab-overview" class="tab-btn active" onclick="openTab('overview')">Overview</button>
        <button id="tab-bugs" class="tab-btn" onclick="openTab('bugs')">Story Bugs</button>
        <button id="tab-charactersheet" class="tab-btn" onclick="openTab('charactersheet')">Character Sheet</button>
        <button id="tab-arcs" class="tab-btn" onclick="openTab('arcs')">Character Arcs</button>
        <button id="tab-timeline" class="tab-btn" onclick="openTab('timeline')">Timeline</button>
        <button id="tab-graph" class="tab-btn" onclick="openTab('graph')">Lore Graph</button>
    </div>

    <section id="overview" class="tab active">
        <div class="card">
            <h2>Chapter Drift Analysis</h2>
            <p><strong>Similarity:</strong> {similarity_score:.4f}</p>
            <p><strong>Drift:</strong> {drift_score:.4f}</p>
            <p>{safe_text(analyze_drift(drift_score))}</p>
            <canvas id="driftChart" height="90"></canvas>
        </div>

        <div class="card">
            <h2>Bug Severity Distribution</h2>
            <canvas id="bugChart" height="90"></canvas>
        </div>
    </section>

    <section id="bugs" class="tab">
        <div class="card">
            <h2>Story Bugs Found</h2>
            <input class="search-box" id="bugSearch" placeholder="Search bug rule/message..." oninput="filterBugs()">
            <div id="bugList">
                {bug_html}
            </div>
        </div>
    </section>

    <section id="charactersheet" class="tab">
        <div class="card">
            <h2>Character Sheet</h2>
            <p>Summary of character state, current actions, potential future plot hole risks, and acquired items/spells in this chapter.</p>
            <div id="characterSheetBox" class="character-sheet-grid">
                <!-- Dynamically filled by JavaScript -->
            </div>
        </div>
    </section>

    <section id="arcs" class="tab">
        <div class="card">
            <h2>Character Arc Analysis</h2>
            {arc_html}
        </div>
    </section>

    <section id="timeline" class="tab">
        <div class="card">
            <h2>Event Timeline</h2>
            <div id="timelineBox" class="timeline"></div>
        </div>
    </section>

    <section id="graph" class="tab">
        <div class="card">
            <h2>Lore Graph</h2>
            <p>Nodes represent characters, scenes, events, items, facts, and locations.</p>
            <div id="graphBox" class="graph-box"></div>
        </div>
    </section>

</div>

<script>
const DASHBOARD_DATA = {data_json};

function openTab(id) {{
    document.querySelectorAll(".tab").forEach(el => el.classList.remove("active"));
    document.querySelectorAll(".tab-btn").forEach(el => el.classList.remove("active"));
    document.getElementById(id).classList.add("active");
    
    // Set correct active tab button
    const btn = document.getElementById("tab-" + id);
    if (btn) btn.classList.add("active");

    if (id === "graph") {{
        setTimeout(renderGraph, 50);
    }}
}}

function safeEscape(str) {{
    if (str === null || str === undefined) return "";
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}}

function formatList(values) {{
    if (!values || !values.length) return "-";
    return safeEscape(values.join(", "));
}}

function severityBadge(severity) {{
    return `<span class="badge badge-${{safeEscape(severity || "low")}}">${{safeEscape(severity || "low")}}</span>`;
}}

function renderTimelineBug(bug) {{
    const evidenceText = bug.evidence && Object.keys(bug.evidence).length
        ? `<details><summary>Bug evidence</summary><pre>${{safeEscape(JSON.stringify(bug.evidence, null, 2))}}</pre></details>`
        : "";

    return `
        <div class="timeline-bug severity-${{safeEscape(bug.severity || "low")}}">
            <div class="timeline-bug-head">
                ${{severityBadge(bug.severity)}}
                <strong>${{safeEscape(bug.rule || "story_bug")}}</strong>
                <span class="item-badge">${{safeEscape(bug.timeline_role || "event")}}</span>
            </div>
            <p>${{safeEscape(bug.message || "-")}}</p>
            ${{evidenceText}}
        </div>
    `;
}}

function renderGraphDetails(ev) {{
    const rows = [];

    if (ev.graph_order) {{
        rows.push(`<div class="graph-detail-row"><strong>Graph order:</strong> ${{safeEscape(ev.graph_order)}}</div>`);
    }}

    if (ev.graph_markers && ev.graph_markers.length) {{
        rows.push(`<div class="graph-detail-row"><strong>Graph markers:</strong> ${{formatList(ev.graph_markers)}}</div>`);
    }}

    if (ev.graph_movements && ev.graph_movements.length) {{
        const movementText = ev.graph_movements
            .map(m => `${{m.character || "?"}}: ${{m.from_location || "Unknown"}} -> ${{m.to_location || "Unknown"}}`)
            .join("; ");
        rows.push(`<div class="graph-detail-row"><strong>Character movement:</strong> ${{safeEscape(movementText)}}</div>`);
    }}

    if (ev.graph_item_events && ev.graph_item_events.length) {{
        const itemText = ev.graph_item_events
            .map(item => `${{item.action || "ITEM"}} ${{item.item || "?"}}${{item.owner ? " by " + item.owner : ""}}${{item.users && item.users.length ? " by " + item.users.join(", ") : ""}}`)
            .join("; ");
        rows.push(`<div class="graph-detail-row"><strong>Item continuity:</strong> ${{safeEscape(itemText)}}</div>`);
    }}

    if (ev.graph_fact_events && ev.graph_fact_events.length) {{
        const factText = ev.graph_fact_events
            .map(fact => `${{fact.action || "FACT"}}: ${{fact.fact || "?"}}${{fact.character ? " / " + fact.character : ""}}`)
            .join("; ");
        rows.push(`<div class="graph-detail-row"><strong>Knowledge/facts:</strong> ${{safeEscape(factText)}}</div>`);
    }}

    return rows.length ? `<div class="graph-detail-grid">${{rows.join("")}}</div>` : "";
}}

function filterBugs() {{
    const query = document.getElementById("bugSearch").value.toLowerCase();
    document.querySelectorAll(".bug-item").forEach(item => {{
        item.style.display = item.innerText.toLowerCase().includes(query) ? "block" : "none";
    }});
}}

if (window.Chart) {{
new Chart(document.getElementById("driftChart"), {{
    type: "bar",
    data: {{
        labels: ["Similarity", "Drift"],
        datasets: [{{
            label: "Semantic Score",
            data: [{similarity_score:.4f}, {drift_score:.4f}],
            backgroundColor: ["#4f46e5", "#ef4444"]
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{
            legend: {{ display: false }}
        }},
        scales: {{
            y: {{
                beginAtZero: true,
                max: 1
            }}
        }}
    }}
}});

new Chart(document.getElementById("bugChart"), {{
    type: "doughnut",
    data: {{
        labels: ["High", "Medium", "Low"],
        datasets: [{{
            data: [
                DASHBOARD_DATA.issueStats.high,
                DASHBOARD_DATA.issueStats.medium,
                DASHBOARD_DATA.issueStats.low
            ],
            backgroundColor: ["#ef4444", "#f59e0b", "#3b82f6"]
        }}]
    }},
    options: {{
        responsive: true
    }}
}});
}}

function renderCharacterSheet() {{
    const box = document.getElementById("characterSheetBox");
    const sheet = DASHBOARD_DATA.characterSheet || [];

    if (!sheet.length) {{
        box.innerHTML = "<p>No character sheet data available for this chapter.</p>";
        return;
    }}

    box.innerHTML = sheet.map(char => {{
        const items = char.acquired_items_or_spells || [];
        const itemsList = items.length 
            ? items.map(item => `<span class="item-badge">${{safeEscape(item)}}</span>`).join(" ")
            : "<span class='no-items'>None</span>";

        const hasRisk = char.risk_of_plot_hole && char.risk_of_plot_hole.toLowerCase() !== "none";
        const riskClass = hasRisk ? "risk-active" : "risk-none";

        return `
            <div class="character-card">
                <div class="char-header">
                    <div class="char-avatar">${{safeEscape(char.character ? char.character[0] : "?")}}</div>
                    <div class="char-name-container">
                        <h3>${{safeEscape(char.character)}}</h3>
                    </div>
                </div>
                <div class="char-body">
                    <div class="char-section">
                        <h4>Narrative Summary</h4>
                        <p>${{safeEscape(char.narrative_summary)}}</p>
                    </div>
                    <div class="char-section">
                        <h4>Current Actions</h4>
                        <p>${{safeEscape(char.current_actions)}}</p>
                    </div>
                    <div class="char-section ${{riskClass}}">
                        <h4>Future Plot Hole Risk</h4>
                        <div class="risk-message">
                            ${{hasRisk 
                                ? `⚠️ ${{safeEscape(char.risk_of_plot_hole)}}`
                                : `✅ No major risks detected`
                            }}
                        </div>
                    </div>

                    <div class="char-section">
                        <h4>Acquired Items & Spells</h4>
                        <div class="items-container">${{itemsList}}</div>
                    </div>
                </div>
            </div>
        `;
    }}).join("");
}}

function renderTimeline() {{
    const box = document.getElementById("timelineBox");
    const rows = DASHBOARD_DATA.timelineData || [];
    const unresolvedBugs = DASHBOARD_DATA.timelineUnresolvedBugs || [];

    if (!rows.length) {{
        box.innerHTML = "<p>No timeline data available.</p>";
        return;
    }}

    const unresolvedHtml = unresolvedBugs.length
        ? `
            <div class="timeline-empty-bugs">
                <h3>Chapter-Level Bugs</h3>
                <p>These story bugs were detected, but the report could not pin them to one exact event.</p>
                <div class="timeline-bugs">
                    ${{unresolvedBugs.map(renderTimelineBug).join("")}}
                </div>
            </div>
        `
        : "";

    box.innerHTML = unresolvedHtml + rows.map(ev => {{
        const bugs = ev.bugs || [];
        const bugHtml = bugs.length
            ? `<div class="timeline-bugs">${{bugs.map(renderTimelineBug).join("")}}</div>`
            : "";
        const graphDetails = renderGraphDetails(ev);
        const cssClass = bugs.length ? "timeline-item has-bugs" : "timeline-item";

        return `
        <div class="${{cssClass}}" id="timeline-${{safeEscape(ev.event_id)}}">
            <h4>${{safeEscape(ev.chapter_id)}} · ${{safeEscape(ev.chapter_title || "-")}}</h4>

            <div class="timeline-meta">
                <span><strong>Scene:</strong> ${{safeEscape(ev.scene_id)}} · ${{safeEscape(ev.scene_title || "-")}}</span>
                <span><strong>Event:</strong> ${{safeEscape(ev.event_id)}} · ${{safeEscape(ev.event_title || "-")}}</span>
                ${{ev.graph_order ? `<span><strong>Graph Order:</strong> ${{safeEscape(ev.graph_order)}}</span>` : ""}}
                <span><strong>POV:</strong> ${{safeEscape(ev.scene_pov || "-")}}</span>
                <span><strong>Mood:</strong> ${{safeEscape(ev.scene_mood || "-")}}</span>
                <span><strong>Location:</strong> ${{safeEscape(ev.event_location || ev.scene_location || "-")}}</span>
            </div>

            <h3>${{safeEscape(ev.event_id)}}: ${{safeEscape(ev.event_title || "-")}} · Event #${{safeEscape(ev.seq)}}</h3>
            ${{bugHtml}}
            <p><strong>Event Summary:</strong> ${{safeEscape(ev.summary || "-")}}</p>
            <p><strong>Scene Summary:</strong> ${{safeEscape(ev.scene_summary || "-")}}</p>

            <div class="timeline-meta" style="margin-top: 12px; background: transparent; border: none; padding: 0;">
                ${{ ev.participants && ev.participants.length ? `<span><strong>Participants:</strong> ${{formatList(ev.participants)}}</span>` : "" }}
                ${{ ev.acquired_items && ev.acquired_items.length ? `<span style="background: var(--accent-green-bg); color: #065f46;"><strong>Acquired:</strong> ${{formatList(ev.acquired_items)}}</span>` : "" }}
                ${{ ev.used_items && ev.used_items.length ? `<span style="background: var(--accent-amber-bg); color: #92400e;"><strong>Used:</strong> ${{formatList(ev.used_items)}}</span>` : "" }}
                ${{ ev.revelations && ev.revelations.length ? `<span style="background: var(--accent-blue-bg); color: #1e40af;"><strong>Revelation:</strong> ${{formatList(ev.revelations)}}</span>` : "" }}
            </div>

            ${{graphDetails}}

            <details>
                <summary>Show extracted details</summary>
                <p><strong>Participants:</strong> ${{formatList(ev.participants)}}</p>
                <p><strong>Acquired Items:</strong> ${{formatList(ev.acquired_items)}}</p>
                <p><strong>Used Items:</strong> ${{formatList(ev.used_items)}}</p>
                <p><strong>Revelations:</strong> ${{formatList(ev.revelations)}}</p>
                <p><strong>Knowledge Gains:</strong> ${{safeEscape(JSON.stringify(ev.knowledge_gains || {{}}))}}</p>
                <p><strong>Causal Parents:</strong> ${{formatList(ev.causal_parents)}}</p>
                <p><strong>Start Time:</strong> ${{safeEscape(ev.start_time || "-")}}</p>
                <p><strong>End Time:</strong> ${{safeEscape(ev.end_time || "-")}}</p>
                <p><strong>Graph Markers:</strong> ${{formatList(ev.graph_markers)}}</p>
                <p><strong>Scene Text:</strong> ${{safeEscape(ev.scene_text || "-")}}</p>
            </details>
        </div>
    `;
    }}).join("");
}}

let graphRendered = false;

function renderGraph() {{
    if (graphRendered) return;
    graphRendered = true;

    const data = DASHBOARD_DATA.graphData || {{nodes: [], links: []}};
    const container = document.getElementById("graphBox");
    const width = container.clientWidth;
    const height = container.clientHeight;

    if (!data.nodes.length) {{
        container.innerHTML = "<p style='padding:20px;'>No graph data available.</p>";
        return;
    }}

    const svg = d3.select("#graphBox")
        .append("svg")
        .attr("width", width)
        .attr("height", height);

    const color = d3.scaleOrdinal()
        .domain(["chapter", "scene", "event", "character", "item", "fact", "location"])
        .range(["#0f172a", "#2563eb", "#7c3aed", "#10b981", "#f59e0b", "#ef4444", "#06b6d4"]);

    const simulation = d3.forceSimulation(data.nodes)
        .force("link", d3.forceLink(data.links).id(d => d.id).distance(105))
        .force("charge", d3.forceManyBody().strength(-300))
        .force("center", d3.forceCenter(width / 2, height / 2));

    const link = svg.append("g")
        .selectAll("line")
        .data(data.links)
        .enter()
        .append("line")
        .attr("stroke", "#e2e8f0")
        .attr("stroke-width", 2);

    const node = svg.append("g")
        .selectAll("circle")
        .data(data.nodes)
        .enter()
        .append("circle")
        .attr("r", 9)
        .attr("fill", d => color(d.group))
        .call(drag(simulation));

    const label = svg.append("g")
        .selectAll("text")
        .data(data.nodes)
        .enter()
        .append("text")
        .attr("class", "node-label")
        .text(d => d.label)
        .attr("dx", 12)
        .attr("dy", 4)
        .attr("fill", "#334155");

    node.append("title").text(d => `${{d.title}} (${{d.group}})`);

    simulation.on("tick", () => {{
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        node
            .attr("cx", d => d.x)
            .attr("cy", d => d.y);

        label
            .attr("x", d => d.x)
            .attr("y", d => d.y);
    }});

    function drag(simulation) {{
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}

        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}

        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}

        return d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended);
    }}
}}

renderCharacterSheet();
renderTimeline();
</script>

</body>
</html>
"""

    Path(output_path).write_text(html, encoding="utf-8")
