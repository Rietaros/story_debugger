# html_report.py
from __future__ import annotations

from pathlib import Path
from html import escape
import json
import os

from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL


def safe_text(value) -> str:
    if value is None:
        return ""
    return escape(str(value))


def analyze_drift(drift_score: float) -> str:
    if drift_score < 0.20:
        return "Low drift. The chapter remains semantically close to the early story baseline."
    if drift_score < 0.40:
        return "Moderate drift. The chapter introduces noticeable development while still staying connected."
    return "High drift. The chapter may strongly shift tone, theme, mood, or narrative direction."


def analyze_bug(issue) -> str:
    rule = getattr(issue, "rule", "")
    severity = getattr(issue, "severity", "")
    message = getattr(issue, "message", "")
    evidence = getattr(issue, "evidence", {}) or {}

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

    return f"""
    <div class="bug-item severity-{safe_text(severity)}" data-rule="{safe_text(rule)}">
        <div class="bug-head">
            <span class="badge">{safe_text(severity)}</span>
            <strong>{safe_text(rule)}</strong>
        </div>
        <p>{safe_text(message)}</p>
        {meaning}
    </div>
    """


def llm_character_arc_analysis(
    character: str,
    chapter_id: str,
    arc_rows: list[dict],
    chapter_summary: str = "",
    character_events: list[dict] | None = None,
    model_name: str | None = None,
) -> str:
    if not OPENAI_API_KEY or not arc_rows:
        return fallback_character_arc_analysis(character)

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
        )
        return response.output_text.strip()
    except Exception as exc:
        return fallback_character_arc_analysis(character) + f"<br><em>LLM analysis unavailable: {safe_text(exc)}</em>"

def fallback_character_arc_analysis(character: str) -> str:
    return (
        f"The emotion arc for <strong>{safe_text(character)}</strong> shows how emotional "
        "intensity changes through the chapter. Spikes may indicate conflict, realization, "
        "danger, or turning points. Sudden drops may indicate trauma, contradiction, or unstable characterization."
    )


def build_timeline_data(extraction) -> list[dict]:
    rows = []

    for scene in extraction.scenes:
        for event in scene.events:
            rows.append(
                {
                    "chapter_id": extraction.chapter_id,
                    "chapter_title": extraction.title,
                    "chapter_synopsis": extraction.synopsis,

                    "scene_id": scene.scene_id,
                    "scene_title": scene.title,
                    "scene_location": scene.location,
                    "scene_pov": scene.pov,
                    "scene_mood": scene.mood,
                    "scene_summary": scene.summary,
                    "scene_text": scene.text,

                    "event_id": event.event_id,
                    "seq": event.seq,
                    "event_location": event.location or scene.location or "Unknown",
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

    return sorted(rows, key=lambda x: (x["scene_id"], x["seq"]))


def build_graph_data(extraction) -> dict:
    nodes = {}
    links = []

    def add_node(node_id: str, label: str, group: str):
        if node_id not in nodes:
            nodes[node_id] = {"id": node_id, "label": label, "group": group}

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
        add_node(scene_id, scene.scene_id, "scene")
        links.append({"source": extraction.chapter_id, "target": scene_id, "label": "HAS_SCENE"})

        if scene.location:
            add_node(scene.location, scene.location, "location")
            links.append({"source": scene_id, "target": scene.location, "label": "SET_IN"})

        for event in scene.events:
            event_id = f"event:{event.event_id}"
            add_node(event_id, event.event_id, "event")
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
):
    issue_count = len(issues)
    high_count = sum(1 for i in issues if getattr(i, "severity", "") == "high")
    medium_count = sum(1 for i in issues if getattr(i, "severity", "") == "medium")
    low_count = sum(1 for i in issues if getattr(i, "severity", "") == "low")

    bug_html = (
        "\n".join(analyze_bug(issue) for issue in issues)
        if issues
        else "<p class='good'>No story bugs detected by current rules.</p>"
    )

    arc_data = []
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

            for row in arc_rows:
                arc_data.append(
                    {
                        "character": character,
                        "sentence_index": row.get("sentence_index"),
                        "emotion": row.get("emotion"),
                        "score": row.get("score"),
                    }
                )
    else:
        arc_html = "<p>No character arc visualization found for this chapter.</p>"

    timeline_data = build_timeline_data(extraction) if extraction else []
    graph_data = build_graph_data(extraction) if extraction else {"nodes": [], "links": []}

    data_json = json.dumps(
        {
            "arcData": arc_data,
            "timelineData": timeline_data,
            "graphData": graph_data,
            "issueStats": {
                "high": high_count,
                "medium": medium_count,
                "low": low_count,
            },
        },
        ensure_ascii=False,
    )

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>{safe_text(chapter_id)} Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>

<style>
body {{
    margin: 0;
    font-family: Inter, Arial, sans-serif;
    background: #f4f6fb;
    color: #1f2937;
}}

header {{
    padding: 28px 40px;
    background: linear-gradient(135deg, #111827, #312e81);
    color: white;
}}

header h1 {{
    margin: 0;
    font-size: 32px;
}}

header p {{
    opacity: 0.9;
}}

.container {{
    padding: 28px 40px;
}}

.grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}}

.metric-card {{
    background: white;
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
}}

.metric-card .label {{
    color: #6b7280;
    font-size: 13px;
}}

.metric-card .value {{
    font-size: 28px;
    font-weight: 800;
    margin-top: 6px;
}}

.card {{
    background: white;
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 24px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
}}

.tabs {{
    display: flex;
    gap: 10px;
    margin-bottom: 18px;
    flex-wrap: wrap;
}}

.tab-btn {{
    border: none;
    padding: 10px 14px;
    border-radius: 999px;
    background: #e5e7eb;
    cursor: pointer;
    font-weight: 700;
}}

.tab-btn.active {{
    background: #312e81;
    color: white;
}}

.tab {{
    display: none;
}}

.tab.active {{
    display: block;
}}

.good {{
    color: #047857;
    font-weight: 700;
}}

.bug-item {{
    border-left: 5px solid #b91c1c;
    padding: 14px 16px;
    margin-bottom: 16px;
    background: #fff5f5;
    border-radius: 10px;
}}
.analysis-box {{
    margin-top: 16px;
    background: #f9fafb;
    border-left: 5px solid #312e81;
    padding: 16px 18px;
    border-radius: 12px;
}}

.arc-analysis h4 {{
    margin: 14px 0 6px 0;
    color: #312e81;
}}

.arc-analysis p {{
    margin: 0 0 10px 0;
    line-height: 1.6;
}}

.bug-head {{
    display: flex;
    gap: 10px;
    align-items: center;
}}

.badge {{
    padding: 4px 8px;
    border-radius: 999px;
    background: #fee2e2;
    color: #991b1b;
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
}}

.arc-card {{
    margin-bottom: 24px;
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 18px;
}}

.arc-card img {{
    max-width: 100%;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}}

.timeline {{
    border-left: 4px solid #312e81;
    margin-left: 12px;
    padding-left: 20px;
}}

.timeline-item {{
    margin-bottom: 18px;
    position: relative;
}}

.timeline-item::before {{
    content: "";
    position: absolute;
    left: -30px;
    top: 6px;
    width: 14px;
    height: 14px;
    background: #312e81;
    border-radius: 50%;
}}

.timeline-item h4 {{
    margin: 0 0 6px 0;
}}

.graph-box {{
    height: 520px;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    overflow: hidden;
}}

.node-label {{
    font-size: 11px;
    pointer-events: none;
}}

.search-box {{
    padding: 10px 12px;
    width: 100%;
    max-width: 400px;
    border: 1px solid #d1d5db;
    border-radius: 10px;
    margin-bottom: 14px;
}}

.timeline-meta {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 8px 0 12px 0;
}}

.timeline-meta span {{
    background: #eef2ff;
    color: #312e81;
    padding: 5px 9px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
}}

details {{
    background: #f9fafb;
    padding: 10px 12px;
    border-radius: 10px;
    margin-top: 10px;
}}

summary {{
    cursor: pointer;
    font-weight: 800;
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
        <button class="tab-btn active" onclick="openTab('overview')">Overview</button>
        <button class="tab-btn" onclick="openTab('bugs')">Story Bugs</button>
        <button class="tab-btn" onclick="openTab('arcs')">Character Arcs</button>
        <button class="tab-btn" onclick="openTab('timeline')">Timeline</button>
        <button class="tab-btn" onclick="openTab('graph')">Lore Graph</button>
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

    <section id="arcs" class="tab">
        <div class="card">
            <h2>Character Arc Analysis</h2>
            <canvas id="arcChart" height="110"></canvas>
            <hr>
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
    event.target.classList.add("active");

    if (id === "graph") {{
        setTimeout(renderGraph, 50);
    }}
}}

function filterBugs() {{
    const query = document.getElementById("bugSearch").value.toLowerCase();
    document.querySelectorAll(".bug-item").forEach(item => {{
        item.style.display = item.innerText.toLowerCase().includes(query) ? "block" : "none";
    }});
}}

new Chart(document.getElementById("driftChart"), {{
    type: "bar",
    data: {{
        labels: ["Similarity", "Drift"],
        datasets: [{{
            label: "Semantic Score",
            data: [{similarity_score:.4f}, {drift_score:.4f}]
        }}]
    }},
    options: {{
        responsive: true,
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
            ]
        }}]
    }},
    options: {{
        responsive: true
    }}
}});

function renderArcChart() {{
    const rows = DASHBOARD_DATA.arcData || [];
    const grouped = {{}};

    rows.forEach(r => {{
        if (!grouped[r.character]) grouped[r.character] = [];
        grouped[r.character].push(r);
    }});

    const labels = [...new Set(rows.map(r => r.sentence_index))].sort((a, b) => a - b);

    const datasets = Object.entries(grouped).map(([character, values]) => {{
        const valueMap = Object.fromEntries(values.map(v => [v.sentence_index, v.score]));
        return {{
            label: character,
            data: labels.map(x => valueMap[x] ?? null),
            tension: 0.3
        }};
    }});

    new Chart(document.getElementById("arcChart"), {{
        type: "line",
        data: {{
            labels,
            datasets
        }},
        options: {{
            responsive: true,
            scales: {{
                y: {{
                    beginAtZero: true,
                    max: 1
                }}
            }}
        }}
    }});
}}

function renderTimeline() {{
    const box = document.getElementById("timelineBox");
    const rows = DASHBOARD_DATA.timelineData || [];

    if (!rows.length) {{
        box.innerHTML = "<p>No timeline data available.</p>";
        return;
    }}

    box.innerHTML = rows.map(ev => `
        <div class="timeline-item">
            <h4>${{ev.chapter_id}} · ${{ev.chapter_title || "-"}}</h4>

            <div class="timeline-meta">
                <span><strong>Scene:</strong> ${{ev.scene_id}} · ${{ev.scene_title || "-"}}</span>
                <span><strong>POV:</strong> ${{ev.scene_pov || "-"}}</span>
                <span><strong>Mood:</strong> ${{ev.scene_mood || "-"}}</span>
                <span><strong>Location:</strong> ${{ev.event_location || ev.scene_location || "-"}}</span>
            </div>

            <h3>${{ev.event_id}} · Event #${{ev.seq}}</h3>
            <p><strong>Event Summary:</strong> ${{ev.summary || "-"}}</p>
            <p><strong>Scene Summary:</strong> ${{ev.scene_summary || "-"}}</p>

            <details>
                <summary>Show extracted details</summary>
                <p><strong>Participants:</strong> ${{(ev.participants || []).join(", ") || "-"}}</p>
                <p><strong>Acquired Items:</strong> ${{(ev.acquired_items || []).join(", ") || "-"}}</p>
                <p><strong>Used Items:</strong> ${{(ev.used_items || []).join(", ") || "-"}}</p>
                <p><strong>Revelations:</strong> ${{(ev.revelations || []).join(", ") || "-"}}</p>
                <p><strong>Knowledge Gains:</strong> ${{JSON.stringify(ev.knowledge_gains || {{}})}}</p>
                <p><strong>Causal Parents:</strong> ${{(ev.causal_parents || []).join(", ") || "-"}}</p>
                <p><strong>Start Time:</strong> ${{ev.start_time || "-"}}</p>
                <p><strong>End Time:</strong> ${{ev.end_time || "-"}}</p>
                <p><strong>Scene Text:</strong> ${{ev.scene_text || "-"}}</p>
            </details>
        </div>
    `).join("");
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
        .range(["#111827", "#2563eb", "#7c3aed", "#059669", "#d97706", "#dc2626", "#0891b2"]);

    const simulation = d3.forceSimulation(data.nodes)
        .force("link", d3.forceLink(data.links).id(d => d.id).distance(95))
        .force("charge", d3.forceManyBody().strength(-350))
        .force("center", d3.forceCenter(width / 2, height / 2));

    const link = svg.append("g")
        .selectAll("line")
        .data(data.links)
        .enter()
        .append("line")
        .attr("stroke", "#cbd5e1")
        .attr("stroke-width", 1.5);

    const node = svg.append("g")
        .selectAll("circle")
        .data(data.nodes)
        .enter()
        .append("circle")
        .attr("r", 8)
        .attr("fill", d => color(d.group))
        .call(drag(simulation));

    const label = svg.append("g")
        .selectAll("text")
        .data(data.nodes)
        .enter()
        .append("text")
        .attr("class", "node-label")
        .text(d => d.label)
        .attr("dx", 11)
        .attr("dy", 4);

    node.append("title").text(d => `${{d.label}} (${{d.group}})`);

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

renderArcChart();
renderTimeline();
</script>

</body>
</html>
"""

    Path(output_path).write_text(html, encoding="utf-8")