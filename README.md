# Story Debugger

Story Debugger is a Python-based narrative analysis system for checking long-form fiction, game scripts, visual novels, and serialized stories for continuity problems. It combines LLM-based extraction, rule-based validation, lore graph tracking, semantic drift analysis, and character arc visualization.

The project is designed for stories written in English, Bahasa Indonesia, or a mix of both.

## What It Does

Story Debugger converts chapter text into structured narrative data, then uses that data to generate a debugging report.

Core capabilities:

- Detect plot holes and continuity risks.
- Build a structured event timeline from extracted scenes and events.
- Build a lore graph of characters, scenes, events, locations, items, facts, and causal links.
- Track item ownership, item usage, knowledge, revelations, movement, and causality.
- Detect character location conflicts, item continuity issues, memory/knowledge inconsistencies, and causal problems.
- Generate character emotion arc charts.
- Generate a character sheet for each chapter.
- Produce an interactive HTML report for each analyzed chapter.

## Plot Hole Categories

The LLM analysis classifies higher-level story issues into these categories:

- `contradiction`: A character acts against established traits, or a previously known fact changes to fit a new scene.
- `missing_details`: A vital detail, item, injury, spell, or condition disappears or reappears without explanation.
- `forgotten_subplot`: A character or subplot is introduced with a major unresolved conflict, then abandoned.
- `out_of_character`: A character behaves outside their established nature, often only to move the plot forward.

Rule-based checks also detect concrete continuity issues such as:

- `double_location_same_step`
- `item_used_before_acquired`
- `item_used_by_non_owner`
- `item_ownership_transfer_without_loss`
- `unknown_causal_parent`
- `causality_cycle`
- `memory_claim_without_knowledge_event`
- `possible_explicit_contradiction`

## How The Pipeline Works

```text
Markdown chapter
  -> LLM extraction
  -> structured scenes, events, characters, items, facts
  -> lore graph ingestion
  -> rule-based plot hole detection
  -> semantic drift analysis
  -> character emotion analysis
  -> LLM plot-hole and character-sheet analysis
  -> HTML dashboard and JSON/CSV outputs
```

## Project Structure

```text
.
|-- arcs.py               # Character mention detection and emotion arc plotting
|-- config.py             # Loads environment variables from .env
|-- extract_llm.py        # LLM extraction and narrative analysis
|-- html_report.py        # Interactive HTML dashboard generation
|-- lore_graph.py         # NetworkX lore graph and continuity memory
|-- rules.py              # Rule-based story bug detector
|-- schemas.py            # Pydantic data models
|-- semantic.py           # Semantic similarity and drift scoring
|-- run_main.ipynb        # Main notebook pipeline
|-- story/                # Input story chapters
|-- outputs/              # Generated reports and artifacts
|-- .env.example          # Safe environment template
`-- .gitignore            # Ignores .env, caches, and generated outputs
```

## Configuration

Secrets and runtime configuration are loaded from `.env`.

Create your local environment file:

```bash
cp .env.example .env
```

Then edit `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

`OPENAI_API_KEY` is required for LLM extraction and LLM-based plot-hole analysis. `OPENAI_MODEL` defaults to `gpt-4o-mini` when unset.

The `.env` file is intentionally ignored by Git. Do not commit API keys or private configuration.

## Dependencies

Recommended runtime: Python 3.10 or newer.

Install the core dependencies in your Python environment:

```bash
pip install \
  openai python-dotenv pydantic pandas numpy spacy \
  transformers torch sentence-transformers matplotlib networkx scikit-learn \
  jupyter
```

Depending on your platform, `torch` may require a platform-specific install command. See the official PyTorch installation instructions if the generic install does not work.

## Input Stories

Place chapter files as Markdown files under the configured story folder.

The current notebook uses:

```python
STORY_DIR = Path("story/story_check")
```

Each chapter file should use a stable filename such as:

```text
chapter_001.md
chapter_002.md
chapter_003.md
```

The filename becomes the chapter ID.

## Running The Project

Open and run:

```text
run_main.ipynb
```

The notebook performs the full pipeline:

1. Loads chapter Markdown files.
2. Extracts structured narrative data using OpenAI.
3. Builds and updates the lore graph.
4. Runs rule-based continuity checks.
5. Runs LLM plot-hole and character-sheet analysis.
6. Computes semantic drift.
7. Builds character emotion arcs.
8. Writes JSON, CSV, PNG, and HTML outputs.

## Generated Outputs

For each chapter, the pipeline writes files under:

```text
outputs/<chapter_id>/
```

Typical output files:

```text
extraction.json              # Structured chapter extraction
issues.json                  # Rule-based and LLM-detected story bugs
semantic.json                # Similarity and drift scores
character_sheet.json         # Per-character chapter summary and risk notes
character_emotions.csv       # Emotion classification rows
arc_<character>.png          # Character emotion arc chart
<chapter_id>_analytics.html  # Interactive report
```

## HTML Report

The generated dashboard includes:

- Overview metrics for drift, similarity, issue count, and severity.
- Story Bugs tab with descriptions, evidence, and timeline links.
- Character Sheet tab with character summaries, actions, future plot-hole risks, and acquired items or spells.
- Character Arcs tab with emotion charts and action-aware arc analysis.
- Timeline tab showing every extracted event, graph order, movement, item continuity, knowledge/fact records, and attached bug markers.
- Lore Graph tab showing graph nodes for chapters, scenes, events, characters, locations, items, and facts.

## Data Model Summary

The extraction output is validated with Pydantic models in `schemas.py`.

Main entities:

- `ChapterExtraction`: chapter title, synopsis, scenes, and character roster.
- `Scene`: scene ID, title, location, POV, mood, summary, text, and events.
- `Event`: event ID, title, sequence, participants, location, items, revelations, knowledge gains, causal parents, and summary.
- `Issue`: severity, rule name, message, and evidence.
- `CharacterSheetItem`: character summary, current actions, future plot-hole risk, and acquired items or spells.

## Notes And Limitations

Story Debugger is an analysis assistant, not a final editorial authority. It is strongest when chapters have enough context for the extractor to identify characters, locations, items, and event causality. LLM extraction can still miss details or overgeneralize, so important findings should be reviewed by the writer.

For best results:

- Keep chapter files ordered by filename.
- Use consistent character names or aliases.
- Mention important item transfers, injuries, spells, promises, and revelations explicitly.
- Review `extraction.json` when a report seems incomplete.
- Treat high-severity issues as review priorities, not automatic proof of an error.

## Development Status

This project is actively evolving. Current focus areas include:

- More reliable multilingual extraction for English and Bahasa Indonesia.
- Better event title generation.
- Stronger timeline-to-bug linking.
- Richer lore graph reporting.
- Improved character sheet risk analysis.
