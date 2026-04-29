# 📖 Narrative Intelligence System  
### *What If Your Novel Had Unit Tests?*

---

## 🚀 Project Description

This project is a **Narrative Intelligence System** built with Python to help writers, game designers, and storytellers maintain **plot consistency, logical integrity, and character coherence** in long-form narratives (e.g., visual novels, RPGs, serialized fiction).

Instead of relying purely on LLM prompting, this system acts as a **“Story Debugger”**, analyzing structured narrative data and detecting:

- Plot holes  
- Timeline inconsistencies  
- Item misuse (used before acquired)  
- Character contradictions  
- Knowledge paradoxes  

Think of it as:

pytest → for code  
StoryDebugger → for narrative  

---

## 🧠 Core Concept

Story Text → LLM Extraction → Structured Events → Lore Graph → Rule Engine → Issues Report

---

## 📁 Project Structure

├── story/  
│   ├── chapter_001.md  
│   ├── chapter_002.md  
│   └── chapter_003.md  
├── outputs/  
├── schemas.py  
├── extract_llm.py  
├── semantic.py  
├── lore_graph.py  
├── rules.py  
├── arcs.py  

---

## 🧩 Module Breakdown

### 📂 story/
Raw narrative text in Markdown format.

### 📂 outputs/
Generated outputs:
- Emotion arcs
- Debug reports
- CSV summaries

### 📄 schemas.py
Defines structured data models (Pydantic).

### 📄 extract_llm.py
LLM-based extraction of:
- Characters
- Events
- Items
- Knowledge
- Causality

### 📄 semantic.py
Handles semantic similarity and drift analysis.

### 📄 lore_graph.py
Builds knowledge graph:
- Character relations
- Item ownership
- Timeline & causality

### 📄 rules.py
Story Debugger engine:
- Detects plot holes
- Validates logic consistency

### 📄 arcs.py
Character emotion tracking:
- NER
- Sentiment analysis
- Visualization

---

## ▶️ Running the Pipeline

```python
from extract_llm import LLMExtractor
from lore_graph import LoreGraph
from rules import PlotHoleDetector
from semantic import SemanticContinuity
from arcs import ArcTracker
from utils import load_chapters, build_auto_roster

STORY_DIR = "story/"

def main():
    extractor = LLMExtractor()
    semantic = SemanticContinuity()
    lore = LoreGraph()
    detector = PlotHoleDetector()

    extractions = []

    for chapter_id, title, text in load_chapters(STORY_DIR):
        extraction = extractor.extract(chapter_id, title, text)
        extractions.append(extraction)

    roster = build_auto_roster(extractions)
    arc = ArcTracker(roster)

    for extraction in extractions:
        lore.ingest(extraction)
        issues = detector.check(extraction, lore)
        drift = semantic.compute_drift(extraction)

        print(f"[{extraction.chapter_id}] drift={drift:.4f} issues={len(issues)}")

        for issue in issues:
            print("-", issue.rule, issue.message)

if __name__ == "__main__":
    main()
```

---

## 📊 Example Output

[chapter_001] drift=0.0000 issues=3  
- item_used_before_acquired  
- double_location_same_step  
- memory_claim_without_knowledge_event  

---

## 🎯 Key Features

- Narrative → Structured Data  
- Knowledge Graph Tracking  
- Rule-Based Story Debugging  
- Semantic Drift Analysis  
- Character Emotion Arc  

---

## 🧠 Why This Matters

Engineering discipline → storytelling consistency

---

"We don’t just write stories anymore — we test them."
