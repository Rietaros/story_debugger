<img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/79edce8d-e457-4ca7-a7d3-f3e693b6986c" /># 📖 Narrative Intelligence System  
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
## Requirements

### Core
- pydantic>=2.6
- pandas>=2.0
- numpy>=1.24

### LLM (OpenAI)
- openai>=1.0

### NLP
- spacy>=3.7

### Emotion model (HuggingFace)
- transformers>=4.40
- torch>=2.1

### Sentence similarity
- sentence-transformers>=2.6

### Visualization
- matplotlib>=3.7

### Graph processing
- networkx>=3.2

### Optional (recommended for performance)
- scikit-learn>=1.3

---

## 🧠 Core Concept

Story Text → LLM Extraction → Structured Events → Lore Graph → Rule Engine → Issues Report

---

## 📁 Project Structure

├── story/  
    -│     ├── chapter_001.md     
    -│     ├── chapter_002.md    
    -│     └── chapter_003.md    
├── outputs/  
├── schemas.py  
├── config.py  
├── html_report.py  
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

### 📄 config.py
Config your OpenAI API

---

## ▶️ Running the Pipeline
### 📄 run_main.ipynb


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

## 🪄 V 1.1

Add .html output to visualize the analysis

---

"We don’t just write stories anymore — we test them."

---
