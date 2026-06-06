# schemas.py
from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict


class CharacterRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    canonical: str
    aliases: list[str] = Field(default_factory=list)


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: str
    chapter_id: str
    scene_id: str
    seq: int
    title: str | None = None
    location: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    participants: list[str] = Field(default_factory=list)
    acquired_items: list[str] = Field(default_factory=list)
    used_items: list[str] = Field(default_factory=list)
    revelations: list[str] = Field(default_factory=list)
    knowledge_gains: dict[str, list[str]] = Field(default_factory=dict)
    causal_parents: list[str] = Field(default_factory=list)
    summary: str


class Scene(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scene_id: str
    chapter_id: str
    title: str | None = None
    location: str | None = None
    pov: str | None = None
    mood: str | None = None
    summary: str
    text: str
    events: list[Event] = Field(default_factory=list)


class ChapterExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chapter_id: str
    title: str
    synopsis: str
    scenes: list[Scene]
    characters: list[CharacterRef] = Field(default_factory=list)


class Issue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    severity: str
    rule: str
    message: str
    evidence: dict = Field(default_factory=dict)


class CharacterSheetItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    character: str
    narrative_summary: str
    current_actions: str
    risk_of_plot_hole: str
    acquired_items_or_spells: list[str] = Field(default_factory=list)


class LLMAnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plot_holes: list[Issue] = Field(default_factory=list)
    character_sheet: list[CharacterSheetItem] = Field(default_factory=list)

