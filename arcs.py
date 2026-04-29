# arcs.py
from __future__ import annotations

import re

import matplotlib.pyplot as plt
import pandas as pd
import spacy
from transformers import pipeline


INDONESIAN_EMOTION_MAP = {
    "LABEL_0": "marah",
    "LABEL_1": "takut",
    "LABEL_2": "senang",
    "LABEL_3": "sedih",
    "LABEL_4": "terkejut",
    "LABEL_5": "netral",
}

ENGLISH_TO_INDONESIAN_MAP = {
    "anger": "marah",
    "fear": "takut",
    "joy": "senang",
    "sadness": "sedih",
    "surprise": "terkejut",
    "disgust": "jijik",
    "neutral": "netral",
    "positive": "positif",
    "negative": "negatif",
}


def normalize_emotion_label(label: str) -> str:
    label = str(label)

    if label in INDONESIAN_EMOTION_MAP:
        return INDONESIAN_EMOTION_MAP[label]

    lowered = label.lower()
    if lowered in ENGLISH_TO_INDONESIAN_MAP:
        return ENGLISH_TO_INDONESIAN_MAP[lowered]

    return label


def build_roster_nlp(roster: dict[str, list[str]]) -> spacy.Language:
    nlp = spacy.blank("xx")
    nlp.add_pipe("sentencizer")
    ruler = nlp.add_pipe("entity_ruler")

    patterns = []
    for canonical, aliases in roster.items():
        for alias in [canonical, *aliases]:
            patterns.append(
                {
                    "label": "CHARACTER",
                    "pattern": alias,
                    "id": canonical,
                }
            )
    ruler.add_patterns(patterns)
    return nlp


class ArcTracker:
    def __init__(
        self,
        roster: dict[str, list[str]],
        lang: str = "id",
        emotion_model: str | None = None,
    ) -> None:
        self.nlp = build_roster_nlp(roster)

        if emotion_model is None:
            emotion_model = (
                "Chipan/indobert-emotion"
                if lang == "id"
                else "j-hartmann/emotion-english-distilroberta-base"
            )

        self.emotion = pipeline(
            "text-classification",
            model=emotion_model,
            top_k=None,
        )

        print("Model emotion labels:", self.emotion.model.config.id2label)

    def analyze(self, chapter_id: str, text: str) -> pd.DataFrame:
        rows = []
        doc = self.nlp(text)

        for sent_idx, sent in enumerate(doc.sents):
            sentence = sent.text.strip()
            if not sentence:
                continue

            characters = []
            for ent in sent.ents:
                if ent.label_ == "CHARACTER":
                    characters.append(ent.ent_id_ or ent.text)

            if not characters:
                continue

            scores = self.emotion(sentence, truncation=True)[0]
            if isinstance(scores, dict):
                scores = [scores]

            top = max(scores, key=lambda x: x["score"])

            raw_emotion = top["label"]
            mapped_emotion = normalize_emotion_label(raw_emotion)

            for character in sorted(set(characters)):
                rows.append(
                    {
                        "chapter_id": chapter_id,
                        "sentence_index": sent_idx,
                        "character": character,
                        "emotion_raw": raw_emotion,
                        "emotion": mapped_emotion,
                        "score": float(top["score"]),
                        "sentence": sentence,
                    }
                )

        return pd.DataFrame(rows)

    def plot_arc(self, df: pd.DataFrame, character: str, out_path: str) -> None:
        char_df = df[df["character"] == character].copy()
        if char_df.empty:
            return

        plt.figure(figsize=(10, 4))
        plt.plot(char_df["sentence_index"], char_df["score"])

        for _, row in char_df.iterrows():
            plt.text(
                row["sentence_index"],
                row["score"],
                row["emotion"],
                fontsize=8,
            )

        plt.title(f"Emotion arc: {character}")
        plt.xlabel("Sentence index")
        plt.ylabel("Top emotion score")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()