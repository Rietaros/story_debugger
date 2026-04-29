# semantic.py
from __future__ import annotations

from collections import defaultdict
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import os

os.environ["HF_TOKEN"] = "your_huggingface_token"


class SemanticContinuity:
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]):
        return self.model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

    def centroid_similarity(self, left_texts: list[str], right_texts: list[str]) -> float:
        left = self.encode(left_texts).mean(dim=0, keepdim=True)
        right = self.encode(right_texts).mean(dim=0, keepdim=True)
        return float(util.cos_sim(left, right)[0][0])

    def chapter_drift(
        self,
        comparisons: list[tuple[str, list[str], list[str]]],
    ) -> pd.DataFrame:
        rows = []
        for label, early_texts, late_texts in comparisons:
            sim = self.centroid_similarity(early_texts, late_texts)
            rows.append(
                {
                    "label": label,
                    "similarity": round(sim, 4),
                    "drift_score": round(1.0 - sim, 4),
                }
            )
        return pd.DataFrame(rows).sort_values("similarity")

    def character_voice_consistency(
        self,
        excerpts_by_character: dict[str, list[str]],
    ) -> pd.DataFrame:
        rows = []
        for character, excerpts in excerpts_by_character.items():
            if len(excerpts) < 2:
                continue

            emb = self.encode(excerpts)
            sim_matrix = util.cos_sim(emb, emb).cpu().numpy()

            for idx, excerpt in enumerate(excerpts):
                mean_sim = float((sim_matrix[idx].sum() - 1.0) / (len(excerpts) - 1))
                rows.append(
                    {
                        "character": character,
                        "excerpt_index": idx,
                        "mean_voice_similarity": round(mean_sim, 4),
                        "excerpt_preview": excerpt[:120].replace("\n", " "),
                    }
                )
        return pd.DataFrame(rows)



