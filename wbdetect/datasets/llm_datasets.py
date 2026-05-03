from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from datasets import load_dataset, Dataset


@dataclass
class LLMGranularityLevel:
    name: str
    positive_categories: Set[str]
    description: str = ""


@dataclass
class LLMHierarchy:
    levels: List[LLMGranularityLevel]

    def get_binary_labels(
        self, categories: List[Dict[str, bool]], level_idx: int
    ) -> np.ndarray:
        pos_cats = self.levels[level_idx].positive_categories
        labels = []
        for cat_dict in categories:
            is_positive = any(cat_dict.get(c, False) for c in pos_cats)
            labels.append(int(is_positive))
        return np.array(labels, dtype=np.int32)


def get_beavertails_hierarchy() -> LLMHierarchy:
    ALL_HARM = {
        "animal_abuse", "child_abuse", "controversial_topics,politics",
        "discrimination,stereotype,injustice", "drug_abuse,weapons,banned_substance",
        "financial_crime,property_crime,theft", "hate_speech,offensive_language",
        "misinformation_regarding_ethics,laws_and_safety",
        "non_violent_unethical_behavior", "privacy_violation",
        "self_harm", "sexually_explicit,adult_content",
        "terrorism,organized_crime", "violence,aiding_and_abetting,incitement",
    }

    VIOLENCE_CLUSTER = {
        "violence,aiding_and_abetting,incitement",
        "terrorism,organized_crime",
        "animal_abuse",
        "child_abuse",
        "self_harm",
    }

    return LLMHierarchy(levels=[
        LLMGranularityLevel(
            name="animal_abuse_only",
            positive_categories={"animal_abuse"},
            description="Single narrow category",
        ),
        LLMGranularityLevel(
            name="violence_cluster",
            positive_categories=VIOLENCE_CLUSTER,
            description="Violence-related categories",
        ),
        LLMGranularityLevel(
            name="all_harmful",
            positive_categories=ALL_HARM,
            description="All 14 harm categories pooled",
        ),
    ])


def load_stemqa(
    subset: str = "biology",
    split: str = "train",
    n: Optional[int] = None,
    seed: int = 42,
) -> Dataset:
    ds = load_dataset("4gate/StemQAMixture", subset, split=split)
    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    return ds


def load_beavertails(
    split: str = "330k_train",
    n: Optional[int] = None,
    seed: int = 42,
) -> Dataset:
    ds = load_dataset("PKU-Alignment/BeaverTails", split=split)
    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    return ds
