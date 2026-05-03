from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np


@dataclass
class GranularityLevel:
    name: str
    positive_class_ids: Set[int]
    description: str = ""


@dataclass
class VisionHierarchy:
    """
    A hierarchy of granularity levels for vision classification.
    Each level defines which class IDs are "positive" (target) vs "negative" (rest).
    Levels go from narrowest (most specific) to broadest.
    """
    levels: List[GranularityLevel]
    class_names: Dict[int, str]

    def get_binary_labels(self, labels: np.ndarray, level_idx: int) -> np.ndarray:
        pos_ids = self.levels[level_idx].positive_class_ids
        return np.isin(labels, list(pos_ids)).astype(np.int32)

    def subsample_balanced(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        level_idx: int,
        n_per_class: int | None = None,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        binary = self.get_binary_labels(labels, level_idx)
        pos_idx = np.where(binary == 1)[0]
        neg_idx = np.where(binary == 0)[0]

        rng = np.random.RandomState(seed)
        if n_per_class is None:
            n_per_class = min(len(pos_idx), len(neg_idx))

        pos_sample = rng.choice(pos_idx, size=min(n_per_class, len(pos_idx)), replace=False)
        neg_sample = rng.choice(neg_idx, size=min(n_per_class, len(neg_idx)), replace=False)
        idx = np.concatenate([pos_sample, neg_sample])
        rng.shuffle(idx)
        return X[idx], binary[idx]


def get_cifar100_hierarchy() -> VisionHierarchy:
    """
    CIFAR-100 built-in hierarchy: 20 superclasses, 100 fine classes.
    Returns a hierarchy with multiple granularity levels using the
    "large_carnivores" branch as the example.

    Fine classes per superclass (5 each):
        large_carnivores: bear(3), leopard(42), lion(43), tiger(88), wolf(97)
    """
    # CIFAR-100 fine label IDs for large_carnivores
    BEAR, LEOPARD, LION, TIGER, WOLF = 3, 42, 43, 88, 97
    large_carnivores = {BEAR, LEOPARD, LION, TIGER, WOLF}

    # Other animal superclasses for broader levels
    aquatic_mammals = {4, 30, 55, 72, 95}  # beaver, dolphin, otter, seal, whale
    small_mammals = {15, 31, 37, 63, 77}   # hamster, mouse, rabbit, shrew, squirrel
    insects = {0, 51, 53, 57, 83}          # bee, mushroom(?), ... (approximate)
    reptiles = {26, 29, 44, 71, 82}        # crocodile, dinosaur, lizard, snake, turtle
    fish = {1, 32, 67, 73, 91}             # aquarium_fish, flatfish, ray, shark, trout
    all_animals = large_carnivores | aquatic_mammals | small_mammals | reptiles | fish

    class_names = {
        BEAR: "bear", LEOPARD: "leopard", LION: "lion",
        TIGER: "tiger", WOLF: "wolf",
    }

    levels = [
        GranularityLevel(
            name="tiger",
            positive_class_ids={TIGER},
            description="Just tiger vs everything else",
        ),
        GranularityLevel(
            name="big_cat",
            positive_class_ids={LEOPARD, LION, TIGER},
            description="Leopard + lion + tiger vs everything else",
        ),
        GranularityLevel(
            name="large_carnivore",
            positive_class_ids=large_carnivores,
            description="Bear + leopard + lion + tiger + wolf vs everything else",
        ),
        GranularityLevel(
            name="animal",
            positive_class_ids=all_animals,
            description="All animal superclasses vs non-animals",
        ),
    ]

    return VisionHierarchy(levels=levels, class_names=class_names)
