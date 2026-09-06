from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping


@dataclass(frozen=True)
class FlavorExperience:
    primary_tastes: Dict[str, float]
    aromatic_notes: Dict[str, float]
    sensations: Dict[str, float]
    overall_intensity: float
    finish: str


PRIMARY_TASTE_RULES = {
    "sweet": {
        "sucrose": 1.0,
        "glucose": 0.85,
        "fructose": 1.1,
        "maltol": 0.35,
        "vanillin": 0.15,
    },
    "sour": {
        "citric_acid": 1.0,
        "malic_acid": 0.85,
        "lactic_acid": 0.65,
        "acetic_acid": 0.75,
    },
    "bitter": {
        "caffeine": 1.0,
        "theobromine": 0.7,
        "quinine": 1.2,
        "catechin": 0.45,
    },
    "salty": {
        "sodium_chloride": 1.0,
        "potassium_chloride": 0.8,
    },
    "umami": {
        "glutamate": 1.0,
        "inosinate": 0.75,
        "guanylate": 0.75,
    },
}

AROMATIC_RULES = {
    "citrus": {"limonene": 1.0, "citral": 0.9},
    "floral": {"linalool": 1.0, "geraniol": 0.85, "nerol": 0.7},
    "fruity": {"isoamyl_acetate": 1.0, "ethyl_butyrate": 0.9, "ethyl_acetate": 0.35},
    "vanilla": {"vanillin": 1.0, "ethyl_vanillin": 1.1},
    "roasted": {"pyrazines": 1.0, "furaneol": 0.65},
    "earthy": {"geosmin": 1.0},
}

SENSATION_RULES = {
    "heat": {"capsaicin": 1.0, "piperine": 0.7},
    "cooling": {"menthol": 1.0},
    "astringent": {"tannin": 1.0, "catechin": 0.75},
    "creamy": {"diacetyl": 1.0},
}


def _normalize(score: float) -> float:
    return round(max(0.0, min(score / 100.0, 1.0)), 3)


def _score_rule_set(
    composition: Mapping[str, float],
    rules: Mapping[str, Mapping[str, float]],
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for label, compounds in rules.items():
        total = 0.0
        for compound, weight in compounds.items():
            total += composition.get(compound, 0.0) * weight
        normalized = _normalize(total)
        if normalized > 0:
            scores[label] = normalized
    return scores


def _finish(primary_tastes: Mapping[str, float], sensations: Mapping[str, float]) -> str:
    heat = sensations.get("heat", 0)
    cooling = sensations.get("cooling", 0)
    if heat >= 0.35 or cooling >= 0.35:
        if cooling >= heat:
            return "cooling"
        return "warming"
    top_score = max(primary_tastes.values(), default=0)
    leaders = {taste for taste, score in primary_tastes.items() if score == top_score}
    if top_score > 0 and "sweet" in leaders:
        return "rounded sweet"
    if top_score > 0 and "bitter" in leaders:
        return "lingering bitter"
    if top_score > 0 and "sour" in leaders:
        return "bright tart"
    if primary_tastes.get("sour", 0) >= 0.3:
        return "bright tart"
    return "clean"


def _overall_intensity(*score_sets: Iterable[float], total_dimensions: int) -> float:
    if total_dimensions <= 0:
        return 0.0
    total = sum(score for scores in score_sets for score in scores)
    return round(total / total_dimensions, 3)


def predict_flavor_experience(composition: Mapping[str, float]) -> FlavorExperience:
    """
    Predict the likely flavor experience from a molecular composition.

    Composition values are relative concentrations on a 0-100 scale.
    Unknown compounds are ignored.
    Returned scores are normalized to the 0.0-1.0 range, rounded to three
    decimals, capped at 1.0 for high inputs, and clamped to 0.0 for negatives.
    Overall intensity is the average across all modeled taste, aroma, and
    sensation dimensions, with absent dimensions treated as zero. A single
    compound may contribute to multiple dimensions when it appears in more than
    one rule set. When strong heat and cooling sensations tie, cooling wins the
    finish label.
    """

    primary_tastes = _score_rule_set(composition, PRIMARY_TASTE_RULES)
    aromatic_notes = _score_rule_set(composition, AROMATIC_RULES)
    sensations = _score_rule_set(composition, SENSATION_RULES)

    return FlavorExperience(
        primary_tastes=primary_tastes,
        aromatic_notes=aromatic_notes,
        sensations=sensations,
        overall_intensity=_overall_intensity(
            primary_tastes.values(),
            aromatic_notes.values(),
            sensations.values(),
            total_dimensions=(
                len(PRIMARY_TASTE_RULES) + len(AROMATIC_RULES) + len(SENSATION_RULES)
            ),
        ),
        finish=_finish(primary_tastes, sensations),
    )
