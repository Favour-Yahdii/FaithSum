from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from .extraction import ReviewBeliefBase
from .schema import Aspect


@dataclass
class MergedAspect:
    aspect: str
    merged_score: float
    label: str
    positive_count: int
    negative_count: int
    mixed_count: int
    observed_count: int
    prevalence: str
    evidence: list[str]


@dataclass
class MergedBeliefBase:
    product_id: str
    aspects: dict[str, MergedAspect]
    min_distance: float
    optimal_world_count: int
    optimal_worlds: list[dict[str, int]]


def merge_product_beliefs(
    product_id: str,
    review_bases: list[ReviewBeliefBase],
    aspects: list[Aspect],
    *,
    enumerate_worlds: bool = False,
) -> MergedBeliefBase:
    aspect_names = [a.name for a in aspects]
    per_aspect_scores: dict[str, list[float]] = {name: [] for name in aspect_names}
    per_aspect_evidence: dict[str, list[str]] = {name: [] for name in aspect_names}

    for base in review_bases:
        for aspect_name, opinion in base.opinions.items():
            if aspect_name not in per_aspect_scores:
                continue
            per_aspect_scores[aspect_name].append(float(opinion.score))
            per_aspect_evidence[aspect_name].extend(opinion.evidence[:2])

    selected_values: dict[str, list[int]] = {}
    min_distance = 0.0
    merged: dict[str, MergedAspect] = {}

    for aspect_name in aspect_names:
        scores = per_aspect_scores[aspect_name]
        if not scores:
            selected_values[aspect_name] = [0, 1]
            merged[aspect_name] = MergedAspect(
                aspect=aspect_name,
                merged_score=0.5,
                label="unknown",
                positive_count=0,
                negative_count=0,
                mixed_count=0,
                observed_count=0,
                prevalence="unobserved",
                evidence=[],
            )
            continue

        cost_if_positive = sum(1.0 - s for s in scores)
        cost_if_negative = sum(s for s in scores)
        best_cost = min(cost_if_positive, cost_if_negative)
        min_distance += best_cost

        eps = 1e-9
        values = []
        if abs(cost_if_negative - best_cost) <= eps:
            values.append(0)
        if abs(cost_if_positive - best_cost) <= eps:
            values.append(1)
        selected_values[aspect_name] = values

        merged_score = sum(values) / len(values)
        pos = sum(1 for s in scores if s >= 0.62)
        neg = sum(1 for s in scores if s <= 0.38)
        mix = len(scores) - pos - neg
        merged[aspect_name] = MergedAspect(
            aspect=aspect_name,
            merged_score=round(float(merged_score), 4),
            label=score_label(merged_score, pos, neg, mix),
            positive_count=pos,
            negative_count=neg,
            mixed_count=mix,
            observed_count=len(scores),
            prevalence=prevalence_label(pos, neg, mix),
            evidence=dedupe(per_aspect_evidence[aspect_name])[:4],
        )

    optimal_worlds = []
    world_count = 1
    for values in selected_values.values():
        world_count *= len(values)
    if enumerate_worlds and world_count <= 256:
        for combo in product(*(selected_values[name] for name in aspect_names)):
            optimal_worlds.append(dict(zip(aspect_names, combo)))

    return MergedBeliefBase(
        product_id=product_id,
        aspects=merged,
        min_distance=round(float(min_distance), 4),
        optimal_world_count=world_count,
        optimal_worlds=optimal_worlds,
    )


def score_label(score: float, pos: int, neg: int, mix: int) -> str:
    if score == 0.5 and (pos or neg):
        return "contested"
    if score >= 0.75:
        return "positive"
    if score <= 0.25:
        return "negative"
    if mix:
        return "mixed"
    return "contested"


def prevalence_label(pos: int, neg: int, mix: int) -> str:
    total = pos + neg + mix
    if total == 0:
        return "unobserved"
    leader = max(pos, neg, mix)
    if leader / total >= 0.7:
        if leader == pos:
            return "positive majority"
        if leader == neg:
            return "negative majority"
        return "mixed majority"
    if pos > 0 and neg > 0:
        return "split"
    return "unclear"


def dedupe(items: list[str]) -> list[str]:
    seen = set()
    output = []
    for item in items:
        key = " ".join(item.lower().split())
        if key and key not in seen:
            seen.add(key)
            output.append(item)
    return output
