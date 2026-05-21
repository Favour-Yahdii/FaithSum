from __future__ import annotations

import json

from .merge import MergedBeliefBase


def merged_beliefs_to_dict(merged: MergedBeliefBase) -> dict:
    return {
        "product_id": merged.product_id,
        "min_distance": merged.min_distance,
        "optimal_world_count": merged.optimal_world_count,
        "optimal_worlds": merged.optimal_worlds,
        "aspects": {
            name: {
                "merged_score": aspect.merged_score,
                "label": aspect.label,
                "prevalence": aspect.prevalence,
                "positive_count": aspect.positive_count,
                "negative_count": aspect.negative_count,
                "mixed_count": aspect.mixed_count,
                "observed_count": aspect.observed_count,
                "evidence": aspect.evidence,
            }
            for name, aspect in merged.aspects.items()
        },
    }


def template_summary(merged: MergedBeliefBase, max_aspects: int = 6) -> str:
    observed = [a for a in merged.aspects.values() if a.observed_count > 0]
    observed.sort(key=lambda a: (a.observed_count, abs(a.positive_count - a.negative_count)), reverse=True)
    pieces = []
    for aspect in observed[:max_aspects]:
        pretty = aspect.aspect.replace("_", " ")
        if aspect.label == "positive":
            prefix = "reviewers lean positive on" if aspect.prevalence == "split" else "reviewers are broadly positive about"
            pieces.append(f"{prefix} {pretty}")
        elif aspect.label == "negative":
            prefix = "reviewers lean negative on" if aspect.prevalence == "split" else "reviewers are broadly negative about"
            pieces.append(f"{prefix} {pretty}")
        elif aspect.label == "contested":
            pieces.append(f"opinions are split on {pretty}")
        elif aspect.label == "mixed":
            pieces.append(f"reviews describe {pretty} in mixed terms")
    if not pieces:
        return "The reviews do not provide enough aspect-level evidence for a grounded summary."
    if len(pieces) == 1:
        return pieces[0].capitalize() + "."
    return "Overall, " + "; ".join(pieces[:-1]) + "; and " + pieces[-1] + "."


def build_llm_prompt(merged: MergedBeliefBase) -> str:
    payload = json.dumps(merged_beliefs_to_dict(merged), indent=2)
    return f"""
You are given natural-language interpretations derived from multiple hypotheses and analyses of amazon product reviews.
Your task is to generate an accurate summarised meta-review that reflects the combined meaning of these interpretations.
Requirements for the summary:
- Capture all key points conveyed across the interpretations.
- Maintain a balanced, neutral tone.
- Represent differing perspectives fairly (positive, negative, mixed).
- Preserve disagreement when an aspect is labelled contested, mixed or split.
- Read fluently like a concise professional summary.
- Do not introduce opinions not grounded in the input.

Merged belief base:
{payload}

Return JSON only:
{{"summary": "..."}}
"""

# BM_FUSION
# You are given natural language interpretations derived from a merged-world representation produced through belief merging on a movie review dataset.

# Your role:
# Act as a **fusor and synthesizer**. Combine all unique insights across the provided aspect-level interpretations into a single, coherent **meta-review**. This meta-review should:
# - Seamlessly integrate the key points from all aspects.
# - Reflect the general tone and balance of the interpretations.
# - Read naturally, like a critic’s concise, well-written summary.
# - Avoid repetition or listing; aim for fluent synthesis.

# Output format (strictly follow this JSON structure):
# {{
#   "Summary": "..."
# }}

# You are given natural-language interpretations derived from multiple hypotheses and analyses of amazon product reviews.
# Your task is to generate an accurate summarised meta-review that reflects the combined meaning of these interpretations.
# Requirements for the summary:
# - Capture all key points conveyed across the interpretations.
# - Maintain a balanced, neutral tone.
# - Represent differing perspectives fairly (positive, negative, mixed).
# - Preserve disagreement when an aspect is labelled contested, mixed or split.
# - Read fluently like a concise professional summary.
# - Do not introduce opinions not grounded in the input.

# Merged belief base:
# {payload}

# Return JSON only:
# {{"summary": "..."}}

# You are writing a concise Amazon review synthesis from an already merged belief base.

# Do not re-aggregate opinions. Do not invent unsupported product claims. 

# Preserve disagreement when an aspect is labelled contested, mixed, or split.

# Merged belief base:
# {payload}

# Return JSON only:
# {{"summary": "..."}}