from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .embeddings import build_embedder
from .extraction import AspectOpinionExtractor, ReviewBeliefBase
from .llm_refine import HFRefiner, OpenAIRefiner
from .merge import merge_product_beliefs
from .schema import Aspect, load_aspects
from .summarise import merged_beliefs_to_dict, template_summary
from tqdm import tqdm

PRODUCT_COLUMNS = ("product_id", "asin", "parent_asin", "item_id")
TEXT_COLUMNS = ("review_text", "reviewBody", "text", "content", "reviews")
RATING_COLUMNS = ("rating", "overall", "stars", "score")


def run_pipeline(
    input_path: str | Path,
    aspect_path: str | Path | list[str | Path],
    output_path: str | Path,
    *,
    embedding_model: str | None = None,
    product_col: str | None = None,
    text_col: str | None = None,
    rating_col: str | None = None,
    review_id_col: str | None = None,
    max_reviews_per_product: int | None = None,
    min_reviews_per_product: int = 2,
    use_llm_extraction: bool = False,
    use_llm_refinement: bool = False,
    llm_model: str = "gpt-4.1-mini",
    hf_model: str | None = None,
    use_hf_inference: bool = False,
    hf_token_env: str = "HF_TOKEN",
    openai_batch_poll_interval: float = 30.0,
    openai_batch_timeout: float | None = None,
) -> None:
    aspects = load_aspects(aspect_path)
    frame = read_reviews(input_path)
    product_col = product_col or infer_column(frame, PRODUCT_COLUMNS, "product id")
    text_col = text_col or infer_column(frame, TEXT_COLUMNS, "review text")
    rating_col = rating_col or first_existing(frame, RATING_COLUMNS)

    embedder = build_embedder(embedding_model)
    extractor = AspectOpinionExtractor(aspects, embedder=embedder)
    refiner = None
    if use_llm_extraction or use_llm_refinement:
        if hf_model or use_hf_inference:
            if not hf_model:
                raise ValueError("hf_model is required when using HF refinement or extraction.")
            refiner = HFRefiner(
                model=hf_model,
                use_inference_api=use_hf_inference,
                token_env=hf_token_env,
            )
        else:
            refiner = OpenAIRefiner(
                llm_model,
                batch_poll_interval=openai_batch_poll_interval,
                batch_timeout=openai_batch_timeout,
            )

    review_records = []
    selected_counts: dict[str, int] = defaultdict(int)
    for idx, row in tqdm(frame.iterrows(), total=len(frame), desc="processing reviews"):
        product_id = str(row[product_col])
        text = str(row[text_col])
        if not text or text.lower() == "nan":
            continue
        rating = parse_float(row[rating_col]) if rating_col else None
        review_id = str(row[review_id_col]) if review_id_col else f"{product_id}:{idx}"
        if max_reviews_per_product and selected_counts[product_id] >= max_reviews_per_product:
            continue
        selected_counts[product_id] += 1
        heuristic_base = extractor.extract(text, product_id=product_id, review_id=review_id, rating=rating)
        review_records.append(
            {
                "product_id": product_id,
                "review_text": text,
                "rating": rating,
                "review_id": review_id,
                "initial_base": heuristic_base,
            }
        )

    if refiner and use_llm_extraction:
        if isinstance(refiner, OpenAIRefiner):
            refined_bases = refiner.extract_review_belief_bases(review_records, aspects)
        else:
            refined_bases = [
                refiner.extract_review_belief_base(
                    review_text=record["review_text"],
                    product_id=record["product_id"],
                    review_id=record["review_id"],
                    rating=record["rating"],
                    aspects=aspects,
                    initial_base=record["initial_base"],
                )
                for record in tqdm(review_records, desc="extracting with llm")
            ]
    elif refiner and use_llm_refinement:
        if isinstance(refiner, OpenAIRefiner):
            refined_bases = refiner.refine_review_belief_bases(review_records, aspects)
        else:
            refined_bases = [
                refiner.refine_review_belief_base(record["initial_base"], record["review_text"], aspects)
                for record in tqdm(review_records, desc="refining with llm")
            ]
    else:
        refined_bases = [record["initial_base"] for record in review_records]

    grouped: dict[str, list[ReviewBeliefBase]] = defaultdict(list)
    for base in refined_bases:
        grouped[base.product_id].append(base)

    merged_rows = []
    for product_id, bases in grouped.items():
        if len(bases) < min_reviews_per_product:
            continue
        merged = merge_product_beliefs(product_id, bases, aspects)
        merged_rows.append(
            {
                "product_id": product_id,
                "review_count": len(bases),
                "merged": merged,
                "merged_beliefs": merged_beliefs_to_dict(merged),
                "review_belief_bases": [belief_base_to_dict(base) for base in bases],
            }
        )

    if refiner and isinstance(refiner, OpenAIRefiner):
        summaries = refiner.summarise_many([row["merged"] for row in merged_rows])
    elif refiner:
        summaries = [refiner.summarise(row["merged"]) for row in tqdm(merged_rows, desc="summarising with llm")]
    else:
        summaries = [template_summary(row["merged"]) for row in merged_rows]

    output = []
    for row, summary in zip(merged_rows, summaries, strict=True):
        output.append(
            {
                "product_id": row["product_id"],
                "review_count": row["review_count"],
                "summary": summary,
                "merged_beliefs": row["merged_beliefs"],
                "review_belief_bases": row["review_belief_bases"],
            }
        )

    write_jsonl(output_path, output)


def read_reviews(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    if path.suffix.lower() in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    raise ValueError(f"Unsupported input format: {path.suffix}. Use csv, tsv, json, or jsonl.")


def infer_column(frame: pd.DataFrame, candidates: Iterable[str], description: str) -> str:
    found = first_existing(frame, candidates)
    if found:
        return found
    raise ValueError(f"Could not infer {description} column. Available columns: {list(frame.columns)}")


def first_existing(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lower_to_original = {str(col).lower(): col for col in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def parse_float(value) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def belief_base_to_dict(base: ReviewBeliefBase) -> dict:
    return {
        "review_id": base.review_id,
        "product_id": base.product_id,
        "rating": base.rating,
        "opinions": {name: asdict(opinion) for name, opinion in base.opinions.items()},
    }


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
