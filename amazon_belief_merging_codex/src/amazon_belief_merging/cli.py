from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from .discover_aspects import discover_candidate_aspects
from .pipeline import run_pipeline


DEFAULT_ASPECTS = Path(__file__).resolve().parents[2] / "configs" / "aspects_amazon_general.json"


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Apply belief-level aggregation to Amazon reviews.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Extract aspect beliefs, merge by product, and write JSONL summaries.")
    run.add_argument("--input", required=True, help="CSV/TSV/JSON/JSONL reviews file.")
    run.add_argument("--output", required=True, help="Output JSONL path.")
    run.add_argument(
        "--aspects",
        action="append",
        default=None,
        help="Aspect schema JSON. Repeat to layer core plus category-specific packs.",
    )
    run.add_argument("--product-col", default=None)
    run.add_argument("--text-col", default=None)
    run.add_argument("--rating-col", default=None)
    run.add_argument("--review-id-col", default=None)
    run.add_argument("--embedding-model", default=None, help="Optional sentence-transformers model name.")
    run.add_argument("--max-reviews-per-product", type=int, default=None)
    run.add_argument("--min-reviews-per-product", type=int, default=2)
    run.add_argument("--llm-extract", action="store_true", help="Use an LLM as the primary aspect sentiment extractor.")
    run.add_argument("--llm-refine", action="store_true", help="Use an LLM to refine heuristic extraction and generate summaries.")
    run.add_argument("--llm-model", default="gpt-4.1-mini", help="OpenAI model name (default: gpt-4.1-mini).")
    run.add_argument(
        "--openai-batch-poll-interval",
        type=float,
        default=30.0,
        help="Seconds between OpenAI Batch API status checks (default: 30).",
    )
    run.add_argument(
        "--openai-batch-timeout",
        type=float,
        default=None,
        help="Optional seconds to wait for each OpenAI batch before timing out.",
    )
    run.add_argument("--hf-model", default=None, help="Hugging Face model name for HF refinement or extraction.")
    run.add_argument("--hf-inference", action="store_true", help="Use Hugging Face Inference API instead of local model.")
    run.add_argument("--hf-token-env", default="HF_TOKEN", help="Environment variable holding the HF token.")

    discover = sub.add_parser("discover-aspects", help="Mine frequent candidate aspect terms from reviews.")
    discover.add_argument("--input", required=True)
    discover.add_argument("--output", required=True)
    discover.add_argument("--text-col", default=None)
    discover.add_argument("--top-n", type=int, default=60)

    args = parser.parse_args()
    if args.command == "run":
        run_pipeline(
            input_path=args.input,
            aspect_path=args.aspects or [str(DEFAULT_ASPECTS)],
            output_path=args.output,
            embedding_model=args.embedding_model,
            product_col=args.product_col,
            text_col=args.text_col,
            rating_col=args.rating_col,
            review_id_col=args.review_id_col,
            max_reviews_per_product=args.max_reviews_per_product,
            min_reviews_per_product=args.min_reviews_per_product,
            use_llm_extraction=args.llm_extract,
            use_llm_refinement=args.llm_refine,
            llm_model=args.llm_model,
            hf_model=args.hf_model,
            use_hf_inference=args.hf_inference,
            hf_token_env=args.hf_token_env,
            openai_batch_poll_interval=args.openai_batch_poll_interval,
            openai_batch_timeout=args.openai_batch_timeout,
        )
    elif args.command == "discover-aspects":
        discover_candidate_aspects(args.input, args.output, text_col=args.text_col, top_n=args.top_n)


if __name__ == "__main__":
    main()
