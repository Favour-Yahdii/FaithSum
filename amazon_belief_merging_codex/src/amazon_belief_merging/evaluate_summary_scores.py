from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable


DEFAULT_OUTPUTS_DIR = Path(
    "/Users/fyahdii/Desktop/PhD/Year 2/acc-synth/amazon_belief_merging_codex/outputs"
)
DEFAULT_GOLD_PATH = Path(
    "/Users/fyahdii/Desktop/PhD/Year 2/acc-synth/amazon_belief_merging_codex/amazon_all_with_product_id.json"
)

MODEL_NAME_RE = re.compile(r"(gpt|llama|qwen)", re.IGNORECASE)

ID_CANDIDATES = (
    "product_id",
    "asin",
    "id",
    "world_input.product_id",
    "world_input.id",
)
GEN_TEXT_CANDIDATES = (
    "summary",
    "synthesis",
    "Proposed Meta-Review",
    "generated_summary",
    "raw_output",
    "prediction",
    "output",
)
GOLD_TEXT_CANDIDATES = (
    "summary",
    "gold",
    "reference",
    "target",
    "critics_consensus",
    "output",
)


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = [json.loads(line) for line in text.splitlines() if line.strip()]

    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ("data", "instances", "predictions", "results", "items"):
            value = obj.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        values = list(obj.values())
        if values and all(isinstance(x, dict) for x in values):
            return values
    raise ValueError(f"Unsupported JSON shape in {path}: {type(obj).__name__}")


def get_path(record: dict[str, Any], dotted_key: str) -> Any:
    value: Any = record
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def pick_key(records: list[dict[str, Any]], candidates: Iterable[str], label: str) -> str:
    for key in candidates:
        for record in records:
            value = get_path(record, key)
            if value not in (None, ""):
                return key
    raise KeyError(f"Could not find {label}. Tried: {', '.join(candidates)}")


def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return str(value[0]).strip() if value else ""
    return str(value).strip()


def index_by_id(
    records: list[dict[str, Any]],
    *,
    id_key: str | None,
    id_candidates: Iterable[str] = ID_CANDIDATES,
    text_key: str,
    excluded_ids: set[str],
) -> dict[str, str]:
    indexed: dict[str, str] = {}
    for record in records:
        if id_key:
            item_id = coerce_text(get_path(record, id_key))
        else:
            item_id = next(
                (
                    coerce_text(get_path(record, candidate))
                    for candidate in id_candidates
                    if coerce_text(get_path(record, candidate))
                ),
                "",
            )
        text = coerce_text(get_path(record, text_key))
        if item_id and text and item_id not in excluded_ids:
            indexed[item_id] = text
    return indexed


def discover_generated_files(outputs_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in outputs_dir.glob("*.json")
        if MODEL_NAME_RE.search(path.name)
    )


def infer_bartscore_path(files: Iterable[Path], gold_path: Path) -> Path | None:
    roots = []
    for path in [gold_path, *files]:
        roots.extend([path.parent, *path.parents])
    for root in roots:
        candidate = root / "bartscore.py"
        if candidate.exists():
            return root
    return None


def load_bart_scorer(device: str, checkpoint: str, bartscore_path: Path | None):
    if bartscore_path:
        sys.path.insert(0, str(bartscore_path))
    try:
        from bartscore import BARTScorer
    except ImportError as exc:
        raise SystemExit(
            "Could not import bartscore. Pass --bartscore-path pointing to the folder "
            "that contains bartscore.py, or install it in this environment."
        ) from exc
    return BARTScorer(device=device, checkpoint=checkpoint)


def load_bert_scorer(model_type: str, lang: str, batch_size: int):
    try:
        from bert_score import BERTScorer
    except ImportError as exc:
        raise SystemExit(
            "Could not import bert_score. Install it with `python -m pip install bert-score`."
        ) from exc
    return BERTScorer(model_type=model_type, lang=lang, batch_size=batch_size)


def stats(prefix: str, values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
            f"{prefix}_std": None,
        }
    return {
        f"{prefix}_mean": round(mean(values), 6),
        f"{prefix}_min": round(min(values), 6),
        f"{prefix}_max": round(max(values), 6),
        f"{prefix}_std": round(pstdev(values), 6),
    }


def evaluate_file(
    path: Path,
    *,
    gold_by_id: dict[str, str],
    shared_id_key: str | None,
    gen_id_key: str | None,
    gen_text_key: str | None,
    excluded_ids: set[str],
    bart_scorer: Any,
    bert_scorer: Any,
) -> dict[str, Any]:
    generated_records = load_records(path)
    id_key = gen_id_key or shared_id_key or pick_key(generated_records, ID_CANDIDATES, "generated ID key")
    text_key = gen_text_key or pick_key(generated_records, GEN_TEXT_CANDIDATES, "generated text")
    generated_by_id = index_by_id(
        generated_records,
        id_key=id_key,
        text_key=text_key,
        excluded_ids=excluded_ids,
    )

    ids = sorted(set(generated_by_id) & set(gold_by_id))
    candidates = [generated_by_id[item_id] for item_id in ids]
    references = [gold_by_id[item_id] for item_id in ids]
    if not ids:
        return {
            "file": str(path),
            "model": path.stem,
            "id_key": id_key,
            "generated_text_key": text_key,
            "n": 0,
            **stats("bart", []),
            "bert_P_mean": None,
            "bert_R_mean": None,
            "bert_F1_mean": None,
        }

    bart_scores = [float(score) for score in bart_scorer.score(candidates, references)]
    precision, recall, f1 = bert_scorer.score(candidates, references)

    return {
        "file": str(path),
        "model": path.stem,
        "id_key": id_key,
        "generated_text_key": text_key,
        "n": len(ids),
        **stats("bart", bart_scores),
        "bert_P_mean": round(float(precision.mean()), 6),
        "bert_R_mean": round(float(recall.mean()), 6),
        "bert_F1_mean": round(float(f1.mean()), 6),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score generated summaries with BARTScore and BERTScore."
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Generated JSON/JSONL files to score. If omitted, discover model JSON files.",
    )
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_PATH)
    parser.add_argument("--id-key", default=None, help="Shared ID key, e.g. product_id.")
    parser.add_argument("--gold-id-key", default=None, help="Gold ID key if it differs from generated.")
    parser.add_argument("--gen-id-key", default=None, help="Generated ID key if it differs from gold.")
    parser.add_argument("--gold-text-key", default=None, help="Gold summary key.")
    parser.add_argument("--gen-text-key", default=None, help="Generated summary key.")
    parser.add_argument("--exclude-id", action="append", default=[])
    parser.add_argument("--bart-device", default="mps:0")
    parser.add_argument("--bart-checkpoint", default="facebook/bart-large-cnn")
    parser.add_argument("--bartscore-path", type=Path, default=None)
    parser.add_argument("--bert-model", default="roberta-large")
    parser.add_argument("--bert-lang", default="en")
    parser.add_argument("--bert-batch-size", type=int, default=1)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = args.files or discover_generated_files(args.outputs_dir)
    if not files:
        raise SystemExit(f"No generated model JSON files found in {args.outputs_dir}")

    gold_records = load_records(args.gold)
    gold_id_key = args.gold_id_key or args.id_key or pick_key(gold_records, ID_CANDIDATES, "gold ID key")
    gold_text_key = args.gold_text_key or pick_key(gold_records, GOLD_TEXT_CANDIDATES, "gold text")
    excluded_ids = set(args.exclude_id)
    gold_by_id = index_by_id(
        gold_records,
        id_key=gold_id_key,
        text_key=gold_text_key,
        excluded_ids=excluded_ids,
    )

    bartscore_path = args.bartscore_path or infer_bartscore_path(files, args.gold)
    bart_scorer = load_bart_scorer(args.bart_device, args.bart_checkpoint, bartscore_path)
    bert_scorer = load_bert_scorer(args.bert_model, args.bert_lang, args.bert_batch_size)

    rows = [
        evaluate_file(
            path,
            gold_by_id=gold_by_id,
            shared_id_key=args.id_key,
            gen_id_key=args.gen_id_key,
            gen_text_key=args.gen_text_key,
            excluded_ids=excluded_ids,
            bart_scorer=bart_scorer,
            bert_scorer=bert_scorer,
        )
        for path in files
    ]
    rows.sort(key=lambda row: (row["bert_F1_mean"] is None, -(row["bert_F1_mean"] or 0)))

    print(json.dumps(rows, indent=2))
    if args.output_csv:
        write_csv(args.output_csv, rows)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
