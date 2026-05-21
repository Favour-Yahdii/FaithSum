# FaithSum
This Repo contains code for the paper Faithful Summarisation under Disagreement via Belief-Level Aggregation

## Install

```bash
cd amazon_belief_merging
python -m pip install -e .
```

Optional stronger backends:

```bash
python -m pip install -e '.[embeddings,llm,hf]'
```

## Expected Input

Use CSV, TSV, JSON, or JSONL. 

## Run Example

```bash
amazon-belief-merge run \
  --input reviews.jsonl \
  --output outputs/merged_reviews.jsonl \
  --aspects configs/aspects_amazon_general.json \
  --min-reviews-per-product 5
```

Layer category-specific aspects on top of the core schema by repeating `--aspects`:

```bash
amazon-belief-merge run \
  --input electronics_reviews.jsonl \
  --output outputs/electronics_merged.jsonl \
  --aspects configs/aspects_amazon_general.json \
  --aspects configs/aspect_packs/electronics.json
```

Included packs:

- `configs/aspect_packs/electronics.json`
- `configs/aspect_packs/clothing_shoes.json`
- `configs/aspect_packs/beauty_personal_care.json`
- `configs/aspect_packs/books_media.json`
- `configs/aspect_packs/home_kitchen.json`
- `configs/aspect_packs/grocery_food.json`

With sentence-transformer routing:

```bash
amazon-belief-merge run \
  --input reviews.jsonl \
  --output outputs/merged_reviews.jsonl \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2
```

With OpenAI refinement:

```bash
export OPENAI_API_KEY=...
amazon-belief-merge run \
  --input reviews.jsonl \
  --output outputs/merged_reviews.jsonl \
  --llm-refine \
  --llm-model gpt-4.1-mini
```

OpenAI extraction/refinement and LLM summaries are submitted through the OpenAI Batch API. The runner waits for each batch to complete before writing the output file. You can tune polling for long runs:

```bash
amazon-belief-merge run \
  --input amazon_all_flat.jsonl \
  --output outputs/amazon_merged_gpt_bm_fusion.json \
  --aspects configs/aspects_amazon_general.json \
  --aspects configs/aspect_packs/beauty_personal_care.json \
  --aspects configs/aspect_packs/books_media.json \
  --aspects configs/aspect_packs/clothing_shoes.json \
  --aspects configs/aspect_packs/electronics.json \
  --aspects configs/aspect_packs/grocery_food.json \
  --aspects configs/aspect_packs/home_kitchen.json \
  --llm-extract \
  --llm-model gpt-5
  --openai-batch-poll-interval 60 \
  --openai-batch-timeout 86400
```

## Discover Candidate Aspects

Before fixing the schema for a category, mine frequent terms:

```bash
amazon-belief-merge discover-aspects \
  --input reviews.jsonl \
  --output outputs/candidate_aspects.json
```

Then use the candidates to revise or create a category pack. 

## Belief Merging Details

For each product and aspect, review-level scores are in `[0,1]`, where `1` means positive and `0` means negative. The distance from a binary world value `w_j` to review score `c(a_j)` is:

```text
delta(w_j, c(a_j)) = 1 - c(a_j), if w_j = 1
                     c(a_j),     if w_j = 0
```

The selected world(s) minimize summed L1 distance across reviews. 
