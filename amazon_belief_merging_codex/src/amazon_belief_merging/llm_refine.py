from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any

from .extraction import AspectOpinion, ReviewBeliefBase
from .schema import Aspect
from .summarise import build_llm_prompt
from .merge import MergedBeliefBase


class OpenAIRefiner:
    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        *,
        batch_poll_interval: float = 30.0,
        batch_timeout: float | None = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the optional llm dependency: pip install -e '.[llm]'") from exc
        if not os.environ.get("GPT_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = OpenAI()
        self.model = model
        self.batch_poll_interval = batch_poll_interval
        self.batch_timeout = batch_timeout

    def extract_review_belief_base(
        self,
        *,
        review_text: str,
        product_id: str,
        review_id: str,
        rating: float | None,
        aspects: list[Aspect],
        initial_base: ReviewBeliefBase | None = None,
    ) -> ReviewBeliefBase:
        prompt = aspect_sentiment_extraction_prompt(review_text, rating, aspects, initial_base)
        data = self._json_chat(prompt)
        opinions = parse_llm_opinions(data, aspects)
        return ReviewBeliefBase(
            review_id=review_id,
            product_id=product_id,
            rating=rating,
            opinions=opinions,
        )

    def extract_review_belief_bases(
        self,
        records: list[dict[str, Any]],
        aspects: list[Aspect],
    ) -> list[ReviewBeliefBase]:
        prompts = [
            aspect_sentiment_extraction_prompt(
                str(record["review_text"]),
                record["rating"],
                aspects,
                record.get("initial_base"),
            )
            for record in records
        ]
        responses = self._json_chat_batch(prompts, task_name="extract")
        bases = []
        for record, data in zip(records, responses, strict=True):
            opinions = parse_llm_opinions(data, aspects)
            bases.append(
                ReviewBeliefBase(
                    review_id=str(record["review_id"]),
                    product_id=str(record["product_id"]),
                    rating=record["rating"],
                    opinions=opinions,
                )
            )
        return bases

    def refine_review_belief_base(self, base: ReviewBeliefBase, review_text: str, aspects: list[Aspect]) -> ReviewBeliefBase:
        prompt = extraction_refinement_prompt(base, review_text, aspects)
        data = self._json_chat(prompt)
        opinions = parse_llm_opinions(data, aspects)
        return ReviewBeliefBase(
            review_id=base.review_id,
            product_id=base.product_id,
            rating=base.rating,
            opinions=opinions or base.opinions,
        )

    def refine_review_belief_bases(
        self,
        records: list[dict[str, Any]],
        aspects: list[Aspect],
    ) -> list[ReviewBeliefBase]:
        prompts = [
            extraction_refinement_prompt(record["initial_base"], str(record["review_text"]), aspects)
            for record in records
        ]
        responses = self._json_chat_batch(prompts, task_name="refine")
        bases = []
        for record, data in zip(records, responses, strict=True):
            base = record["initial_base"]
            opinions = parse_llm_opinions(data, aspects)
            bases.append(
                ReviewBeliefBase(
                    review_id=base.review_id,
                    product_id=base.product_id,
                    rating=base.rating,
                    opinions=opinions or base.opinions,
                )
            )
        return bases

    def summarise(self, merged: MergedBeliefBase) -> str:
        data = self._json_chat(build_llm_prompt(merged))
        return str(data.get("summary", "")).strip()

    def summarise_many(self, merged_bases: list[MergedBeliefBase]) -> list[str]:
        responses = self._json_chat_batch(
            [build_llm_prompt(merged) for merged in merged_bases],
            task_name="summarise",
        )
        return [str(data.get("summary", "")).strip() for data in responses]

    def _json_chat(self, prompt: str) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            # temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return self._parse_json_text(response.choices[0].message.content or "{}")

    def _json_chat_batch(self, prompts: list[str], *, task_name: str) -> list[dict[str, Any]]:
        if not prompts:
            return []
        custom_ids = [f"{task_name}-{idx}" for idx in range(len(prompts))]
        requests = [
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
            for custom_id, prompt in zip(custom_ids, prompts, strict=True)
        ]
        with tempfile.NamedTemporaryFile("w+b", suffix=".jsonl") as handle:
            for request in requests:
                handle.write((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
            handle.flush()
            handle.seek(0)
            input_file = self.client.files.create(file=handle.file, purpose="batch")

        batch = self.client.batches.create(
            input_file_id=input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        batch = self._wait_for_batch(batch.id)
        if not getattr(batch, "output_file_id", None):
            message = f"OpenAI batch {batch.id} finished without an output file."
            if getattr(batch, "error_file_id", None):
                message += f" Error file: {batch.error_file_id}."
            raise RuntimeError(message)

        output_text = self._download_file_text(batch.output_file_id)
        by_custom_id: dict[str, dict[str, Any]] = {}
        for line in output_text.splitlines():
            if not line.strip():
                continue
            result = json.loads(line)
            custom_id = result["custom_id"]
            if result.get("error"):
                raise RuntimeError(f"OpenAI batch request {custom_id} failed: {result['error']}")
            response = result.get("response", {})
            if response.get("status_code") != 200:
                raise RuntimeError(f"OpenAI batch request {custom_id} returned {response.get('status_code')}: {response}")
            by_custom_id[custom_id] = response.get("body", {})
        return [
            self._parse_json_text(by_custom_id[custom_id]["choices"][0]["message"].get("content") or "{}")
            for custom_id in custom_ids
        ]

    def _wait_for_batch(self, batch_id: str) -> Any:
        started = time.monotonic()
        terminal_statuses = {"completed", "failed", "expired", "cancelled"}
        while True:
            batch = self.client.batches.retrieve(batch_id)
            status = getattr(batch, "status", None)
            if status in terminal_statuses:
                if status != "completed":
                    raise RuntimeError(f"OpenAI batch {batch_id} ended with status {status}.")
                return batch
            if self.batch_timeout is not None and time.monotonic() - started > self.batch_timeout:
                raise TimeoutError(f"Timed out waiting for OpenAI batch {batch_id}.")
            time.sleep(self.batch_poll_interval)

    def _download_file_text(self, file_id: str) -> str:
        content = self.client.files.content(file_id)
        if hasattr(content, "text"):
            text = content.text
            return text() if callable(text) else text
        if hasattr(content, "content"):
            data = content.content
        elif hasattr(content, "read"):
            data = content.read()
        else:
            data = content
        if isinstance(data, bytes):
            return data.decode("utf-8")
        return str(data)

    def _parse_json_text(self, text: str) -> dict[str, Any]:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= start:
            text = text[start : end + 1]
        return json.loads(text)


class HFRefiner:
    def __init__(
        self,
        model: str,
        *,
        use_inference_api: bool = False,
        token_env: str = "HF_TOKEN",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.use_inference_api = use_inference_api
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.token = os.environ.get(token_env)
        if use_inference_api and not self.token:
            raise RuntimeError(f"{token_env} is not set.")

        self._client = None
        self._tokenizer = None
        self._model = None

        if use_inference_api:
            try:
                from huggingface_hub import InferenceClient
            except ImportError as exc:
                raise RuntimeError("Install the optional hf dependency: pip install -e '.[hf]'") from exc
            self._client = InferenceClient(model=self.model, token=self.token)
        else:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError("Install the optional hf dependency: pip install -e '.[hf]'") from exc
            self._tokenizer = AutoTokenizer.from_pretrained(self.model)
            self._model = AutoModelForCausalLM.from_pretrained(self.model)

    def extract_review_belief_base(
        self,
        *,
        review_text: str,
        product_id: str,
        review_id: str,
        rating: float | None,
        aspects: list[Aspect],
        initial_base: ReviewBeliefBase | None = None,
    ) -> ReviewBeliefBase:
        prompt = aspect_sentiment_extraction_prompt(review_text, rating, aspects, initial_base)
        data = self._json_chat(prompt)
        opinions = parse_llm_opinions(data, aspects)
        return ReviewBeliefBase(
            review_id=review_id,
            product_id=product_id,
            rating=rating,
            opinions=opinions,
        )

    def refine_review_belief_base(self, base: ReviewBeliefBase, review_text: str, aspects: list[Aspect]) -> ReviewBeliefBase:
        prompt = extraction_refinement_prompt(base, review_text, aspects)
        data = self._json_chat(prompt)
        opinions = parse_llm_opinions(data, aspects)
        return ReviewBeliefBase(
            review_id=base.review_id,
            product_id=base.product_id,
            rating=base.rating,
            opinions=opinions or base.opinions,
        )

    def summarise(self, merged: MergedBeliefBase) -> str:
        data = self._json_chat(build_llm_prompt(merged))
        return str(data.get("summary", "")).strip()

    def _json_chat(self, prompt: str) -> dict[str, Any]:
        text = self._generate(prompt)
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= start:
            text = text[start : end + 1]
        return json.loads(text or "{}")

    def _generate(self, prompt: str) -> str:
        if self.use_inference_api:
            return self._generate_inference(prompt)
        return self._generate_local(prompt)

    def _generate_inference(self, prompt: str) -> str:
        if not self._client:
            raise RuntimeError("Inference client not initialized.")
        try:
            response = self._client.text_generation(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                return_full_text=False,
            )
        except Exception as exc:
            raise RuntimeError(f"HF Inference API call failed: {exc}") from exc
        return str(response)

    def _generate_local(self, prompt: str) -> str:
        if hasattr(self._tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            enc = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
        else:
            enc = self._tokenizer(prompt, return_tensors="pt")

        enc = enc.to(self._model.device)

        outputs = self._model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
        )

        generated = outputs[0][enc["input_ids"].shape[-1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)


def parse_llm_opinions(data: dict[str, Any], aspects: list[Aspect]) -> dict[str, AspectOpinion]:
    valid_aspects = {a.name for a in aspects}
    opinions = {}
    for item in data.get("opinions", []):
        aspect = item.get("aspect")
        if aspect not in valid_aspects:
            continue
        try:
            score = float(item.get("score", 0.5))
        except (TypeError, ValueError):
            score = 0.5
        score = max(0.0, min(1.0, score))
        polarity = item.get("polarity") or score_to_polarity(score)
        if polarity not in {"positive", "negative", "mixed", "neutral"}:
            polarity = score_to_polarity(score)
        evidence = item.get("evidence") or []
        if isinstance(evidence, str):
            evidence = [evidence]
        try:
            confidence = float(item.get("confidence", 0.75))
        except (TypeError, ValueError):
            confidence = 0.75
        opinions[aspect] = AspectOpinion(
            aspect=aspect,
            score=round(score, 4),
            polarity=polarity,
            evidence=[str(e) for e in evidence[:3]],
            confidence=round(max(0.0, min(1.0, confidence)), 4),
        )
    return opinions


def score_to_polarity(score: float) -> str:
    if score >= 0.62:
        return "positive"
    if score <= 0.38:
        return "negative"
    return "mixed"


def aspect_sentiment_extraction_prompt(
    review_text: str,
    rating: float | None,
    aspects: list[Aspect],
    initial_base: ReviewBeliefBase | None = None,
) -> str:
    aspect_payload = [
        {
            "name": a.name,
            "description": a.description,
            "keywords": list(a.keywords),
        }
        for a in aspects
    ]
    initial = {}
    if initial_base:
        initial = {
            name: {
                "score": opinion.score,
                "polarity": opinion.polarity,
                "evidence": opinion.evidence,
            }
            for name, opinion in initial_base.opinions.items()
        }
    return f"""You are an aspect-based sentiment extractor for Amazon product reviews.

Extract only opinions explicitly supported by the review text. Use the fixed aspect schema below; do not create new aspects. Omit aspects that are not discussed.

For each extracted aspect:
- score must be a float in [0, 1], where 1.0 is strongly positive, 0.0 is strongly negative, and 0.5 is neutral/mixed.
- polarity must be one of: positive, negative, mixed, neutral.
- confidence must be in [0, 1].
- evidence must contain short exact snippets or very close paraphrases from the review.

Important rules:
- Handle contrastive clauses separately. Example: "great sound but terrible battery" is positive for sound_quality and negative for battery_life.
- Do not let the star rating override explicit text. Use it only as weak context when the wording is ambiguous.
- If an aspect has both praise and criticism in the same review, use polarity "mixed" and a score near 0.5.
- If the text says a product arrived broken, map that to shipping_packaging only when the complaint is about arrival/packaging; map it to product_quality or reliability_durability when it is about the item itself.

Aspect schema:
{json.dumps(aspect_payload, indent=2)}

Review rating: {rating}

Review text:
{review_text}

Optional initial heuristic extraction, for hints only:
{json.dumps(initial, indent=2)}

Return JSON only:
{{
  "opinions": [
    {{
      "aspect": "aspect_name_from_schema",
      "score": 0.0,
      "polarity": "negative",
      "confidence": 0.0,
      "evidence": ["short supported snippet"]
    }}
  ]
}}
"""


def extraction_refinement_prompt(base: ReviewBeliefBase, review_text: str, aspects: list[Aspect]) -> str:
    aspect_payload = [{"name": a.name, "description": a.description} for a in aspects]
    initial = {
        name: {
            "score": opinion.score,
            "polarity": opinion.polarity,
            "evidence": opinion.evidence,
        }
        for name, opinion in base.opinions.items()
    }
    return f"""Refine aspect-level opinions for one Amazon review using a fixed aspect schema.

Use only the review text as evidence. Keep scores in [0,1], where 1 is positive and 0 is negative. Omit aspects not discussed.

Aspect schema:
{json.dumps(aspect_payload, indent=2)}

Review rating: {base.rating}
Review text:
{review_text}

Initial embedding/lexicon extraction:
{json.dumps(initial, indent=2)}

Return JSON only:
{{
  "opinions": [
    {{"aspect": "product_quality", "score": 0.0, "polarity": "negative", "confidence": 0.0, "evidence": ["short quote or paraphrase"]}}
  ]
}}
"""
