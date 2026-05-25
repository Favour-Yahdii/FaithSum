from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Aspect:
    name: str
    description: str
    keywords: tuple[str, ...] = ()

    @property
    def embedding_text(self) -> str:
        return " ".join([self.name.replace("_", " "), self.description, *self.keywords])


def load_aspects(path: str | Path | Iterable[str | Path]) -> list[Aspect]:
    paths = [path] if isinstance(path, (str, Path)) else list(path)
    data = []
    for one_path in paths:
        data.extend(json.loads(Path(one_path).read_text(encoding="utf-8")))
    aspects = [
        Aspect(
            name=row["name"],
            description=row.get("description", ""),
            keywords=tuple(row.get("keywords", [])),
        )
        for row in data
    ]
    if not aspects:
        raise ValueError(f"No aspects found in {paths}")
    return merge_duplicate_aspects(aspects)


def merge_duplicate_aspects(aspects: list[Aspect]) -> list[Aspect]:
    by_name: dict[str, Aspect] = {}
    order: list[str] = []
    for aspect in aspects:
        if aspect.name not in by_name:
            by_name[aspect.name] = aspect
            order.append(aspect.name)
            continue
        previous = by_name[aspect.name]
        keywords = tuple(dict.fromkeys([*previous.keywords, *aspect.keywords]))
        description = previous.description
        if aspect.description and aspect.description not in previous.description:
            description = f"{previous.description}; {aspect.description}".strip("; ")
        by_name[aspect.name] = Aspect(aspect.name, description, keywords)
    return [by_name[name] for name in order]
