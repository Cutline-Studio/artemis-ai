"""Loads the brain definitions in brains/*.yaml and builds each brain's system prompt."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

BRAINS_DIR = Path(__file__).resolve().parent.parent / "brains"
SPECIALIST_IDS = ("apollo", "mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto")


@dataclass(frozen=True)
class Brain:
    id: str
    name: str
    kind: str
    role: str
    instructions: tuple[str, ...]
    rules: tuple[str, ...]
    keywords: tuple[str, ...] = ()
    title: str = ""
    model: dict = field(default_factory=dict)
    training_sets: tuple[str, ...] = ()

    def system_prompt(self) -> str:
        label = f"{self.name} ({self.title})" if self.title else self.name
        lines = [f"You are {label}, part of Artemis AI.", self.role, "", "Instructions:"]
        lines += [f"- {i}" for i in self.instructions]
        lines += ["", "Rules:"] + [f"- {r}" for r in self.rules]
        return "\n".join(lines)


def load_brains(directory: Path = BRAINS_DIR) -> dict[str, Brain]:
    brains = {}
    for path in sorted(directory.glob("*.yaml")):
        d = yaml.safe_load(path.read_text())
        brains[d["id"]] = Brain(
            id=d["id"], name=d["name"], kind=d["kind"], role=d["role"],
            instructions=tuple(d["instructions"]), rules=tuple(d.get("rules", ())),
            keywords=tuple(k.lower() for k in d.get("keywords", ())),
            title=d.get("title", ""), model=d.get("model", {}),
            training_sets=tuple(d.get("training_sets", ())),
        )
    missing = set(SPECIALIST_IDS) - brains.keys()
    if missing or "artemis" not in brains:
        raise ValueError(f"missing brain definitions: {sorted(missing | ({'artemis'} - brains.keys()))}")
    return brains
