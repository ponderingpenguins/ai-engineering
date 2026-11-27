from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ConfidenceMap:
    frame_count: int
    scores: dict[str, np.ndarray] = field(default_factory=dict)

    def mean_confidence(self, task: str) -> float:
        if task not in self.scores or len(self.scores[task]) == 0:
            return 0.0
        val = float(np.nanmean(self.scores[task]))
        return val if not np.isnan(val) else 0.0

    def summary(self) -> dict[str, float]:
        return {task: self.mean_confidence(task) for task in self.scores}

    def weights(self, task: str) -> Optional[np.ndarray]:
        return self.scores.get(task)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.scores).to_parquet(path, index=False)

    @classmethod
    def load(cls, path: Path) -> ConfidenceMap:
        df = pd.read_parquet(Path(path))
        scores = {col: df[col].to_numpy(dtype=float) for col in df.columns}
        return cls(frame_count=len(df), scores=scores)

    @classmethod
    def merge(cls, maps: list[ConfidenceMap]) -> ConfidenceMap:
        """Aggregate runs — take element-wise max confidence per task."""
        if not maps:
            raise ValueError("No maps to merge")
        all_tasks: set[str] = set()
        for m in maps:
            all_tasks.update(m.scores.keys())
        scores: dict[str, np.ndarray] = {}
        for task in all_tasks:
            arrays = [m.scores[task] for m in maps if task in m.scores]
            scores[task] = np.max(np.stack(arrays, axis=0), axis=0)
        return cls(frame_count=maps[0].frame_count, scores=scores)
