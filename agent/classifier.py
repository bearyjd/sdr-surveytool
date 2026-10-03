# agent/classifier.py
"""The modulation-classifier seam (design decision 2).

v1 ships only UnavailableClassifier. The TorchSig model trained on the DGX
Spark and exported to the Jetson plugs in here later, behind the same
protocol; until then routing can only ground a result in the band table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ModulationPrediction:
    label: str
    confidence: float


class ModulationClassifier(Protocol):
    def predict(self, iq: np.ndarray, sample_rate: float) -> ModulationPrediction | None: ...


class UnavailableClassifier:
    """No trained model exists yet: never a prediction."""

    def predict(self, iq: np.ndarray, sample_rate: float) -> ModulationPrediction | None:
        return None
