# agent/classifier.py
"""The modulation-classifier seam (design decision 2).

v1 ships only UnavailableClassifier. The TorchSig model trained on the DGX
Spark and exported to the Jetson plugs in here later, behind the same
protocol. It is handed the channelized primary region (one emitter at
baseband) and its rate. In v1 routing grounds a result only in the band
table: a label never grounds a tag by itself.
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
