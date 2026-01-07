from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

ARKIT_BLENDSHAPES = [
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
    "tongueOut",
]


class BlendshapeCalibrator:
    def __init__(self, scales: np.ndarray, biases: np.ndarray):
        self.scales = scales
        self.biases = biases

    @classmethod
    def from_file(cls, path: Path) -> "BlendshapeCalibrator":
        obj = json.loads(path.read_text(encoding="utf-8"))
        mapping = obj.get("affine", obj)
        scales = np.ones(len(ARKIT_BLENDSHAPES), dtype=np.float32)
        biases = np.zeros(len(ARKIT_BLENDSHAPES), dtype=np.float32)
        for name, params in mapping.items():
            if name not in ARKIT_BLENDSHAPES:
                continue
            idx = ARKIT_BLENDSHAPES.index(name)
            scales[idx] = float(params.get("scale", 1.0))
            biases[idx] = float(params.get("bias", 0.0))
        return cls(scales, biases)

    def apply(self, values: np.ndarray) -> np.ndarray:
        """
        values: (..., 52)
        """
        return np.clip(values * self.scales + self.biases, 0.0, 1.0)


def maybe_load_calibrator(path: Optional[Path]) -> Optional[BlendshapeCalibrator]:
    if path and Path(path).exists():
        return BlendshapeCalibrator.from_file(Path(path))
    return None
