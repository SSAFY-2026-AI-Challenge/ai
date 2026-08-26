from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from credit_ai.features.schema import FEATURE_DIM, CreditFeatures
from credit_ai.model.network import CreditMLP

SCORE_MIN = 300
SCORE_MAX = 1000


@dataclass(slots=True)
class StandardScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        return (matrix - self.mean) / np.maximum(self.std, 1e-6)

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, list[float]]) -> StandardScaler:
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
        )

    @classmethod
    def fit(cls, matrix: np.ndarray) -> StandardScaler:
        return cls(
            mean=matrix.mean(axis=0).astype(np.float32),
            std=matrix.std(axis=0).astype(np.float32),
        )


@dataclass(slots=True)
class ScoreOutput:
    score: int
    grade: str
    grade_code: str
    features: dict[str, float | int]


def grade_for(score: float) -> tuple[str, str]:
    if score >= 900:
        return "최우수", "A+"
    if score >= 800:
        return "우수", "A"
    if score >= 700:
        return "양호", "B"
    if score >= 600:
        return "보통", "C"
    return "노력", "D"


class CreditScorer:
    def __init__(self, model: CreditMLP, scaler: StandardScaler, device: torch.device) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.scaler = scaler
        self.device = device

    def predict_many(self, features_list: list[CreditFeatures]) -> np.ndarray:
        if not features_list:
            return np.array([], dtype=np.float32)
        matrix = np.stack([item.to_model_vector() for item in features_list])
        scaled = self.scaler.transform(matrix).astype(np.float32)
        tensor = torch.from_numpy(scaled).to(self.device)
        with torch.no_grad():
            scores = self.model(tensor).cpu().numpy().reshape(-1)
        return np.clip(scores, SCORE_MIN, SCORE_MAX)

    def predict(self, features: CreditFeatures) -> float:
        return float(self.predict_many([features])[0])

    def score(self, features: CreditFeatures) -> ScoreOutput:
        raw = self.predict(features)
        value = int(round(raw))
        grade, code = grade_for(value)
        return ScoreOutput(
            score=value,
            grade=grade,
            grade_code=code,
            features=features.as_dict(),
        )

    def save(self, directory: Path, extra: dict | None = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / "credit_mlp.pt")
        scaler_doc = self.scaler.to_dict()
        scaler_doc["in_dim"] = FEATURE_DIM
        scaler_doc["hidden"] = [64, 32, 16]
        if extra:
            scaler_doc["metrics"] = extra
        (directory / "scaler.json").write_text(
            json.dumps(scaler_doc, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path, device: str | None = None) -> CreditScorer:
        path = Path(directory)
        artifact = path / "credit_mlp.pt"
        scaler_path = path / "scaler.json"
        if not artifact.exists() or not scaler_path.exists():
            raise FileNotFoundError(
                f"학습된 모델이 없습니다. `{directory}` 에서 `credit-ai train` 을 실행하세요."
            )
        torch_device = torch.device(device or "cpu")
        scaler_doc = json.loads(scaler_path.read_text(encoding="utf-8"))
        hidden = tuple(scaler_doc.get("hidden", [64, 32, 16]))
        model = CreditMLP(in_dim=int(scaler_doc.get("in_dim", FEATURE_DIM)), hidden=hidden)
        state = torch.load(artifact, map_location=torch_device, weights_only=True)
        model.load_state_dict(state)
        scaler = StandardScaler.from_dict(scaler_doc)
        return cls(model=model, scaler=scaler, device=torch_device)

    @classmethod
    def try_load(cls, directory: Path) -> CreditScorer | None:
        try:
            return cls.load(directory)
        except FileNotFoundError:
            return None

    try_load = try_load
