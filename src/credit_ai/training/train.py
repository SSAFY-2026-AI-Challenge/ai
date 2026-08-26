from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from credit_ai.features.schema import CreditFeatures
from credit_ai.model.network import CreditMLP
from credit_ai.model.scorer import CreditScorer, StandardScaler
from credit_ai.training.synthetic import generate_dataset


def _matrices(
    features: list[CreditFeatures], labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.stack([item.to_model_vector() for item in features]).astype(np.float32)
    return matrix, labels.astype(np.float32).reshape(-1, 1)


def train_model(
    out_dir: Path,
    n_samples: int = 8_000,
    epochs: int = 45,
    batch_size: int = 128,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    features, labels = generate_dataset(n_samples, seed=seed, with_noise=True)
    matrix, target = _matrices(features, labels)

    n_train = int(0.8 * len(matrix))
    perm = np.random.default_rng(seed).permutation(len(matrix))
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    scaler = StandardScaler.fit(matrix[train_idx])
    x_train = scaler.transform(matrix[train_idx])
    x_val = scaler.transform(matrix[val_idx])
    y_train = target[train_idx]
    y_val = target[val_idx]

    torch_device = torch.device(device)
    model = CreditMLP().to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.SmoothL1Loss()

    train_ds = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_mae = float("inf")
    best_state = None
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(torch_device)
            batch_y = batch_y.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.item()) * len(batch_x)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(torch.from_numpy(x_val).to(torch_device)).cpu().numpy()
        val_mae = float(np.mean(np.abs(val_pred - y_val)))
        train_loss = running / max(len(train_ds), 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_mae": val_mae})
        if val_mae < best_mae:
            best_mae = val_mae
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    scorer = CreditScorer(model=model, scaler=scaler, device=torch_device)
    metrics = {
        "val_mae": round(best_mae, 3),
        "val_mae": round(best_mae, 3),
        "n_samples": float(n_samples),
        "epochs": float(epochs),
        "score_min": 300.0,
        "score_max": 1000.0,
    }
    out_dir = Path(out_dir)
    scorer.save(out_dir, extra=metrics)
    (out_dir / "metrics.json").write_text(
        json.dumps({"best": metrics, "history": history[-8:]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics
