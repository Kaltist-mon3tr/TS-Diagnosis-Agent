from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor


@dataclass
class ScoreEngine:
    model: torch.nn.Module
    bank: dict[str, Tensor]
    mask_ratios: tuple[float, ...] = (0.0, 0.15, 0.30)
    reconstruction_weight: float = 0.45
    uncertainty_weight: float = 0.15
    anchor_weight: float = 0.15
    scale_weight: float = 0.25

    @torch.no_grad()
    def score_batch(self, x: Tensor, raw_log_scale: Tensor | None = None) -> dict[str, Tensor]:
        # Deployment path deliberately excludes anomaly_decoder.
        errors: list[Tensor] = []
        predictions: list[Tensor] = []
        for ratio in self.mask_ratios:
            masked = _mask_input(x, ratio)
            outputs = self.model.deployment_forward(masked)
            prediction = outputs["normal_recon"]
            predictions.append(prediction)
            errors.append(((prediction - x) ** 2).mean(dim=(1, 2)))
        error_stack = torch.stack(errors, dim=1)
        reconstruction = error_stack[:, 0]
        uncertainty = error_stack.std(dim=1)
        embedding = self.model.deployment_forward(x)["shared_global"]
        vectors = self.bank["vectors"].to(x.device)
        distance = torch.cdist(embedding, vectors).min(dim=1).values
        rec_scale = self.bank["rec_scale"].to(x.device).clamp_min(1e-4)
        unc_scale = self.bank["unc_scale"].to(x.device).clamp_min(1e-4)
        dist_scale = self.bank["dist_scale"].to(x.device).clamp_min(1e-4)
        if raw_log_scale is None:
            raw_log_scale = torch.zeros(x.shape[0], device=x.device)
        scale_center = self.bank["scale_center"].to(x.device)
        scale_scale = self.bank["scale_scale"].to(x.device).clamp_min(1e-4)
        scale_deviation = (raw_log_scale - scale_center).abs()
        score = (
            (self.reconstruction_weight * reconstruction / rec_scale
            + self.uncertainty_weight * uncertainty / unc_scale
            + self.anchor_weight * distance / dist_scale
            + self.scale_weight * scale_deviation / scale_scale)
        )
        return {
            "score": score,
            "reconstruction": reconstruction,
            "uncertainty": uncertainty,
            "anchor_distance": distance,
            "scale_deviation": scale_deviation,
            "residual": (predictions[0] - x).abs(),
            "predictions": torch.stack(predictions, dim=1),
        }


def _mask_input(x: Tensor, ratio: float) -> Tensor:
    if ratio <= 0:
        return x
    masked = x.clone()
    length = x.shape[-1]
    span = max(1, int(length * ratio))
    start = (length // 2 - span // 2) % max(1, length - span + 1)
    masked[..., start : start + span] = 0.0
    return masked


@torch.no_grad()
def build_normal_bank(model: torch.nn.Module, loader: Iterable[dict], device: str | torch.device = "cpu") -> dict[str, Tensor]:
    model.eval()
    vectors: list[Tensor] = []
    reconstruction: list[Tensor] = []
    uncertainty: list[Tensor] = []
    scales: list[Tensor] = []
    for batch in loader:
        x = batch["x"].to(device)
        outputs = model.deployment_forward(x)
        vectors.append(outputs["shared_global"].cpu())
        reconstruction.append(((outputs["normal_recon"] - x) ** 2).mean(dim=(1, 2)).cpu())
        masked_errors = []
        for ratio in (0.0, 0.15, 0.30):
            masked = _mask_input(x, ratio)
            pred = model.deployment_forward(masked)["normal_recon"]
            masked_errors.append(((pred - x) ** 2).mean(dim=(1, 2)))
        uncertainty.append(torch.stack(masked_errors, dim=1).std(dim=1).cpu())
        scales.append(batch["raw_log_scale"].cpu())
    vectors_tensor = torch.cat(vectors)
    rec = torch.cat(reconstruction)
    unc = torch.cat(uncertainty)
    raw_scales = torch.cat(scales)
    if vectors_tensor.shape[0] > 1:
        pairwise = torch.cdist(vectors_tensor, vectors_tensor)
        distance = pairwise.masked_fill(torch.eye(len(vectors_tensor), dtype=torch.bool), float("inf")).min(dim=1).values
    else:
        distance = torch.ones(1)
    return {
        "vectors": vectors_tensor,
        "rec_scale": rec.median().clamp_min(1e-4),
        "unc_scale": unc.median().clamp_min(1e-4),
        "dist_scale": distance.median().clamp_min(1e-4),
        "scale_center": raw_scales.median(),
        "scale_scale": (raw_scales - raw_scales.median()).abs().median().clamp_min(1e-4),
    }


def score_rows(engine: ScoreEngine, loader: Iterable[dict], device: str | torch.device = "cpu") -> list[dict]:
    engine.model.eval()
    results: list[dict] = []
    for batch in loader:
        values = engine.score_batch(batch["x"].to(device), batch["raw_log_scale"].to(device))
        for i, sample_id in enumerate(batch["sample_id"]):
            results.append({
                "sample_id": sample_id,
                "score": float(values["score"][i].cpu()),
                "reconstruction": float(values["reconstruction"][i].cpu()),
                "uncertainty": float(values["uncertainty"][i].cpu()),
                "anchor_distance": float(values["anchor_distance"][i].cpu()),
                "scale_deviation": float(values["scale_deviation"][i].cpu()),
                "row": batch["row"][i],
            })
    return results


def rank_auc(labels: list[int], scores: list[float]) -> float:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return float("nan")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def best_f1(labels: list[int], scores: list[float]) -> tuple[float, float]:
    candidates = sorted(set(scores))
    best = (0.0, candidates[0] if candidates else 0.0)
    for threshold in candidates:
        predicted = [int(score >= threshold) for score in scores]
        tp = sum(p == y == 1 for p, y in zip(predicted, labels))
        fp = sum(p == 1 and y == 0 for p, y in zip(predicted, labels))
        fn = sum(p == 0 and y == 1 for p, y in zip(predicted, labels))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        value = 2 * precision * recall / max(1e-8, precision + recall)
        if value > best[0]:
            best = (value, threshold)
    return best

