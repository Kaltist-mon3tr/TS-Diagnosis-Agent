from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    if not bool(mask.any()):
        return value.new_zeros(())
    return value[mask].mean()


def stage1_loss(outputs: dict[str, Tensor | None], x: Tensor, labels: Tensor, domains: Tensor, weights: dict[str, float] | None = None):
    weights = weights or {}
    normal = labels < 0.5
    abnormal = ~normal
    normal_error = ((outputs["normal_recon"] - x) ** 2).mean(dim=(1, 2))
    anomaly_recon = outputs["anomaly_recon"]
    if anomaly_recon is None:
        raise RuntimeError("stage1 requires include_anomaly=True")
    anomaly_error = ((anomaly_recon - x) ** 2).mean(dim=(1, 2))
    reconstruction = _masked_mean(normal_error, normal)
    anomaly_branch = _masked_mean(anomaly_error, abnormal)
    shared_domain_loss = F.cross_entropy(outputs["domain_logits"], domains)
    private_domain_loss = F.cross_entropy(outputs["private_domain_logits"], domains)
    domain_loss = shared_domain_loss + private_domain_loss
    shared = F.normalize(outputs["shared"].flatten(1), dim=1)
    private = F.normalize(outputs["private"].flatten(1), dim=1)
    disentangle = (shared * private).sum(dim=1).abs().mean()
    total = (
        reconstruction
        + weights.get("anomaly", 1.0) * anomaly_branch
        + weights.get("domain", 0.1) * domain_loss
        + weights.get("disentangle", 0.01) * disentangle
    )
    stats = {
        "loss": float(total.detach()),
        "normal_reconstruction": float(reconstruction.detach()),
        "anomaly_training_branch": float(anomaly_branch.detach()),
        "domain": float(domain_loss.detach()),
        "disentangle": float(disentangle.detach()),
    }
    return total, stats


def stage2_loss(outputs: dict[str, Tensor], x: Tensor, anchor: Tensor | None = None):
    reconstruction = ((outputs["normal_recon"] - x) ** 2).mean()
    consistency = outputs["shared"].std(dim=-1, unbiased=False).mean() * 0.01
    anchor_loss = x.new_zeros(())
    if anchor is not None:
        anchor_loss = ((outputs["shared_global"] - anchor) ** 2).mean() * 0.01
    total = reconstruction + consistency + anchor_loss
    return total, {
        "loss": float(total.detach()),
        "normal_reconstruction": float(reconstruction.detach()),
        "adapter_consistency": float(consistency.detach()),
        "anchor": float(anchor_loss.detach()),
    }

