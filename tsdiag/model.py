from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: Tensor, coefficient: float) -> Tensor:
        ctx.coefficient = coefficient
        return value.view_as(value)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return -ctx.coefficient * grad_output, None


def grad_reverse(value: Tensor, coefficient: float) -> Tensor:
    return GradientReverse.apply(value, coefficient)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        groups = next((g for g in (8, 4, 2, 1) if out_channels % g == 0), 1)
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 5, stride=stride, padding=2),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.block(value)


class ConvDecoder(nn.Module):
    def __init__(self, latent_channels: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(latent_channels, hidden),
            ConvBlock(hidden, hidden),
            nn.Conv1d(hidden, 1, 3, padding=1),
        )

    def forward(self, value: Tensor, output_length: int) -> Tensor:
        value = F.interpolate(value, size=output_length, mode="linear", align_corners=False)
        return self.net(value)


class MultiDomainReconstructor(nn.Module):
    """Shared/private temporal encoder with a training-only adversarial anomaly decoder.

    Channels are processed with shared weights, so the model supports different
    sensor counts as long as each batch uses a consistent channel count.
    """

    def __init__(self, in_channels: int = 1, hidden: int = 48, latent: int = 24, num_domains: int = 3):
        super().__init__()
        self.config = {
            "in_channels": in_channels,  # retained for explicit checkpoint schema; shared weights process channels independently
            "hidden": hidden,
            "latent": latent,
            "num_domains": num_domains,
        }
        self.feature = nn.Sequential(
            ConvBlock(1, hidden, stride=2),
            ConvBlock(hidden, hidden, stride=2),
            ConvBlock(hidden, hidden),
        )
        self.shared_projection = nn.Conv1d(hidden, latent, 1)
        self.private_projection = nn.Conv1d(hidden, latent, 1)
        self.normal_decoder = ConvDecoder(latent, hidden)
        self.anomaly_decoder = ConvDecoder(latent, hidden)
        self.domain_head = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_domains),
        )
        self.private_domain_head = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_domains),
        )
        self.adapter_scale = nn.Parameter(torch.zeros(1, 1, latent, 1))
        self.adapter_shift = nn.Parameter(torch.zeros(1, 1, latent, 1))

    @staticmethod
    def _as_channels(x: Tensor) -> tuple[Tensor, int, int, int]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"expected [batch, channels, time], got {tuple(x.shape)}")
        batch, channels, length = x.shape
        return x.reshape(batch * channels, 1, length), batch, channels, length

    def encode(self, x: Tensor, use_adapter: bool = True) -> dict[str, Tensor]:
        flat, batch, channels, _ = self._as_channels(x)
        features = self.feature(flat)
        shared = self.shared_projection(features)
        private = self.private_projection(features)
        latent_length = shared.shape[-1]
        shared = shared.reshape(batch, channels, shared.shape[1], latent_length)
        private = private.reshape(batch, channels, private.shape[1], latent_length)
        if use_adapter:
            shared = shared * (1.0 + self.adapter_scale) + self.adapter_shift
        return {
            "features": features,
            "shared": shared,
            "private": private,
            "shared_global": shared.mean(dim=(1, 3)),
            "private_global": private.mean(dim=(1, 3)),
        }

    def _decode(self, decoder: nn.Module, latent: Tensor, output_length: int) -> Tensor:
        batch, channels, width, steps = latent.shape
        flat = latent.reshape(batch * channels, width, steps)
        return decoder(flat, output_length).reshape(batch, channels, output_length)

    def forward(
        self,
        x: Tensor,
        domain_lambda: float = 0.0,
        anomaly_lambda: float = 0.0,
        include_anomaly: bool = True,
        use_adapter: bool = True,
    ) -> dict[str, Optional[Tensor]]:
        encoded = self.encode(x, use_adapter=use_adapter)
        normal_recon = self._decode(self.normal_decoder, encoded["shared"], x.shape[-1])
        anomaly_recon = None
        if include_anomaly:
            anomaly_recon = self._decode(
                self.anomaly_decoder,
                grad_reverse(encoded["shared"], anomaly_lambda),
                x.shape[-1],
            )
        domain_logits = self.domain_head(grad_reverse(encoded["shared_global"], domain_lambda))
        private_domain_logits = self.private_domain_head(encoded["private_global"])
        return {
            **encoded,
            "normal_recon": normal_recon,
            "anomaly_recon": anomaly_recon,
            "domain_logits": domain_logits,
            "private_domain_logits": private_domain_logits,
        }

    def deployment_forward(self, x: Tensor, use_adapter: bool = True) -> dict[str, Tensor]:
        # Deployment path does not even invoke anomaly_decoder or domain_head.
        encoded = self.encode(x, use_adapter=use_adapter)
        return {
            "shared": encoded["shared"],
            "shared_global": encoded["shared_global"],
            "normal_recon": self._decode(self.normal_decoder, encoded["shared"], x.shape[-1]),
        }

