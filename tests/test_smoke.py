from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsdiag.model import MultiDomainReconstructor
from tsdiag.scoring import ScoreEngine


def main() -> None:
    model = MultiDomainReconstructor(in_channels=1, hidden=16, latent=8, num_domains=2)
    x = torch.randn(4, 1, 128)
    calls = {"anomaly_decoder": 0}
    hook = model.anomaly_decoder.register_forward_hook(lambda *_: calls.__setitem__("anomaly_decoder", calls["anomaly_decoder"] + 1))
    training = model(x, domain_lambda=0.1, anomaly_lambda=0.2, include_anomaly=True)
    assert training["anomaly_recon"] is not None
    assert training["normal_recon"].shape == x.shape
    bank = {
        "vectors": torch.randn(3, 8),
        "rec_scale": torch.tensor(1.0),
        "unc_scale": torch.tensor(1.0),
        "dist_scale": torch.tensor(1.0),
        "scale_center": torch.tensor(0.0),
        "scale_scale": torch.tensor(1.0),
    }
    deployment = model.deployment_forward(x)
    assert "anomaly_recon" not in deployment
    assert calls["anomaly_decoder"] == 1, "only the explicit training forward should call anomaly decoder"
    scores = ScoreEngine(model, bank).score_batch(x)
    assert scores["score"].shape == (4,)
    assert calls["anomaly_decoder"] == 1, "deployment scoring must not call anomaly decoder"
    assert scores["residual"].shape == x.shape
    multi_channel = model.deployment_forward(torch.randn(2, 3, 128))
    assert multi_channel["normal_recon"].shape == (2, 3, 128)
    hook.remove()
    print("smoke test passed: anomaly decoder is training-only")


if __name__ == "__main__":
    main()

