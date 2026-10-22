from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsdiag.data import ManifestDataset, assert_disjoint_rows, collate_fixed, load_manifest
from tsdiag.losses import stage2_loss
from tsdiag.model import MultiDomainReconstructor
from tsdiag.scoring import ScoreEngine, build_normal_bank, score_rows


def load_model(path: str, device: torch.device) -> MultiDomainReconstructor:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = MultiDomainReconstructor(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model


def average_normal_reconstruction(model, loader, device) -> float:
    model.eval()
    values = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            output = model.deployment_forward(x)
            values.append(float(((output["normal_recon"] - x) ** 2).mean().cpu()))
    return sum(values) / max(1, len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: target normal adaptation")
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--normal-manifest", required=True, help="known target-domain normal support")
    parser.add_argument("--calibration-manifest", required=True, help="held-out target normals; no gradients")
    parser.add_argument("--unlabeled-manifest", required=True, help="mixed pool; labels are ignored")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", default="", help="resume from this stage-two last.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pseudo-quantile", type=float, default=0.20)
    parser.add_argument("--no-pseudo-normal", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = load_model(args.pretrained, device)

    normal_rows = load_manifest(args.normal_manifest)
    calibration_rows = load_manifest(args.calibration_manifest)
    unlabeled_rows = load_manifest(args.unlabeled_manifest)
    assert_disjoint_rows({
        "target_normal_adaptation": normal_rows,
        "target_normal_calibration": calibration_rows,
        "target_unlabeled_pool": unlabeled_rows,
    })
    normal_loader = DataLoader(ManifestDataset(normal_rows), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fixed)
    calibration_loader = DataLoader(ManifestDataset(calibration_rows), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fixed)
    unlabeled_loader = DataLoader(ManifestDataset(unlabeled_rows), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fixed)

    bank_before = build_normal_bank(model, normal_loader, device)
    pseudo_rows = []
    if not args.no_pseudo_normal:
        scored = score_rows(ScoreEngine(model, bank_before), unlabeled_loader, device)
        cutoff = float(torch.quantile(torch.tensor([item["score"] for item in scored]), args.pseudo_quantile))
        pseudo_rows = [item["row"] for item in scored if item["score"] <= cutoff]

    train_loader = DataLoader(
        ManifestDataset(normal_rows + pseudo_rows),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fixed,
    )
    # Freeze every source parameter except the affine target adapter.
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("adapter_")
    optimizer = torch.optim.AdamW([model.adapter_scale, model.adapter_shift], lr=5e-3, weight_decay=1e-4)
    anchor = bank_before["vectors"].mean(dim=0, keepdim=True).to(device)
    history_path = output / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if args.resume and history_path.exists() else []
    start_epoch, best, bad_epochs = 1, float("inf"), 0
    if args.resume:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        start_epoch = int(resume["epoch"]) + 1
        best = float(resume.get("best_calibration_loss", best))
        bad_epochs = int(resume.get("bad_epochs", 0))
    patience = 5

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        values = []
        for batch in train_loader:
            x = batch["x"].to(device)
            encoded = model.encode(x)
            outputs = {
                "shared": encoded["shared"],
                "shared_global": encoded["shared_global"],
                "normal_recon": model._decode(model.normal_decoder, encoded["shared"], x.shape[-1]),
            }
            loss, stats = stage2_loss(outputs, x, anchor.expand(x.shape[0], -1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            values.append(stats)

        train_stats = {key: sum(item[key] for item in values) / len(values) for key in values[0]}
        calibration_loss = average_normal_reconstruction(model, calibration_loader, device)
        record = {
            "epoch": epoch,
            "train": train_stats,
            "heldout_normal_calibration_reconstruction": calibration_loss,
            "pseudo_normal_rows": len(pseudo_rows),
        }
        history.append(record)
        checkpoint = {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "config": model.config,
            "pseudo_normal_rows": len(pseudo_rows),
            "best_calibration_loss": min(best, calibration_loss),
            "optimizer_state": optimizer.state_dict(),
            "bad_epochs": bad_epochs,
        }
        torch.save(checkpoint, output / "last.pt")
        if calibration_loss < best:
            best = calibration_loss
            bad_epochs = 0
            checkpoint["best_calibration_loss"] = best
            checkpoint["bad_epochs"] = bad_epochs
            torch.save(checkpoint, output / "best.pt")
        else:
            bad_epochs += 1
            checkpoint["bad_epochs"] = bad_epochs
        torch.save(checkpoint, output / "last.pt")
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if bad_epochs >= patience:
            break

    # Restore selected checkpoint before constructing the deployment normal bank.
    selected = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(selected["model_state"])
    final_bank = build_normal_bank(model, normal_loader, device)
    torch.save(final_bank, output / "normal_bank.pt")
    (output / "protocol.json").write_text(json.dumps({
        "stage": "stage2_target_adaptation",
        "known_target_data": "trusted normal support only",
        "heldout_calibration": "normal only; used for model selection, never gradients",
        "unlabeled_pool": len(unlabeled_rows),
        "pseudo_normal_rows": len(pseudo_rows),
        "pseudo_normal_rule": f"lowest {args.pseudo_quantile:.2f} fraction of deployment scores; labels were not read",
        "trainable_parameters": ["adapter_scale", "adapter_shift"],
        "anomaly_decoder_used": False,
        "selected_epoch": selected["epoch"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

