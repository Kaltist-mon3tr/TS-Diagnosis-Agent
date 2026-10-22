from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsdiag.data import ManifestDataset, collate_fixed, load_manifest
from tsdiag.losses import stage1_loss
from tsdiag.model import MultiDomainReconstructor


def evaluate(model, loader, device, weights):
    model.eval()
    values = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            outputs = model(x, domain_lambda=0.0, anomaly_lambda=0.0, include_anomaly=True)
            _, stats = stage1_loss(outputs, x, batch["label"].to(device), batch["domain"].to(device), weights)
            values.append(stats)
    return {key: sum(item[key] for item in values) / len(values) for key in values[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: multi-source training")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--latent", type=int, default=24)
    parser.add_argument("--anomaly-lambda", type=float, default=0.25)
    parser.add_argument("--domain-lambda", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    train_rows = load_manifest(args.manifest)
    val_rows = load_manifest(args.val_manifest)
    train_dataset = ManifestDataset(train_rows)
    val_dataset = ManifestDataset(val_rows)
    domains = sorted({str(row.get("domain", "unknown")) for row in train_rows})
    domain_to_id = {name: i for i, name in enumerate(domains)}
    for row in train_dataset.rows:
        row["_domain_id"] = domain_to_id[str(row.get("domain", "unknown"))]
    for row in val_dataset.rows:
        row["_domain_id"] = domain_to_id.get(str(row.get("domain", "unknown")), 0)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fixed)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fixed)
    model = MultiDomainReconstructor(1, args.hidden, args.latent, len(domains)).to(device)
    # The target-domain adapter must remain untouched during source pretraining.
    model.adapter_scale.requires_grad_(False)
    model.adapter_shift.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    start_epoch, best = 1, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best = float(checkpoint.get("best_val_loss", best))
    weights = {"anomaly": 1.0, "domain": 0.1, "disentangle": 0.01}
    history = []
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = []
        for batch in train_loader:
            x = batch["x"].to(device)
            outputs = model(x, domain_lambda=args.domain_lambda, anomaly_lambda=args.anomaly_lambda, include_anomaly=True)
            loss, stats = stage1_loss(outputs, x, batch["label"].to(device), batch["domain"].to(device), weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running.append(stats)
        train_stats = {key: sum(item[key] for item in running) / len(running) for key in running[0]}
        val_stats = evaluate(model, val_loader, device, weights)
        record = {"epoch": epoch, "train": train_stats, "val": val_stats}
        history.append(record)
        checkpoint = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch, "best_val_loss": best, "config": model.config, "domains": domains}
        torch.save(checkpoint, output / "last.pt")
        if val_stats["loss"] < best:
            best = val_stats["loss"]
            checkpoint["best_val_loss"] = best
            torch.save(checkpoint, output / "best.pt")
        print(json.dumps(record, ensure_ascii=False), flush=True)
    (output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "protocol.json").write_text(json.dumps({
        "stage": "stage1_source_pretraining",
        "anomaly_decoder": "training_only",
        "deployment_score": "normal_reconstruction + mask_uncertainty + target_anchor_distance",
        "domains": domains,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

