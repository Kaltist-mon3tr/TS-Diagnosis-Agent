from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsdiag.data import ManifestDataset, assert_disjoint_rows, collate_fixed, load_manifest
from tsdiag.model import MultiDomainReconstructor
from tsdiag.scoring import ScoreEngine, best_f1, build_normal_bank, rank_auc, score_rows


def average_precision(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    hits = 0
    total = 0.0
    for rank, idx in enumerate(order, 1):
        if labels[idx] == 1:
            hits += 1
            total += hits / rank
    return total / positives


def main() -> None:
    parser = argparse.ArgumentParser(description="Target-domain deployment evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--normal-manifest", required=True)
    parser.add_argument("--calibration-manifest", required=True, help="held-out target normals used only for threshold calibration")
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bank", default="")
    parser.add_argument("--false-alarm-rate", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = MultiDomainReconstructor(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    def loader(manifest):
        return DataLoader(ManifestDataset(load_manifest(manifest)), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fixed)
    normal_loader = loader(args.normal_manifest)
    calibration_rows = load_manifest(args.calibration_manifest)
    test_rows = load_manifest(args.test_manifest)
    assert_disjoint_rows({
        "target_normal_support": load_manifest(args.normal_manifest),
        "target_normal_calibration": calibration_rows,
        "target_test": test_rows,
    })
    bank_path = Path(args.bank) if args.bank else Path(args.checkpoint).with_name("normal_bank.pt")
    bank = torch.load(bank_path, map_location="cpu", weights_only=False) if bank_path.exists() else build_normal_bank(model, normal_loader, device)
    engine = ScoreEngine(model, bank)
    calibration = score_rows(engine, DataLoader(ManifestDataset(calibration_rows), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fixed), device)
    threshold = float(torch.quantile(torch.tensor([item["score"] for item in calibration]), 1.0 - args.false_alarm_rate))
    results = score_rows(engine, DataLoader(ManifestDataset(test_rows), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fixed), device)
    labels = [int(item["row"].get("label", 0)) for item in results]
    scores = [item["score"] for item in results]
    predicted = [int(score >= threshold) for score in scores]
    tp = sum(p == y == 1 for p, y in zip(predicted, labels))
    fp = sum(p == 1 and y == 0 for p, y in zip(predicted, labels))
    fn = sum(p == 0 and y == 1 for p, y in zip(predicted, labels))
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "windows": len(results),
        "threshold_source": f"held-out target normal calibration quantile for nominal FAR={args.false_alarm_rate}",
        "threshold": threshold,
        "anomaly_decoder_used": False,
        "roc_auc": rank_auc(labels, scores),
        "average_precision": average_precision(labels, scores),
        "f1_at_calibrated_threshold": f1,
        "false_alarm_rate_on_test_normals": fp / max(1, labels.count(0)),
        "mean_score_normal": sum(s for s, y in zip(scores, labels) if y == 0) / max(1, labels.count(0)),
        "mean_score_anomaly": sum(s for s, y in zip(scores, labels) if y == 1) / max(1, labels.count(1)),
        "score_definition": "0.45*normal_reconstruction + 0.15*mask_uncertainty + 0.15*target_normal_anchor_distance + 0.25*target_normal_scale_deviation, robustly scaled",
        "test_threshold_optimization": False,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

