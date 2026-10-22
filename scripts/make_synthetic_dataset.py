from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def make_signal(domain: str, label: int, index: int, rng: np.random.Generator, length: int = 256) -> np.ndarray:
    t = np.arange(length, dtype=np.float32)
    base = {"hydraulic": 0.08, "paderborn": 0.13, "pmsm": 0.21, "target": 0.16}[domain]
    frequency = base + rng.normal(0, 0.004)
    signal = np.sin(2 * np.pi * frequency * t) + 0.25 * np.sin(2 * np.pi * frequency * 3 * t)
    signal += rng.normal(0, 0.12, size=length)
    if label:
        kind = (index // 3) % 3
        if kind == 0:
            signal[length // 2 : length // 2 + 5] += 4.0
        elif kind == 1:
            signal += 0.45 * np.sin(2 * np.pi * (frequency * 1.8) * t)
        else:
            signal *= 1.8
    return signal.astype(np.float32)[None, :]


def write_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.root
    signal_dir = root / "signals"
    signal_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    def create(name: str, rows: list[dict]) -> None:
        write_manifest(root / f"{name}.jsonl", rows)

    def make_rows(domain: str, count: int, label_rule, prefix: str) -> list[dict]:
        rows = []
        for i in range(count):
            label = int(label_rule(i))
            path = signal_dir / f"{prefix}_{i:04d}.npy"
            np.save(path, make_signal(domain, label, i, rng))
            rows.append({"sample_id": f"{prefix}_{i:04d}", "path": str(path.resolve()), "domain": domain, "label": label})
        return rows

    source_train, source_val = [], []
    for domain in ("hydraulic", "paderborn", "pmsm"):
        source_train.extend(make_rows(domain, 180, lambda i: i % 3 == 0, f"{domain}_train"))
        source_val.extend(make_rows(domain, 45, lambda i: i % 3 == 0, f"{domain}_val"))
    target_normal = make_rows("target", 70, lambda _: 0, "target_normal_adapt")
    target_calibration = make_rows("target", 30, lambda _: 0, "target_normal_calibration")
    target_unlabeled = make_rows("target", 200, lambda i: i % 2, "target_unlabeled")
    target_test = make_rows("target", 240, lambda i: i % 2, "target_test")
    for name, rows in (
        ("source_train", source_train), ("source_val", source_val),
        ("target_normal", target_normal), ("target_calibration", target_calibration),
        ("target_unlabeled", target_unlabeled), ("target_test", target_test),
    ):
        create(name, rows)
    print(json.dumps({
        "root": str(root.resolve()), "source_train": len(source_train), "source_val": len(source_val),
        "target_normal": len(target_normal), "target_calibration": len(target_calibration),
        "target_unlabeled": len(target_unlabeled), "target_test": len(target_test),
    }))


if __name__ == "__main__":
    main()

