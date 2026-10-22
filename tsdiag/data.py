from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def load_manifest(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row.setdefault("sample_id", f"{Path(path).stem}-{line_no:06d}")
            row.setdefault("label", 0)
            row.setdefault("domain", "unknown")
            row.setdefault("split", Path(path).stem)
            rows.append(row)
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return rows


def _load_array(row: dict) -> tuple[np.ndarray, float]:
    value = row.get("path")
    if value is None:
        raise KeyError(f"row {row.get('sample_id')} has no path")
    array = np.load(value)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"{value} must have shape [channels, time], got {array.shape}")
    start = int(row.get("start", 0))
    length = row.get("length")
    if length is not None:
        array = array[:, start : start + int(length)]
    elif start:
        array = array[:, start:]
    if array.shape[1] < 8:
        raise ValueError(f"{value} contains fewer than 8 time points after slicing")
    array = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mean = array.mean(axis=1, keepdims=True)
    std = array.std(axis=1, keepdims=True)
    # Keep raw scale as a separate signal statistic: per-window normalization
    # makes waveform shape easier to share across domains but would erase
    # amplitude anomalies if scale were discarded.
    raw_scale = float(np.log(np.maximum(std.mean(), 1e-8)))
    return (array - mean) / (std + 1e-6), raw_scale


class ManifestDataset(Dataset):
    def __init__(self, rows: Sequence[dict], domain_to_id: dict[str, int] | None = None):
        self.rows = list(rows)
        if not self.rows:
            raise ValueError("dataset has no rows")
        domains = sorted({str(row.get("domain", "unknown")) for row in self.rows})
        self.domain_to_id = domain_to_id or {name: i for i, name in enumerate(domains)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        domain_name = str(row.get("domain", "unknown"))
        signal, raw_log_scale = _load_array(row)
        return {
            "x": torch.from_numpy(signal),
            "raw_log_scale": torch.tensor(raw_log_scale, dtype=torch.float32),
            "label": torch.tensor(float(row.get("label", 0)), dtype=torch.float32),
            "domain": torch.tensor(int(row.get("_domain_id", self.domain_to_id.get(domain_name, 0))), dtype=torch.long),
            "sample_id": row.get("sample_id", str(index)),
            "row": row,
        }


def collate_fixed(batch: list[dict]) -> dict:
    lengths = {item["x"].shape[-1] for item in batch}
    channels = {item["x"].shape[0] for item in batch}
    if len(lengths) != 1 or len(channels) != 1:
        raise ValueError("all windows in one batch must have the same channel and time dimensions")
    return {
        "x": torch.stack([item["x"] for item in batch]),
        "raw_log_scale": torch.stack([item["raw_log_scale"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "domain": torch.stack([item["domain"] for item in batch]),
        "sample_id": [item["sample_id"] for item in batch],
        "row": [item["row"] for item in batch],
    }


def rows_by_label(rows: Iterable[dict], label: int) -> list[dict]:
    return [row for row in rows if int(row.get("label", 0)) == label]


def assert_disjoint_rows(named_rows: dict[str, Sequence[dict]]) -> None:
    """Reject sample, declared-group, and overlapping-window leakage."""
    seen_ids: dict[str, str] = {}
    seen_groups: dict[str, str] = {}
    seen_windows: dict[str, list[tuple[int, float, str]]] = {}
    for split, rows in named_rows.items():
        for row in rows:
            sample_id = str(row.get("sample_id", ""))
            if sample_id and sample_id in seen_ids:
                raise ValueError(f"sample_id overlap between {seen_ids[sample_id]} and {split}: {sample_id}")
            if sample_id:
                seen_ids[sample_id] = split
            group_id = row.get("group_id")
            if group_id:
                if str(group_id) in seen_groups and seen_groups[str(group_id)] != split:
                    raise ValueError(f"group_id overlap between {seen_groups[str(group_id)]} and {split}: {group_id}")
                seen_groups[str(group_id)] = split
            path = str(Path(row["path"]).resolve())
            start = int(row.get("start", 0))
            end = start + int(row["length"]) if row.get("length") is not None else float("inf")
            for previous_start, previous_end, previous_split in seen_windows.get(path, []):
                overlaps = start < previous_end and previous_start < end
                if overlaps and previous_split != split:
                    raise ValueError(f"overlapping windows across {previous_split} and {split}: {path}")
            seen_windows.setdefault(path, []).append((start, end, split))

