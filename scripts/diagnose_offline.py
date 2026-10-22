from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsdiag.data import ManifestDataset, collate_fixed, load_manifest
from tsdiag.evidence import summarize_signal
from tsdiag.knowledge import KnowledgeRetriever
from tsdiag.model import MultiDomainReconstructor
from tsdiag.scoring import ScoreEngine, build_normal_bank


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline structured diagnosis record")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--normal-manifest", required=True)
    parser.add_argument("--calibration-manifest", default="")
    parser.add_argument("--signal", required=True)
    parser.add_argument("--condition", default="unknown")
    parser.add_argument("--knowledge", default="knowledge/graph.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ollama-model", default="", help="optional local Ollama model; omit to emit evidence-only JSON")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/generate")
    parser.add_argument("--neo4j", action="store_true", help="retrieve from local Neo4j instead of the JSON graph")
    parser.add_argument("--neo4j-database", default="neo4j")
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = MultiDomainReconstructor(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    normal_loader = DataLoader(ManifestDataset(load_manifest(args.normal_manifest)), batch_size=64, shuffle=False, collate_fn=collate_fixed)
    bank_path = Path(args.checkpoint).with_name("normal_bank.pt")
    bank = torch.load(bank_path, map_location="cpu") if bank_path.exists() else build_normal_bank(model, normal_loader, device)
    signal = np.load(args.signal).astype(np.float32)
    if signal.ndim == 1:
        signal = signal[None, :]
    x = torch.from_numpy((signal - signal.mean(axis=1, keepdims=True)) / (signal.std(axis=1, keepdims=True) + 1e-6)).unsqueeze(0)
    raw_log_scale = torch.tensor([float(np.log(np.maximum(signal.std(axis=1).mean(), 1e-8)))], dtype=torch.float32)
    values = ScoreEngine(model, bank).score_batch(x.to(device), raw_log_scale.to(device))
    score = float(values["score"][0].cpu())
    residual = values["residual"][0].mean(dim=0).cpu().numpy()
    evidence = summarize_signal(signal, score, args.condition, residual=residual)
    calibration_threshold = None
    if args.calibration_manifest:
        calibration_rows = load_manifest(args.calibration_manifest)
        calibration_loader = DataLoader(ManifestDataset(calibration_rows), batch_size=64, shuffle=False, collate_fn=collate_fixed)
        calibration_bank = torch.load(bank_path, map_location="cpu", weights_only=False) if bank_path.exists() else build_normal_bank(model, normal_loader, device)
        calibration_results = []
        for batch in calibration_loader:
            batch_scores = ScoreEngine(model, calibration_bank).score_batch(batch["x"].to(device), batch["raw_log_scale"].to(device))
            calibration_results.extend([float(v) for v in batch_scores["score"].cpu()])
        calibration_threshold = float(torch.quantile(torch.tensor(calibration_results), 0.95))
    verdict = "anomaly_candidate" if calibration_threshold is not None and score >= calibration_threshold else ("within_calibrated_normal_range" if calibration_threshold is not None else "threshold_not_calibrated")
    evidence["calibrated_verdict"] = verdict
    evidence["normal_calibration_threshold_far_5pct"] = calibration_threshold
    terms = [args.condition]
    if evidence["time_domain"]["crest_factor"] >= 3.0:
        terms.extend(["impulse", "transient", "crest factor"])
    if float(values["scale_deviation"][0]) > 1.0:
        terms.extend(["amplitude", "scale", "rms"])
    if evidence["frequency_domain"]["dominant_bin"] > 0:
        terms.extend(["frequency", "spectrum"])
    candidates = KnowledgeRetriever(
        args.knowledge, use_neo4j=args.neo4j, database=args.neo4j_database
    ).retrieve(terms, limit=6)
    record = {
        "schema_version": "1.0",
        "diagnosis_input": {
            "anomaly_score": score,
            "score_components": {key: float(values[key][0].cpu()) for key in ("reconstruction", "uncertainty", "anchor_distance", "scale_deviation")},
            "evidence": evidence,
            "condition": args.condition,
            "knowledge_candidates": candidates,
        },
        "report_contract": {
            "must_cite": ["anomaly_score", "score_components", "candidate_anomaly_segments", "knowledge_candidates"],
            "available_evidence_ids": [
                "anomaly_score", "reconstruction", "uncertainty", "anchor_distance", "scale_deviation",
                "time_domain.rms", "time_domain.std", "time_domain.peak", "time_domain.crest_factor",
                "frequency_domain.dominant_bin", "candidate_anomaly_segments",
            ] + [str(item.get("id")) for item in candidates if item.get("id")],
            "offline_only": True,
            "llm_role": "explain and rank evidence; do not invent measurements",
        },
    }
    if args.ollama_model:
        from urllib.parse import urlparse
        parsed_url = urlparse(args.ollama_url)
        if parsed_url.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Offline mode permits only a local Ollama endpoint.")
        prompt = (
            "你是离线工业异常诊断助手。只能依据下面的结构化证据和知识候选生成中文JSON报告。"
            "不要编造测量值、故障部件或因果关系；证据不足时明确写需复核。"
            "只可引用 report_contract.available_evidence_ids 中的ID。输出字段：summary, diagnosis_candidates(含name/rationale/evidence_ids/confidence), "
            "maintenance_actions(含action/evidence_ids), uncertainty_notes。\n\n"
            + json.dumps({
                "diagnosis_input": record["diagnosis_input"],
                "available_evidence_ids": record["report_contract"]["available_evidence_ids"],
            }, ensure_ascii=False)
        )
        payload = json.dumps({"model": args.ollama_model, "prompt": prompt, "stream": False, "format": "json"}).encode("utf-8")
        request = urllib.request.Request(args.ollama_url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=180) as response:
            generated = json.loads(response.read().decode("utf-8")).get("response", "")
        try:
            parsed_report = json.loads(generated)
        except json.JSONDecodeError:
            parsed_report = None
        allowed_evidence_ids = {
            "anomaly_score", "reconstruction", "uncertainty", "anchor_distance", "scale_deviation",
            "time_domain.rms", "time_domain.std", "time_domain.peak", "time_domain.crest_factor",
            "frequency_domain.dominant_bin", "candidate_anomaly_segments",
        } | {str(item.get("id")) for item in candidates if item.get("id")}
        cited = []
        if isinstance(parsed_report, dict):
            for item in parsed_report.get("diagnosis_candidates", []):
                cited.extend(item.get("evidence_ids", []))
            for item in parsed_report.get("maintenance_actions", []):
                cited.extend(item.get("evidence_ids", []))
        invalid_citations = sorted(set(cited) - allowed_evidence_ids)
        record["llm_report"] = parsed_report if parsed_report is not None else {"raw": generated}
        record["llm_model"] = args.ollama_model
        record["llm_citation_ids_validated"] = bool(parsed_report is not None and cited and not invalid_citations)
        record["llm_invalid_citations"] = invalid_citations
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

