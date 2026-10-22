from __future__ import annotations

import numpy as np


def summarize_signal(signal: np.ndarray, score: float, condition: str | None = None, residual: np.ndarray | None = None) -> dict:
    array = np.asarray(signal, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    channel = array[0]
    centered = channel - channel.mean()
    rms = float(np.sqrt(np.mean(channel ** 2)))
    std = float(channel.std())
    peak = float(np.max(np.abs(channel)))
    crest = peak / max(rms, 1e-8)
    spectrum = np.abs(np.fft.rfft(centered))
    frequencies = np.fft.rfftfreq(len(centered), d=1.0)
    dominant = int(np.argmax(spectrum[1:]) + 1) if len(spectrum) > 1 else 0
    if residual is None:
        residual = np.abs(centered)
        segment_source = "signal_amplitude_candidates"
    else:
        residual = np.asarray(residual, dtype=np.float64).reshape(-1)
        segment_source = "normal_decoder_absolute_residual"
    threshold = float(np.quantile(residual, 0.95))
    indices = np.flatnonzero(residual >= threshold)
    segments = []
    if len(indices):
        start = prev = int(indices[0])
        for value in indices[1:]:
            value = int(value)
            if value > prev + 1:
                segments.append({"start": start, "end": prev})
                start = value
            prev = value
        segments.append({"start": start, "end": prev})
    return {
        "score": float(score),
        "condition": condition,
        "time_domain": {"rms": rms, "std": std, "peak": peak, "crest_factor": crest},
        "frequency_domain": {
            "dominant_bin": dominant,
            "dominant_normalized_frequency": float(frequencies[dominant]) if len(frequencies) else 0.0,
        },
        "candidate_anomaly_segments": segments,
        "segment_source": segment_source,
    }

