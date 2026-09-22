"""Single-file AI-music detection using the project's frozen final models."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

# librosa/numba must use a writable cache in restricted notebook environments.
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "ai_music_numba_cache"))
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import joblib
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, Wav2Vec2FeatureExtractor


SR = 24_000
SEGMENT_SECONDS = 10.0
TARGET_SAMPLES = int(SR * SEGMENT_SECONDS)
N_FFT = 1024
HOP_LENGTH = 240
N_MELS = 128
N_MFCC = 40
FMAX = 12_000
MERT_NAME = "m-a-p/MERT-v1-95M"
MERT_REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"

# These are frozen Validation track thresholds from the final tuning run.
THRESHOLDS = {
    "Logistic Regression": {"segment": 0.5676900245181252, "track": 0.5634561570614823},
    "RBF-SVM": {"segment": 0.5847561880627183, "track": 0.6615054850320587},
    "Log-Mel CNN": {"segment": 0.142173171043396, "track": 0.2410693895071745},
    "Frozen MERT + LR": {"segment": 0.5685665621681771, "track": 0.5887511404871456},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _planned_starts(duration: float) -> list[tuple[str, float]]:
    if duration >= 30:
        return [("start", 0.0), ("middle", (duration - 10.0) / 2.0), ("end", duration - 10.0)]
    if duration >= 20:
        return [("start", 0.0), ("end", duration - 10.0)]
    if duration >= 10:
        return [("center", (duration - 10.0) / 2.0)]
    raise ValueError(f"오디오 길이가 10초 미만입니다: {duration:.3f}s")


def _add_mean_std(out: dict[str, float], prefix: str, values: np.ndarray) -> None:
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[None, :]
    for index in range(values.shape[0]):
        name = f"{prefix}_{index + 1:02d}" if values.shape[0] > 1 else prefix
        out[f"{name}_mean"] = float(np.mean(values[index]))
        out[f"{name}_std"] = float(np.std(values[index]))


def handcrafted_features(waveform: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    mfcc = librosa.feature.mfcc(y=waveform, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT,
                                hop_length=HOP_LENGTH, n_mels=N_MELS, fmax=FMAX)
    _add_mean_std(out, "mfcc", mfcc)
    _add_mean_std(out, "mfcc_delta", librosa.feature.delta(mfcc, order=1))
    _add_mean_std(out, "mfcc_delta2", librosa.feature.delta(mfcc, order=2))
    _add_mean_std(out, "spectral_centroid", librosa.feature.spectral_centroid(y=waveform, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH))
    _add_mean_std(out, "spectral_bandwidth", librosa.feature.spectral_bandwidth(y=waveform, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH))
    _add_mean_std(out, "spectral_rolloff", librosa.feature.spectral_rolloff(y=waveform, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, roll_percent=0.85))
    _add_mean_std(out, "spectral_flatness", librosa.feature.spectral_flatness(y=waveform, n_fft=N_FFT, hop_length=HOP_LENGTH))
    _add_mean_std(out, "spectral_contrast", librosa.feature.spectral_contrast(y=waveform, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH))
    _add_mean_std(out, "rms", librosa.feature.rms(y=waveform, frame_length=N_FFT, hop_length=HOP_LENGTH))
    _add_mean_std(out, "zcr", librosa.feature.zero_crossing_rate(y=waveform, frame_length=N_FFT, hop_length=HOP_LENGTH))
    if len(out) != 266:
        raise ValueError(f"handcrafted feature count mismatch: {len(out)}")
    return out


def logmel(waveform: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=waveform, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                         n_mels=N_MELS, fmax=FMAX, power=2.0, center=True)
    values = (librosa.power_to_db(mel, ref=np.max, top_db=80.0) + 40.0) / 40.0
    if values.shape != (128, 1001) or not np.isfinite(values).all():
        raise ValueError(f"unexpected Log-Mel shape/values: {values.shape}")
    return values.astype(np.float32)


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class LogMelCNN(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(_ConvBlock(1, 16), _ConvBlock(16, 32), _ConvBlock(32, 64), _ConvBlock(64, 128))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x))).squeeze(1)


def load_inputs(paths: Iterable[str | Path]) -> tuple[pd.DataFrame, list[np.ndarray]]:
    metadata: list[dict] = []
    segments: list[np.ndarray] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        info = sf.info(path)
        waveform, _ = librosa.load(path, sr=SR, mono=True)
        waveform = np.asarray(waveform, dtype=np.float32)
        if not np.isfinite(waveform).all() or len(waveform) < TARGET_SAMPLES:
            raise ValueError(f"오디오 디코딩 실패 또는 10초 미만: {path}")
        duration = len(waveform) / SR
        starts = _planned_starts(duration)
        for role, start in starts:
            begin = int(round(start * SR))
            segment = waveform[begin: begin + TARGET_SAMPLES]
            if len(segment) < TARGET_SAMPLES:
                segment = np.pad(segment, (0, TARGET_SAMPLES - len(segment)))
            metadata.append({
                "file_name": path.name, "segment_role": role, "start_sec": float(start),
                "end_sec": float(start + SEGMENT_SECONDS), "duration_sec": float(duration),
                "source_sample_rate_hz": int(info.samplerate), "channels": int(info.channels),
                "format": str(info.format), "subtype": str(info.subtype), "bytes": path.stat().st_size,
                "average_bitrate_bps": float(path.stat().st_size * 8 / duration), "sha256": sha256(path),
            })
            segments.append(segment.astype(np.float32, copy=False))
    return pd.DataFrame(metadata), segments


def _load_mert(project_root: Path):
    local_snapshot = Path.home() / ".cache/huggingface/hub/models--m-a-p--MERT-v1-95M/snapshots" / MERT_REVISION
    local_only = local_snapshot.is_dir()
    config = AutoConfig.from_pretrained(MERT_NAME, revision=MERT_REVISION, trust_remote_code=True, local_files_only=local_only)
    config.conv_pos_batch_norm = False
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MERT_NAME, revision=MERT_REVISION, trust_remote_code=True, local_files_only=local_only)
    encoder = AutoModel.from_pretrained(MERT_NAME, revision=MERT_REVISION, config=config, trust_remote_code=True, local_files_only=local_only)
    encoder.eval()
    return processor, encoder


def run_inference(input_paths: Iterable[str | Path], project_root: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(project_root)
    metadata, waveforms = load_inputs(input_paths)
    feature_header = pd.read_csv(root / "data/processed/features/handcrafted_features_10s.csv", nrows=0).columns
    feature_columns = sorted([c for c in feature_header if c.startswith(("mfcc_", "mfcc_delta_", "mfcc_delta2_", "spectral_", "rms_", "zcr_"))])
    matrix = np.asarray([[handcrafted_features(w)[c] for c in feature_columns] for w in waveforms], dtype=np.float64)
    matrix_frame = pd.DataFrame(matrix, columns=feature_columns)
    scaler_path = root / "checkpoints/optimized/logistic_best.joblib"
    lr = joblib.load(scaler_path)
    svm = joblib.load(root / "checkpoints/optimized/svm_best.joblib")
    scores: dict[str, np.ndarray] = {
        "Logistic Regression": lr.predict_proba(matrix_frame)[:, 1],
        "RBF-SVM": svm.decision_function(matrix_frame),
    }
    cnn = LogMelCNN()
    checkpoint = torch.load(root / "checkpoints/optimized/cnn_best.pt", map_location="cpu", weights_only=False)
    cnn.load_state_dict(checkpoint["model_state_dict"])
    cnn.eval()
    with torch.inference_mode():
        scores["Log-Mel CNN"] = torch.sigmoid(cnn(torch.from_numpy(np.stack([logmel(w) for w in waveforms]))[:, None])).numpy()

    mert_artifact = joblib.load(root / "checkpoints/optimized/mert_best.joblib")
    processor, encoder = _load_mert(root)
    mert_embeddings = []
    for waveform in waveforms:
        inputs = processor(waveform, sampling_rate=SR, return_tensors="pt", padding=False)
        with torch.inference_mode():
            output = encoder(input_values=inputs["input_values"].float(), attention_mask=inputs.get("attention_mask"), output_hidden_states=True, return_dict=True)
        mert_embeddings.append(output.hidden_states[int(mert_artifact["layer"])].float().mean(dim=1).squeeze(0).numpy())
    emb = np.asarray(mert_embeddings, dtype=np.float32)
    scores["Frozen MERT + LR"] = mert_artifact["classifier"].predict_proba(mert_artifact["scaler"].transform(emb))[:, 1]

    segment_rows = []
    for model_name, values in scores.items():
        for index, value in enumerate(values):
            row = metadata.iloc[index]
            threshold = THRESHOLDS[model_name]["segment"]
            segment_rows.append({**row.to_dict(), "model": model_name, "score": float(value),
                                 "score_type": "SVM decision margin" if model_name == "RBF-SVM" else "AI score",
                                 "segment_threshold": threshold, "segment_prediction": "AI 생성" if value >= threshold else "인간 제작"})
    segment_frame = pd.DataFrame(segment_rows)
    track_rows = []
    for (file_name, model_name), group in segment_frame.groupby(["file_name", "model"], sort=False):
        threshold = THRESHOLDS[model_name]["track"]
        mean_score = float(group["score"].mean())
        track_rows.append({"file_name": file_name, "model": model_name, "n_segments": len(group),
                           "ai_score": mean_score, "track_threshold": threshold,
                           "margin_to_threshold": mean_score - threshold,
                           "segment_min": float(group["score"].min()), "segment_max": float(group["score"].max()),
                           "score_type": group["score_type"].iloc[0],
                           "prediction": "AI 생성" if mean_score >= threshold else "인간 제작"})
    track_frame = pd.DataFrame(track_rows)
    consensus = track_frame.assign(is_ai=track_frame["prediction"].eq("AI 생성")).groupby("file_name", sort=False).agg(
        ai_votes=("is_ai", "sum"), model_count=("is_ai", "size"),
    ).reset_index()
    consensus["consensus_prediction"] = np.where(consensus["ai_votes"] == consensus["model_count"], "AI 생성", "모델 불일치")
    return {"input_metadata": metadata.drop_duplicates("file_name"), "segment_predictions": segment_frame,
            "track_predictions": track_frame, "consensus": consensus}


def save_results(results: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in results.items():
        frame.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    (output / "run_config.json").write_text(json.dumps({"sample_rate": SR, "segment_seconds": SEGMENT_SECONDS,
        "aggregation": "mean of start/middle/end 10-second segments", "thresholds": THRESHOLDS,
        "models": list(THRESHOLDS)}, ensure_ascii=False, indent=2), encoding="utf-8")
