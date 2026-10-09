"""
Passive acoustic flow-state monitoring for a CAPD benchtop challenge prototype.

This script is a research prototype. It is designed for water-bag and flexible-
tubing experiments only. It must not be connected to a patient or used to make
medical decisions.

The proposed workflow is:

    contact sensor -> WAV recording -> signal features -> Random Forest
                   -> normal / reduced_flow / stopped_flow / noise

Why start with a Random Forest?

    The first challenge question is whether passive tubing vibration contains a
    repeatable signal at all. A Random Forest works well with a modest tabular
    dataset, models nonlinear feature relationships, provides feature
    importance, and is easier to inspect than a neural network. Deep learning
    would add complexity before the sensor principle has been demonstrated.

Why use metadata and session groups?

    Each recording produces many overlapping windows. Randomly splitting those
    windows would leak almost identical audio into training and testing. This
    script keeps every recording from one experimental session in the same
    split. Record each flow state in at least two independent sessions.

Quick start:

    py -3 capd_flow_monitor.py init
    # Add WAV recordings and complete capd_challenge_data/metadata.csv.
    py -3 capd_flow_monitor.py check
    py -3 capd_flow_monitor.py train
    py -3 capd_flow_monitor.py predict --audio capd_challenge_data/audio/test.wav
    py -3 capd_flow_monitor.py visualize --audio capd_challenge_data/audio/test.wav

Running the file with no command performs a readiness check and exits normally,
even when no recordings have been collected yet.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path 
from typing import Iterable

# Third-party imports are optional during readiness and project initialization.
# This lets VS Code run the file successfully before the dataset or packages are
# ready.
try:
    import joblib
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import signal, stats
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        classification_report,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        log_loss,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import Pipeline
except ModuleNotFoundError as exc:
    DEPENDENCY_ERROR = exc
else:
    DEPENDENCY_ERROR = None


VERSION = "0.1.0"
SUPPORTED_AUDIO = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aiff", ".aif"}
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_DIR = SCRIPT_DIR / "capd_challenge_data"
DEFAULT_METADATA = DEFAULT_PROJECT_DIR / "metadata.csv"
DEFAULT_MODEL = SCRIPT_DIR / "capd_flow_model.joblib"

METADATA_COLUMNS = [
    "audio_path",
    "flow_state",
    "session_id",
    "setup_id",
    "restriction_percent",
    "reference_flow_ml_min",
    "notes",
]


@dataclass(frozen=True)
class AudioConfig:
    """Audio settings saved inside the model package."""

    sample_rate: int = 22_050
    window_seconds: float = 2.0
    hop_seconds: float = 1.0
    lowcut_hz: float = 20.0
    highcut_hz: float = 6_000.0
    n_mfcc: int = 20


class DatasetNotReadyError(RuntimeError):
    """Raised for an incomplete dataset without treating readiness as a crash."""


# =============================================================================
# PROJECT INITIALIZATION AND METADATA
# =============================================================================
def initialize_project(args: argparse.Namespace) -> None:
    """Create a safe dataset layout without overwriting existing work."""
    project_dir = Path(args.project_dir).resolve()
    audio_dir = project_dir / "audio"
    metadata_path = project_dir / "metadata.csv"
    guide_path = project_dir / "DATA_GUIDE.txt"

    audio_dir.mkdir(parents=True, exist_ok=True)

    if not metadata_path.exists():
        with metadata_path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(METADATA_COLUMNS)
        print(f"[CREATED] {metadata_path}")
    else:
        print(f"[EXISTS]  {metadata_path}")

    if not guide_path.exists():
        guide_path.write_text(
            "CAPD benchtop acoustic dataset\n"
            "================================\n\n"
            "1. Put independent WAV recordings in the audio folder.\n"
            "2. Add one metadata.csv row per recording.\n"
            "3. Recommended labels: normal, reduced_flow, stopped_flow, noise.\n"
            "4. session_id means one independent day or full sensor remount.\n"
            "5. Record every label in at least two different sessions.\n"
            "6. Never duplicate a recording to increase the sample count.\n"
            "7. reference_flow_ml_min is optional ground truth from a scale.\n",
            encoding="utf-8",
        )
        print(f"[CREATED] {guide_path}")
    else:
        print(f"[EXISTS]  {guide_path}")

    print(f"[READY]   Add recordings to {audio_dir}")


def read_metadata(metadata_path: Path, audio_root: Path) -> list[dict[str, str]]:
    """Read and validate one metadata row per independent recording."""
    if not metadata_path.is_file():
        raise DatasetNotReadyError(f"Metadata CSV not found: {metadata_path}")

    with metadata_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        missing_columns = [name for name in ("audio_path", "flow_state", "session_id") if name not in fieldnames]
        if missing_columns:
            raise DatasetNotReadyError(
                "metadata.csv is missing required column(s): " + ", ".join(missing_columns)
            )
        rows = []
        for raw_row in reader:
            row = {key: (value or "").strip() for key, value in raw_row.items() if key is not None}
            if any(row.values()):
                rows.append(row)

    if not rows:
        raise DatasetNotReadyError(f"Metadata CSV has no experiment rows yet: {metadata_path}")

    for row_number, row in enumerate(rows, start=2):
        for required in ("audio_path", "flow_state", "session_id"):
            if not row.get(required):
                raise ValueError(f"metadata.csv row {row_number} has an empty {required} value.")

        path = Path(row["audio_path"])
        resolved = path if path.is_absolute() else audio_root / path
        row["_resolved_audio_path"] = str(resolved.resolve())

    return rows


def resolve_audio_root(metadata_path: Path, audio_root_arg: str | None) -> Path:
    """Use an explicit audio root or default to the metadata folder."""
    return Path(audio_root_arg).resolve() if audio_root_arg else metadata_path.resolve().parent


# =============================================================================
# SIGNAL PROCESSING AND FEATURE EXTRACTION
# =============================================================================
def integrate(values: np.ndarray, coordinates: np.ndarray) -> float:
    """Integrate with NumPy 1.x or 2.x without requiring one exact version."""
    if values.size < 2:
        return 0.0
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, coordinates))
    return float(np.trapz(values, coordinates))


def bandpass_filter(y: np.ndarray, sr: int, lowcut_hz: float, highcut_hz: float) -> np.ndarray:
    """Remove DC/handling drift and frequencies outside the analysis band."""
    nyquist = sr / 2.0
    low = max(lowcut_hz / nyquist, 1e-5)
    high = min(highcut_hz / nyquist, 0.999)
    if not 0 < low < high < 1:
        raise ValueError(
            f"Invalid filter range {lowcut_hz}-{highcut_hz} Hz for sample rate {sr} Hz."
        )

    # sosfiltfilt needs enough samples for edge padding. Normal experiment
    # windows are much longer, but padding keeps short test clips usable.
    minimum_samples = 128
    if y.size < minimum_samples:
        y = np.pad(y, (0, minimum_samples - y.size))
    sos = signal.butter(4, [low, high], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, y)


def summarize(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Reduce a time-varying feature to stable tabular statistics."""
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def spectral_entropy(power: np.ndarray) -> float:
    """Describe whether spectral power is concentrated or broadly distributed."""
    power = np.asarray(power, dtype=float)
    power = power[power > 0]
    if power.size <= 1:
        return 0.0
    probabilities = power / np.sum(power)
    return float(-np.sum(probabilities * np.log2(probabilities)) / np.log2(power.size))


def extract_features(y: np.ndarray, sr: int, cfg: AudioConfig) -> dict[str, float]:
    """Extract amplitude, spectral, MFCC, envelope, and PSD descriptors."""
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        raise ValueError("Cannot extract features from an empty audio window.")

    y = y - float(np.mean(y))
    filtered = bandpass_filter(y, sr, cfg.lowcut_hz, cfg.highcut_hz)

    features: dict[str, float] = {}

    # Absolute amplitude is intentionally retained. In the tap-opening script,
    # normalization reduced microphone-volume differences. Here, vibration
    # strength may be essential for separating flow from stopped flow. The
    # experiment must therefore keep sensor gain and clamping pressure stable.
    frame_length = min(2048, filtered.size)
    hop_length = max(1, min(512, frame_length // 4))
    raw_rms = librosa.feature.rms(
        y=filtered,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]
    absolute = np.abs(filtered)
    features.update(summarize(raw_rms, "raw_rms"))
    features["raw_peak"] = float(np.max(absolute))
    features["raw_abs_median"] = float(np.median(absolute))
    features["raw_abs_p95"] = float(np.percentile(absolute, 95))
    features["raw_crest_factor"] = float(np.max(absolute) / (np.sqrt(np.mean(filtered**2)) + 1e-12))

    # Pearson kurtosis describes how impulsive or heavy-tailed the vibration is.
    # Gaussian-like noise has a value near 3, while bubbles, impacts or abrupt
    # flow disturbances can produce larger values. A flat signal has no defined
    # kurtosis, so report 0 rather than allowing NaN into the ML feature matrix.
    if filtered.size >= 4 and float(np.std(filtered)) > 1e-12:
        features["raw_kurtosis"] = float(stats.kurtosis(filtered, fisher=False, bias=False))
    else:
        features["raw_kurtosis"] = 0.0

    features["clipped_fraction"] = float(np.mean(np.abs(y) >= 0.999))

    # Spectral-shape features use a normalized copy so they describe timbre and
    # flow texture rather than duplicating raw amplitude measurements.
    normalized = librosa.util.normalize(filtered)
    zcr = librosa.feature.zero_crossing_rate(normalized)[0]
    centroid = librosa.feature.spectral_centroid(y=normalized, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=normalized, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=normalized, sr=sr, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(y=normalized)[0]
    contrast = librosa.feature.spectral_contrast(y=normalized, sr=sr)
    mfcc = librosa.feature.mfcc(y=normalized, sr=sr, n_mfcc=cfg.n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc) if mfcc.shape[1] >= 9 else np.zeros_like(mfcc)

    for name, values in {
        "zcr": zcr,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff,
        "flatness": flatness,
    }.items():
        features.update(summarize(values, name))

    for index, values in enumerate(contrast, start=1):
        features.update(summarize(values, f"contrast_{index:02d}"))
    for index, values in enumerate(mfcc, start=1):
        features.update(summarize(values, f"mfcc_{index:02d}"))
    for index, values in enumerate(mfcc_delta, start=1):
        features.update(summarize(values, f"mfcc_delta_{index:02d}"))

    frequencies, psd = signal.welch(filtered, fs=sr, nperseg=min(2048, filtered.size))
    total_power = integrate(psd, frequencies) + 1e-15
    features["welch_total_power"] = total_power
    features["welch_entropy"] = spectral_entropy(psd)
    features["welch_peak_hz"] = float(frequencies[int(np.argmax(psd))])

    bands = [(20, 100), (100, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 6000)]
    for low, high in bands:
        high = min(high, int(sr / 2))
        mask = (frequencies >= low) & (frequencies < high)
        band_power = integrate(psd[mask], frequencies[mask]) if np.any(mask) else 0.0
        features[f"band_{low}_{high}_ratio"] = band_power / total_power

    envelope = np.abs(signal.hilbert(filtered))
    features.update(summarize(envelope, "amplitude_envelope"))
    if envelope.size > 1:
        features["envelope_trend"] = float(
            np.polyfit(np.linspace(0.0, 1.0, envelope.size), envelope, 1)[0]
        )
    else:
        features["envelope_trend"] = 0.0

    # Replace numerical edge cases so the model package remains predictable.
    return {name: float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)) for name, value in features.items()}


def split_into_windows(y: np.ndarray, sr: int, cfg: AudioConfig) -> Iterable[np.ndarray]:
    """Yield overlapping windows while keeping one short recording usable."""
    window = int(cfg.window_seconds * sr)
    hop = int(cfg.hop_seconds * sr)
    if window <= 0 or hop <= 0:
        raise ValueError("window_seconds and hop_seconds must be greater than zero.")
    if y.size <= window:
        yield y
        return
    for start in range(0, y.size - window + 1, hop):
        yield y[start : start + window]


def load_audio_windows(path: Path, cfg: AudioConfig) -> list[np.ndarray]:
    """Load a mono recording and split it into model-ready windows."""
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_AUDIO:
        raise ValueError(f"Unsupported audio format: {path.suffix}. WAV is recommended.")
    y, sr = librosa.load(path, sr=cfg.sample_rate, mono=True)
    if y.size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    return list(split_into_windows(y, sr, cfg))


# =============================================================================
# DATASET, SPLITTING, AND MODEL TRAINING
# =============================================================================
def collect_dataset(
    metadata_path: Path,
    audio_root: Path,
    cfg: AudioConfig,
    quiet: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, list[dict[str, str]]]:
    """Convert labelled recordings into window-level feature rows."""
    metadata_rows = read_metadata(metadata_path, audio_root)
    feature_rows: list[dict[str, float]] = []
    labels: list[str] = []
    groups: list[str] = []
    recording_ids: list[str] = []

    for row in metadata_rows:
        audio_path = Path(row["_resolved_audio_path"])
        if not quiet:
            print(f"Processing {audio_path}", flush=True)
        windows = load_audio_windows(audio_path, cfg)
        if not quiet:
            print(f"  state={row['flow_state']}, session={row['session_id']}, windows={len(windows)}")

        for window_audio in windows:
            feature_rows.append(extract_features(window_audio, cfg.sample_rate, cfg))
            labels.append(row["flow_state"])
            groups.append(row["session_id"])
            recording_ids.append(str(audio_path))

    if not feature_rows:
        raise DatasetNotReadyError("No usable audio windows were extracted.")

    feature_names = sorted(feature_rows[0])
    x = np.asarray([[row[name] for name in feature_names] for row in feature_rows], dtype=float)
    y = np.asarray(labels, dtype=str)
    return (
        x,
        y,
        feature_names,
        np.asarray(groups, dtype=str),
        np.asarray(recording_ids, dtype=str),
        metadata_rows,
    )


def validate_group_coverage(y: np.ndarray, groups: np.ndarray) -> None:
    """Require every class in at least two independent sessions."""
    classes = sorted(set(y.tolist()))
    if len(classes) < 2:
        raise DatasetNotReadyError("Record at least two flow_state classes before training.")

    insufficient = []
    for class_name in classes:
        class_groups = set(groups[y == class_name].tolist())
        if len(class_groups) < 2:
            insufficient.append(f"{class_name} ({len(class_groups)} session)")
    if insufficient:
        raise DatasetNotReadyError(
            "Each class needs recordings from at least two independent sessions. Missing: "
            + ", ".join(insufficient)
        )


def choose_group_holdout(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Find a session-level split containing every class on both sides."""
    all_classes = set(y.tolist())
    for offset in range(200):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state + offset,
        )
        train_index, test_index = next(splitter.split(x, y, groups=groups))
        if set(y[train_index].tolist()) == all_classes and set(y[test_index].tolist()) == all_classes:
            return train_index, test_index
    raise DatasetNotReadyError(
        "Could not create a session-level holdout containing every class. "
        "Record all states in more independent sessions or adjust --test-size."
    )


def make_model(random_state: int) -> Pipeline:
    """Create the deliberately simple first-pass flow-state classifier."""
    forest = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("model", forest),
        ]
    )


def audio_config_from_args(args: argparse.Namespace) -> AudioConfig:
    return AudioConfig(
        sample_rate=args.sample_rate,
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
        lowcut_hz=args.lowcut_hz,
        highcut_hz=args.highcut_hz,
    )


def evaluate_probabilistic_classifier(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> dict:
    """Calculate discrimination, agreement, and probability-quality metrics."""
    y_true = np.asarray(y_true, dtype=str)
    probabilities = np.asarray(probabilities, dtype=float)
    classes = np.asarray(classes, dtype=str)
    predictions = classes[np.argmax(probabilities, axis=1)]
    one_hot = (y_true[:, None] == classes[None, :]).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=classes)

    per_class: dict[str, dict[str, float | int | None]] = {}
    roc_auc_values: list[float] = []
    average_precision_values: list[float] = []
    supports: list[int] = []

    for index, class_name in enumerate(classes):
        binary_true = one_hot[:, index]
        true_positive = int(matrix[index, index])
        false_negative = int(np.sum(matrix[index, :]) - true_positive)
        false_positive = int(np.sum(matrix[:, index]) - true_positive)
        true_negative = int(np.sum(matrix) - true_positive - false_negative - false_positive)
        support = int(np.sum(binary_true))

        # AUC is undefined when an evaluation set contains only one side of a
        # one-vs-rest comparison. The grouped split normally prevents this,
        # but the guard keeps future smaller datasets JSON-safe.
        if np.unique(binary_true).size == 2:
            class_roc_auc = float(roc_auc_score(binary_true, probabilities[:, index]))
            class_average_precision = float(
                average_precision_score(binary_true, probabilities[:, index])
            )
            roc_auc_values.append(class_roc_auc)
            average_precision_values.append(class_average_precision)
            supports.append(support)
        else:
            class_roc_auc = None
            class_average_precision = None

        per_class[str(class_name)] = {
            "support": support,
            "sensitivity_recall": true_positive / max(true_positive + false_negative, 1),
            "specificity": true_negative / max(true_negative + false_positive, 1),
            "roc_auc_ovr": class_roc_auc,
            "average_precision_ovr": class_average_precision,
        }

    support_array = np.asarray(supports, dtype=float)
    weighted_roc_auc = (
        float(np.average(roc_auc_values, weights=support_array)) if roc_auc_values else None
    )
    weighted_average_precision = (
        float(np.average(average_precision_values, weights=support_array))
        if average_precision_values
        else None
    )

    return {
        "num_samples": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_precision": float(
            precision_score(y_true, predictions, labels=classes, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, predictions, labels=classes, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, predictions, labels=classes, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, predictions, labels=classes, average="weighted", zero_division=0)
        ),
        "matthews_correlation_coefficient": float(matthews_corrcoef(y_true, predictions)),
        "cohen_kappa": float(cohen_kappa_score(y_true, predictions, labels=classes)),
        "macro_roc_auc_ovr": float(np.mean(roc_auc_values)) if roc_auc_values else None,
        "weighted_roc_auc_ovr": weighted_roc_auc,
        "macro_average_precision_ovr": (
            float(np.mean(average_precision_values)) if average_precision_values else None
        ),
        "weighted_average_precision_ovr": weighted_average_precision,
        "log_loss": float(log_loss(y_true, probabilities, labels=classes)),
        # Multiclass Brier score: lower values mean better-calibrated class
        # probabilities. This is the mean summed squared probability error.
        "multiclass_brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=classes,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": classes.tolist(),
        "per_class": per_class,
    }


def aggregate_recording_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    recording_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Collapse correlated windows into one robust prediction per WAV file."""
    recording_labels: list[str] = []
    recording_probabilities: list[np.ndarray] = []
    held_out_recordings: list[str] = []

    for recording_id in sorted(set(recording_ids.tolist())):
        mask = recording_ids == recording_id
        labels = sorted(set(y_true[mask].tolist()))
        if len(labels) != 1:
            raise ValueError(f"Recording has conflicting labels: {recording_id}")
        aggregate = np.median(probabilities[mask], axis=0)
        aggregate = aggregate / (np.sum(aggregate) + 1e-12)
        recording_labels.append(labels[0])
        recording_probabilities.append(aggregate)
        held_out_recordings.append(recording_id)

    return (
        np.asarray(recording_labels, dtype=str),
        np.asarray(recording_probabilities, dtype=float),
        held_out_recordings,
    )


def train(args: argparse.Namespace) -> None:
    """Evaluate on held-out sessions, then fit and save the complete dataset."""
    cfg = audio_config_from_args(args)
    metadata_path = Path(args.metadata).resolve()
    audio_root = resolve_audio_root(metadata_path, args.audio_root)
    x, y, feature_names, groups, recording_ids, metadata_rows = collect_dataset(
        metadata_path,
        audio_root,
        cfg,
        quiet=args.quiet,
    )

    validate_group_coverage(y, groups)
    train_index, test_index = choose_group_holdout(
        x,
        y,
        groups,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    evaluation_model = make_model(args.random_state)
    evaluation_model.fit(x[train_index], y[train_index])
    probabilities = evaluation_model.predict_proba(x[test_index])
    classes = np.asarray(evaluation_model.classes_, dtype=str)

    recording_y, recording_probabilities, held_out_recordings = (
        aggregate_recording_probabilities(
            y[test_index],
            probabilities,
            recording_ids[test_index],
        )
    )

    metrics = {
        "primary_evaluation_unit": "recording",
        "recording_level": evaluate_probabilistic_classifier(
            recording_y,
            recording_probabilities,
            classes,
        ),
        "window_level_secondary": evaluate_probabilistic_classifier(
            y[test_index],
            probabilities,
            classes,
        ),
        "held_out_sessions": sorted(set(groups[test_index].tolist())),
        "held_out_recordings": held_out_recordings,
        "evaluation_note": (
            "Recording-level metrics are primary. Window-level metrics are secondary because "
            "overlapping windows from the same recording are correlated."
        ),
    }

    # The saved model uses all available recordings. The metrics above remain
    # from untouched sessions and therefore describe the honest evaluation.
    final_model = make_model(args.random_state)
    final_model.fit(x, y)
    forest = final_model.named_steps["model"]
    importances = sorted(
        zip(feature_names, forest.feature_importances_.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )[:15]

    package = {
        "package_version": VERSION,
        "purpose": "CAPD-like benchtop passive acoustic flow-state research",
        "model": final_model,
        "classes": final_model.classes_.tolist(),
        "feature_names": feature_names,
        "audio_config": asdict(cfg),
        "normal_label": args.normal_label,
        "metrics": metrics,
        "top_feature_importances": [
            {"feature": name, "importance": float(value)} for name, value in importances
        ],
        "num_recordings": len(metadata_rows),
        "num_sessions": len(set(groups.tolist())),
        "num_windows": int(x.shape[0]),
        "num_features": int(x.shape[1]),
    }

    model_path = Path(args.model).resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, model_path)

    printable = {key: value for key, value in package.items() if key != "model"}
    print(json.dumps(printable, indent=2))
    print(f"[SAVED] Model package: {model_path}")


# =============================================================================
# PREDICTION AND VISUALIZATION
# =============================================================================
def prediction_features(audio_path: Path, package: dict) -> np.ndarray:
    """Reproduce training-time features for one unseen recording."""
    cfg = AudioConfig(**package["audio_config"])
    windows = load_audio_windows(audio_path, cfg)
    feature_rows = [extract_features(window, cfg.sample_rate, cfg) for window in windows]
    feature_names = package["feature_names"]
    return np.asarray([[row[name] for name in feature_names] for row in feature_rows], dtype=float)


def predict(args: argparse.Namespace) -> None:
    """Classify an unseen recording and expose uncertainty explicitly."""
    model_path = Path(args.model).resolve()
    audio_path = Path(args.audio).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    package = joblib.load(model_path)
    x = prediction_features(audio_path, package)
    model = package["model"]
    probabilities = model.predict_proba(x)
    classes = np.asarray(model.classes_, dtype=str)

    # The median resists a single bumped or noisy window. A later real-time
    # version can add a sustained-event state machine rather than one aggregate.
    aggregate = np.median(probabilities, axis=0)
    aggregate = aggregate / (np.sum(aggregate) + 1e-12)
    best_index = int(np.argmax(aggregate))
    predicted_state = str(classes[best_index])
    confidence = float(aggregate[best_index])
    reported_state = predicted_state if confidence >= args.confidence_threshold else "uncertain"

    window_states = classes[np.argmax(probabilities, axis=1)]
    counts = Counter(window_states.tolist())
    normal_label = package.get("normal_label", "normal")

    result = {
        "prototype_only": True,
        "audio": str(audio_path),
        "reported_state": reported_state,
        "highest_probability_state": predicted_state,
        "confidence": confidence,
        "possible_flow_anomaly": reported_state not in {normal_label, "uncertain"},
        "aggregate_probabilities": {
            str(class_name): float(aggregate[index]) for index, class_name in enumerate(classes)
        },
        "window_state_counts": dict(sorted(counts.items())),
        "num_windows": int(x.shape[0]),
        "warning": (
            "Research output only. This model does not diagnose catheter blockage, "
            "dialysis adequacy, or any medical condition."
        ),
    }
    print(json.dumps(result, indent=2))


def visualize(args: argparse.Namespace) -> None:
    """Save waveform, envelope, spectrogram, PSD, and MFCC evidence."""
    audio_path = Path(args.audio).resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    cfg = AudioConfig(
        sample_rate=args.sample_rate,
        lowcut_hz=args.lowcut_hz,
        highcut_hz=args.highcut_hz,
    )
    y, sr = librosa.load(audio_path, sr=cfg.sample_rate, mono=True)
    filtered = bandpass_filter(y - float(np.mean(y)), sr, cfg.lowcut_hz, cfg.highcut_hz)
    envelope = np.abs(signal.hilbert(filtered))
    stft = librosa.stft(librosa.util.normalize(filtered), n_fft=2048, hop_length=512)
    spectrogram_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    mfcc = librosa.feature.mfcc(y=librosa.util.normalize(filtered), sr=sr, n_mfcc=cfg.n_mfcc)
    frequencies, psd = signal.welch(filtered, fs=sr, nperseg=min(4096, filtered.size))
    times = np.arange(filtered.size) / sr

    figure, axes = plt.subplots(4, 1, figsize=(12, 15))
    figure.suptitle(f"CAPD-like passive acoustic analysis: {audio_path.name}", fontsize=15)
    figure.subplots_adjust(top=0.94, hspace=0.38)

    axes[0].plot(times, filtered, color="#176B87", linewidth=0.65, label="Filtered vibration")
    axes[0].plot(times, envelope, color="#D95F59", linewidth=0.9, alpha=0.8, label="Envelope")
    axes[0].set(title="Waveform and amplitude envelope", xlabel="Time (s)", ylabel="Amplitude")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25)

    spectrogram_image = librosa.display.specshow(
        spectrogram_db,
        sr=sr,
        hop_length=512,
        x_axis="time",
        y_axis="log",
        ax=axes[1],
        cmap="magma",
    )
    axes[1].set_title("Log-frequency spectrogram")
    figure.colorbar(spectrogram_image, ax=axes[1], format="%+2.0f dB")

    axes[2].semilogy(frequencies, psd + 1e-18, color="#5A4E9C", linewidth=1.0)
    axes[2].set(title="Welch power spectral density", xlabel="Frequency (Hz)", ylabel="Power / Hz")
    axes[2].grid(alpha=0.25)

    mfcc_image = librosa.display.specshow(mfcc, x_axis="time", sr=sr, ax=axes[3], cmap="viridis")
    axes[3].set(title="MFCC spectral-shape features", ylabel="MFCC coefficient")
    figure.colorbar(mfcc_image, ax=axes[3])

    output_path = (
        Path(args.output).resolve()
        if args.output
        else audio_path.with_name(f"{audio_path.stem}_capd_analysis.png")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"[SAVED] Acoustic graphs: {output_path}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)


# =============================================================================
# READINESS CHECK AND COMMAND-LINE INTERFACE
# =============================================================================
def readiness_check(args: argparse.Namespace) -> None:
    """Report readiness without failing when recordings do not exist yet."""
    metadata_path = Path(args.metadata).resolve()
    audio_root = resolve_audio_root(metadata_path, args.audio_root)

    print(f"CAPD passive acoustic monitor v{VERSION} readiness check")
    print("[OK] Script loaded successfully")
    if DEPENDENCY_ERROR is None:
        print("[OK] Python dependencies are available")
    else:
        print(f"[WAIT] Missing Python dependency: {DEPENDENCY_ERROR.name}")
        print("       Install with: py -3 -m pip install -r requirements.txt")

    if not metadata_path.is_file():
        print(f"[WAIT] Metadata not found: {metadata_path}")
        print("       Create the project layout with: py -3 capd_flow_monitor.py init")
        return

    print(f"[OK] Metadata found: {metadata_path}")
    try:
        rows = read_metadata(metadata_path, audio_root)
    except DatasetNotReadyError as exc:
        print(f"[WAIT] {exc}")
        return
    except ValueError as exc:
        print(f"[FIX] {exc}")
        return

    present = 0
    missing = []
    for row in rows:
        path = Path(row["_resolved_audio_path"])
        if path.is_file():
            present += 1
        else:
            missing.append(str(path))

    states = Counter(row["flow_state"] for row in rows)
    sessions = sorted({row["session_id"] for row in rows})
    print(f"[INFO] Metadata rows: {len(rows)}")
    print(f"[INFO] Audio files found: {present}/{len(rows)}")
    print(f"[INFO] Sessions: {len(sessions)} ({', '.join(sessions)})")
    print(f"[INFO] Flow states: {dict(sorted(states.items()))}")
    if missing:
        print(f"[FIX] Missing audio files: {len(missing)}")
        for path in missing[:5]:
            print(f"      {path}")
    elif len(sessions) < 2:
        print("[WAIT] Record every state in at least two independent sessions.")
    else:
        print("[READY] Dataset paths are valid; run the train command when class coverage is sufficient.")


def add_audio_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sample-rate", type=int, default=22_050)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--hop-seconds", type=float, default=1.0)
    parser.add_argument("--lowcut-hz", type=float, default=20.0)
    parser.add_argument("--highcut-hz", type=float, default=6_000.0)


def build_parser() -> argparse.ArgumentParser:
    """Define commands for PowerShell, VS Code, and remote teammates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(
        func=readiness_check,
        require_dependencies=False,
        metadata=str(DEFAULT_METADATA),
        audio_root=None,
    )
    subparsers = parser.add_subparsers()

    init_parser = subparsers.add_parser("init", help="Create the dataset folders and metadata template.")
    init_parser.add_argument("--project-dir", default=str(DEFAULT_PROJECT_DIR))
    init_parser.set_defaults(func=initialize_project, require_dependencies=False)

    check_parser = subparsers.add_parser("check", help="Check dependencies, metadata, sessions, and audio paths.")
    check_parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    check_parser.add_argument("--audio-root")
    check_parser.set_defaults(func=readiness_check, require_dependencies=False)

    train_parser = subparsers.add_parser("train", help="Train and evaluate the flow-state classifier.")
    train_parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    train_parser.add_argument("--audio-root")
    train_parser.add_argument("--model", default=str(DEFAULT_MODEL))
    train_parser.add_argument("--normal-label", default="normal")
    train_parser.add_argument("--test-size", type=float, default=0.25)
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.add_argument("--quiet", action="store_true")
    add_audio_arguments(train_parser)
    train_parser.set_defaults(func=train, require_dependencies=True)

    predict_parser = subparsers.add_parser("predict", help="Classify one unseen recording.")
    predict_parser.add_argument("--model", default=str(DEFAULT_MODEL))
    predict_parser.add_argument("--audio", required=True)
    predict_parser.add_argument("--confidence-threshold", type=float, default=0.60)
    predict_parser.set_defaults(func=predict, require_dependencies=True)

    visualize_parser = subparsers.add_parser("visualize", help="Save acoustic evidence plots for one WAV file.")
    visualize_parser.add_argument("--audio", required=True)
    visualize_parser.add_argument("--output")
    visualize_parser.add_argument("--sample-rate", type=int, default=22_050)
    visualize_parser.add_argument("--lowcut-hz", type=float, default=20.0)
    visualize_parser.add_argument("--highcut-hz", type=float, default=6_000.0)
    visualize_parser.add_argument("--show", action="store_true")
    visualize_parser.set_defaults(func=visualize, require_dependencies=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.require_dependencies and DEPENDENCY_ERROR is not None:
        print(
            f"Missing dependency: {DEPENDENCY_ERROR.name}\n"
            "Install dependencies with: py -3 -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(1)

    try:
        args.func(args)
    except DatasetNotReadyError as exc:
        print(f"[WAIT] {exc}")
        raise SystemExit(0)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
