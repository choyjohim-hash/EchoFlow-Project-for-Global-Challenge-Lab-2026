"""
Unified EchoFlow digital proof-of-concept pipeline.

This file provides one command-line entry point for the two tested modules:

    Simscape mass-flow CSV
        -> simulated acoustic-proxy WAV
        -> librosa/scipy feature extraction
        -> Random Forest training or prediction

The proxy WAV files are synthetic. They do not prove that a contact microphone
can detect tubing flow, and the classifier must not be used for medical
decisions. Real Korg contact-microphone recordings can later enter at the WAV
feature-extraction stage without using the proxy generator.

Examples:

    py -3 echoflow_combined_pipeline.py generate --csv-dir "PATH_TO_CSV_FOLDER"
    py -3 echoflow_combined_pipeline.py features --audio "recording.wav"
    py -3 echoflow_combined_pipeline.py train
    py -3 echoflow_combined_pipeline.py predict --audio "unseen.wav"

Generate proxy audio and train in one command:

    py -3 echoflow_combined_pipeline.py generate-train \
        --csv-dir "PATH_TO_CSV_FOLDER"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import capd_flow_monitor as monitor
import simscape_acoustic_proxy as proxy


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_DIR = SCRIPT_DIR / "capd_simulated_data"
DEFAULT_METADATA = DEFAULT_PROJECT_DIR / "metadata.csv"
DEFAULT_MODEL = DEFAULT_PROJECT_DIR / "capd_simulated_model.joblib"


def require_signal_dependencies() -> None:
    """Fail clearly only when a command needs the scientific Python stack."""
    if monitor.DEPENDENCY_ERROR is not None:
        missing_name = monitor.DEPENDENCY_ERROR.name
        raise RuntimeError(
            f"Missing dependency: {missing_name}. Install the project packages with "
            "'py -3 -m pip install -r requirements.txt'."
        )


# =============================================================================
# STAGE 1: SIMSCAPE FLOW CSV -> SYNTHETIC ACOUSTIC-PROXY WAV
# =============================================================================
def generate_proxy_audio(args: argparse.Namespace) -> None:
    """Delegate proxy generation to the tested Simscape conversion module."""
    proxy.generate(args)


# =============================================================================
# STAGE 2: WAV -> LIBROSA/SCIPY FEATURE TABLE
# =============================================================================
def extract_feature_table(args: argparse.Namespace) -> None:
    """Extract and optionally save one row of features per audio window."""
    require_signal_dependencies()

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    cfg = monitor.AudioConfig(
        sample_rate=args.sample_rate,
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
        lowcut_hz=args.lowcut_hz,
        highcut_hz=args.highcut_hz,
        n_mfcc=args.n_mfcc,
    )
    windows = monitor.load_audio_windows(audio_path, cfg)
    feature_rows = [
        monitor.extract_features(window, cfg.sample_rate, cfg)
        for window in windows
    ]
    feature_names = sorted(feature_rows[0])

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else audio_path.with_name(f"{audio_path.stem}_features.csv")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["window_number", *feature_names],
        )
        writer.writeheader()
        for window_number, row in enumerate(feature_rows, start=1):
            writer.writerow({"window_number": window_number, **row})

    preview_names = [
        "raw_rms_mean",
        "raw_peak",
        "raw_crest_factor",
        "raw_kurtosis",
        "zcr_mean",
        "centroid_mean",
        "bandwidth_mean",
        "flatness_mean",
        "welch_total_power",
        "welch_entropy",
        "welch_peak_hz",
        "mfcc_01_mean",
        "amplitude_envelope_mean",
    ]
    displayed_names = feature_names if args.print_all else preview_names
    feature_values = [
        {
            "window_number": window_number,
            "values": {
                name: float(row[name])
                for name in displayed_names
                if name in row
            },
        }
        for window_number, row in enumerate(feature_rows, start=1)
    ]

    report = {
        "prototype_only": True,
        "audio": str(audio_path),
        "feature_table": str(output_path),
        "num_windows": len(feature_rows),
        "num_features_per_window": len(feature_names),
        "displayed_values": (
            "all extracted features"
            if args.print_all
            else "selected numerical preview; complete values are in feature_table"
        ),
        "feature_values_by_window": feature_values,
        "feature_groups": [
            "amplitude and RMS",
            "crest factor and kurtosis",
            "zero-crossing rate",
            "spectral shape",
            "MFCC and MFCC delta",
            "Welch PSD and frequency-band ratios",
            "amplitude envelope",
        ],
    }
    print(json.dumps(report, indent=2))


# =============================================================================
# STAGE 3: FEATURE TABLES -> RANDOM FOREST MODEL
# =============================================================================
def train_classifier(args: argparse.Namespace) -> None:
    """Extract dataset features, evaluate by session, and save the model."""
    require_signal_dependencies()
    monitor.train(args)


def predict_recording(args: argparse.Namespace) -> None:
    """Extract matching features and classify one unseen WAV recording."""
    require_signal_dependencies()
    monitor.predict(args)


def visualize_recording(args: argparse.Namespace) -> None:
    """Save waveform, envelope, PSD, spectrogram, and MFCC plots."""
    require_signal_dependencies()
    monitor.visualize(args)


def compare_flow_conditions(args: argparse.Namespace) -> None:
    """Plot a representative waveform and spectrogram for every flow label."""
    require_signal_dependencies()

    # These imports live here so the no-command readiness check remains usable
    # before the optional scientific Python packages have been installed.
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import numpy as np

    metadata_path = Path(args.metadata).expanduser().resolve()
    audio_root = monitor.resolve_audio_root(metadata_path, args.audio_root)
    rows = monitor.read_metadata(metadata_path, audio_root)

    rows_by_label: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        rows_by_label.setdefault(row["flow_state"], []).append(row)

    preferred_order = ["normal", "reduced_flow", "stopped_flow", "noise"]
    labels = [
        label for label in preferred_order if label in rows_by_label
    ]
    labels.extend(
        sorted(label for label in rows_by_label if label not in labels)
    )
    if not labels:
        raise ValueError("No flow-state labels were found in the metadata.")

    cfg = monitor.AudioConfig(
        sample_rate=args.sample_rate,
        lowcut_hz=args.lowcut_hz,
        highcut_hz=args.highcut_hz,
    )
    figure, axes = plt.subplots(
        len(labels),
        2,
        figsize=(13, max(3.0 * len(labels), 5.5)),
        squeeze=False,
        constrained_layout=True,
    )
    colours = {
        "normal": "#16825D",
        "reduced_flow": "#D28A16",
        "stopped_flow": "#C13B3B",
        "noise": "#666A73",
    }
    selected_recordings: dict[str, str] = {}

    for row_index, label in enumerate(labels):
        row = rows_by_label[label][0]
        audio_path = Path(row["_resolved_audio_path"])
        if not audio_path.is_file():
            raise FileNotFoundError(
                f"Audio file listed in metadata was not found: {audio_path}"
            )
        selected_recordings[label] = str(audio_path)

        y, sr = librosa.load(audio_path, sr=cfg.sample_rate, mono=True)
        centered = y - float(np.mean(y))
        filtered = monitor.bandpass_filter(
            centered,
            sr,
            cfg.lowcut_hz,
            cfg.highcut_hz,
        )
        times = np.arange(filtered.size) / sr

        waveform_axis = axes[row_index, 0]
        waveform_axis.plot(
            times,
            filtered,
            color=colours.get(label, "#176B87"),
            linewidth=0.8,
        )
        waveform_axis.set(
            title=f"{label}: waveform",
            xlabel="Time (s)",
            ylabel="Amplitude",
        )
        waveform_axis.grid(alpha=0.18)

        stft = librosa.stft(
            librosa.util.normalize(filtered),
            n_fft=2048,
            hop_length=512,
        )
        spectrogram_db = librosa.amplitude_to_db(
            np.abs(stft),
            ref=np.max,
        )
        spectrogram_axis = axes[row_index, 1]
        image = librosa.display.specshow(
            spectrogram_db,
            sr=sr,
            hop_length=512,
            x_axis="time",
            y_axis="hz",
            cmap="magma",
            ax=spectrogram_axis,
        )
        spectrogram_axis.set(
            title=f"{label}: spectrogram",
            xlabel="Time (s)",
            ylabel="Frequency (Hz)",
        )
        figure.colorbar(
            image,
            ax=spectrogram_axis,
            format="%+2.0f dB",
        )

    source_kind = (
        "simulated acoustic proxies"
        if any(
            "SIMULATED" in row.get("notes", "").upper()
            for row in rows
        )
        else "recorded audio"
    )
    figure.suptitle(
        f"EchoFlow flow-condition comparison ({source_kind})",
        fontsize=15,
    )

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else metadata_path.with_name("flow_condition_comparison.png")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    if args.show:
        plt.show()
    else:
        plt.close(figure)

    print(
        json.dumps(
            {
                "prototype_only": True,
                "source": source_kind,
                "labels": labels,
                "selected_recordings": selected_recordings,
                "saved_figure": str(output_path),
                "warning": (
                    "Simulated proxy plots do not establish real contact-sensor "
                    "or clinical performance."
                    if source_kind == "simulated acoustic proxies"
                    else "Measured tubing results still require repeatability and "
                    "independent-session validation."
                ),
            },
            indent=2,
        )
    )


# =============================================================================
# COMPLETE DIGITAL WORKFLOW
# =============================================================================
def generate_and_train(args: argparse.Namespace) -> None:
    """Generate proxies and immediately train the classifier on that dataset."""
    require_signal_dependencies()
    proxy.generate(args)

    project_dir = Path(args.project_dir).expanduser().resolve()
    train_args = argparse.Namespace(
        metadata=str(project_dir / "metadata.csv"),
        audio_root=None,
        model=str(
            Path(args.model).expanduser().resolve()
            if args.model
            else project_dir / "capd_simulated_model.joblib"
        ),
        normal_label=args.normal_label,
        test_size=args.test_size,
        random_state=args.random_state,
        quiet=args.quiet,
        sample_rate=args.sample_rate,
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
        lowcut_hz=args.lowcut_hz,
        highcut_hz=args.highcut_hz,
    )
    monitor.train(train_args)


def readiness_check(_: argparse.Namespace) -> None:
    """Exit successfully when the Run triangle is pressed without arguments."""
    dependencies = (
        "available"
        if monitor.DEPENDENCY_ERROR is None
        else f"missing: {monitor.DEPENDENCY_ERROR.name}"
    )
    print("EchoFlow combined pipeline is ready.")
    print(f"Scientific Python dependencies: {dependencies}")
    print("No files were changed.")
    print(
        "Stages: Simscape CSV -> proxy WAV -> feature extraction "
        "-> Random Forest classification"
    )
    print(
        "Important: generated proxy audio is simulated and is not evidence of "
        "contact-sensor or clinical performance."
    )


def add_audio_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--hop-seconds", type=float, default=1.0)
    parser.add_argument("--lowcut-hz", type=float, default=20.0)
    parser.add_argument("--highcut-hz", type=float, default=6_000.0)


def add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--csv-dir",
        help="Folder containing the four standard Simscape CSV files.",
    )
    parser.add_argument(
        "--input",
        action="append",
        help="Manual input as LABEL=PERCENT=CSV_PATH; may be repeated.",
    )
    parser.add_argument("--project-dir", default=str(DEFAULT_PROJECT_DIR))
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--clips-per-state", type=int, default=2)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--density", type=float, default=997.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-noise",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(func=readiness_check)
    subparsers = parser.add_subparsers()

    generate_parser = subparsers.add_parser(
        "generate",
        help="Convert Simscape CSV traces into proxy WAV files and metadata.",
    )
    add_generation_arguments(generate_parser)
    generate_parser.set_defaults(func=generate_proxy_audio)

    features_parser = subparsers.add_parser(
        "features",
        help="Extract a visible per-window feature table from one WAV file.",
    )
    features_parser.add_argument("--audio", required=True)
    features_parser.add_argument("--output")
    features_parser.add_argument("--n-mfcc", type=int, default=20)
    features_parser.add_argument(
        "--print-all",
        action="store_true",
        help="Print all feature values in the terminal as well as saving the CSV.",
    )
    add_audio_arguments(features_parser)
    features_parser.set_defaults(func=extract_feature_table)

    train_parser = subparsers.add_parser(
        "train",
        help="Extract features, evaluate the classifier, and save a model.",
    )
    train_parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    train_parser.add_argument("--audio-root")
    train_parser.add_argument("--model", default=str(DEFAULT_MODEL))
    train_parser.add_argument("--normal-label", default="normal")
    train_parser.add_argument("--test-size", type=float, default=0.25)
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.add_argument("--quiet", action="store_true")
    add_audio_arguments(train_parser)
    train_parser.set_defaults(func=train_classifier)

    predict_parser = subparsers.add_parser(
        "predict",
        help="Classify one unseen proxy or measured WAV recording.",
    )
    predict_parser.add_argument("--model", default=str(DEFAULT_MODEL))
    predict_parser.add_argument("--audio", required=True)
    predict_parser.add_argument("--confidence-threshold", type=float, default=0.60)
    predict_parser.set_defaults(func=predict_recording)

    visualize_parser = subparsers.add_parser(
        "visualize",
        help="Save waveform, envelope, PSD, spectrogram, and MFCC evidence.",
    )
    visualize_parser.add_argument("--audio", required=True)
    visualize_parser.add_argument("--output")
    visualize_parser.add_argument("--sample-rate", type=int, default=16_000)
    visualize_parser.add_argument("--lowcut-hz", type=float, default=20.0)
    visualize_parser.add_argument("--highcut-hz", type=float, default=6_000.0)
    visualize_parser.add_argument("--show", action="store_true")
    visualize_parser.set_defaults(func=visualize_recording)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Plot representative waveforms and spectrograms for all conditions.",
    )
    compare_parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    compare_parser.add_argument("--audio-root")
    compare_parser.add_argument("--output")
    compare_parser.add_argument("--sample-rate", type=int, default=16_000)
    compare_parser.add_argument("--lowcut-hz", type=float, default=20.0)
    compare_parser.add_argument("--highcut-hz", type=float, default=6_000.0)
    compare_parser.add_argument("--show", action="store_true")
    compare_parser.set_defaults(func=compare_flow_conditions)

    complete_parser = subparsers.add_parser(
        "generate-train",
        help="Generate proxy WAV files and train the Random Forest in one run.",
    )
    add_generation_arguments(complete_parser)
    complete_parser.add_argument("--model")
    complete_parser.add_argument("--normal-label", default="normal")
    complete_parser.add_argument("--test-size", type=float, default=0.25)
    complete_parser.add_argument("--random-state", type=int, default=42)
    complete_parser.add_argument("--quiet", action="store_true")
    complete_parser.add_argument("--window-seconds", type=float, default=2.0)
    complete_parser.add_argument("--hop-seconds", type=float, default=1.0)
    complete_parser.add_argument("--lowcut-hz", type=float, default=20.0)
    complete_parser.add_argument("--highcut-hz", type=float, default=6_000.0)
    complete_parser.set_defaults(func=generate_and_train)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except monitor.DatasetNotReadyError as exc:
        print(f"[WAIT] {exc}")
        raise SystemExit(0)
    except (
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
