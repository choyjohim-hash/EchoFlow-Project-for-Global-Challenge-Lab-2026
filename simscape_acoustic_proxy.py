"""
Convert Simscape mass-flow CSV traces into acoustic proxy WAV files.

This utility bridges the physics demonstration and the existing
``capd_flow_monitor.py`` signal-processing pipeline:

    Simscape flow trace -> acoustic proxy WAV -> librosa/scipy features
                         -> Random Forest flow-state classification

The generated sounds are synthetic, representative signals for a digital
proof of concept. They are not recordings from dialysis tubing, do not model a
particular contact microphone, and must not be presented as clinical evidence.

Running this file with no command performs a readiness check and exits normally.
The generator itself uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import array
import csv
import json
import math
import random
import statistics
import sys
import wave
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_DIR = SCRIPT_DIR / "capd_simulated_data"
METADATA_COLUMNS = [
    "audio_path",
    "flow_state",
    "session_id",
    "setup_id",
    "restriction_percent",
    "reference_flow_ml_min",
    "notes",
]

# The exported filenames describe two restriction severities, but the first
# classifier intentionally uses the broader and more defensible reduced_flow
# class. Distinguishing exact blockage causes requires physical validation.
KNOWN_EXPORTS = {
    "normal_flow_001.csv": ("normal", 0.0),
    "restricted_flow_001.csv": ("reduced_flow", 55.0),
    "severely_restricted_flow_001.csv": ("reduced_flow", 85.0),
    "stopped_flow_001.csv": ("stopped_flow", 100.0),
}


@dataclass(frozen=True)
class FlowTrace:
    """One Simscape time series and its intended proof-of-concept label."""

    path: Path
    label: str
    restriction_percent: float
    time_s: tuple[float, ...]
    mass_flow_kg_s: tuple[float, ...]
    steady_flow_kg_s: float


def read_flow_csv(path: Path, label: str, restriction_percent: float) -> FlowTrace:
    """Read and validate the two columns exported from the Simulink Scope."""
    if not path.is_file():
        raise FileNotFoundError(f"Simscape CSV not found: {path}")

    samples: list[tuple[float, float]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"time_s", "mass_flow_kg_s"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{path.name} must contain the columns time_s and mass_flow_kg_s"
            )
        for row_number, row in enumerate(reader, start=2):
            try:
                time_value = float(row["time_s"])
                flow_value = float(row["mass_flow_kg_s"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Non-numeric value in {path.name} at row {row_number}"
                ) from exc
            if math.isfinite(time_value) and math.isfinite(flow_value):
                samples.append((time_value, flow_value))

    if len(samples) < 3:
        raise ValueError(f"{path.name} contains too few valid samples")

    samples.sort(key=lambda item: item[0])
    deduplicated: list[tuple[float, float]] = []
    for sample in samples:
        if deduplicated and sample[0] == deduplicated[-1][0]:
            deduplicated[-1] = sample
        else:
            deduplicated.append(sample)

    times = tuple(item[0] for item in deduplicated)
    flows = tuple(item[1] for item in deduplicated)
    if times[-1] <= times[0]:
        raise ValueError(f"{path.name} has no positive time span")

    steady_start = max(1, len(flows) // 2)
    steady_flow = statistics.median(abs(value) for value in flows[steady_start:])
    return FlowTrace(
        path=path,
        label=label,
        restriction_percent=restriction_percent,
        time_s=times,
        mass_flow_kg_s=flows,
        steady_flow_kg_s=steady_flow,
    )


def discover_traces(csv_dir: Path) -> list[FlowTrace]:
    """Find the four filenames created during the Simscape exercise."""
    traces = []
    missing = []
    for filename, (label, restriction) in KNOWN_EXPORTS.items():
        path = csv_dir / filename
        if path.is_file():
            traces.append(read_flow_csv(path, label, restriction))
        else:
            missing.append(filename)
    if missing:
        raise FileNotFoundError(
            "Missing expected Simscape export(s) in "
            f"{csv_dir}: {', '.join(missing)}"
        )
    return traces


def parse_manual_input(specification: str) -> tuple[str, Path, float]:
    """Parse LABEL=PERCENT=PATH while allowing '=' inside the path."""
    parts = specification.split("=", 2)
    if len(parts) != 3:
        raise ValueError(
            f"Invalid --input {specification!r}; expected LABEL=PERCENT=CSV_PATH"
        )
    label, restriction_text, path_text = parts
    try:
        restriction = float(restriction_text)
    except ValueError as exc:
        raise ValueError(
            f"Restriction percentage is not numeric in --input {specification!r}"
        ) from exc
    return label.strip(), Path(path_text).expanduser().resolve(), restriction


def interpolate(times: tuple[float, ...], values: tuple[float, ...], point: float) -> float:
    """Linear interpolation for the relatively short variable-step CSV traces."""
    if point <= times[0]:
        return values[0]
    if point >= times[-1]:
        return values[-1]

    low = 0
    high = len(times) - 1
    while high - low > 1:
        middle = (low + high) // 2
        if times[middle] <= point:
            low = middle
        else:
            high = middle
    span = times[high] - times[low]
    fraction = 0.0 if span == 0 else (point - times[low]) / span
    return values[low] + fraction * (values[high] - values[low])


def make_envelope(
    trace: FlowTrace,
    normal_flow_kg_s: float,
    duration_s: float,
    control_rate_hz: int,
    start_fraction: float,
) -> list[float]:
    """Resample a CSV flow trace to a low-rate acoustic amplitude envelope."""
    source_duration = trace.time_s[-1] - trace.time_s[0]
    usable_duration = min(duration_s, source_duration)
    maximum_start = max(0.0, source_duration - usable_duration)
    source_start = trace.time_s[0] + maximum_start * start_fraction
    points = max(2, int(duration_s * control_rate_hz) + 1)

    envelope = []
    for index in range(points):
        output_time = index / control_rate_hz
        source_time = source_start + min(output_time, usable_duration)
        flow = abs(interpolate(trace.time_s, trace.mass_flow_kg_s, source_time))
        envelope.append(min(1.35, flow / max(normal_flow_kg_s, 1e-12)))
    return envelope


def envelope_value(envelope: list[float], sample_index: int, sample_rate: int, control_rate: int) -> float:
    """Interpolate the low-rate envelope at one audio sample."""
    position = sample_index * control_rate / sample_rate
    left = min(int(position), len(envelope) - 1)
    right = min(left + 1, len(envelope) - 1)
    fraction = position - left
    return envelope[left] + fraction * (envelope[right] - envelope[left])


def synthesize_flow_audio(
    trace: FlowTrace,
    normal_flow_kg_s: float,
    duration_s: float,
    sample_rate: int,
    rng: random.Random,
    sensor_gain: float,
    resonance_hz: float,
    start_fraction: float,
) -> list[float]:
    """Create a physically informed, but deliberately modest, acoustic proxy."""
    control_rate = 100
    envelope = make_envelope(
        trace,
        normal_flow_kg_s,
        duration_s,
        control_rate,
        start_fraction,
    )
    sample_count = int(duration_s * sample_rate)
    # Infer restriction severity from the simulated steady flow itself. The
    # metadata percentage is descriptive only and must not leak the expected
    # class into an unseen-condition prediction test.
    restriction = max(
        0.0,
        min(1.0, 1.0 - trace.steady_flow_kg_s / max(normal_flow_kg_s, 1e-12)),
    )
    phase_1 = rng.uniform(0.0, 2.0 * math.pi)
    phase_2 = rng.uniform(0.0, 2.0 * math.pi)
    resonance_2 = resonance_hz * rng.uniform(1.55, 2.05)
    low_noise = 0.0
    output: list[float] = []

    for index in range(sample_count):
        time_value = index / sample_rate
        flow_ratio = envelope_value(envelope, index, sample_rate, control_rate)

        # Flow produces broadband turbulence and excites tube/contact resonances.
        # Restriction changes both level and spectral roughness; stopped flow is
        # intentionally close to the sensor floor rather than mathematically zero.
        # Uniform noise is sufficient for this proxy and is substantially
        # faster than drawing millions of Gaussian samples in standard Python.
        white = rng.random() * 2.0 - 1.0
        low_noise = 0.985 * low_noise + 0.015 * white
        high_noise = white - low_noise
        flow_level = math.sqrt(max(flow_ratio, 0.0))
        turbulence_gain = flow_level * (0.18 + 0.32 * restriction)
        resonance_gain = flow_level * (0.055 + 0.035 * restriction)
        modulation = 1.0 + 0.10 * math.sin(2.0 * math.pi * 3.1 * time_value + phase_1)
        resonance = (
            math.sin(2.0 * math.pi * resonance_hz * time_value + phase_1)
            + 0.45 * math.sin(2.0 * math.pi * resonance_2 * time_value + phase_2)
        )
        sensor_floor = 0.006 * (rng.random() * 2.0 - 1.0)
        value = sensor_gain * (
            turbulence_gain * modulation * high_noise
            + resonance_gain * resonance
            + 0.018 * flow_level * low_noise
        ) + sensor_floor
        output.append(value)

    return output


def synthesize_noise_audio(
    duration_s: float,
    sample_rate: int,
    rng: random.Random,
    sensor_gain: float,
) -> list[float]:
    """Create an unrelated-noise control so disturbances are not called flow."""
    sample_count = int(duration_s * sample_rate)
    hum_hz = rng.choice((50.0, 60.0, 100.0, 120.0))
    hum_phase = rng.uniform(0.0, 2.0 * math.pi)
    event_centres = [
        rng.randrange(sample_rate // 2, max(sample_rate // 2 + 1, sample_count))
        for _ in range(rng.randint(2, 5))
    ]
    output = []
    for index in range(sample_count):
        time_value = index / sample_rate
        value = 0.035 * math.sin(2.0 * math.pi * hum_hz * time_value + hum_phase)
        value += 0.025 * (rng.random() * 2.0 - 1.0)
        for centre in event_centres:
            distance = abs(index - centre)
            if distance < 0.015 * sample_rate:
                value += 0.35 * math.exp(-distance / (0.003 * sample_rate)) * rng.choice((-1.0, 1.0))
        output.append(sensor_gain * value)
    return output


def write_pcm_wav(path: Path, samples: list[float], sample_rate: int) -> None:
    """Write mono 16-bit PCM without requiring NumPy or SciPy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = max((abs(value) for value in samples), default=1.0)
    limiter = max(1.0, peak / 0.92)
    frames = array.array("h")
    for value in samples:
        clipped = max(-1.0, min(1.0, value / limiter))
        frames.append(int(round(clipped * 32767.0)))
    if sys.byteorder != "little":
        frames.byteswap()

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames.tobytes())


def write_waveform_svg(path: Path, examples: dict[str, list[float]]) -> None:
    """Create a dependency-free visual check of one waveform per class."""
    width = 1100
    row_height = 170
    margin = 65
    height = margin + row_height * len(examples)
    colours = {
        "normal": "#188977",
        "reduced_flow": "#D08B24",
        "stopped_flow": "#697386",
        "noise": "#C4475D",
    }
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FAFBFC"/>',
        '<text x="35" y="38" font-family="Arial" font-size="23" font-weight="700" '
        'fill="#172B4D">EchoFlow simulated acoustic proxies</text>',
        '<text x="1065" y="38" text-anchor="end" font-family="Arial" font-size="13" '
        'fill="#5E6C84">NOT MEASURED SENSOR AUDIO</text>',
    ]
    for row_index, (label, samples) in enumerate(examples.items()):
        top = margin + row_index * row_height
        centre = top + row_height / 2
        plot_left = 210
        plot_right = width - 35
        elements.append(
            f'<text x="35" y="{centre + 6:.1f}" font-family="Arial" font-size="18" '
            f'fill="#172B4D">{label}</text>'
        )
        elements.append(
            f'<line x1="{plot_left}" y1="{centre:.1f}" x2="{plot_right}" '
            f'y2="{centre:.1f}" stroke="#DFE1E6" stroke-width="1"/>'
        )
        point_count = min(900, len(samples))
        step = max(1, len(samples) // point_count)
        selected = samples[::step][:point_count]
        peak = max((abs(value) for value in selected), default=1.0) or 1.0
        points = []
        for point_index, value in enumerate(selected):
            x_value = plot_left + (plot_right - plot_left) * point_index / max(1, len(selected) - 1)
            y_value = centre - (value / peak) * (row_height * 0.35)
            points.append(f"{x_value:.1f},{y_value:.1f}")
        elements.append(
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="{colours.get(label, "#176B87")}" stroke-width="1.2"/>'
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def water_mass_flow_to_ml_min(mass_flow_kg_s: float, density_kg_m3: float) -> float:
    """Convert mass flow to approximate water-equivalent volumetric flow."""
    return mass_flow_kg_s / density_kg_m3 * 1_000_000.0 * 60.0


def generate(args: argparse.Namespace) -> None:
    """Generate balanced proxy sessions and metadata for the existing monitor."""
    if args.csv_dir and args.input:
        raise ValueError("Use either --csv-dir or --input, not both")
    if not args.csv_dir and not args.input:
        raise ValueError("Provide --csv-dir or one or more --input values")
    if args.sessions < 2:
        raise ValueError("--sessions must be at least 2 for grouped evaluation")
    if args.clips_per_state < 1:
        raise ValueError("--clips-per-state must be at least 1")
    if args.duration <= 0 or args.sample_rate < 4_000:
        raise ValueError("Use a positive duration and a sample rate of at least 4000 Hz")

    if args.csv_dir:
        traces = discover_traces(Path(args.csv_dir).expanduser().resolve())
    else:
        traces = []
        for specification in args.input:
            label, path, restriction = parse_manual_input(specification)
            traces.append(read_flow_csv(path, label, restriction))

    normal_candidates = [
        trace.steady_flow_kg_s for trace in traces if trace.label == "normal"
    ]
    if not normal_candidates:
        raise ValueError("At least one input must use the label normal")
    normal_flow_kg_s = statistics.median(normal_candidates)
    if normal_flow_kg_s <= 0:
        raise ValueError("The normal trace has zero steady mass flow")

    project_dir = Path(args.project_dir).expanduser().resolve()
    audio_dir = project_dir / "audio"
    metadata_path = project_dir / "metadata.csv"
    manifest_path = project_dir / "SIMULATION_MANIFEST.json"
    plot_path = project_dir / "proxy_waveforms.svg"
    audio_dir.mkdir(parents=True, exist_ok=True)

    if metadata_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{metadata_path} already exists; use --overwrite to replace generated data"
        )

    metadata_rows: list[dict[str, str]] = []
    example_waveforms: dict[str, list[float]] = {}
    root_rng = random.Random(args.seed)

    for session_number in range(1, args.sessions + 1):
        session_id = f"sim_session_{session_number:02d}"
        # Shared session parameters emulate remounting the same sensor/tube setup.
        session_seed = root_rng.randrange(0, 2**31)
        session_rng = random.Random(session_seed)
        sensor_gain = session_rng.uniform(0.78, 1.18)
        resonance_hz = session_rng.uniform(380.0, 1_250.0)

        for trace_number, trace in enumerate(traces, start=1):
            for clip_number in range(1, args.clips_per_state + 1):
                clip_seed = (
                    args.seed
                    + session_number * 100_000
                    + trace_number * 1_000
                    + clip_number
                )
                clip_rng = random.Random(clip_seed)
                start_fraction = clip_rng.random()
                samples = synthesize_flow_audio(
                    trace=trace,
                    normal_flow_kg_s=normal_flow_kg_s,
                    duration_s=args.duration,
                    sample_rate=args.sample_rate,
                    rng=clip_rng,
                    sensor_gain=sensor_gain * clip_rng.uniform(0.92, 1.08),
                    resonance_hz=resonance_hz * clip_rng.uniform(0.94, 1.06),
                    start_fraction=start_fraction,
                )
                source_name = trace.path.stem.replace("_001", "")
                filename = (
                    f"{session_id}_{source_name}_{clip_number:02d}.wav"
                )
                output_path = audio_dir / filename
                write_pcm_wav(output_path, samples, args.sample_rate)
                example_waveforms.setdefault(trace.label, samples)

                metadata_rows.append(
                    {
                        "audio_path": f"audio/{filename}",
                        "flow_state": trace.label,
                        "session_id": session_id,
                        "setup_id": "simscape_acoustic_proxy_v1",
                        "restriction_percent": f"{trace.restriction_percent:.1f}",
                        "reference_flow_ml_min": (
                            f"{water_mass_flow_to_ml_min(trace.steady_flow_kg_s, args.density):.3f}"
                        ),
                        "notes": (
                            "SIMULATED ACOUSTIC PROXY; not measured sensor audio; "
                            f"source_csv={trace.path.name}; seed={clip_seed}"
                        ),
                    }
                )

        if args.include_noise:
            for clip_number in range(1, args.clips_per_state + 1):
                clip_seed = args.seed + session_number * 100_000 + 90_000 + clip_number
                samples = synthesize_noise_audio(
                    args.duration,
                    args.sample_rate,
                    random.Random(clip_seed),
                    sensor_gain,
                )
                filename = f"{session_id}_noise_{clip_number:02d}.wav"
                write_pcm_wav(audio_dir / filename, samples, args.sample_rate)
                example_waveforms.setdefault("noise", samples)
                metadata_rows.append(
                    {
                        "audio_path": f"audio/{filename}",
                        "flow_state": "noise",
                        "session_id": session_id,
                        "setup_id": "simscape_acoustic_proxy_v1",
                        "restriction_percent": "",
                        "reference_flow_ml_min": "",
                        "notes": (
                            "SIMULATED NON-FLOW NOISE CONTROL; not measured sensor audio; "
                            f"seed={clip_seed}"
                        ),
                    }
                )

    with metadata_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(metadata_rows)

    manifest = {
        "prototype_only": True,
        "warning": (
            "Generated WAV files are simulated acoustic proxies. They do not validate "
            "a contact sensor, dialysis tubing, catheter obstruction, or clinical use."
        ),
        "source_csv_files": [str(trace.path) for trace in traces],
        "project_dir": str(project_dir),
        "sample_rate_hz": args.sample_rate,
        "duration_s": args.duration,
        "sessions": args.sessions,
        "clips_per_state_per_session": args.clips_per_state,
        "seed": args.seed,
        "normal_reference_mass_flow_kg_s": normal_flow_kg_s,
        "normal_reference_water_equivalent_ml_min": water_mass_flow_to_ml_min(
            normal_flow_kg_s, args.density
        ),
        "metadata_rows": len(metadata_rows),
        "labels": sorted({row["flow_state"] for row in metadata_rows}),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_waveform_svg(plot_path, example_waveforms)

    print(json.dumps(manifest, indent=2))
    print(f"[SAVED] WAV files: {audio_dir}")
    print(f"[SAVED] Metadata:  {metadata_path}")
    print(f"[SAVED] Manifest:  {manifest_path}")
    print(f"[SAVED] Waveforms: {plot_path}")
    monitor_path = SCRIPT_DIR / "capd_flow_monitor.py"
    model_path = project_dir / "capd_simulated_model.joblib"
    print(
        "[NEXT] Train with:\n"
        f'       py -3 "{monitor_path}" train '
        f'--metadata "{metadata_path}" --model "{model_path}"'
    )


def readiness_check(_: argparse.Namespace) -> None:
    """Provide a green-tick-friendly entry point before arguments are supplied."""
    print("Simscape-to-acoustic proxy generator is ready.")
    print("No files were changed.")
    print(
        "Generate data with:\n"
        '  py -3 simscape_acoustic_proxy.py generate --csv-dir "PATH_TO_CSV_FOLDER"'
    )
    print(
        "Important: outputs are simulated acoustic proxies, not measured tubing audio."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(func=readiness_check)
    subparsers = parser.add_subparsers()

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate WAV proxies and CAPD-compatible metadata from flow CSVs.",
    )
    generate_parser.add_argument(
        "--csv-dir",
        help="Folder containing the four standard Simscape CSV filenames.",
    )
    generate_parser.add_argument(
        "--input",
        action="append",
        help="Manual input as LABEL=PERCENT=CSV_PATH; may be repeated.",
    )
    generate_parser.add_argument(
        "--project-dir",
        default=str(DEFAULT_PROJECT_DIR),
        help="Output dataset folder.",
    )
    generate_parser.add_argument("--sessions", type=int, default=4)
    generate_parser.add_argument("--clips-per-state", type=int, default=2)
    generate_parser.add_argument("--duration", type=float, default=3.0)
    generate_parser.add_argument("--sample-rate", type=int, default=16_000)
    generate_parser.add_argument("--density", type=float, default=997.0)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument(
        "--include-noise",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    generate_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing generated metadata file and same-name WAVs.",
    )
    generate_parser.set_defaults(func=generate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
