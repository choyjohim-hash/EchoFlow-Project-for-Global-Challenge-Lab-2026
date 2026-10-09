# EchoFlow — Global Challenge Lab 2026

Digital proof of concept connecting MATLAB/Simulink (Simscape) flow simulations to Python signal processing and random forest classification.

## Workflow

Simscape mass-flow CSV -> synthetic acoustic proxy WAV -> librosa/SciPy features -> random forest flow-state classification.

The WAV signals are synthetic proxies, not measured tubing acoustics. Results demonstrate the software workflow; they do not validate a contact microphone or clinical performance.

## Files

- `echoflow_combined_pipeline.py`: main command-line entry point.
- `simscape_acoustic_proxy.py`: converts simulated flow traces to proxy audio and metadata.
- `capd_flow_monitor.py`: feature extraction, model training, evaluation and prediction.
- `Matlab_simulation.slx`: Simulink model.
- `Matlab_code.mlx`: MATLAB live script.
- Four labelled flow CSVs: normal, restricted, severely restricted and stopped flow.
- `unseen_restriction_001.csv`: additional simulation trace, excluded from default four-file discovery.
- Technical summary and pitch slides: project explanation.

## Setup

Use Python 3.10 or later. In this folder:

```shell
python -m venv .venv
```

Activate the environment (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

Check imports and dependencies:

```shell
python echoflow_combined_pipeline.py
```

Generate proxy recordings from the four standard CSVs and train:

```shell
python echoflow_combined_pipeline.py generate-train --csv-dir .
```

Outputs default to `capd_simulated_data/`. Keep recordings from a simulation/session grouped when evaluating to avoid treating correlated windows as independent experiments.

Additional commands:

```shell
python echoflow_combined_pipeline.py --help
python echoflow_combined_pipeline.py features --audio recording.wav
python echoflow_combined_pipeline.py predict --audio unseen.wav
```

MATLAB/Simulink and the model's Simscape dependencies are required to rerun the physics simulation. The supplied CSVs allow the Python workflow to run independently.
