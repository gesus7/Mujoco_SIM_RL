# MuJoCo Panda RL

**English** | [简体中文](README.zh-CN.md)

A reinforcement learning project for Franka Emika Panda manipulation, built on Google DeepMind's **MuJoCo Playground** and **Brax PPO**. The task is to grasp a cube and hold it near a target above the table.

The repository provides GPU training, controlled perturbations, fixed-scenario evaluation, and ONNX export for local CPU inference. Policies use simulator state; vision, language conditioning, object release, and real-robot deployment are outside the current scope.

## Features

- **Parallel simulation:** MuJoCo MJX environments with PPO training on one GPU per experiment.
- **Controlled experiments:** upstream and shaped rewards; mass, friction, observation noise, and action-delay perturbations.
- **Bounded runs:** wall-clock supervision, periodic checkpoints, and complete model bundles.
- **Portable inference:** native MuJoCo and ONNX Runtime, without JAX or CUDA on the demo machine.
- **Repeatable evaluation:** seeded scenarios, success-rate confidence intervals, invalid-episode counts, and inference latency.

## Requirements

| Component | Environment |
| --- | --- |
| Training | Linux, Python 3.12 recommended, NVIDIA GPU, driver compatible with the pinned JAX CUDA 12 dependencies |
| Reference hardware | Ubuntu 24.04; RTX 2080 Ti with 11 GB VRAM per worker |
| Local evaluation | Python, CPU, dependencies in [requirements-local.txt](requirements-local.txt) |
| Rendering | An OpenGL-capable environment for the interactive viewer or video export |

Two GPUs run independent experiments; their memory is not pooled. The default run budget is **7.5 hours**, including initialization, compilation, training, evaluation, and export. This is a time limit, not a convergence guarantee.

## Quick start

### 1. Install on the training server

Git, an NVIDIA driver, and Python with the `venv` module must already be available. Run commands from the repository root.

```bash
git clone https://github.com/gesus7/Mujoco_SIM_RL.git
cd Mujoco_SIM_RL
bash scripts/setup_server.sh
source .venv-server/bin/activate
python -m panda_rl.doctor --gpu 0 --physics --output reports/doctor-gpu0.json
```

The setup script installs pinned dependencies and downloads the Panda assets from MuJoCo Menagerie. Installation requires network access. Source revisions are recorded in [source-provenance.json](scripts/source-provenance.json).

### 2. Train

Start with the shaped reward that encourages grasping, lifting, and remaining near the target:

```bash
python -m panda_rl.train --config configs/hold.json --gpu 0 --seed 0 --output runs/hold-s0
```

Alternatively, compare nominal and randomized training on two GPUs:

```bash
python -m panda_rl.launch --configs configs/hold.json configs/hold-combined.json --gpus 0 1 --seed 0 --output runs
```

Choose one command; run directories must not already exist. Add `--dry-run` to validate configuration without training or checking GPU availability.

| Configuration | Reward | Training perturbations |
| --- | --- | --- |
| [hold.json](configs/hold.json) | Progress and target occupancy | None |
| [hold-combined.json](configs/hold-combined.json) | Progress and target occupancy | Physics, observation noise, action delay |
| [progress.json](configs/progress.json) | Progress only | None |
| [baseline.json](configs/baseline.json) | Upstream reward | None |

The first two configurations request 40 million environment steps with 2,048 parallel environments. Other configurations support ablations; their training budgets and batch settings differ. See [config.py](panda_rl/config.py) for defaults and validation.

### 3. Transfer a model bundle

A completed run writes its final bundle to `runs/hold-s0/bundle/`. Copy **the entire directory** to `bundles/hold-s0/` in a local checkout of this repository. It includes the ONNX policy, scene assets, metadata, and licenses. With the two-GPU command, the second bundle is written to `runs/hold-combined-s0/bundle/`.

Within each run directory, `latest_bundle.json` identifies the latest complete periodic bundle, if one was produced, including after interruption or timeout. Its bundle path is relative to that run directory. The same directory contains `status.json` with the run status. The repository does **not** include pretrained weights.

### 4. Evaluate and demonstrate locally

Create a separate local environment. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-local.txt
```

On Linux, activate it with `source .venv/bin/activate`. Then generate scenarios, evaluate the transferred policy, and open the viewer:

```bash
python -m panda_rl.scenarios --count 200 --seed 10000 --profile nominal --output reports/nominal.json
python -m panda_rl.evaluate --bundle bundles/hold-s0 --scenarios reports/nominal.json --output reports/hold-nominal.json
python -m panda_rl.demo --bundle bundles/hold-s0 --scenarios reports/nominal.json --episode 0
```

Append `--video reports/demo.mp4` to the demo command to save a video instead. Evaluation does not require rendering. To test robustness, generate another scenario file with `--profile combined`; use the same seed and count to preserve the initial object and target positions across profiles.

## Evaluation protocol and limitations

Each episode lasts up to **150 control steps (3 seconds)**. Success requires the cube to remain within **5 cm** of the target for **15 consecutive steps (0.3 seconds)**; an invalid episode fails. The cube need not remain at the target until the final step. Evaluation writes JSON and per-episode CSV reports.

Training reward and evaluation success are separate quantities. Physics multipliers are sampled per parallel environment and remain fixed during a training run. Checkpoints omit optimizer and random-number-generator state, so exact training resume is not supported. Simulation performance does not establish real-world transfer.

## Source map

| File | Responsibility |
| --- | --- |
| [envs.py](panda_rl/envs.py) | Task extensions, rewards, perturbations, and numerical termination |
| [train.py](panda_rl/train.py), [launch.py](panda_rl/launch.py) | PPO training, run supervision, and independent GPU workers |
| [export.py](panda_rl/export.py), [runtime.py](panda_rl/runtime.py) | ONNX bundle export and native MuJoCo inference |
| [scenarios.py](panda_rl/scenarios.py), [metrics.py](panda_rl/metrics.py) | Fixed scenarios and success criterion |
| [evaluate.py](panda_rl/evaluate.py), [demo.py](panda_rl/demo.py) | Batch evaluation, viewer, and video |

## Tests

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
```

Tests cover configuration, training supervision, metrics, and export/runtime consistency. Integration tests additionally require the training dependencies and downloaded assets; they skip when those prerequisites are absent. Tests do not run PPO optimization.

## License and acknowledgments

Project code is licensed under [Apache-2.0](LICENSE). The task and training stack build on [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground), [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), and [Brax](https://github.com/google/brax). See [NOTICE.md](NOTICE.md) for attribution and pinned upstream revisions. Retain the accompanying licenses when redistributing model bundles.
