"""GPU-only Brax PPO entry point with an external wall-clock supervisor.

No resume is exposed: Brax inference checkpoints omit optimizer and RNG state.
All heavyweight imports occur in the GPU-masked child, never in the launcher.
"""
import argparse
from dataclasses import replace
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def gpu_environment(gpu, memory_fraction, inherited=None):
    env = dict(os.environ if inherited is None else inherited)
    gpu = str(gpu)
    if not gpu or "," in gpu or not (gpu.isdecimal() or gpu.startswith("GPU-")):
        raise ValueError("--gpu must be one physical GPU index or GPU UUID")
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and gpu not in visible.split(","):
        raise ValueError(f"Physical GPU {gpu} is not in inherited CUDA_VISIBLE_DEVICES={visible}; use an allowed ID")
    if not 0 < memory_fraction < 1:
        raise ValueError("memory_fraction must be between 0 and 1")
    env.update(CUDA_VISIBLE_DEVICES=gpu, JAX_PLATFORMS="cuda",
               XLA_PYTHON_CLIENT_MEM_FRACTION=str(memory_fraction),
               XLA_PYTHON_CLIENT_PREALLOCATE="false")
    env.setdefault("OMP_NUM_THREADS", "6")
    return env


def supervise(command, output, budget_seconds, started, env=None):
    """Budget includes all work since started; last complete bundles survive."""
    output = Path(output)
    deadline = started + budget_seconds
    process = None
    status, returncode = "failed", 1
    old_handlers = {}

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    try:
        for name in (signal.SIGTERM, signal.SIGINT):
            old_handlers[name] = signal.signal(name, interrupt)
        if time.monotonic() >= deadline:
            status, returncode = "timed_out", 124
        else:
            process = subprocess.Popen(command, env=env, start_new_session=(os.name != "nt"))
            try:
                returncode = process.wait(timeout=max(0, deadline - time.monotonic()))
                status = "completed" if returncode == 0 else "failed"
            except subprocess.TimeoutExpired:
                status, returncode = "timed_out", 124
    except KeyboardInterrupt:
        status, returncode = "interrupted", 130
    finally:
        if process is not None and process.poll() is None:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
        for name, handler in old_handlers.items():
            signal.signal(name, handler)
        write_json(output / "status.json", {
            "status": status, "returncode": returncode,
            "wall_seconds": time.monotonic() - started,
            "budget_seconds": budget_seconds,
            "budget_scope": "initialization+training+evaluation+checkpoint+export",
            "resume_supported": False,
            "worker_pid": process.pid if process else None,
        })
    return returncode


def process_gpu_memory_mb():
    """Read this worker's memory only; unavailable values remain null."""
    try:
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                                 "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=2, check=True)
        samples = [int(line.split(",")[1].strip()) for line in result.stdout.splitlines()
                   if line.split(",")[0].strip() == str(os.getpid())]
        return sum(samples) if samples else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def portable_checkpoint_config(observation_size, action_size, network_factory):
    from brax.training.agents.ppo import checkpoint
    config = checkpoint.network_config(observation_size, action_size, True, network_factory)
    # Brax 0.14.2 saves null initializers but load_config tries to look up None.
    # Omitting these default-valued keys preserves network semantics on reload.
    for key in list(config.network_factory_kwargs):
        if key.endswith("kernel_init_fn") and config.network_factory_kwargs[key] is None:
            del config.network_factory_kwargs[key]
    return config


def run_worker(cfg, output, started):
    from functools import partial
    import jax
    from brax.training.agents.ppo import checkpoint, networks, train as ppo
    from panda_rl.envs import load_environment, make_randomizer, wrap_environment
    from panda_rl.export import save_bundle

    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError(f"Require exactly one CUDA GPU; CPU fallback forbidden: {devices}")
    versions = {}
    for package in ("jax", "jaxlib", "mujoco", "mujoco-mjx", "playground", "brax", "flax", "orbax-checkpoint", "numpy", "onnx", "onnxruntime"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    write_json(output / "versions.json", {"python": sys.version, "platform": platform.platform(),
               "packages": versions, "device": str(devices[0]), "device_kind": devices[0].device_kind,
               "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
               "expected_playground_commit": "86bfcd999619ad8d058383feefc1dd16d4cc57a7",
               "playground_install_origin": importlib.metadata.distribution("playground").read_text("direct_url.json")})
    environment = load_environment(cfg)
    eval_environment = load_environment(cfg, evaluation=True)
    network_factory = partial(networks.make_ppo_networks,
                              policy_hidden_layer_sizes=(32, 32, 32, 32),
                              value_hidden_layer_sizes=(256, 256, 256, 256, 256))
    checkpoint_config = portable_checkpoint_config(environment.observation_size,
                                                   environment.action_size, network_factory)
    last_step = 0
    last_metrics = {}

    def progress(step, metrics):
        nonlocal last_step, last_metrics
        last_step = int(step)
        last_metrics = {str(key): float(value) for key, value in metrics.items()}
        # JSON null explicitly records non-finite diagnostics without invalid JSON.
        import math
        clean = {key: value if math.isfinite(value) else None for key, value in last_metrics.items()}
        row = {"env_steps": last_step, "wall_seconds": time.monotonic() - started,
               "worker_gpu_memory_mb": process_gpu_memory_mb(), "metrics": clean}
        with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)

    def training_metadata(step):
        return {"config": cfg.to_dict(), "env_steps": int(step),
                "wall_seconds": time.monotonic() - started,
                "budget_seconds": cfg.max_wall_hours * 3600,
                "algorithm": "brax.ppo", "resume_supported": False,
                "checkpoint_contents": "normalizer, policy, value; no optimizer or RNG",
                "physics_sampling": "per-environment fixed for run; one shared mass/inertia multiplier and one shared contact sliding-friction multiplier",
                "observation_noise": "independent Gaussian on every raw actor/critic observation component",
                "action_delay": "normalized-action FIFO, zero-filled at episode reset"}

    def snapshot(step, make_policy, params):
        nonlocal last_step
        last_step = int(step)
        # Save the official Brax inference checkpoint before the slower export.
        checkpoint.save(str((output / "checkpoints").resolve()), int(step), params, checkpoint_config)
        write_json(output / "latest_checkpoint.json", {
            "path": f"checkpoints/{int(step):012d}", "env_steps": int(step),
            "resume_supported": False})
        destination = output / "bundles" / f"step-{step:012d}"
        save_bundle(destination, eval_environment, params, make_policy,
                    training=training_metadata(step))
        write_json(destination / "COMPLETE.json", {"env_steps": int(step)})
        write_json(output / "latest_bundle.json", {"path": str(destination.relative_to(output)), "env_steps": int(step)})

    make_policy, params, _ = ppo.train(
        environment=environment, eval_env=eval_environment,
        num_timesteps=cfg.num_timesteps, num_envs=cfg.num_envs,
        num_eval_envs=cfg.num_eval_envs, num_evals=cfg.num_evals,
        batch_size=cfg.batch_size, num_minibatches=cfg.num_minibatches,
        unroll_length=cfg.unroll_length, num_updates_per_batch=cfg.num_updates_per_batch,
        episode_length=150, action_repeat=1, num_resets_per_eval=1,
        learning_rate=1e-3, entropy_cost=2e-2, discounting=.97,
        max_grad_norm=cfg.max_grad_norm,
        normalize_observations=True, network_factory=network_factory,
        max_devices_per_host=1, seed=cfg.seed, deterministic_eval=True,
        wrap_env_fn=wrap_environment, randomization_fn=make_randomizer(environment, cfg),
        progress_fn=progress, policy_params_fn=snapshot)
    save_bundle(output / "bundle", eval_environment, params, make_policy,
                training=training_metadata(last_step))
    write_json(output / "bundle" / "COMPLETE.json", {"env_steps": last_step})


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpu", required=True, help="Physical GPU index or UUID (not an index into CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--started", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    from panda_rl.config import ExperimentConfig
    cfg = ExperimentConfig.load(args.config)
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    cfg.validate()
    env = gpu_environment(args.gpu, cfg.memory_fraction)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "config": cfg.to_dict(), "output": str(args.output),
                          "physical_gpu": args.gpu, "budget_seconds": cfg.max_wall_hours * 3600,
                          "gpu_runtime_verified": False}, indent=2))
        return 0
    if platform.system() != "Linux":
        raise RuntimeError("Training is restricted to the Ubuntu server; use --dry-run locally")
    if args.worker:
        os.environ.update(env)
        run_worker(cfg, args.output, args.started if args.started is not None else started)
        return 0
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", cfg.to_dict())
    write_json(output / "status.json", {"status": "running", "started_unix": time.time(),
                                        "budget_seconds": cfg.max_wall_hours * 3600})
    command = [sys.executable, "-m", "panda_rl.train", "--config", str(output / "config.json"),
               "--gpu", args.gpu, "--output", str(output), "--worker", "--started", str(started)]
    return supervise(command, output, cfg.max_wall_hours * 3600, started, env)


if __name__ == "__main__":
    raise SystemExit(main())
