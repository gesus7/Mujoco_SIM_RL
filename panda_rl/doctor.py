"""Server GPU preflight; small inference/physics probes, never PPO updates."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--physics", action="store_true", help="Compile and step one MJX Panda environment")
    a = p.parse_args()
    from panda_rl.train import gpu_environment
    os.environ.update(gpu_environment(a.gpu, 0.8))
    import jax
    import jax.numpy as jp
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError(f"One CUDA GPU required: {devices}")
    start = time.monotonic()
    jax.jit(lambda x: x @ x)(jp.ones((64, 64))).block_until_ready()
    result = {"gpu": str(devices[0]), "physical_gpu": str(a.gpu), "kind": devices[0].device_kind,
              "platform": platform.platform(), "python": platform.python_version(),
              "gpu_probe_seconds": time.monotonic() - start,
              "versions": {k: importlib.metadata.version(k) for k in ("jax", "jaxlib", "mujoco", "mujoco-mjx", "brax", "playground")},
              "nvidia_smi": subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True).stdout}
    if a.physics:
        from panda_rl.config import ExperimentConfig
        from panda_rl.envs import load_environment
        env = load_environment(ExperimentConfig())
        start = time.monotonic()
        state = jax.jit(env.reset)(jax.random.PRNGKey(0))
        state = jax.jit(env.step)(state, jp.zeros(env.action_size))
        state.obs.block_until_ready()
        if not bool(jp.isfinite(state.obs).all()):
            raise RuntimeError("nonfinite MJX state")
        result.update(mjx_single_step_seconds=time.monotonic()-start, observation_size=env.observation_size,
                      action_size=env.action_size)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
