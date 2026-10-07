"""Launch independent experiments, one physical GPU per worker."""
import argparse
from pathlib import Path
import subprocess
import sys


def build_commands(configs, gpus, seed, output):
    if len(configs) != len(gpus) or not configs or len(set(gpus)) != len(gpus):
        raise ValueError("Provide one config per distinct physical GPU")
    names = [Path(config).stem for config in configs]
    if len(set(names)) != len(names):
        raise ValueError("Config file stems must be distinct to avoid output collisions")
    return [[sys.executable, "-m", "panda_rl.train", "--config", str(config),
             "--gpu", str(gpu), "--seed", str(seed), "--output", str(Path(output) / f"{name}-s{seed}")]
            for config, gpu, name in zip(configs, gpus, names)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    commands = build_commands(args.configs, args.gpus, args.seed, args.output)
    if args.dry_run:
        commands = [command + ["--dry-run"] for command in commands]
    processes = []
    try:
        for command in commands:
            processes.append(subprocess.Popen(command))
        codes = [process.wait() for process in processes]
        return 0 if all(code == 0 for code in codes) else 1
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
