"""Plot actual logged reward and throughput; never synthesize missing points."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for run in a.runs:
        rows = [json.loads(line) for line in (run / "progress.jsonl").read_text().splitlines() if line.strip()]
        rows = [r for r in rows if r["metrics"].get("eval/episode_reward") is not None]
        if not rows:
            continue
        axes[0].plot([r["env_steps"] / 1e6 for r in rows], [r["metrics"]["eval/episode_reward"] for r in rows], marker="o", label=run.name)
        axes[1].plot([r["wall_seconds"] / 60 for r in rows], [r["env_steps"] / 1e6 for r in rows], marker="o", label=run.name)
    axes[0].set(xlabel="Environment steps (million)", ylabel="Nominal evaluation shaped return")
    axes[1].set(xlabel="Wall time (minutes, including compilation)", ylabel="Environment steps (million)")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
