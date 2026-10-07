"""Evaluate a portable policy against a frozen scenario file on native MuJoCo."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from panda_rl.metrics import SuccessTracker, wilson_interval
from panda_rl.runtime import NativePolicy
from panda_rl.scenarios import load_scenarios


def evaluate(bundle, scenarios):
    policy = NativePolicy(bundle)
    rows = []
    inference_ms = []
    started = time.monotonic()
    for scenario in scenarios["scenarios"]:
        obs = policy.reset(scenario)
        tracker = SuccessTracker(dt=policy.meta["ctrl_dt"])
        distances, first_success, invalid = [], None, False
        max_height, closest_gripper = 0., float("inf")
        for step in range(policy.meta["episode_length"]):
            infer_start = time.perf_counter()
            action = policy.action(obs)
            inference_ms.append((time.perf_counter() - infer_start) * 1000)
            obs, distance, valid = policy.step(action)
            distances.append(distance)
            box = policy.data.xpos[policy.meta["object_body"]]
            if np.isfinite(box).all():
                max_height = max(max_height, float(box[2]))
                closest_gripper = min(closest_gripper, float(np.linalg.norm(box - policy.data.site_xpos[policy.meta["gripper_site"]])))
            if tracker.update(distance, valid) and first_success is None:
                first_success = (step + 1) * policy.meta["ctrl_dt"]
            if not valid:
                invalid = True
                break
        rows.append({"scenario": scenario["id"], "success": bool(tracker.success and not invalid),
                     "invalid": invalid, "first_success_seconds": first_success,
                     "max_object_height_m": max_height,
                     "min_gripper_distance_m": closest_gripper if np.isfinite(closest_gripper) else None,
                     "ever_lifted_5cm": max_height >= policy.meta["init_object_pos"][2] + .05,
                     "final_distance_m": distances[-1] if np.isfinite(distances[-1]) else None,
                     "min_distance_m": float(np.min(distances)) if np.isfinite(distances).all() else None})
    count = len(rows)
    successes = sum(r["success"] for r in rows)
    final_distances = [r["final_distance_m"] for r in rows if r["final_distance_m"] is not None]
    return {"backend": "native-mujoco+onnx", "profile": scenarios["profile"],
            "scenario_sha256": hashlib.sha256(json.dumps(scenarios, sort_keys=True).encode()).hexdigest(),
            "policy_sha256": policy.meta["sha256"]["policy.onnx"], "training": policy.meta["training"],
            "criterion": "distance <= 0.05 m for 15 consecutive 0.02 s steps; invalid episode fails",
            "episodes": count, "successes": successes, "success_rate": successes / count,
            "success_rate_wilson_95": wilson_interval(successes, count),
            "invalid_rate": sum(r["invalid"] for r in rows) / count,
            "ever_lifted_5cm_rate": sum(r["ever_lifted_5cm"] for r in rows) / count,
            "inference_ms_p50": float(np.percentile(inference_ms, 50)),
            "inference_ms_p95": float(np.percentile(inference_ms, 95)),
            "mean_final_distance_m": float(np.mean(final_distances)) if final_distances else None,
            "wall_seconds": time.monotonic() - started, "episodes_detail": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--scenarios", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = evaluate(a.bundle, load_scenarios(a.scenarios))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    with a.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["episodes_detail"][0]))
        writer.writeheader()
        writer.writerows(report["episodes_detail"])
    print(json.dumps({k: v for k, v in report.items() if k not in {"episodes_detail", "training"}}, indent=2))


if __name__ == "__main__":
    main()
