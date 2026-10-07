"""Fixed native MuJoCo evaluation scenarios, shared across policies."""
import argparse
import json
from pathlib import Path
import numpy as np

PROFILES = {"nominal": (1.0, 1.0, 0.0, 0), "physics": (0.8, 1.2, 0.0, 0),
            "sensor": (1.0, 1.0, 0.01, 1), "combined": (0.8, 1.2, 0.01, 1),
            "stress": (0.7, 1.3, 0.02, 2),"delay1": (1.0, 1.0, 0.0, 1),
            "delay2": (1.0, 1.0, 0.0, 2),"delay4": (1.0, 1.0, 0.0, 4)}


def generate_scenarios(count=200, seed=10000, profile="nominal"):
    if count <= 0 or seed < 0 or profile not in PROFILES:
        raise ValueError("invalid scenario configuration")
    low, high, noise, delay = PROFILES[profile]
    # Separate streams keep initial conditions identical across disturbance profiles.
    geometry = np.random.default_rng(seed)
    physics = np.random.default_rng(seed + 1)
    rows = []
    for i in range(count):
        rows.append({"id": i, "object_offset": geometry.uniform([-.2, -.2, 0], [.2, .2, 0]).tolist(),
                     "target_offset": geometry.uniform([-.2, -.2, .2], [.2, .2, .4]).tolist(),
                     "mass_scale": float(physics.uniform(low, high)),
                     "friction_scale": float(physics.uniform(low, high)),
                     "obs_noise_std": noise, "delay_steps": delay, "noise_seed": seed + 100000 + i})
    return {"schema_version": 1, "seed": seed, "profile": profile, "scenarios": rows}


def load_scenarios(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema_version") != 1 or not doc.get("scenarios"):
        raise ValueError("invalid scenario file")
    for row in doc["scenarios"]:
        for key in ("object_offset", "target_offset"):
            a = np.asarray(row[key])
            if a.shape != (3,) or not np.isfinite(a).all():
                raise ValueError(f"invalid {key}")
        for key in ("mass_scale", "friction_scale"):
            if not np.isfinite(row[key]) or row[key] <= 0:
                raise ValueError(f"invalid {key}")
        if type(row["delay_steps"]) is not int or row["delay_steps"] < 0 or not np.isfinite(row["obs_noise_std"]) or row["obs_noise_std"] < 0:
            raise ValueError("invalid sensor disturbance")
    return doc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=200)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--profile", choices=PROFILES, default="nominal")
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(generate_scenarios(a.count, a.seed, a.profile), indent=2), encoding="utf-8")
    print(a.output)


if __name__ == "__main__":
    main()
