"""Fetch just Panda meshes at the exact revision expected by Playground."""
import argparse
from pathlib import Path
import subprocess

COMMIT = "1b86ece576591213e2b666ebf59508454200ca97"


def run(*args):
    return subprocess.run(["git", *map(str, args)], check=True, text=True, capture_output=True).stdout.strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", type=Path, default=Path(__file__).resolve().parents[1] / ".cache" / "menagerie")
    a = p.parse_args()
    path = a.destination.resolve()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading fixed Panda assets...", flush=True)
        run("clone", "--filter=blob:none", "--no-checkout", "--depth", "1", "https://github.com/google-deepmind/mujoco_menagerie.git", path)
        run("-C", path, "sparse-checkout", "set", "franka_emika_panda")
        run("-C", path, "fetch", "--depth", "1", "origin", COMMIT)
        run("-C", path, "checkout", COMMIT)
    if run("-C", path, "rev-parse", "HEAD") != COMMIT or not (path / "franka_emika_panda" / "LICENSE").is_file():
        raise RuntimeError(f"Existing asset directory is incomplete or has a different revision: {path}. Use a new --destination and PANDA_MENAGERIE_PATH.")
    print(f"Assets verified: {path} @ {COMMIT}")


if __name__ == "__main__":
    main()
