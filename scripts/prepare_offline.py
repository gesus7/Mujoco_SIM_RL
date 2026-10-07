"""Verify and prepare the specific source wheel/assets used on the deployed server."""
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    provenance = json.loads((root / "scripts" / "source-provenance.json").read_text())
    wheel = root / "playground-0.2.0-py3-none-any.whl"
    assets = root / "panda-assets.zip"
    for file, key in ((wheel, "playground_wheel_sha256"), (assets, "panda_assets_zip_sha256")):
        if hashlib.sha256(file.read_bytes()).hexdigest() != provenance[key]:
            raise ValueError(f"Unexpected offline artifact: {file}")
    destination = root / ".cache" / "menagerie"
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(assets) as archive:
        for member in archive.namelist():
            target = (destination / member).resolve()
            if not target.is_relative_to(destination.resolve()) or not member.startswith("franka_emika_panda/"):
                raise ValueError(f"Unexpected archive entry: {member}")
        archive.extractall(destination)
    text = (root / "requirements-server.txt").read_text().replace(
        "-r requirements-local.txt", "-r " + (root / "requirements-local.txt").as_posix())
    text = "\n".join(wheel.as_posix() if line.startswith("playground @") else line for line in text.splitlines()) + "\n"
    (root / "reports").mkdir(exist_ok=True)
    (root / "reports" / "requirements-resolved.txt").write_text(text)
    print("Fixed-revision wheel and Panda assets verified")


if __name__ == "__main__":
    main()
