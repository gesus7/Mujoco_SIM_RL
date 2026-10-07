"""Show a policy or record an MP4. This entry point cannot train."""
import argparse
from pathlib import Path
import time
from panda_rl.runtime import NativePolicy
from panda_rl.scenarios import load_scenarios, generate_scenarios
from panda_rl.metrics import SuccessTracker


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--scenarios", type=Path)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--video", type=Path, help="Write MP4 instead of opening a window")
    a = p.parse_args()
    scenarios = load_scenarios(a.scenarios) if a.scenarios else generate_scenarios(1)
    policy = NativePolicy(a.bundle)
    tracker = SuccessTracker(dt=policy.meta["ctrl_dt"])
    obs = policy.reset(scenarios["scenarios"][a.episode])
    print(f"Training metadata: {policy.meta['training']}")
    if a.video:
        import imageio.v2 as imageio
        import mujoco
        import numpy as np
        from PIL import Image, ImageDraw
        a.video.parent.mkdir(parents=True, exist_ok=True)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.45, 0, 0.3]
        camera.distance, camera.azimuth, camera.elevation = 1.8, 135, -25
        with mujoco.Renderer(policy.model, height=480, width=640) as renderer, imageio.get_writer(str(a.video), fps=50) as video:
            for step in range(policy.meta["episode_length"]):
                obs, distance, valid = policy.step(policy.action(obs))
                tracker.update(distance, valid)
                renderer.update_scene(policy.render_data(), camera=camera)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 640, 45), fill=(20, 25, 32))
                draw.text((10, 6), f"Panda PPO | {a.bundle.name} | {scenarios['profile']} | scene {a.episode}", fill="white")
                held = "YES" if tracker.success and valid else "NO"
                draw.text((10, 25), f"{(step+1)*policy.meta['ctrl_dt']:.2f}s | distance {distance:.3f}m | held goal: {held} (<=5cm, 0.3s)", fill="white")
                video.append_data(np.asarray(frame))
                if not valid:
                    break
        print(a.video)
    else:
        import mujoco.viewer
        display = policy.render_data()
        with mujoco.viewer.launch_passive(policy.model, display) as viewer:
            for _ in range(policy.meta["episode_length"]):
                if not viewer.is_running():
                    break
                started = time.monotonic()
                obs, distance, valid = policy.step(policy.action(obs))
                tracker.update(distance, valid)
                display.qpos[:] = policy.data.qpos
                display.qvel[:] = policy.data.qvel
                mujoco.mj_forward(policy.model, display)
                viewer.sync()
                time.sleep(max(0, policy.meta["ctrl_dt"] - (time.monotonic() - started)))
                if not valid:
                    break
    print(f"Held-goal success: {tracker.success and not tracker.invalid}")


if __name__ == "__main__":
    main()
