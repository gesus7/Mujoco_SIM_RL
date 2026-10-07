"""Native MuJoCo + ONNX inference; no JAX or training dependency."""
from collections import deque
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco
from panda_rl.export import cpu_session


class NativePolicy:
    def __init__(self, bundle):
        root = Path(bundle)
        self.meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if self.meta.get("schema_version") != 1 or self.meta.get("task") != "PandaPickCube":
            raise ValueError("unsupported policy bundle")
        if self.meta["mujoco_version"] != mujoco.__version__:
            raise ValueError(f"Install mujoco=={self.meta['mujoco_version']} to match the bundle")
        for name, digest in self.meta["sha256"].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"bundle checksum mismatch: {name}")
        assets = {p.name: p.read_bytes() for p in (root / "assets").iterdir() if p.is_file()}
        self.model = mujoco.MjModel.from_xml_string((root / "scene.xml").read_text(encoding="utf-8"), assets)
        self.model.opt.timestep = self.meta["sim_dt"]
        self.data = mujoco.MjData(self.model)
        self.session = cpu_session(root / "policy.onnx")
        self.base_mass = self.model.body_mass.copy()
        self.base_inertia = self.model.body_inertia.copy()
        self.base_friction = self.model.geom_friction.copy()
        self.body_mask = self.base_mass > 0
        self.contact_mask = (self.model.geom_contype != 0) | (self.model.geom_conaffinity != 0)

    def reset(self, scenario):
        m, d, meta = self.model, self.data, self.meta
        self.model.body_mass[:] = self.base_mass * np.where(self.body_mask, scenario["mass_scale"], 1)
        self.model.body_inertia[:] = self.base_inertia * np.where(self.body_mask, scenario["mass_scale"], 1)[:, None]
        self.model.geom_friction[:] = self.base_friction
        self.model.geom_friction[:, 0] *= np.where(self.contact_mask, scenario["friction_scale"], 1)
        mujoco.mj_setConst(m, d)
        mujoco.mj_resetData(m, d)
        d.qpos[:] = meta["init_qpos"]
        d.ctrl[:] = meta["init_ctrl"]
        obj = meta["object_qposadr"]
        d.qpos[obj:obj + 3] = np.asarray(meta["init_object_pos"]) + scenario["object_offset"]
        self.target = np.asarray(meta["init_object_pos"]) + scenario["target_offset"]
        d.mocap_pos[meta["target_mocap"]] = self.target
        d.mocap_quat[meta["target_mocap"]] = [1, 0, 0, 0]
        # Upstream reset uses mjx.make_data without forward; keep derived fields
        # zero until the first physics step. Rendering uses a separate data copy.
        self.noise = scenario["obs_noise_std"]
        self.rng = np.random.default_rng(scenario["noise_seed"])
        self.delay = deque([np.zeros(meta["action_size"]) for _ in range(scenario["delay_steps"])])
        return self.observation()

    def observation(self):
        d, meta = self.data, self.meta
        gripper = d.site_xpos[meta["gripper_site"]]
        box = d.xpos[meta["object_body"]]
        box_matrix = d.xmat[meta["object_body"]].ravel()
        obs = np.concatenate([d.qpos, d.qvel, gripper,
            d.site_xmat[meta["gripper_site"]].ravel()[3:], box_matrix[3:],
            box - gripper, self.target - box, np.eye(3).ravel()[:6] - box_matrix[:6],
            d.ctrl - d.qpos[np.asarray(meta["robot_qposadr"][:-1])]])
        if self.noise:
            obs = obs + self.rng.normal(0, self.noise, obs.shape)
        return obs.astype(np.float32)

    def action(self, observation):
        action = self.session.run(None, {"observation": np.asarray(observation, dtype=np.float32)[None]})[0][0]
        if action.shape != (self.meta["action_size"],) or not np.isfinite(action).all():
            raise ValueError("nonfinite or invalid policy action")
        return action

    def step(self, action):
        action = np.clip(action, -1, 1)
        if self.delay:
            self.delay.append(action)
            action = self.delay.popleft()
        self.data.ctrl[:] = np.clip(self.data.ctrl + action * self.meta["action_scale"],
                                   self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
        for _ in range(self.meta["substeps"]):
            mujoco.mj_step(self.model, self.data)
        # mj_step leaves derived poses at the same integration stage as MJX.
        box = self.data.xpos[self.meta["object_body"]]
        valid = bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
                     and np.all(np.abs(self.data.qvel) <= 100.)
                     and np.all(np.abs(box) <= 1) and box[2] >= 0)
        distance = float(np.linalg.norm(box - self.target))
        return self.observation(), distance, valid

    def render_data(self):
        d = mujoco.MjData(self.model)
        d.qpos[:] = self.data.qpos
        d.qvel[:] = self.data.qvel
        d.ctrl[:] = self.data.ctrl
        d.mocap_pos[:] = self.data.mocap_pos
        d.mocap_quat[:] = self.data.mocap_quat
        mujoco.mj_forward(self.model, d)
        return d
