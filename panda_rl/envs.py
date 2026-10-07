"""Official Panda task with optional disturbances and explicit progress shaping."""
import os
from pathlib import Path
import jax
import jax.numpy as jp
import numpy as np


def wrap_environment(env, **kwargs):
    from mujoco_playground import wrapper
    # Brax forwards the training randomizer to eval_env as well.
    if getattr(env, "evaluation", False):
        kwargs["randomization_fn"] = None
    return wrapper.wrap_for_brax_training(env, full_reset=True, **kwargs)


def delay_action(action, queue):
    """FIFO delay; reset queue consists of zero actions, in normalized units."""
    if queue.shape[0] == 0:
        return action, queue
    return queue[0], jp.concatenate((queue[1:], action[None]), axis=0)


def progress_reward(potential, previous_best, first_step):
    return jp.where(first_step, 0., jp.maximum(potential - previous_best, 0.)), jp.maximum(potential, previous_best)


def invalid_physics(qpos, qvel, obs):
    # 100 rad/s (joints) or m/s (free body) is well beyond this tabletop task.
    # Finite-but-exploded velocities otherwise poison the running normalizer.
    return (~jp.all(jp.isfinite(qpos)) | ~jp.all(jp.isfinite(obs)) |
            ~jp.all(jp.isfinite(qvel)) | jp.any(jp.abs(qvel) > 100.))


def load_environment(cfg, evaluation=False):
    from mujoco_playground._src import mjx_env
    from mujoco_playground._src.manipulation.franka_emika_panda import pick
    assets = Path(os.environ.get("PANDA_MENAGERIE_PATH", Path(__file__).resolve().parents[1] / ".cache" / "menagerie"))
    if not (assets / "franka_emika_panda" / "LICENSE").is_file():
        raise FileNotFoundError(f"Panda assets missing at {assets}; run python scripts/fetch_assets.py or set PANDA_MENAGERIE_PATH")
    mjx_env.MENAGERIE_PATH = assets

    class TrainingPanda(pick.PandaPickCube):
        def __init__(self):
            config = pick.default_config()
            config.impl = "jax"
            self.evaluation = evaluation
            self.sensor_enabled = not evaluation and cfg.profile in ("sensor", "combined")
            super().__init__(config=config)

        def _get_obs(self, data, info):
            obs = super()._get_obs(data, info)
            if self.sensor_enabled:
                info["rng"], noise_key = jax.random.split(info["rng"])
                obs = obs + cfg.obs_noise_std * jax.random.normal(noise_key, obs.shape)
            return obs

        def reset(self, rng):
            state = super().reset(rng)
            state.metrics["numerical_failure"] = jp.array(0.)
            if cfg.reward_mode in {"progress", "progress_hold"}:
                state.info.update(best_potential=jp.array(0.), first_step=jp.array(True))
                state.metrics.update(closure=jp.array(0.), lifted=jp.array(0.))
            if self.sensor_enabled:
                state.info["action_fifo"] = jp.zeros((cfg.delay_steps, self.action_size))
            return state

        def step(self, state, action):
            if self.sensor_enabled:
                action, queue = delay_action(action, state.info["action_fifo"])
                state = state.replace(info={**state.info, "action_fifo": queue})
            state = super().step(state, action)
            if cfg.reward_mode in {"progress", "progress_hold"}:
                box = state.data.xpos[self._obj_body]
                gap = jp.linalg.norm(box - state.data.site_xpos[self._gripper_site])
                near = 1. - jp.tanh(20. * gap)
                # Closure is only a shaping proxy, never the reported success.
                closure = near * jp.clip((.04 - state.data.qpos[self._robot_qposadr[7]]) / .02, 0., 1.)
                lifted = near * jp.clip((box[2] - .03) / .2, 0., 1.)
                potential = state.reward + 2. * closure + 4. * lifted
                reward, peak = progress_reward(potential, state.info["best_potential"], state.info["first_step"])
                if cfg.reward_mode == "progress_hold":
                    reward += .2 * (jp.linalg.norm(box - state.info["target_pos"]) <= .05)
                state.info.update(best_potential=peak, first_step=jp.array(False))
                state.metrics.update(closure=closure, lifted=lifted)
                state = state.replace(reward=reward)
            invalid = invalid_physics(state.data.qpos, state.data.qvel, state.obs) | ~jp.isfinite(state.reward)
            state.metrics["numerical_failure"] = invalid.astype(float)
            metrics = {k: jp.nan_to_num(v, nan=0., posinf=0., neginf=0.) for k, v in state.metrics.items()}
            return state.replace(metrics=metrics, done=jp.maximum(state.done, invalid.astype(float)),
                                 reward=jp.where(invalid, 0., state.reward),
                                 obs=jp.where(invalid, jp.zeros_like(state.obs), state.obs))

    return TrainingPanda()


def make_randomizer(env, cfg):
    """One uniform mass and one friction multiplier per parallel environment.

    Mass and inertia share a multiplier for every positive-mass body. Sliding
    friction shares a multiplier for all contact-enabled geoms. These samples
    are fixed for the run by Brax's domain-randomization wrapper, not per reset.
    """
    if cfg.profile not in ("physics", "combined"):
        return None
    body_mask = jp.asarray(np.asarray(env.mj_model.body_mass) > 0)
    contact_mask = jp.asarray((np.asarray(env.mj_model.geom_contype) != 0) |
                              (np.asarray(env.mj_model.geom_conaffinity) != 0))

    def randomize(model, rng):
        def sample(key):
            mass_key, friction_key = jax.random.split(key)
            mass = jax.random.uniform(mass_key, (), minval=cfg.mass_range[0], maxval=cfg.mass_range[1])
            friction = jax.random.uniform(friction_key, (), minval=cfg.friction_range[0], maxval=cfg.friction_range[1])
            masses = model.body_mass * jp.where(body_mask, mass, 1)
            inertia = model.body_inertia * jp.where(body_mask, mass, 1)[:, None]
            frictions = model.geom_friction.at[:, 0].multiply(jp.where(contact_mask, friction, 1))
            return masses, inertia, frictions

        masses, inertia, frictions = jax.vmap(sample)(rng)
        axes = jax.tree.map(lambda _: None, model).tree_replace(
            {"body_mass": 0, "body_inertia": 0, "geom_friction": 0})
        return model.tree_replace({"body_mass": masses, "body_inertia": inertia,
                                   "geom_friction": frictions}), axes

    return randomize
