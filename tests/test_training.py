"""Local checks only: these tests never call PPO training."""
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from panda_rl.train import gpu_environment, supervise
from panda_rl.launch import build_commands


class TrainingControlTests(unittest.TestCase):
    def test_gpu_selection_uses_physical_id_without_nested_remap(self):
        env = gpu_environment("1", 0.8, {"CUDA_VISIBLE_DEVICES": "0,1"})
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(env["JAX_PLATFORMS"], "cuda")
        with self.assertRaises(ValueError):
            gpu_environment("0", 0.8, {"CUDA_VISIBLE_DEVICES": "1"})

    def test_import_does_not_initialize_jax(self):
        result = subprocess.run([sys.executable, "-c", "import panda_rl.train, panda_rl.launch, sys; assert 'jax' not in sys.modules"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_dry_run_validates_without_creating_run_or_importing_jax(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = Path(directory) / "config.json"
            cfg.write_text('{"profile":"combined"}', encoding="utf-8")
            output = Path(directory) / "run"
            command = [sys.executable, "-c", "from panda_rl.train import main; import sys; main(); assert 'jax' not in sys.modules",
                       "--config", str(cfg), "--output", str(output), "--gpu", "0", "--seed", "7", "--dry-run"]
            env = dict(os.environ)
            env.pop("CUDA_VISIBLE_DEVICES", None)
            result = subprocess.run(command, capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["config"]["seed"], 7)
            self.assertFalse(output.exists())

    def test_two_gpu_commands_keep_matching_outputs_and_seed(self):
        commands = build_commands(["configs/baseline.json", "configs/combined.json"], ["0", "1"], 4, Path("runs"))
        self.assertEqual(len(commands), 2)
        for command, gpu, name in zip(commands, ["0", "1"], ["baseline", "combined"]):
            self.assertEqual(command[command.index("--gpu") + 1], gpu)
            self.assertEqual(command[command.index("--seed") + 1], "4")
            self.assertEqual(Path(command[command.index("--output") + 1]).name, name + "-s4")
        with self.assertRaises(ValueError):
            build_commands(["a.json", "b.json"], ["0", "0"], 0, Path("runs"))

    def test_timeout_includes_prior_initialization_time(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            start = time.monotonic() - 1
            code = supervise([sys.executable, "-c", "import time; time.sleep(60)"], output, 0.1, start)
            self.assertEqual(code, 124)
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["status"], "timed_out")
            self.assertGreaterEqual(status["wall_seconds"], 1)

    def test_success_and_failure_have_distinct_status(self):
        for returncode, expected in [(0, "completed"), (3, "failed")]:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                result = supervise([sys.executable, "-c", f"raise SystemExit({returncode})"], output, 10, time.monotonic())
                self.assertEqual(result, returncode)
                self.assertEqual(json.loads((output / "status.json").read_text())["status"], expected)

    def test_running_worker_is_killed_at_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            started = time.monotonic()
            result = supervise([sys.executable, "-c", "import time; time.sleep(60)"], Path(directory), .2, started)
            self.assertEqual(result, 124)
            self.assertLess(time.monotonic() - started, 5)


@unittest.skipUnless(importlib.util.find_spec("jax") and importlib.util.find_spec("mujoco_playground")
                    and (Path(os.environ.get("PANDA_MENAGERIE_PATH", Path(__file__).resolve().parents[1] / ".cache" / "menagerie")) / "franka_emika_panda" / "LICENSE").exists(),
                    "requires optional training dependencies and Panda assets; no training is performed")
class DisturbanceTests(unittest.TestCase):
    def test_progress_reward_does_not_pay_for_waiting_or_repeating_old_peak(self):
        from panda_rl.envs import progress_reward, invalid_physics
        import jax.numpy as jp
        import numpy as np
        peak = jp.array(0.)
        rewards = []
        for i, value in enumerate([2., 2., 3., 2., 3., 4.]):
            reward, peak = progress_reward(jp.array(value), peak, i == 0)
            rewards.append(float(reward))
        np.testing.assert_array_equal(rewards, [0, 0, 1, 0, 0, 1])
        self.assertFalse(bool(invalid_physics(jp.zeros(16), jp.ones(15), jp.zeros(66))))
        self.assertTrue(bool(invalid_physics(jp.zeros(16), jp.ones(15) * 101, jp.zeros(66))))
        self.assertTrue(bool(invalid_physics(jp.zeros(16), jp.zeros(15), jp.full(66, float('nan')))))

    def test_real_randomized_batched_wrapper_traces_with_full_reset(self):
        import jax
        import jax.numpy as jp
        from functools import partial
        from panda_rl.config import ExperimentConfig
        from panda_rl.envs import load_environment, make_randomizer, wrap_environment
        cfg = ExperimentConfig(profile="combined", reward_mode="progress_hold")
        cfg.validate()
        env = load_environment(cfg)
        keys = jax.random.split(jax.random.PRNGKey(11), 2)
        randomizer = partial(make_randomizer(env, cfg), rng=keys)
        wrapped = wrap_environment(env, episode_length=150, action_repeat=1, randomization_fn=randomizer)
        state = wrapped.reset(keys)
        traced = jax.eval_shape(wrapped.step, state, jp.zeros((2, 8)))
        self.assertEqual(traced.obs.shape, (2, env.observation_size))
        self.assertEqual(traced.info["action_fifo"].shape, (2, cfg.delay_steps, 8))

    def test_real_panda_reset_sensor_reproducibility_and_step_trace(self):
        import jax
        import jax.numpy as jp
        import numpy as np
        from panda_rl.config import ExperimentConfig
        from panda_rl.envs import load_environment
        from mujoco_playground._src.manipulation.franka_emika_panda import pick

        cfg = ExperimentConfig(profile="sensor", delay_steps=2)
        nominal = load_environment(cfg, evaluation=True)
        reference_cfg = pick.default_config()
        reference_cfg.impl = "jax"
        reference = pick.PandaPickCube(config=reference_cfg)
        key = jax.random.PRNGKey(9)
        nominal_state, reference_state = nominal.reset(key), reference.reset(key)
        np.testing.assert_array_equal(nominal_state.obs, reference_state.obs)
        np.testing.assert_array_equal(nominal_state.data.qpos, reference_state.data.qpos)
        # Upstream intentionally has no forward pass at reset: preserve its obs.
        np.testing.assert_array_equal(nominal_state.data.site_xpos, np.zeros_like(nominal_state.data.site_xpos))
        sensor = load_environment(cfg)
        state = sensor.reset(key)
        np.testing.assert_array_equal(state.obs, sensor.reset(key).obs)
        self.assertFalse(np.array_equal(state.obs, nominal_state.obs))
        np.testing.assert_array_equal(state.info["action_fifo"], np.zeros((2, 8)))
        traced = jax.eval_shape(sensor.step, state, jp.zeros(8))
        self.assertEqual(traced.obs.shape, state.obs.shape)
        self.assertEqual(traced.info["action_fifo"].shape, (2, 8))

    def test_evaluation_wrapper_disables_randomization_and_resets_info(self):
        import jax
        import jax.numpy as jp
        import numpy as np
        from panda_rl.envs import wrap_environment
        from mujoco_playground._src.mjx_env import State

        class MinimalEnv:
            evaluation = True
            @property
            def unwrapped(self):
                return self
            def reset(self, key):
                return State(jp.zeros(1), jp.zeros(1), jp.array(0.), jp.array(0.), {},
                             {"action_fifo": jp.zeros((1, 2)), "reached_box": jp.array(0.)})
            def step(self, state, action):
                return state.replace(done=jp.array(1.), info={**state.info,
                    "action_fifo": jp.ones((1, 2)), "reached_box": jp.array(1.)})

        def forbidden_randomizer(*args, **kwargs):
            raise AssertionError("Nominal evaluation must not invoke randomization")
        env = wrap_environment(MinimalEnv(), episode_length=150, action_repeat=1,
                               randomization_fn=forbidden_randomizer)
        state = env.reset(jax.random.split(jax.random.PRNGKey(0), 2))
        state = env.step(state, jp.zeros((2, 2)))
        np.testing.assert_array_equal(state.info["action_fifo"], np.zeros((2, 1, 2)))
        np.testing.assert_array_equal(state.info["reached_box"], np.zeros(2))

    def test_fifo_delay_and_zero_delay(self):
        import numpy as np
        import jax.numpy as jp
        from panda_rl.envs import delay_action
        queue = jp.zeros((2, 3))
        for value, expected in [(1, 0), (2, 0), (3, 1), (4, 2)]:
            action, queue = delay_action(jp.full((3,), value), queue)
            np.testing.assert_array_equal(action, np.full(3, expected))
        action, queue = delay_action(jp.ones(3), jp.zeros((0, 3)))
        np.testing.assert_array_equal(action, np.ones(3))
        self.assertEqual(queue.shape, (0, 3))

    def test_mass_inertia_and_contact_friction_randomize_together(self):
        from types import SimpleNamespace
        import jax
        import mujoco
        from mujoco import mjx
        import numpy as np
        from panda_rl.envs import make_randomizer
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom name="floor" type="plane" size="1 1 .1"/><body pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1" friction="1 .03 .003"/></body></worldbody></mujoco>')
        mjx_model = mjx.put_model(model, impl="jax")
        cfg = SimpleNamespace(profile="physics", mass_range=(.8, 1.2), friction_range=(.8, 1.2))
        randomize = make_randomizer(SimpleNamespace(mj_model=model), cfg)
        randomized, axes = randomize(mjx_model, jax.random.split(jax.random.PRNGKey(3), 8))
        factors = np.asarray(randomized.body_mass[:, 1] / mjx_model.body_mass[1])
        np.testing.assert_allclose(randomized.body_inertia[:, 1, 0] / mjx_model.body_inertia[1, 0], factors)
        self.assertGreater(float(np.std(factors)), 0)
        self.assertGreater(float(np.std(randomized.geom_friction[:, 0, 0])), 0)
        np.testing.assert_allclose(randomized.geom_friction[:, :, 1:], np.broadcast_to(mjx_model.geom_friction[:, 1:], (8, 2, 2)))
        self.assertEqual(axes.body_mass, 0)
        self.assertEqual(axes.body_inertia, 0)
        self.assertEqual(axes.geom_friction, 0)

    def test_official_checkpoint_roundtrips_initialized_network_without_updates(self):
        import jax
        import jax.numpy as jp
        import numpy as np
        from functools import partial
        from brax.training.acme import running_statistics, specs
        from brax.training.agents.ppo import checkpoint, networks
        factory = partial(networks.make_ppo_networks, policy_hidden_layer_sizes=(32, 32, 32, 32),
                          value_hidden_layer_sizes=(256, 256, 256, 256, 256))
        network = factory(4, 2)
        params = (running_statistics.init_state(specs.Array((4,), jp.float32)),
                  network.policy_network.init(jax.random.PRNGKey(1)),
                  network.value_network.init(jax.random.PRNGKey(2)))
        from panda_rl.train import portable_checkpoint_config
        config = portable_checkpoint_config(4, 2, factory)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint.save(directory, 0, params, config)
            path = Path(directory) / "000000000000"
            restored = checkpoint.load(path)
            for expected, actual in zip(jax.tree.leaves(params), jax.tree.leaves(restored)):
                np.testing.assert_array_equal(expected, actual)
            from panda_rl.export import load_checkpoint_cpu
            host_restored = load_checkpoint_cpu(path)
            for expected, actual in zip(jax.tree.leaves(params), jax.tree.leaves(host_restored)):
                np.testing.assert_array_equal(expected, actual)
            self.assertEqual(checkpoint.load_config(path).observation_size, 4)
            restored_policy = checkpoint.load_policy(path)
            action, _ = restored_policy(jp.zeros(4), jax.random.PRNGKey(0))
            self.assertTrue(np.isfinite(action).all())
            self.assertEqual(action.shape, (2,))


if __name__ == "__main__":
    unittest.main()
