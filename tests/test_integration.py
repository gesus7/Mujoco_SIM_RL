"""Real scene and randomly initialized policy checks; no optimizer is run."""
import json
import os
from pathlib import Path
import numpy as np
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    pytest.importorskip("jax")
    pytest.importorskip("mujoco_playground")
    root = Path(__file__).resolve().parents[1]
    if not (Path(os.environ.get("PANDA_MENAGERIE_PATH", root / ".cache" / "menagerie")) / "franka_emika_panda" / "LICENSE").is_file():
        pytest.skip("run scripts/fetch_assets.py for real-scene integration tests")
    import jax
    import jax.numpy as jp
    from brax.training.agents.ppo import networks
    from brax.training.acme import running_statistics
    from panda_rl.config import ExperimentConfig
    from panda_rl.envs import load_environment
    from panda_rl.export import save_bundle
    env = load_environment(ExperimentConfig(), evaluation=True)
    net = networks.make_ppo_networks(env.observation_size, env.action_size,
                                    preprocess_observations_fn=running_statistics.normalize)
    params = (running_statistics.init_state(jp.zeros(env.observation_size)),
              net.policy_network.init(jax.random.PRNGKey(123)),
              net.value_network.init(jax.random.PRNGKey(456)))
    path = tmp_path_factory.mktemp("random-policy") / "bundle"
    metadata = save_bundle(path, env, params, networks.make_inference_fn(net),
                           training={"status": "UNTRAINED_TEST_FIXTURE", "env_steps": 0})
    return path, env, metadata


def test_real_brax_export_and_native_reset_observation_match(exported):
    import jax
    from panda_rl.runtime import NativePolicy
    path, env, meta = exported
    assert meta["observation_size"] == 66 and meta["action_size"] == 8
    assert meta["onnx_max_abs_error"] < 2e-5
    state = env.reset(jax.random.PRNGKey(77))
    scenario = {"object_offset": (np.asarray(state.data.qpos[env._obj_qposadr:env._obj_qposadr + 3]) - np.asarray(env._init_obj_pos)).tolist(),
                "target_offset": (np.asarray(state.info["target_pos"]) - np.asarray(env._init_obj_pos)).tolist(),
                "mass_scale": 1.0, "friction_scale": 1.0, "obs_noise_std": 0.0, "delay_steps": 0, "noise_seed": 0}
    policy = NativePolicy(path)
    native_obs = policy.reset(scenario)
    np.testing.assert_allclose(native_obs, state.obs, atol=2e-6, rtol=0)
    # Render must not change the observation or simulation state.
    policy.render_data()
    np.testing.assert_array_equal(policy.observation(), native_obs)


def test_native_finite_rollout_fixed_scenarios_and_tamper_detection(exported, tmp_path):
    from panda_rl.evaluate import evaluate
    from panda_rl.scenarios import generate_scenarios
    from panda_rl.runtime import NativePolicy
    path, _, _ = exported
    report = evaluate(path, generate_scenarios(3, profile="combined"))
    assert report["episodes"] == 3
    assert report["invalid_rate"] == 0
    assert np.isfinite(report["mean_final_distance_m"])
    json.dumps(report, allow_nan=False)
    import shutil
    damaged = tmp_path / "bad-bundle"
    shutil.copytree(path, damaged)
    with (damaged / "policy.onnx").open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        NativePolicy(damaged)


def test_native_and_mjx_observation_layout_from_same_physical_state(exported):
    import jax.numpy as jp
    import mujoco
    from mujoco import mjx
    from panda_rl.runtime import NativePolicy
    from panda_rl.scenarios import generate_scenarios
    path, env, _ = exported
    native = NativePolicy(path)
    native.reset(generate_scenarios(1)["scenarios"][0])
    native.step(np.zeros(8))
    # Copy the identical state, isolating observation construction from physics drift.
    data = mjx.put_data(env.mj_model, native.data)
    expected = env._get_obs(data, {"target_pos": jp.asarray(native.target)})
    np.testing.assert_allclose(native.observation(), expected, atol=2e-6, rtol=0)
