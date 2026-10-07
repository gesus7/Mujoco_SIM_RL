import json
import math

import numpy as np
import pytest


def test_configuration_rejects_unsafe_or_incompatible_budget(tmp_path):
    from panda_rl.config import ExperimentConfig

    ExperimentConfig().validate()
    for change in ({"max_wall_hours": 8.1}, {"max_wall_hours": math.nan},
                   {"num_envs": 300}, {"mass_range": [1.2, 0.8]},
                   {"delay_steps": -1}, {"seed": -1}):
        with pytest.raises(ValueError):
            ExperimentConfig(**change).validate()
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"num_env": 128}), encoding="utf-8")
    with pytest.raises((ValueError, TypeError)):
        ExperimentConfig.load(path)


def test_success_requires_contiguous_hold_and_no_invalid_state():
    from panda_rl.metrics import SuccessTracker

    tracker = SuccessTracker(dt=0.02, threshold=0.05, hold_seconds=0.3)
    for _ in range(14):
        assert not tracker.update(0.02)
    assert not tracker.update(0.2)
    for _ in range(14):
        assert not tracker.update(0.02)
    assert tracker.update(0.02)
    broken = SuccessTracker(dt=0.02)
    assert not broken.update(float("nan"))
    assert broken.invalid


def test_onnx_has_normalization_swish_and_deterministic_tanh(tmp_path):
    import onnxruntime as ort
    from panda_rl.export import write_onnx, cpu_session

    rng = np.random.default_rng(7)
    weights = [(rng.normal(size=(5, 7)).astype("f"), rng.normal(size=7).astype("f")),
               (rng.normal(size=(7, 4)).astype("f"), rng.normal(size=4).astype("f"))]
    mean = np.arange(5, dtype="f")
    std = np.arange(1, 6, dtype="f")
    x = rng.normal(size=(16, 5)).astype("f")
    path = tmp_path / "policy.onnx"
    write_onnx(path, weights, mean, std, action_size=2)
    y = (x - mean) / std
    y = y @ weights[0][0] + weights[0][1]
    y = y / (1 + np.exp(-y))
    y = np.tanh((y @ weights[1][0] + weights[1][1])[:, :2])
    actual = cpu_session(path).run(None, {"observation": x})[0]
    np.testing.assert_allclose(actual, y, rtol=2e-5, atol=2e-6)


def test_scenario_file_is_reproducible_and_contains_fixed_initial_conditions(tmp_path):
    from panda_rl.scenarios import generate_scenarios, load_scenarios

    a = generate_scenarios(count=8, seed=10000, profile="nominal")
    assert a == generate_scenarios(count=8, seed=10000, profile="nominal")
    assert a != generate_scenarios(count=8, seed=10001, profile="nominal")
    path = tmp_path / "scenes.json"
    path.write_text(json.dumps(a), encoding="utf-8")
    assert load_scenarios(path) == a
    assert all(len(s["object_offset"]) == len(s["target_offset"]) == 3 for s in a["scenarios"])


def test_export_tolerance_distinguishes_extreme_probes_from_normal_inputs():
    from panda_rl.export import validate_actions
    expected = np.zeros((65, 8), dtype=np.float32)
    actual = expected.copy()
    actual[14, 7] = 2.5e-5  # Observed GPU/CPU FP32 difference on ~493-sigma probe.
    assert validate_actions(expected, actual)["unscaled_random"] < 1e-4
    actual[0, 0] = 2.5e-5
    with pytest.raises(ValueError):
        validate_actions(expected, actual)
    with pytest.raises(ValueError):
        validate_actions(expected, np.zeros((65, 1)))
    actual[0, 0] = 0
    actual[14, 7] = float("nan")
    with pytest.raises(ValueError):
        validate_actions(expected, actual)
