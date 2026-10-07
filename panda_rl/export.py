"""Export the pinned Brax swish MLP to portable deterministic ONNX."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from panda_rl import PLAYGROUND_COMMIT, MENAGERIE_COMMIT


def cpu_session(path):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def write_onnx(path, weights, mean, std, action_size):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    mean, std = np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)
    if mean.ndim != 1 or std.shape != mean.shape or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("invalid observation normalization")
    nodes, initializers = [], []

    def const(name, value):
        initializers.append(numpy_helper.from_array(np.asarray(value), name))
        return name

    nodes.append(helper.make_node("Sub", ["observation", const("mean", mean)], ["centered"]))
    nodes.append(helper.make_node("Div", ["centered", const("std", std)], ["normalized"]))
    previous = "normalized"
    for i, (kernel, bias) in enumerate(weights):
        kernel, bias = np.asarray(kernel, dtype=np.float32), np.asarray(bias, dtype=np.float32)
        if not np.isfinite(kernel).all() or not np.isfinite(bias).all():
            raise ValueError("nonfinite policy parameters")
        out = f"linear_{i}"
        nodes.append(helper.make_node("Gemm", [previous, const(f"w{i}", kernel), const(f"b{i}", bias)], [out]))
        if i < len(weights) - 1:
            nodes.append(helper.make_node("Sigmoid", [out], [f"sigmoid_{i}"]))
            nodes.append(helper.make_node("Mul", [out, f"sigmoid_{i}"], [f"swish_{i}"]))
            previous = f"swish_{i}"
        else:
            previous = out
    if weights[-1][1].shape != (2 * action_size,):
        raise ValueError("expected tanh-normal policy with mean and scale outputs")
    nodes.append(helper.make_node("Slice", [previous, const("starts", np.array([0], dtype=np.int64)),
                 const("ends", np.array([action_size], dtype=np.int64)), const("axes", np.array([1], dtype=np.int64))], ["action_mean"]))
    nodes.append(helper.make_node("Tanh", ["action_mean"], ["action"]))
    graph = helper.make_graph(nodes, "panda_deterministic_policy",
        [helper.make_tensor_value_info("observation", TensorProto.FLOAT, [None, len(mean)])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [None, action_size])], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="panda_rl")
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def validate_actions(reference, actual):
    reference, actual = np.asarray(reference), np.asarray(actual)
    if reference.shape != actual.shape or reference.ndim != 2:
        raise ValueError("Mismatched ONNX action shapes")
    error = np.abs(reference - actual)
    if error.shape[0] != 65 or not np.isfinite(error).all():
        raise ValueError("Invalid ONNX validation outputs")
    errors = {"real_reset": float(error[:1].max()), "unscaled_random": float(error[1:33].max()),
              "mean_std_scaled_random": float(error[33:].max())}
    # Raw random coordinates can be hundreds of standard deviations outside the
    # scene manifold. GPU/CPU float32 arithmetic amplifies error there; 1e-4 in
    # normalized action corresponds to at most 4e-6 in a control increment.
    for name, value in errors.items():
        tolerance = 1e-4 if name == "unscaled_random" else 2e-5
        if value > tolerance:
            raise ValueError(f"ONNX {name} error {value} exceeds {tolerance}")
    return errors


def save_bundle(output, env, params, make_policy, *, training):
    """Publish only after ONNX agrees with Brax on synthetic and real observations."""
    import jax
    import mujoco
    from mujoco_playground._src import mjx_env
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable bundle: {output}")
    staging = output.with_name(output.name + ".partial")
    staging.mkdir(parents=True, exist_ok=False)
    (staging / "assets").mkdir()
    normalizer, policy = params[:2]
    layers = policy["params"]
    expected = [f"hidden_{i}" for i in range(len(layers))]
    if set(layers) != set(expected):
        raise ValueError("unsupported policy architecture; expected a plain swish MLP")
    weights = [(np.asarray(layers[k]["kernel"]), np.asarray(layers[k]["bias"])) for k in expected]
    mean, std = np.asarray(normalizer.mean), np.asarray(normalizer.std)
    write_onnx(staging / "policy.onnx", weights, mean, std, env.action_size)
    state = env.reset(jax.random.PRNGKey(98765))
    real_obs = np.asarray(state.obs, dtype=np.float32).reshape(1, -1)
    rng = np.random.default_rng(4321)
    probes = np.concatenate([real_obs, rng.normal(size=(32, mean.size)).astype(np.float32),
                             (mean + rng.normal(size=(32, mean.size)) * std).astype(np.float32)])
    action_jax = np.asarray(make_policy(params, deterministic=True)(probes, jax.random.PRNGKey(0))[0])
    action_onnx = cpu_session(staging / "policy.onnx").run(None, {"observation": probes})[0]
    errors = validate_actions(action_jax, action_onnx)
    max_error = max(errors.values())
    (staging / "scene.xml").write_bytes(Path(env.xml_path).read_bytes())
    for name, content in env.model_assets.items():
        if Path(name).name != name:
            raise ValueError(f"unexpected nested asset name: {name}")
        (staging / "assets" / name).write_bytes(content)
    # Preserve the licenses accompanying the upstream robot meshes and scene.
    licenses = staging / "licenses"
    licenses.mkdir()
    robot_license = Path(mjx_env.MENAGERIE_PATH) / "franka_emika_panda" / "LICENSE"
    if not robot_license.exists():
        raise FileNotFoundError("Robot asset LICENSE is required in portable bundles")
    shutil.copyfile(robot_license, licenses / "menagerie-franka-LICENSE")
    shutil.copyfile(Path(__file__).resolve().parents[1] / "LICENSE", licenses / "project-LICENSE")
    shutil.copyfile(Path(__file__).resolve().parents[1] / "NOTICE.md", staging / "NOTICE.md")
    model = env.mj_model
    metadata = {"schema_version": 1, "task": "PandaPickCube", "playground_commit": PLAYGROUND_COMMIT,
        "menagerie_commit": MENAGERIE_COMMIT, "mujoco_version": mujoco.__version__,
        "observation_size": int(mean.size), "action_size": int(env.action_size),
        "ctrl_dt": float(env.dt), "sim_dt": float(model.opt.timestep),
        "substeps": int(env.n_substeps), "action_scale": float(env._action_scale), "episode_length": 150,
        "init_qpos": np.asarray(env._init_q).tolist(), "init_ctrl": np.asarray(env._init_ctrl).tolist(),
        "init_object_pos": np.asarray(env._init_obj_pos).tolist(),
        "robot_qposadr": np.asarray(env._robot_qposadr).tolist(), "object_qposadr": int(env._obj_qposadr),
        "object_body": int(env._obj_body), "gripper_site": int(env._gripper_site),
        "target_mocap": int(np.asarray(env._mocap_target).item()), "onnx_max_abs_error": max_error,
        "onnx_error_by_probe": errors,
        "onnx_tolerance_by_probe": {"real_reset": 2e-5, "mean_std_scaled_random": 2e-5, "unscaled_random": 1e-4},
        "training": training, "normalization": "(observation-mean)/std; no clipping",
        "inference": "deterministic tanh-normal mode; swish hidden layers"}
    metadata["sha256"] = {p.relative_to(staging).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(staging.rglob("*")) if p.is_file()}
    (staging / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    staging.rename(output)
    return metadata


def load_checkpoint_cpu(path):
    """Restore arrays on the host, regardless of the GPU used when saving."""
    import jax
    from orbax import checkpoint as ocp
    from brax.training import types
    from brax.training.acme import running_statistics
    path = Path(path).resolve()
    with ocp.PyTreeCheckpointer() as reader:
        metadata = reader.metadata(path).item_metadata
        restore_args = jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), metadata)
        # Brax 0.14.2 computes these arguments but does not pass them to restore.
        target = reader.restore(path, args=ocp.args.PyTreeRestore(restore_args=restore_args))
    state = target[0]
    if isinstance(state["count"], dict) and "hi" in state["count"]:
        state["count"] = types.UInt64(**state["count"])
    target[0] = running_statistics.RunningStatisticsState(**state)
    return target


def main():
    """Recover a portable bundle from a completed Brax inference checkpoint."""
    import argparse
    import os
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from brax.training import checkpoint as common_checkpoint
    from brax.training.agents.ppo import checkpoint, networks
    from panda_rl.config import ExperimentConfig
    from panda_rl.envs import load_environment
    cfg = ExperimentConfig.load(args.config)
    network_config = checkpoint.load_config(args.checkpoint.resolve())
    network = common_checkpoint.get_network(network_config, networks.make_ppo_networks)
    params = load_checkpoint_cpu(args.checkpoint)
    result = save_bundle(args.output, load_environment(cfg, evaluation=True), params,
                         networks.make_inference_fn(network),
                         training={"config": cfg.to_dict(), "env_steps": int(args.checkpoint.name),
                                   "recovered_from_checkpoint": str(args.checkpoint), "resume_supported": False})
    print(json.dumps({"bundle": str(args.output), "onnx_max_abs_error": result["onnx_max_abs_error"]}))


if __name__ == "__main__":
    main()
