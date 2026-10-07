# MuJoCo Panda RL

[English](README.md) | **简体中文**

基于 Google DeepMind **MuJoCo Playground** 与 **Brax PPO** 的 Franka Emika Panda 机械臂强化学习项目。任务是抓取方块，并使其在桌面上方的目标附近保持一段时间。

仓库提供 GPU 训练、可控扰动、固定场景评测及 ONNX 导出，支持本地 CPU 推理。策略使用仿真状态；当前不涉及视觉输入、语言条件、松爪放置或真机部署。

## 功能

- **并行仿真**：使用 MuJoCo MJX 与 PPO，每个实验使用一张 GPU。
- **对照实验**：提供上游奖励与塑形奖励，支持质量、摩擦、观测噪声及动作延迟扰动。
- **训练预算**：限制运行时长，定期保存检查点与完整模型包。
- **独立推理**：通过原生 MuJoCo 与 ONNX Runtime 运行，本地演示无需 JAX 或 CUDA。
- **可复现评测**：固定随机场景，输出成功率置信区间、无效回合统计及推理延迟。

## 环境要求

| 用途 | 环境 |
| --- | --- |
| 训练 | Linux，建议 Python 3.12，NVIDIA GPU，兼容仓库固定版本 JAX CUDA 12 依赖的驱动 |
| 参考硬件 | Ubuntu 24.04；每个训练进程使用一张 RTX 2080 Ti 11 GB |
| 本地评测 | Python、CPU 及 [requirements-local.txt](requirements-local.txt) 中的依赖 |
| 渲染 | 交互窗口和视频导出需要可用的 OpenGL 环境 |

双卡用于并行运行独立实验，显存不合并。默认单次运行预算为 **7.5 小时**，包括初始化、编译、训练、评测和导出；该预算是时间上限，不保证策略在限时内收敛。

## 快速开始

### 1. 在训练服务器上安装

预先安装 Git、NVIDIA 驱动及支持 `venv` 的 Python。以下命令均在仓库根目录运行。

```bash
git clone https://github.com/gesus7/Mujoco_SIM_RL.git
cd Mujoco_SIM_RL
bash scripts/setup_server.sh
source .venv-server/bin/activate
python -m panda_rl.doctor --gpu 0 --physics --output reports/doctor-gpu0.json
```

安装脚本会安装固定版本依赖，并从 MuJoCo Menagerie 下载 Panda 资产，需要网络连接。上游版本记录见 [source-provenance.json](scripts/source-provenance.json)。

### 2. 训练

建议从鼓励接近、抓取、抬升及目标驻留的塑形奖励开始：

```bash
python -m panda_rl.train --config configs/hold.json --gpu 0 --seed 0 --output runs/hold-s0
```

也可使用两张 GPU，同时训练无扰动与随机化策略：

```bash
python -m panda_rl.launch --configs configs/hold.json configs/hold-combined.json --gpus 0 1 --seed 0 --output runs
```

上述命令二选一，实验输出目录不能已存在。添加 `--dry-run` 可仅检查配置，不启动训练，也不验证 GPU 是否可用。

| 配置 | 奖励 | 训练扰动 |
| --- | --- | --- |
| [hold.json](configs/hold.json) | 进展奖励与目标驻留奖励 | 无 |
| [hold-combined.json](configs/hold-combined.json) | 进展奖励与目标驻留奖励 | 物理参数、观测噪声、动作延迟 |
| [progress.json](configs/progress.json) | 仅进展奖励 | 无 |
| [baseline.json](configs/baseline.json) | 上游原始奖励 | 无 |

前两个配置使用 2,048 个并行环境，目标训练步数为 4,000 万。其他配置用于消融对照，训练预算与批量设置有所不同。默认值与校验规则见 [config.py](panda_rl/config.py)。

### 3. 传输模型包

训练正常结束后，最终模型包位于 `runs/hold-s0/bundle/`。将**整个目录**复制到本地仓库的 `bundles/hold-s0/`，保留 ONNX 策略、场景资产、元数据及许可证。使用双卡命令时，另一个模型包位于 `runs/hold-combined-s0/bundle/`。

每个实验输出目录内的 `latest_bundle.json` 指向最近一次完整导出的模型包，其中路径相对于该实验目录；训练中断或超时后也可使用，前提是此前已完成导出。同一目录内的 `status.json` 记录运行状态。仓库**不包含预训练权重**。

### 4. 本地评测与演示

建立独立的本地环境。Windows PowerShell 示例：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-local.txt
```

Linux 下使用 `source .venv/bin/activate` 激活环境。随后生成场景、评测已传输的策略并打开交互窗口：

```bash
python -m panda_rl.scenarios --count 200 --seed 10000 --profile nominal --output reports/nominal.json
python -m panda_rl.evaluate --bundle bundles/hold-s0 --scenarios reports/nominal.json --output reports/hold-nominal.json
python -m panda_rl.demo --bundle bundles/hold-s0 --scenarios reports/nominal.json --episode 0
```

在演示命令末尾添加 `--video reports/demo.mp4` 可改为导出视频。批量评测无需渲染。测试鲁棒性时，使用 `--profile combined` 生成另一份场景文件；保持种子和数量一致，可在不同扰动条件下复用相同的方块与目标初始位置。

## 评测协议与局限

每个回合最多 **150 个控制步（3 秒）**。方块需连续 **15 步（0.3 秒）** 位于目标 **5 cm** 范围内才算成功，无效回合计为失败；不要求方块持续停留至回合结束。评测输出 JSON 汇总及逐回合 CSV 数据。

训练奖励与评测成功率是不同指标。物理参数按并行环境采样，在单次训练期间保持不变。检查点不包含优化器及随机数生成器状态，因此不支持精确断点续训。仿真表现不能证明真机迁移能力。

## 源码结构

| 文件 | 职责 |
| --- | --- |
| [envs.py](panda_rl/envs.py) | 任务扩展、奖励、扰动及数值异常终止 |
| [train.py](panda_rl/train.py)、[launch.py](panda_rl/launch.py) | PPO 训练、时长监督及独立 GPU 进程 |
| [export.py](panda_rl/export.py)、[runtime.py](panda_rl/runtime.py) | ONNX 模型包导出与原生 MuJoCo 推理 |
| [scenarios.py](panda_rl/scenarios.py)、[metrics.py](panda_rl/metrics.py) | 固定场景及成功判据 |
| [evaluate.py](panda_rl/evaluate.py)、[demo.py](panda_rl/demo.py) | 批量评测、交互窗口及视频 |

## 测试

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
```

测试覆盖配置、训练监督、指标及导出与推理一致性。集成测试还需要训练依赖和已下载的机器人资产，条件不满足时跳过。测试不会执行 PPO 优化。

## 许可证与致谢

项目代码使用 [Apache-2.0](LICENSE) 许可证。任务与训练框架基于 [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)、[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) 和 [Brax](https://github.com/google/brax)。上游归属及固定版本见 [NOTICE.md](NOTICE.md)。再分发模型包时，请保留随包许可证。
