本项目在 Google DeepMind MuJoCo Playground 的 PandaPickCube 上增加实验配置、扰动、预算监督、模型导出和本地评测。

- Playground: https://github.com/google-deepmind/mujoco_playground/tree/86bfcd999619ad8d058383feefc1dd16d4cc57a7 ，Apache-2.0。
- Panda meshes/XML: https://github.com/google-deepmind/mujoco_menagerie/tree/1b86ece576591213e2b666ebf59508454200ca97/franka_emika_panda ，Apache-2.0。
- Brax: https://github.com/google/brax ，Apache-2.0。
- 本项目新增代码使用 Apache-2.0，见 LICENSE。runtime.py 中观测布局和控制语义依据上述 Playground 源码改写。
- envs.py 的历史最佳进展奖励借鉴 Playground 的 pick_cartesian.py；闭合/抬升代理势函数、目标内驻留奖励及数值终止门槛为本项目修改，原始奖励对照保留。
- 模型包携带机器人资产许可证；发布时保留 NOTICE.md、LICENSE 和 bundle/licenses。各 Python 依赖的许可证见其发行包。
