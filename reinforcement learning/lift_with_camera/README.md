# Franka Lift Camera Fusion 修改记录

## 1. 目标
在 IsaacLab 的 Franka Lift 任务中，引入相机视觉并与机械臂状态融合训练（skrl PPO），实现稳定的抓取与抬升，并改善抬起后的目标跟踪。

## 2. 初始改动（从 CNN 结构开始）
最初将策略输入改为 `state + image` 融合：

- `image` 分支：3 层 CNN  
  - `conv2d(32, k8, s4)`  
  - `conv2d(64, k4, s2)`  
  - `conv2d(64, k3, s1)`  
  - `flatten -> linear(512) -> linear(64)`
- `state` 分支：`linear(128) -> linear(64)`
- 融合分支：`concat(image_feat, state_feat) -> linear(256) -> linear(128)`

对应文件：

- `config/franka/agents/skrl_camera_fusion_ppo_cfg.yaml`

## 3. 第一次训练遇到的问题

### 3.1 启动/配置问题
- 报错：未启用相机渲染（`A camera was spawned without the --enable_cameras flag`）  
  - 处理：训练和回放统一加 `--enable_cameras`
- 报错：`AttributeError: 'dict' object has no attribute 'shape'`（早期使用不匹配框架/观测结构）  
  - 处理：切换到 skrl，并按 skrl 的观测格式重构
- 报错：`unexpected character after line continuation character`  
  - 原因：Python 文件误粘贴 Windows 路径文本（如 `C:\Users\...`）

### 3.2 训练行为问题
- 可以抓起，但抬起后高频抖动明显
- `object_goal_tracking` 中后期退化
- 总回报中期出现漂移/坠崖，`min reward` 变差
- `action_rate` 惩罚变得更负，和抖动现象一致

## 4. 核心修改（按时间线）

### 4.1 建立 camera fusion 独立配置链
新增独立文件，避免污染原始 lift 配置：

- `lift_camera_env_cfg.py`
- `lift_camera_fusion_env_cfg.py`
- `lift_camera_fusion_adapted_env_cfg.py`
- `config/franka/joint_pos_camera_env_cfg.py`
- `config/franka/joint_pos_camera_fusion_adapted_env_cfg.py`
- `mdp_camera_fusion/`（`observations/rewards/terminations`）

并注册任务：

- `Isaac-Lift-Cube-Franka-Camera-Fusion-Adapted-v0`

### 4.2 修正观测结构与网络输入
将观测统一到 `policy` 字典（`concatenate_terms=False`），包含：

- `joint_pos, joint_vel, object_position, target_object_position, actions, image`

使 skrl YAML 中的 `OBSERVATIONS["image"]` 与 `OBSERVATIONS["...state..."]` 可直接索引，解决输入维度/索引错误。

### 4.3 稳定性调参（不破坏 checkpoint 续训）
针对抖动与中期漂移，调整 reward 与 PPO 超参：

- Reward（`lift_camera_fusion_adapted_env_cfg.py`）
  - `reaching_object: 2.0 -> 1.5`
  - `object_goal_tracking: 8.0 -> 10.0`
  - `object_goal_tracking_fine_grained: 2.0 -> 3.0`
  - `action_rate: -1e-5 -> -5e-5`
  - `joint_vel: -1e-5 -> -3e-5`
- PPO（`skrl_camera_fusion_ppo_cfg.yaml`）
  - `learning_rate: 1e-4 -> 6e-5`
  - `learning_epochs: 8 -> 5`
  - `ratio_clip: 0.2 -> 0.15`
  - `grad_norm_clip: 1.0 -> 0.5`
  - `entropy_loss_scale: 0.001 -> 0.0005`
  - KL 阈值收紧

### 4.4 抬起后视觉门控（gating）
新增函数：

- `mdp_camera_fusion/observations_camera_fusion.py::gated_image_after_lift`

逻辑：

- 抬起前视觉权重高（`1.0`）
- 抬起后平滑降到低权重（默认 `0.1`）
- 使用 `sigmoid` 平滑过渡，抑制抬起后的高频视觉驱动抖动
- **图像张量形状不变**，可继续沿用现有模型结构

## 5. 续训兼容性说明
为了继续从已有 checkpoint 训练，后期改动遵守：

- 不改 action 维度
- 不改网络层形状与参数命名
- 不改观测张量最终维度（gating 仅做按样本缩放）

因此可继续使用：

- `.../checkpoints/best_agent.pt`

## 6. 常用命令

训练（续训）：

```bash
cd /root/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task Isaac-Lift-Cube-Franka-Camera-Fusion-Adapted-v0 \
  --algorithm PPO \
  --headless \
  --enable_cameras \
  --num_envs 64 \
  --checkpoint /root/IsaacLab/logs/skrl/franka_lift_camera_fusion/<run>/checkpoints/best_agent.pt \
  --max_iterations 30000
```

回放：

```bash
cd /root/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
  --task Isaac-Lift-Cube-Franka-Camera-Fusion-Adapted-v0 \
  --algorithm PPO \
  --enable_cameras \
  --num_envs 1 \
  --checkpoint /root/IsaacLab/logs/skrl/franka_lift_camera_fusion/<run>/checkpoints/best_agent.pt
```

TensorBoard：

```bash
cd /root/IsaacLab
tensorboard --logdir /root/IsaacLab/logs/skrl/franka_lift_camera_fusion --host 0.0.0.0 --port 6006
```
