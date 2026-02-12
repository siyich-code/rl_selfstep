import random
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from gymnasium.error import NameNotFound, NamespaceNotFound
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F

class replay_buffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones

    def size(self):
        return len(self.buffer)

class env_agent:
    ''' 环境与智能体交互 '''
    def __init__(self, env_name):
        # Gymnasium + ale-py 在部分版本里需要显式注册 ALE 环境
        if env_name.startswith("ALE/"):
            try:
                import ale_py  # type: ignore
                gym.register_envs(ale_py)
            except Exception:
                pass
        try:
            # AtariPreprocessing 会处理 frame skip，这里把 ALE 原生 frameskip 关掉避免重复
            if env_name.startswith("ALE/"):
                base_env = gym.make(env_name, frameskip=1)
            else:
                base_env = gym.make(env_name)
        except (NameNotFound, NamespaceNotFound) as e:
            raise RuntimeError(
                f"环境 `{env_name}` 不可用: {e}. "
                "当前代码使用 Atari 预处理，请先安装并注册 Atari 环境，例如: "
                "`pip install gymnasium[atari,accept-rom-license] ale-py`。"
            ) from e
        # DQN 2015 standard preprocessing for Atari
        try:
            base_env = AtariPreprocessing(
                base_env,
                frame_skip=4,
                grayscale_obs=True,
                scale_obs=False,
                terminal_on_life_loss=False
            )
        except gym.error.DependencyNotInstalled as e:
            raise RuntimeError(
                "AtariPreprocessing 需要 opencv-python。请安装: "
                "`pip install opencv-python` 或 `pip install \"gymnasium[other]\"`。"
            ) from e
        try:
            self.env = FrameStackObservation(base_env, num_stack=4)
        except TypeError:
            # gymnasium 版本差异：参数名可能是 stack_size
            self.env = FrameStackObservation(base_env, stack_size=4)
        self.state = self.env.reset()[0]
        self.done = False

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.done = terminated or truncated
        return next_state, reward, self.done, info

    def reset(self):
        self.state = self.env.reset()[0]
        self.done = False
        return self.state
    
class Q_NET(nn.Module):
    """
    DQN (Nature 2015) network:
    Input:  (B, 4, 84, 84)  uint8/float
    Conv1:  32 filters, 8x8, stride 4, ReLU
    Conv2:  64 filters, 4x4, stride 2, ReLU
    Conv3:  64 filters, 3x3, stride 1, ReLU
    FC:     512, ReLU
    Output: num_actions (linear)
    """
    def __init__(self, num_actions):
        super(Q_NET, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.fc1 = nn.Linear(7*7*64, 512)
        self.out = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        return self.out(x)
class dqn:
    ''' DQN算法 '''
    def __init__(self,learning_rate, gamma,
                 epsilon, target_update, device, specific_actions):
        
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device # 设备
        self.q_net = Q_NET(num_actions=specific_actions).to(device)  #最少6个动作
        self.target_q_net = Q_NET(num_actions=specific_actions).to(device)# 目标网络
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        #todo use gradiant decent optimizer
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        
    def _state_to_tensor(self, state):
        state_np = np.asarray(state, dtype=np.float32) / 255.0
        state_t = torch.from_numpy(state_np).unsqueeze(0).to(self.device)
        return state_t

    def _batch_to_tensor(self, batch_state):
        batch_np = np.asarray(batch_state, dtype=np.float32) / 255.0
        return torch.from_numpy(batch_np).to(self.device)

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.q_net.out.out_features)
        else:
            state = self._state_to_tensor(state)
            action = self.q_net(state).argmax().item()
        return action
    
    def update(self, replay_buffer, batch_size):
        if replay_buffer.size() < batch_size:
            return None

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = self._batch_to_tensor(states)
        actions = torch.tensor(actions, dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1).to(self.device)
        next_states = self._batch_to_tensor(next_states)
        dones = torch.tensor(dones, dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q(s,a)
        with torch.no_grad():
            max_next_q_values = self.target_q_net(next_states).max(dim=1)[0].view(
                -1, 1)  # max_a' Q_target(s',a')
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)  # 目标Q值

        loss = F.mse_loss(q_values, target_q_values)  # 均方误差损失

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新目标网络
        self.count += 1
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return loss.item()
