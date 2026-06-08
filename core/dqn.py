"""
DQN网络 - Deep Q-Network for Relationship Management

结合反思注意力的DQN架构
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
from collections import deque
import random

from .mdp import ActionType, RelationshipState, ACTION_SPACE


class ReflectionAttention(nn.Module):
    """
    反思注意力模块

    将历史反思经验作为上下文，指导当前决策
    """

    def __init__(self, embed_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self,
                state_embed: torch.Tensor,
                reflection_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            state_embed: (batch, embed_dim) 状态嵌入
            reflection_context: (batch, seq_len, embed_dim) 反思上下文
        """
        if reflection_context is None:
            return state_embed

        # 扩展维度以适配attention
        state_embed = state_embed.unsqueeze(1)  # (batch, 1, embed_dim)

        # 注意力融合
        attn_output, _ = self.attention(
            state_embed,
            reflection_context,
            reflection_context
        )

        # 残差连接
        output = self.norm(state_embed + attn_output)

        return output.squeeze(1)  # (batch, embed_dim)


class RelationshipDQN(nn.Module):
    """
    关系管理DQN网络

    架构：
    1. 状态编码器：提取状态特征
    2. 反思注意力：融合历史经验
    3. Q值头：输出各动作价值
    """

    def __init__(self,
                 state_dim: int = 9,
                 action_dim: int = 9,
                 hidden_dim: int = 128,
                 num_reflection_layers: int = 2):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 反思注意力
        self.reflection_attention = ReflectionAttention(
            embed_dim=hidden_dim,
            num_heads=4
        )

        # 反思上下文编码器
        self.reflection_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_reflection_layers,
            batch_first=True
        )

        # Q值头
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # 价值流和优势流（Dueling DQN）
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        # 使用Dueling架构
        self.use_dueling = True

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """编码状态"""
        return self.state_encoder(state)

    def encode_reflections(self,
                           reflections: torch.Tensor) -> torch.Tensor:
        """
        编码反思序列

        Args:
            reflections: (batch, seq_len, hidden_dim) 历史反思嵌入
        """
        output, _ = self.reflection_encoder(reflections)
        return output  # (batch, seq_len, hidden_dim)

    def forward(self,
                state: torch.Tensor,
                reflection_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            state: (batch, state_dim) 状态向量
            reflection_context: (batch, seq_len, hidden_dim) 反思上下文

        Returns:
            q_values: (batch, action_dim) Q值
        """
        # 编码状态
        state_embed = self.encode_state(state)

        # 融合反思上下文
        if reflection_context is not None:
            state_embed = self.reflection_attention(state_embed, reflection_context)

        # 计算Q值
        if self.use_dueling:
            # Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
            value = self.value_stream(state_embed)
            advantage = self.advantage_stream(state_embed)
            q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            q_values = self.q_head(state_embed)

        return q_values

    def get_action(self,
                    state: RelationshipState,
                    reflection_context: Optional[torch.Tensor] = None,
                    epsilon: float = 0.0) -> Tuple[ActionType, float]:
        """
        选择动作

        Args:
            state: 当前状态
            reflection_context: 反思上下文
            epsilon: 探索率

        Returns:
            action: 选择的动作
            q_value: 动作Q值
        """
        # 转换为tensor
        state_vec = torch.from_numpy(state.to_vector()).unsqueeze(0)

        with torch.no_grad():
            q_values = self.forward(state_vec, reflection_context)

        # epsilon-greedy
        if random.random() < epsilon:
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            action_idx = q_values.argmax(dim=-1).item()

        action = ActionType(action_idx)
        q_value = q_values[0, action_idx].item()

        return action, q_value


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self,
             state: np.ndarray,
             action: int,
             reward: float,
             next_state: np.ndarray,
             done: bool,
             reflection: Optional[np.ndarray] = None):
        """存储转移"""
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'reflection': reflection
        })

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """采样批次"""
        batch = random.sample(self.buffer, batch_size)

        states = torch.from_numpy(
            np.stack([t['state'] for t in batch])
        ).float()
        actions = torch.tensor([t['action'] for t in batch])
        rewards = torch.tensor([t['reward'] for t in batch]).float()
        next_states = torch.from_numpy(
            np.stack([t['next_state'] for t in batch])
        ).float()
        dones = torch.tensor([t['done'] for t in batch]).float()

        reflections = None
        if batch[0]['reflection'] is not None:
            reflections = torch.from_numpy(
                np.stack([t['reflection'] for t in batch])
            ).float()

        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'reflections': reflections
        }

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer(ReplayBuffer):
    """优先级经验回放"""

    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        super().__init__(capacity)
        self.alpha = alpha
        self.priorities = deque(maxlen=capacity)

    def push(self, *args, priority: float = 1.0, **kwargs):
        super().push(*args, **kwargs)
        self.priorities.append(priority ** self.alpha)

    def _process_batch(self, batch):
        """处理批次数据"""
        states = torch.from_numpy(
            np.stack([t['state'] for t in batch])
        ).float()
        actions = torch.tensor([t['action'] for t in batch])
        rewards = torch.tensor([t['reward'] for t in batch]).float()
        next_states = torch.from_numpy(
            np.stack([t['next_state'] for t in batch])
        ).float()
        dones = torch.tensor([t['done'] for t in batch]).float()

        reflections = None
        if batch[0]['reflection'] is not None:
            reflections = torch.from_numpy(
                np.stack([t['reflection'] for t in batch])
            ).float()

        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'reflections': reflections
        }

    def sample(self, batch_size: int, beta: float = 0.4):
        # 计算采样概率
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()

        # 采样索引
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # 计算重要性权重
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights = weights / weights.max()

        # 获取批次
        batch = [self.buffer[i] for i in indices]

        return {
            **self._process_batch(batch),
            'indices': indices,
            'weights': torch.tensor(weights).float()
        }

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority ** self.alpha


class DQNAgent:
    """
    DQN智能体

    特性：
    - Double DQN
    - Dueling Architecture
    - Prioritized Experience Replay
    - Soft Target Update
    """

    def __init__(self,
                 state_dim: int = 9,
                 action_dim: int = 9,
                 hidden_dim: int = 128,
                 lr: float = 1e-4,
                 gamma: float = 0.95,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.995,
                 target_update_freq: int = 100,
                 soft_update_tau: float = 0.005,
                 device: str = 'auto'):

        # 设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 网络
        self.q_network = RelationshipDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = RelationshipDQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # 优化器
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        # 经验回放
        self.memory = PrioritizedReplayBuffer()

        # 超参数
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.soft_update_tau = soft_update_tau

        self.action_dim = action_dim
        self.step_counter = 0

    def select_action(self,
                      state: RelationshipState,
                      reflection_context: Optional[torch.Tensor] = None) -> ActionType:
        """选择动作"""
        action, _ = self.q_network.get_action(
            state, reflection_context, self.epsilon
        )
        return action

    def act(self, state: RelationshipState) -> ActionType:
        """执行动作（接口）"""
        return self.select_action(state)

    def store_transition(self,
                         state: np.ndarray,
                         action: int,
                         reward: float,
                         next_state: np.ndarray,
                         done: bool,
                         td_error: float = 1.0):
        """存储转移"""
        self.memory.push(state, action, reward, next_state, done,
                         priority=abs(td_error) + 1e-6)

    def learn(self, batch_size: int = 32) -> Optional[float]:
        """学习"""
        if len(self.memory) < batch_size:
            return None

        # 采样
        batch = self.memory.sample(batch_size, beta=0.4)

        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        weights = batch['weights'].to(self.device)
        indices = batch['indices']

        # 计算当前Q值
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: 用主网络选择动作，用目标网络评估
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=-1)
            next_q = self.target_network(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # 计算损失
        td_errors = current_q - target_q
        loss = (weights * td_errors ** 2).mean()

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # 更新优先级
        new_priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
        self.memory.update_priorities(indices, new_priorities)

        # 软更新目标网络
        self._soft_update()

        # 衰减探索率
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        self.step_counter += 1

        return loss.item()

    def _soft_update(self):
        """软更新目标网络"""
        for target_param, param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.soft_update_tau * param.data +
                (1 - self.soft_update_tau) * target_param.data
            )

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_counter': self.step_counter
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.step_counter = checkpoint['step_counter']


__all__ = ['RelationshipDQN', 'DQNAgent', 'ReplayBuffer', 'PrioritizedReplayBuffer']
