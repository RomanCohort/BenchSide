"""
奖励模型 - 从用户反馈学习
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class FeedbackRecord:
    """用户反馈记录"""
    state: np.ndarray
    action: int
    user_rating: float  # -1到+1
    outcome_success: bool
    relationship_change: float  # loved_index变化
    them_initiated: bool  # 对方是否主动


class RewardModel(nn.Module):
    """
    奖励模型

    从用户反馈学习奖励函数
    用于：
    1. 计算即时奖励
    2. 预测长期奖励
    3. 评估建议质量
    """

    REWARD_COMPONENTS = {
        'user_feedback': {
            'weight': 10.0,
            'range': (-1, 1),
            'description': '用户显式反馈'
        },
        'relationship_progress': {
            'weight': 5.0,
            'range': (-10, 10),
            'description': '关系指标变化'
        },
        'timeliness': {
            'weight': 2.0,
            'range': (0, 1),
            'description': '回复及时性'
        },
        'over_active_penalty': {
            'weight': -3.0,
            'range': (0, 1),
            'description': '过度主动惩罚'
        },
        'them_initiated': {
            'weight': 8.0,
            'range': (0, 1),
            'description': '对方主动（强正向信号）'
        },
        'cold_recovery': {
            'weight': 3.0,
            'range': (0, 1),
            'description': '冷战修复奖励'
        },
        'action_risk': {
            'weight': -2.0,
            'range': (0, 1),
            'description': '高风险动作惩罚'
        }
    }

    def __init__(self, state_dim: int = 9, action_dim: int = 9):
        super().__init__()

        # 状态-动作编码器
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # 奖励预测头
        self.reward_head = nn.Linear(64, 1)

        # 各分量预测头
        self.component_heads = nn.ModuleDict({
            'user_feedback': nn.Linear(64, 1),
            'relationship_progress': nn.Linear(64, 1),
            'timeliness': nn.Linear(64, 1),
            'them_initiated': nn.Linear(64, 1)
        })

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        预测奖励

        Args:
            state: (batch, state_dim)
            action: (batch, action_dim) one-hot

        Returns:
            reward: (batch, 1)
        """
        # 编码
        x = torch.cat([state, action], dim=-1)
        embed = self.encoder(x)

        # 预测总奖励
        reward = self.reward_head(embed)

        return reward

    def predict_components(self,
                           state: torch.Tensor,
                           action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        预测各奖励分量

        Returns:
            各分量的预测值
        """
        x = torch.cat([state, action], dim=-1)
        embed = self.encoder(x)

        components = {}
        for name, head in self.component_heads.items():
            components[name] = head(embed)

        return components

    def compute_reward(self,
                       feedback: FeedbackRecord,
                       outcome: Dict) -> float:
        """
        计算奖励信号

        Args:
            feedback: 用户反馈
            outcome: 结果信息

        Returns:
            奖励值
        """
        reward = 0.0

        # 1. 用户反馈
        reward += feedback.user_rating * self.REWARD_COMPONENTS['user_feedback']['weight']

        # 2. 关系进展
        reward += feedback.relationship_change * self.REWARD_COMPONENTS['relationship_progress']['weight']

        # 3. 对方主动
        if feedback.them_initiated:
            reward += self.REWARD_COMPONENTS['them_initiated']['weight']

        # 4. 成功结果
        if feedback.outcome_success:
            reward += 5.0

        return reward

    def compute_delayed_reward(self,
                                episode: List[Dict],
                                final_state: Dict) -> float:
        """
        计算延迟奖励（基于整个episode）

        Args:
            episode: 整个交互序列
            final_state: 最终状态

        Returns:
            总延迟奖励
        """
        delayed_reward = 0.0

        # 评估关系变化
        initial_loved = episode[0]['state']['loved_index']
        final_loved = final_state['loved_index']
        loved_change = final_loved - initial_loved

        if loved_change > 5:
            delayed_reward += 10.0  # 显著升温
        elif loved_change > 0:
            delayed_reward += 5.0   # 略有升温
        elif loved_change < -5:
            delayed_reward -= 5.0   # 显著降温

        # 评估对方主动性
        them_initiated_count = sum(
            1 for step in episode if step.get('them_initiated', False)
        )
        delayed_reward += them_initiated_count * 3.0

        return delayed_reward


class FeedbackCollector:
    """
    用户反馈收集器

    支持多种反馈形式：
    1. 显式评分（按钮点击）
    2. 隐式反馈（行为采纳）
    3. 结果追踪（后续变化）
    """

    def __init__(self):
        self.feedback_buffer: List[FeedbackRecord] = []

    def collect_explicit(self,
                         state: np.ndarray,
                         action: int,
                         rating: float) -> FeedbackRecord:
        """
        收集显式反馈

        Args:
            rating: 用户评分 (-1到+1)
        """
        feedback = FeedbackRecord(
            state=state,
            action=action,
            user_rating=rating,
            outcome_success=False,  # 待定
            relationship_change=0.0,  # 待定
            them_initiated=False   # 待定
        )
        self.feedback_buffer.append(feedback)
        return feedback

    def collect_implicit(self,
                         state: np.ndarray,
                         action: int,
                         adopted: bool) -> FeedbackRecord:
        """
        收集隐式反馈

        Args:
            adopted: 是否采纳建议
        """
        # 采纳 = +0.5, 不采纳 = -0.2
        rating = 0.5 if adopted else -0.2

        return self.collect_explicit(state, action, rating)

    def update_outcome(self,
                       feedback_idx: int,
                       outcome: Dict):
        """
        更新反馈的结果信息

        Args:
            feedback_idx: 反馈索引
            outcome: {
                'success': bool,
                'relationship_change': float,
                'them_initiated': bool
            }
        """
        if feedback_idx < len(self.feedback_buffer):
            feedback = self.feedback_buffer[feedback_idx]
            feedback.outcome_success = outcome.get('success', False)
            feedback.relationship_change = outcome.get('relationship_change', 0.0)
            feedback.them_initiated = outcome.get('them_initiated', False)

    def get_training_data(self) -> Dict:
        """
        获取训练数据
        """
        states = np.stack([f.state for f in self.feedback_buffer])
        actions = np.zeros((len(self.feedback_buffer), 9))
        for i, f in enumerate(self.feedback_buffer):
            actions[i, f.action] = 1.0

        # 计算奖励
        reward_model = RewardModel()
        rewards = np.array([
            reward_model.compute_reward(f, {}) for f in self.feedback_buffer
        ])

        return {
            'states': torch.from_numpy(states).float(),
            'actions': torch.from_numpy(actions).float(),
            'rewards': torch.from_numpy(rewards).float()
        }


class RewardLearner:
    """
    奖励学习器

    从用户反馈学习奖励函数
    """

    def __init__(self,
                 reward_model: RewardModel,
                 lr: float = 1e-3):
        self.model = reward_model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self,
              feedback_data: Dict,
              epochs: int = 10,
              batch_size: int = 32):
        """
        训练奖励模型
        """
        states = feedback_data['states']
        actions = feedback_data['actions']
        rewards = feedback_data['rewards']

        n_samples = len(states)

        for epoch in range(epochs):
            total_loss = 0.0

            # 随机采样batch
            indices = torch.randperm(n_samples)

            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i+batch_size]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_rewards = rewards[batch_idx]

                # 预测
                pred_rewards = self.model(batch_states, batch_actions)

                # 损失
                loss = nn.MSELoss()(pred_rewards.squeeze(), batch_rewards)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / (n_samples // batch_size + 1)
            print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))


__all__ = [
    'RewardModel', 'FeedbackCollector', 'FeedbackRecord',
    'RewardLearner'
]