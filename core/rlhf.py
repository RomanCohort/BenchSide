"""
RLHF (Reinforcement Learning from Human Feedback) 扩展模块

将本项目升级为 RLHF 框架，增强论文创新性和技术深度

核心思想：
1. 收集人类对决策建议的偏好反馈
2. 训练奖励模型学习人类偏好
3. 使用 PPO 微调策略网络

这比普通 RL 更符合当前 AI 研究趋势！
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import random


# ============================================================
# 1. 人类反馈数据结构
# ============================================================

@dataclass
class HumanFeedback:
    """单条人类反馈"""
    state: np.ndarray                    # 状态
    action_a: int                        # 动作A
    action_b: int                        # 动作B
    preference: int                      # 0=A更好, 1=B更好, 0.5=差不多
    confidence: float                    # 置信度
    user_id: str                         # 用户ID
    context: Optional[str] = None        # 上下文描述
    timestamp: float = 0.0               # 时间戳


@dataclass
class PreferenceDataset:
    """偏好数据集"""
    feedbacks: List[HumanFeedback] = field(default_factory=list)

    def add_feedback(self, feedback: HumanFeedback):
        self.feedbacks.append(feedback)

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """获取训练数据"""
        states = []
        actions_a = []
        actions_b = []
        preferences = []

        for f in self.feedbacks:
            states.append(f.state)
            actions_a.append(f.action_a)
            actions_b.append(f.action_b)
            preferences.append(f.preference)

        return (
            np.array(states),
            np.array(actions_a),
            np.array(actions_b),
            np.array(preferences)
        )

    def split(self, ratio: float = 0.8) -> Tuple['PreferenceDataset', 'PreferenceDataset']:
        """划分训练/验证集"""
        n = len(self.feedbacks)
        split_idx = int(n * ratio)

        train = PreferenceDataset(feedbacks=self.feedbacks[:split_idx])
        val = PreferenceDataset(feedbacks=self.feedbacks[split_idx:])

        return train, val


# ============================================================
# 2. 奖励模型 (Reward Model)
# ============================================================

class RewardModel(nn.Module):
    """
    奖励模型

    从人类偏好中学习奖励函数

    架构：
    - State Encoder: 编码状态
    - Action Embedding: 动作嵌入
    - Reward Head: 输出奖励分数
    """

    def __init__(self,
                 state_dim: int = 9,
                 action_dim: int = 9,
                 hidden_dim: int = 128,
                 num_layers: int = 3):
        super().__init__()

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Action embedding
        self.action_embedding = nn.Embedding(action_dim, hidden_dim)

        # Reward network
        layers = []
        input_dim = hidden_dim * 2  # state + action
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        layers.append(nn.Linear(hidden_dim, 1))

        self.reward_head = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        计算奖励分数

        Args:
            state: (batch, state_dim)
            action: (batch,) action indices

        Returns:
            reward: (batch, 1)
        """
        # Encode state
        state_feat = self.state_encoder(state)  # (batch, hidden_dim)

        # Embed action
        action_feat = self.action_embedding(action)  # (batch, hidden_dim)

        # Concatenate
        combined = torch.cat([state_feat, action_feat], dim=-1)  # (batch, hidden_dim*2)

        # Compute reward
        reward = self.reward_head(combined)  # (batch, 1)

        return reward

    def get_reward(self, state: np.ndarray, action: int) -> float:
        """获取单个奖励值（推理用）"""
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            action_tensor = torch.tensor([action])
            reward = self.forward(state_tensor, action_tensor)
            return reward.item()


class RewardModelTrainer:
    """奖励模型训练器"""

    def __init__(self,
                 reward_model: RewardModel,
                 lr: float = 1e-4,
                 device: str = "cpu"):
        self.model = reward_model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_epoch(self,
                    states: np.ndarray,
                    actions_a: np.ndarray,
                    actions_b: np.ndarray,
                    preferences: np.ndarray,
                    batch_size: int = 32) -> float:
        """
        训练一个epoch

        使用 Bradley-Terry 模型：
        P(A > B) = sigmoid(r(A) - r(B))
        """
        self.model.train()

        n = len(states)
        indices = list(range(n))
        random.shuffle(indices)

        total_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            batch_indices = indices[i:i+batch_size]

            # 获取batch数据
            batch_states = torch.from_numpy(states[batch_indices]).float().to(self.device)
            batch_actions_a = torch.from_numpy(actions_a[batch_indices]).long().to(self.device)
            batch_actions_b = torch.from_numpy(actions_b[batch_indices]).long().to(self.device)
            batch_prefs = torch.from_numpy(preferences[batch_indices]).float().to(self.device)

            # 计算奖励
            reward_a = self.model(batch_states, batch_actions_a).squeeze(-1)  # (batch,)
            reward_b = self.model(batch_states, batch_actions_b).squeeze(-1)  # (batch,)

            # Bradley-Terry loss: -log(sigmoid(r_a - r_b)) when pref=1
            # pref=1: A更好, pref=0: B更好
            logits = reward_a - reward_b
            loss = F.binary_cross_entropy_with_logits(logits, batch_prefs)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def evaluate(self,
                 states: np.ndarray,
                 actions_a: np.ndarray,
                 actions_b: np.ndarray,
                 preferences: np.ndarray) -> Dict:
        """评估模型"""
        self.model.eval()

        with torch.no_grad():
            states_tensor = torch.from_numpy(states).float().to(self.device)
            actions_a_tensor = torch.from_numpy(actions_a).long().to(self.device)
            actions_b_tensor = torch.from_numpy(actions_b).long().to(self.device)
            prefs_tensor = torch.from_numpy(preferences).float().to(self.device)

            reward_a = self.model(states_tensor, actions_a_tensor).squeeze(-1)
            reward_b = self.model(states_tensor, actions_b_tensor).squeeze(-1)

            logits = reward_a - reward_b
            probs = torch.sigmoid(logits)

            # 计算准确率
            predictions = (probs > 0.5).float()
            accuracy = (predictions == prefs_tensor).float().mean().item()

            # 计算loss
            loss = F.binary_cross_entropy_with_logits(logits, prefs_tensor).item()

        return {
            "loss": loss,
            "accuracy": accuracy
        }


# ============================================================
# 3. PPO 策略优化
# ============================================================

class PPOActorCritic(nn.Module):
    """PPO Actor-Critic 网络"""

    def __init__(self,
                 state_dim: int = 9,
                 action_dim: int = 9,
                 hidden_dim: int = 128):
        super().__init__()

        # 共享特征提取
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor (策略网络)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Critic (价值网络)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        features = self.shared(state)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def get_action(self, state: np.ndarray) -> Tuple[int, float]:
        """采样动作"""
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            logits, value = self.forward(state_tensor)
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
            log_prob = F.log_softmax(logits, dim=-1)[0, action].item()
            return action, log_prob

    def evaluate_actions(self,
                         states: torch.Tensor,
                         actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """评估动作"""
        logits, values = self.forward(states)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        return action_log_probs, values.squeeze(-1), entropy


class PPOTrainer:
    """PPO训练器（使用学习到的奖励模型）"""

    def __init__(self,
                 actor_critic: PPOActorCritic,
                 reward_model: RewardModel,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 device: str = "cpu"):
        self.actor_critic = actor_critic.to(device)
        self.reward_model = reward_model.to(device)
        self.device = device

        self.optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

    def compute_gae(self,
                    rewards: torch.Tensor,
                    values: torch.Tensor,
                    dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算 GAE (Generalized Advantage Estimation)"""
        advantages = []
        returns = []

        gae = 0
        next_value = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            next_value = values[t]

        return torch.stack(advantages), torch.stack(returns)

    def train_step(self,
                   states: torch.Tensor,
                   actions: torch.Tensor,
                   old_log_probs: torch.Tensor,
                   advantages: torch.Tensor,
                   returns: torch.Tensor,
                   epochs: int = 4) -> Dict:
        """PPO训练步骤"""

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for _ in range(epochs):
            # 评估当前策略
            log_probs, values, entropy = self.actor_critic.evaluate_actions(states, actions)

            # 计算ratio
            ratio = torch.exp(log_probs - old_log_probs)

            # PPO clip loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values, returns)

            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            # 优化
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        return {
            "policy_loss": total_policy_loss / epochs,
            "value_loss": total_value_loss / epochs,
            "entropy": total_entropy / epochs
        }


# ============================================================
# 4. RLHF 训练流程
# ============================================================

class RLHFTrainer:
    """
    完整的 RLHF 训练流程

    Step 1: 收集人类偏好反馈
    Step 2: 训练奖励模型
    Step 3: 使用 PPO 微调策略
    """

    def __init__(self,
                 state_dim: int = 9,
                 action_dim: int = 9,
                 hidden_dim: int = 128,
                 device: str = "cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # 初始化模型
        self.reward_model = RewardModel(state_dim, action_dim, hidden_dim)
        self.actor_critic = PPOActorCritic(state_dim, action_dim, hidden_dim)

        # 训练器
        self.reward_trainer = RewardModelTrainer(self.reward_model, device=device)
        self.ppo_trainer = PPOTrainer(self.actor_critic, self.reward_model, device=device)

        # 数据
        self.preference_data = PreferenceDataset()

    def collect_feedback(self,
                         state: np.ndarray,
                         action_a: int,
                         action_b: int,
                         preference: int,
                         confidence: float = 1.0,
                         user_id: str = "default"):
        """收集人类反馈"""
        feedback = HumanFeedback(
            state=state,
            action_a=action_a,
            action_b=action_b,
            preference=preference,
            confidence=confidence,
            user_id=user_id
        )
        self.preference_data.add_feedback(feedback)

    def train_reward_model(self,
                            epochs: int = 10,
                            batch_size: int = 32) -> Dict:
        """训练奖励模型"""
        if len(self.preference_data.feedbacks) < 10:
            return {"error": "Not enough feedback data"}

        # 划分数据集
        train_data, val_data = self.preference_data.split(0.8)

        train_states, train_actions_a, train_actions_b, train_prefs = train_data.get_training_data()
        val_states, val_actions_a, val_actions_b, val_prefs = val_data.get_training_data()

        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

        for epoch in range(epochs):
            # 训练
            train_loss = self.reward_trainer.train_epoch(
                train_states, train_actions_a, train_actions_b, train_prefs, batch_size
            )

            # 验证
            val_metrics = self.reward_trainer.evaluate(
                val_states, val_actions_a, val_actions_b, val_prefs
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])

            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_metrics['loss']:.4f}, "
                  f"Val Acc={val_metrics['accuracy']:.2%}")

        return history

    def train_policy_with_ppo(self,
                               env_episodes: int = 100,
                               ppo_epochs: int = 4) -> Dict:
        """使用 PPO 微调策略"""
        # 这里需要与环境交互
        # 简化：使用奖励模型计算奖励

        history = {"rewards": [], "policy_loss": [], "value_loss": []}

        for episode in range(env_episodes):
            # 收集轨迹（需要实际环境）
            # 这里简化处理
            pass

        return history

    def save_models(self, path: str):
        """保存模型"""
        torch.save({
            "reward_model": self.reward_model.state_dict(),
            "actor_critic": self.actor_critic.state_dict()
        }, path)

    def load_models(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.reward_model.load_state_dict(checkpoint["reward_model"])
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])


# ============================================================
# 5. 反馈收集界面
# ============================================================

def create_comparison_prompt(state: np.ndarray,
                              action_a: int,
                              action_b: int,
                              action_names: List[str]) -> str:
    """创建比较提示"""
    prompt = f"""
请比较以下两个决策建议，选择更好的一个：

当前状态：
- 主动指数: {state[7]:.0f}
- 被爱指数: {state[8]:.0f}
- 冷淡指数: {state[3]:.0f}

建议 A: {action_names[action_a]}
建议 B: {action_names[action_b]}

请选择：
1. A 更好
2. B 更好
3. 差不多

您的选择（1/2/3）: """
    return prompt


# ============================================================
# 6. 测试
# ============================================================

def test_rlhf():
    """测试 RLHF 框架"""
    print("=" * 50)
    print("RLHF Framework Test")
    print("=" * 50)

    # 初始化
    trainer = RLHFTrainer()

    # 模拟收集反馈
    print("\n[1] Collecting simulated feedback...")
    action_names = [
        "立即回复", "稍后回复", "主动发起话题", "分享生活",
        "询问近况", "表达关心", "给彼此空间", "寻求见面", "保持当前节奏"
    ]

    for i in range(100):
        # 随机状态
        state = np.random.rand(9) * 100

        # 随机两个动作
        action_a = random.randint(0, 8)
        action_b = random.randint(0, 8)
        while action_b == action_a:
            action_b = random.randint(0, 8)

        # 模拟偏好（基于简单规则）
        if state[7] > 70 and action_a == 7:  # simp高，给空间
            preference = 1.0
        elif state[7] > 70 and action_b == 7:
            preference = 0.0
        else:
            preference = random.choice([0.0, 0.5, 1.0])

        trainer.collect_feedback(state, action_a, action_b, preference)

    print(f"  Collected {len(trainer.preference_data.feedbacks)} feedbacks")

    # 训练奖励模型
    print("\n[2] Training reward model...")
    history = trainer.train_reward_model(epochs=10, batch_size=16)

    # 测试奖励模型
    print("\n[3] Testing reward model...")
    test_state = np.array([0.5, 0.5, 0.5, 10, 0.5, 0.5, 100, 70, 60])
    for action in range(9):
        reward = trainer.reward_model.get_reward(test_state, action)
        print(f"  {action_names[action]}: reward = {reward:.3f}")

    # 保存模型
    print("\n[4] Saving models...")
    trainer.save_models("models/rlhf_model.pth")
    print("  Models saved to models/rlhf_model.pth")

    print("\n" + "=" * 50)
    print("RLHF test complete!")
    print("=" * 50)


if __name__ == "__main__":
    test_rlhf()
