"""
RL智能体 - 整合DQN和反思推理
"""
import torch
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from .mdp import ActionType, RelationshipState, RelationshipMDP, ACTION_SPACE
from .dqn import DQNAgent
from .reflection import ReflectionReasoner, ReflectionResult


@dataclass
class AgentDecision:
    """智能体决策结果"""
    action: ActionType
    action_description: str
    q_value: float
    reflection: ReflectionResult
    final_confidence: float
    reasoning_trace: str


class ReflectiveRLAgent:
    """
    反思强化学习智能体

    结合：
    1. DQN 用于动作价值估计
    2. 反思推理器 用于深度思考
    3. 经验回放 用于学习历史

    决策流程：
    1. DQN计算Q值
    2. 反思推理生成建议
    3. 融合两者做决策
    4. 记录反馈学习
    """

    def __init__(self,
                 dqn_agent: DQNAgent,
                 reflection_reasoner: ReflectionReasoner,
                 mdp: RelationshipMDP,
                 reflection_weight: float = 0.3):
        """
        Args:
            dqn_agent: DQN智能体
            reflection_reasoner: 反思推理器
            mdp: MDP环境
            reflection_weight: 反思结果权重 (0-1)
        """
        self.dqn = dqn_agent
        self.reflector = reflection_reasoner
        self.mdp = mdp
        self.reflection_weight = reflection_weight

        # 历史反思缓存
        self.reflection_history: List[np.ndarray] = []

    def decide(self,
               state: RelationshipState,
               chat_context: str,
               stats: Dict,
               training: bool = True) -> AgentDecision:
        """
        做出决策

        Args:
            state: 当前状态
            chat_context: 聊天上下文
            stats: 统计数据
            training: 是否训练模式

        Returns:
            决策结果
        """
        # 1. DQN计算Q值
        reflection_context = self._get_reflection_context()
        q_values = self._compute_q_values(state, reflection_context)

        # 2. 反思推理
        state_dict = self._state_to_dict(state)
        reflection = self.reflector.reflect(state_dict, chat_context, stats)

        # 3. 融合决策
        action, final_confidence = self._fuse_decision(
            q_values, reflection, training
        )

        # 4. 获取动作描述
        action_desc = ACTION_SPACE[action].description

        # 5. 构建推理链
        reasoning_trace = self._build_reasoning_trace(
            q_values, reflection, action
        )

        return AgentDecision(
            action=action,
            action_description=action_desc,
            q_value=q_values[action.value].item(),
            reflection=reflection,
            final_confidence=final_confidence,
            reasoning_trace=reasoning_trace
        )

    def _compute_q_values(self,
                          state: RelationshipState,
                          reflection_context: Optional[torch.Tensor]) -> torch.Tensor:
        """计算Q值"""
        state_vec = torch.from_numpy(state.to_vector()).unsqueeze(0)
        state_vec = state_vec.to(self.dqn.device)

        with torch.no_grad():
            q_values = self.dqn.q_network(state_vec, reflection_context)

        return q_values.squeeze(0)

    def _get_reflection_context(self, max_len: int = 10) -> Optional[torch.Tensor]:
        """获取反思上下文"""
        if not self.reflection_history:
            return None

        # 取最近max_len个反思
        recent = self.reflection_history[-max_len:]

        # 转换为tensor
        context = torch.from_numpy(
            np.stack(recent)
        ).unsqueeze(0).float().to(self.dqn.device)

        return context

    def _fuse_decision(self,
                        q_values: torch.Tensor,
                        reflection: ReflectionResult,
                        training: bool) -> Tuple[ActionType, float]:
        """
        融合DQN和反思的决策

        Q值决定：基础决策
        反思调整：根据反思置信度调整
        """
        # 从反思建议映射动作
        action_map = {
            "立即回复": ActionType.REPLY_NOW,
            "等待": ActionType.WAIT_THEN_REPLY,
            "表情": ActionType.SEND_MEME,
            "提问": ActionType.ASK_QUESTION,
            "分享": ActionType.SHARE_LIFE,
            "约见面": ActionType.SUGGEST_MEETING,
            "关心": ActionType.BE_SUPPORTIVE,
            "空间": ActionType.TAKE_SPACE,
            "不行动": ActionType.NO_ACTION
        }

        # 找到反思建议的动作
        reflection_action = None
        for keyword, action in action_map.items():
            if keyword in reflection.recommended_action:
                reflection_action = action
                break

        if reflection_action is None:
            reflection_action = ActionType.NO_ACTION

        # 调整Q值
        adjusted_q = q_values.clone()

        if reflection.confidence > 0.7 and reflection_action:
            # 高置信度反思，增加对应动作的Q值
            boost = reflection.confidence * 2.0
            adjusted_q[reflection_action.value] += boost

        # 选择动作
        action_idx = adjusted_q.argmax().item()
        action = ActionType(action_idx)

        # 计算最终置信度
        q_confidence = torch.softmax(q_values, dim=-1)[action_idx].item()
        final_confidence = (
            (1 - self.reflection_weight) * q_confidence +
            self.reflection_weight * reflection.confidence
        )

        return action, final_confidence

    def _state_to_dict(self, state: RelationshipState) -> Dict:
        """状态转换为字典"""
        return {
            'simp_index': state.simp_index,
            'loved_index': state.loved_index,
            'cold_index': state.last_cold_score,
            'pending_duration': state.pending_duration,
            'recent_ratio': state.recent_sent_ratio,
            'conversation_stage': state.conversation_stage
        }

    def _build_reasoning_trace(self,
                                q_values: torch.Tensor,
                                reflection: ReflectionResult,
                                action: ActionType) -> str:
        """构建推理链"""
        # Q值排名
        q_ranking = torch.argsort(q_values, descending=True)
        top_actions = [
            f"{ACTION_SPACE[ActionType(idx.item())].description}: {q_values[idx.item()].item():.2f}"
            for idx in q_ranking[:3]
        ]

        trace = f"""
## 决策推理链

### DQN评估（Top 3）
{chr(10).join('- ' + a for a in top_actions)}

### 反思推理
{reflection.reasoning}

### 最终决策
选择: {ACTION_SPACE[action].description}
置信度: {reflection.confidence:.2%}

### 风险提示
{reflection.warning if reflection.warning else '无明显风险'}
"""
        return trace.strip()

    def learn(self, batch_size: int = 32) -> Optional[float]:
        """学习"""
        return self.dqn.learn(batch_size)

    def store_transition(self,
                         state: RelationshipState,
                         action: ActionType,
                         reward: float,
                         next_state: RelationshipState,
                         done: bool):
        """存储转移"""
        self.dqn.store_transition(
            state.to_vector(),
            action.value,
            reward,
            next_state.to_vector(),
            done
        )

    def store_reflection(self, reflection: ReflectionResult):
        """存储反思到历史"""
        embedding = self.reflector.get_reflection_embedding(reflection)
        self.reflection_history.append(embedding)

        # 限制历史长度
        if len(self.reflection_history) > 100:
            self.reflection_history = self.reflection_history[-100:]

    def save(self, path: str):
        """保存模型"""
        self.dqn.save(path)

    def load(self, path: str):
        """加载模型"""
        self.dqn.load(path)


class AgentTrainer:
    """
    智能体训练器

    支持在线学习和离线训练
    """

    def __init__(self,
                 agent: ReflectiveRLAgent,
                 save_path: str = "./models"):
        self.agent = agent
        self.save_path = save_path
        self.episode_rewards = []

    def train_episode(self,
                      env_state: RelationshipState,
                      max_steps: int = 100) -> float:
        """
        训练一个episode

        Args:
            env_state: 初始状态
            max_steps: 最大步数

        Returns:
            总奖励
        """
        total_reward = 0.0
        state = env_state

        for step in range(max_steps):
            # 决策
            decision = self.agent.decide(state, "", {}, training=True)

            # 执行动作，获取环境反馈
            # 这里需要真实的交互数据或模拟器
            # 简化版本：使用MDP的转移函数

            # 学习
            loss = self.agent.learn()

            total_reward += decision.q_value

            if loss:
                print(f"Step {step}, Loss: {loss:.4f}")

        self.episode_rewards.append(total_reward)
        return total_reward

    def train_offline(self,
                      episodes: List[Dict],
                      epochs: int = 10,
                      batch_size: int = 32):
        """
        离线训练

        Args:
            episodes: 历史交互数据
            epochs: 训练轮数
            batch_size: 批次大小
        """
        # 预填充经验回放
        for episode in episodes:
            for transition in episode.get('transitions', []):
                self.agent.store_transition(
                    RelationshipState(**transition['state']),
                    ActionType(transition['action']),
                    transition['reward'],
                    RelationshipState(**transition['next_state']),
                    transition.get('done', False)
                )

        # 训练
        for epoch in range(epochs):
            loss = self.agent.learn(batch_size)
            if loss:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

        # 保存
        self.agent.save(f"{self.save_path}/agent.pth")


def create_agent(config: Optional[Dict] = None) -> ReflectiveRLAgent:
    """
    工厂函数：创建智能体

    Args:
        config: 配置字典

    Returns:
        配置好的智能体
    """
    config = config or {}

    # 创建MDP
    mdp = RelationshipMDP(gamma=config.get('gamma', 0.95))

    # 创建DQN
    dqn = DQNAgent(
        state_dim=9,
        action_dim=9,
        hidden_dim=config.get('hidden_dim', 128),
        lr=config.get('lr', 1e-4),
        gamma=config.get('gamma', 0.95),
        epsilon_start=config.get('epsilon_start', 1.0),
        epsilon_end=config.get('epsilon_end', 0.05),
        device=config.get('device', 'auto')
    )

    # 创建反思推理器
    reflector = ReflectionReasoner(
        llm_client=config.get('llm_client')
    )

    # 创建智能体
    agent = ReflectiveRLAgent(
        dqn_agent=dqn,
        reflection_reasoner=reflector,
        mdp=mdp,
        reflection_weight=config.get('reflection_weight', 0.3)
    )

    return agent


__all__ = [
    'ReflectiveRLAgent', 'AgentDecision', 'AgentTrainer',
    'create_agent'
]
