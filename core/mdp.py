"""
MDP定义 - 关系管理马尔可夫决策过程

State (状态空间): 关系当前状态
Action (动作空间): 可执行的行为建议
Reward (奖励函数): 行为质量评估
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class ActionType(Enum):
    """动作类型枚举"""
    REPLY_NOW = 0           # 立即回复
    WAIT_THEN_REPLY = 1     # 等待后回复
    SEND_MEME = 2           # 发表情包
    ASK_QUESTION = 3        # 提问引导
    SHARE_LIFE = 4          # 分享生活
    SUGGEST_MEETING = 5     # 约见面
    BE_SUPPORTIVE = 6       # 表达关心
    TAKE_SPACE = 7          # 给彼此空间
    NO_ACTION = 8           # 不建议


@dataclass
class RelationshipState:
    """
    关系状态向量

    维度说明：
    - reply_pending: 是否有未回复消息
    - pending_duration: 等待时长（秒）
    - recent_sent_ratio: 近期发送比例 (0-1)
    - last_cold_score: 最近冷淡程度 (0-100)
    - conversation_stage: 关系阶段 (0-6)
    - time_of_day: 时间段 (0-23)
    - message_importance: 消息重要度 (0-1)
    - simp_index: 主动指数 (0-100)
    - loved_index: 被爱指数 (0-100)
    """
    reply_pending: bool = False
    pending_duration: float = 0.0
    recent_sent_ratio: float = 0.5
    last_cold_score: float = 0.0
    conversation_stage: int = 0
    time_of_day: int = 12
    message_importance: float = 0.5
    simp_index: float = 50.0
    loved_index: float = 50.0

    def to_vector(self) -> np.ndarray:
        """转换为状态向量"""
        return np.array([
            float(self.reply_pending),
            self.pending_duration / 86400.0,  # 归一化到天
            self.recent_sent_ratio,
            self.last_cold_score / 100.0,
            self.conversation_stage / 6.0,
            self.time_of_day / 23.0,
            self.message_importance,
            self.simp_index / 100.0,
            self.loved_index / 100.0
        ], dtype=np.float32)

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> 'RelationshipState':
        """从向量恢复状态"""
        return cls(
            reply_pending=bool(vec[0] > 0.5),
            pending_duration=vec[1] * 86400.0,
            recent_sent_ratio=vec[2],
            last_cold_score=vec[3] * 100.0,
            conversation_stage=int(vec[4] * 6.0),
            time_of_day=int(vec[5] * 23.0),
            message_importance=vec[6],
            simp_index=vec[7] * 100.0,
            loved_index=vec[8] * 100.0
        )


@dataclass
class Action:
    """动作定义"""
    action_type: ActionType
    description: str
    parameters: Dict = field(default_factory=dict)

    # 动作元信息
    risk_level: float = 0.5  # 风险等级 (0-1)
    expected_outcome: str = ""
    prerequisites: List[str] = field(default_factory=list)


# 动作空间定义
ACTION_SPACE: Dict[ActionType, Action] = {
    ActionType.REPLY_NOW: Action(
        action_type=ActionType.REPLY_NOW,
        description="立即回复消息",
        risk_level=0.2,
        expected_outcome="保持对话活跃",
        prerequisites=["对方发了消息"]
    ),
    ActionType.WAIT_THEN_REPLY: Action(
        action_type=ActionType.WAIT_THEN_REPLY,
        description="等待一段时间再回复",
        risk_level=0.4,
        expected_outcome="降低主动感，观察对方反应",
        parameters={"wait_minutes": 30}
    ),
    ActionType.SEND_MEME: Action(
        action_type=ActionType.SEND_MEME,
        description="发表情包调节气氛",
        risk_level=0.1,
        expected_outcome="缓解紧张或增加趣味"
    ),
    ActionType.ASK_QUESTION: Action(
        action_type=ActionType.ASK_QUESTION,
        description="主动提问引导对话",
        risk_level=0.3,
        expected_outcome="延续话题，表达关注",
        parameters=["question_type"]
    ),
    ActionType.SHARE_LIFE: Action(
        action_type=ActionType.SHARE_LIFE,
        description="分享生活趣事",
        risk_level=0.3,
        expected_outcome="增加亲密度，展示生活"
    ),
    ActionType.SUGGEST_MEETING: Action(
        action_type=ActionType.SUGGEST_MEETING,
        description="约线下见面",
        risk_level=0.7,
        expected_outcome="推进关系，或可能被拒",
        prerequisites=["关系温度足够"]
    ),
    ActionType.BE_SUPPORTIVE: Action(
        action_type=ActionType.BE_SUPPORTIVE,
        description="表达关心和支持",
        risk_level=0.2,
        expected_outcome="增加信任感"
    ),
    ActionType.TAKE_SPACE: Action(
        action_type=ActionType.TAKE_SPACE,
        description="给彼此空间，暂不主动",
        risk_level=0.5,
        expected_outcome="降低压迫感，测试对方主动性"
    ),
    ActionType.NO_ACTION: Action(
        action_type=ActionType.NO_ACTION,
        description="暂时不建议行动",
        risk_level=0.0,
        expected_outcome="等待更好的时机"
    )
}


class RelationshipMDP:
    """
    关系管理MDP

    状态转移：基于行为和对方响应
    奖励计算：多维度综合评估
    """

    # 奖励权重
    REWARD_WEIGHTS = {
        'user_feedback': 10.0,      # 用户反馈
        'relationship_progress': 5.0, # 关系进展
        'timeliness': 2.0,          # 及时性
        'over_active_penalty': -3.0, # 过度主动惩罚
        'cold_recovery': 3.0,       # 冷战修复
        'them_initiated': 8.0       # 对方主动
    }

    def __init__(self, gamma: float = 0.95):
        self.gamma = gamma
        self.state_dim = 9
        self.action_dim = len(ActionType)

    def get_state(self, stats: Dict, context: Dict) -> RelationshipState:
        """
        从统计数据和上下文提取状态

        Args:
            stats: stats.json 的内容
            context: 当前对话上下文
        """
        return RelationshipState(
            reply_pending=context.get('reply_pending', False),
            pending_duration=context.get('pending_duration', 0.0),
            recent_sent_ratio=stats.get('basic', {}).get('my_ratio', 0.5),
            last_cold_score=stats.get('scores', {}).get('cold_index', 0.0),
            conversation_stage=self._predict_stage(stats),
            time_of_day=context.get('hour', 12),
            message_importance=context.get('message_importance', 0.5),
            simp_index=stats.get('scores', {}).get('simp_index', 50.0),
            loved_index=stats.get('scores', {}).get('loved_index', 50.0)
        )

    def _predict_stage(self, stats: Dict) -> int:
        """
        预测关系阶段

        0: 初识试探期
        1: 暧昧升温期
        2: 拉锯确认期
        3: 实名化前夜
        4: 正式确认期
        5: 关系维护期
        6: 降温衰退期
        """
        scores = stats.get('scores', {})
        simp = scores.get('simp_index', 50)
        loved = scores.get('loved_index', 50)
        cold = scores.get('cold_index', 0)

        # 简单规则判断
        if loved > 70 and simp < 60:
            return 4  # 正式确认
        elif loved > 50 and simp > 60:
            return 2  # 拉锯确认
        elif loved > 40 and cold < 30:
            return 1  # 暧昧升温
        elif cold > 50:
            return 6  # 降温衰退
        else:
            return 0  # 初识试探

    def compute_reward(self,
                       state: RelationshipState,
                       action: ActionType,
                       next_state: RelationshipState,
                       user_feedback: float = 0.0,
                       outcome: Optional[Dict] = None) -> float:
        """
        计算奖励信号

        Args:
            state: 当前状态
            action: 执行的动作
            next_state: 下一状态
            user_feedback: 用户显式反馈 (-1到+1)
            outcome: 结果信息
        """
        reward = 0.0
        outcome = outcome or {}

        # 1. 用户反馈奖励
        reward += user_feedback * self.REWARD_WEIGHTS['user_feedback']

        # 2. 关系进展奖励
        if next_state.loved_index > state.loved_index:
            reward += self.REWARD_WEIGHTS['relationship_progress']
        elif next_state.loved_index < state.loved_index:
            reward -= self.REWARD_WEIGHTS['relationship_progress']

        # 3. 及时性奖励
        if action == ActionType.REPLY_NOW:
            if state.pending_duration < 300:  # 5分钟内
                reward += self.REWARD_WEIGHTS['timeliness']

        # 4. 过度主动惩罚
        if action not in [ActionType.TAKE_SPACE, ActionType.NO_ACTION]:
            if state.recent_sent_ratio > 0.7:
                reward += self.REWARD_WEIGHTS['over_active_penalty']

        # 5. 冷战修复奖励
        if outcome.get('cold_war_ended', False):
            reward += self.REWARD_WEIGHTS['cold_recovery']

        # 6. 对方主动奖励（最强正向信号）
        if outcome.get('them_initiated', False):
            reward += self.REWARD_WEIGHTS['them_initiated']

        return reward

    def step(self,
             state: RelationshipState,
             action: ActionType,
             outcome: Dict) -> Tuple[RelationshipState, float, bool]:
        """
        执行一步转移

        Returns:
            next_state: 下一状态
            reward: 奖励
            done: 是否终止（关系结束或成功）
        """
        # 计算奖励
        reward = self.compute_reward(state, action, None,
                                     outcome.get('user_feedback', 0),
                                     outcome)

        # 状态转移（简化模型，实际应由真实数据驱动）
        next_state = self._transition(state, action, outcome)

        # 终止条件
        done = outcome.get('relationship_ended', False) or \
               outcome.get('relationship_success', False)

        return next_state, reward, done

    def _transition(self,
                    state: RelationshipState,
                    action: ActionType,
                    outcome: Dict) -> RelationshipState:
        """状态转移函数"""
        # 基本转移逻辑
        next_state = RelationshipState(
            reply_pending=outcome.get('reply_pending', False),
            pending_duration=outcome.get('pending_duration', 0.0),
            recent_sent_ratio=self._update_ratio(state, action),
            last_cold_score=outcome.get('cold_score', state.last_cold_score),
            conversation_stage=outcome.get('stage', state.conversation_stage),
            time_of_day=outcome.get('hour', state.time_of_day),
            message_importance=outcome.get('importance', 0.5),
            simp_index=outcome.get('simp_index', state.simp_index),
            loved_index=outcome.get('loved_index', state.loved_index)
        )
        return next_state

    def _update_ratio(self, state: RelationshipState, action: ActionType) -> float:
        """更新发送比例"""
        if action in [ActionType.REPLY_NOW, ActionType.ASK_QUESTION,
                      ActionType.SHARE_LIFE, ActionType.BE_SUPPORTIVE]:
            # 主动行为增加比例
            return min(state.recent_sent_ratio + 0.05, 1.0)
        elif action == ActionType.TAKE_SPACE:
            # 给空间降低比例
            return max(state.recent_sent_ratio - 0.1, 0.0)
        return state.recent_sent_ratio


# 导出
__all__ = [
    'ActionType', 'Action', 'ACTION_SPACE',
    'RelationshipState', 'RelationshipMDP'
]
