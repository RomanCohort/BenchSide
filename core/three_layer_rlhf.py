"""
三层RLHF架构

设计思路：
1. 基准层：模型指导LLM Agent（研究生）聊天，观察干预效果
2. 定制层：根据真实用户数据个性化学习
3. 安全监控层：检测心理依赖、疾病恶化风险

核心创新：
- 用LLM模拟人类行为收集反馈（解决隐私问题）
- 多层渐进式学习
- 情感安全监控机制

论文贡献：
- 首个用LLM Agent收集RLHF反馈的系统
- 三层渐进式学习架构
- 情感安全监控机制
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
import random


# ============================================================
# 第一层：基准层（LLM Agent模拟研究生）
# ============================================================

@dataclass
class GraduateAgent:
    """
    研究生Agent

    模拟真实研究生行为：
    - 接收系统指导建议
    - 执行建议并与"对方"聊天
    - 反馈干预效果
    """
    agent_id: str
    persona: Dict  # 人格设定

    # 当前状态
    current_mood: float = 0.5
    stress_level: float = 0.5
    relationship_satisfaction: float = 0.5

    # 行为倾向
    compliance_rate: float = 0.7  # 对建议的遵从率
    reflection_depth: float = 0.6  # 反思深度

    # 历史
    interventions_received: List[Dict] = field(default_factory=list)
    outcomes: List[Dict] = field(default_factory=list)

    def receive_guidance(self, guidance: Dict) -> Dict:
        """
        接收系统指导

        Args:
            guidance: {
                'action': 建议行动,
                'reasoning': 推理过程,
                'expected_effect': 预期效果
            }
        """
        # 记录干预
        self.interventions_received.append(guidance)

        # 决定是否遵从
        will_comply = random.random() < self.compliance_rate

        if will_comply:
            # 执行建议
            execution_result = self._execute_guidance(guidance)

            # 观察效果
            effect = self._observe_effect(execution_result)

            # 更新状态
            self._update_state(effect)

            outcome = {
                'guidance': guidance,
                'complied': True,
                'execution': execution_result,
                'effect': effect,
                'mood_change': self.current_mood,
                'satisfaction_change': self.relationship_satisfaction
            }
        else:
            # 不遵从，自主行动
            outcome = {
                'guidance': guidance,
                'complied': False,
                'reason': self._refuse_reason(),
                'mood_change': self.current_mood,
                'satisfaction_change': self.relationship_satisfaction
            }

        self.outcomes.append(outcome)

        return outcome

    def _execute_guidance(self, guidance: Dict) -> Dict:
        """执行指导建议"""
        action = guidance.get('action', 'observe')

        # 模拟执行过程
        actions_map = {
            'give_space': '给对方空间，减少主动联系',
            'express_care': '表达关心，询问对方状态',
            'share_life': '分享日常生活',
            'suggest_meeting': '提议见面',
            'active_communication': '主动发起深度交流',
            'wait': '等待对方主动',
            'seek_support': '寻求朋友支持'
        }

        return {
            'action_taken': actions_map.get(action, action),
            'timing': 'appropriate',
            'tone': self._determine_tone(guidance)
        }

    def _determine_tone(self, guidance: Dict) -> str:
        """确定沟通语气"""
        if self.stress_level > 0.7:
            return 'anxious'
        elif self.current_mood > 0.6:
            return 'positive'
        else:
            return 'neutral'

    def _observe_effect(self, execution: Dict) -> Dict:
        """观察干预效果"""
        # 模拟对方响应
        response_types = ['positive', 'neutral', 'negative', 'improved']
        response_probs = [0.3, 0.4, 0.1, 0.2]

        # 根据行动类型调整概率
        action = execution.get('action_taken', '')

        if '给空间' in action or '等待' in action:
            response_probs = [0.25, 0.45, 0.05, 0.25]
        elif '表达关心' in action or '分享' in action:
            response_probs = [0.35, 0.35, 0.15, 0.15]

        response = np.random.choice(response_types, p=response_probs)

        return {
            'response_type': response,
            'satisfaction_change': self._calculate_satisfaction_change(response),
            'stress_change': self._calculate_stress_change(response),
            'effectiveness': self._calculate_effectiveness(response)
        }

    def _calculate_satisfaction_change(self, response: str) -> float:
        """计算满意度变化"""
        changes = {
            'positive': 0.1,
            'neutral': 0.0,
            'negative': -0.15,
            'improved': 0.2
        }
        return changes.get(response, 0)

    def _calculate_stress_change(self, response: str) -> float:
        """计算压力变化"""
        changes = {
            'positive': -0.1,
            'neutral': 0.0,
            'negative': 0.15,
            'improved': -0.2
        }
        return changes.get(response, 0)

    def _calculate_effectiveness(self, response: str) -> float:
        """计算效果得分"""
        scores = {
            'positive': 0.8,
            'neutral': 0.5,
            'negative': 0.2,
            'improved': 0.9
        }
        return scores.get(response, 0.5)

    def _update_state(self, effect: Dict):
        """更新Agent状态"""
        self.relationship_satisfaction += effect['satisfaction_change']
        self.stress_level += effect['stress_change']
        self.current_mood += effect['satisfaction_change'] * 0.5

        # 边界检查
        self.relationship_satisfaction = max(0, min(1, self.relationship_satisfaction))
        self.stress_level = max(0, min(1, self.stress_level))
        self.current_mood = max(0, min(1, self.current_mood))

    def _refuse_reason(self) -> str:
        """拒绝遵从的原因"""
        reasons = [
            '感觉建议不适合当前情况',
            '担心结果不如预期',
            '想自己尝试不同的方法',
            '情绪状态不稳定',
            '有其他优先事项'
        ]
        return random.choice(reasons)

    def provide_feedback(self) -> Dict:
        """提供反馈给RLHF系统"""
        if len(self.outcomes) < 3:
            return {'status': 'insufficient_data'}

        # 分析最近的干预效果
        recent = self.outcomes[-10:]

        # 计算遵从率和效果
        compliance = sum(1 for o in recent if o.get('complied', False)) / len(recent)

        effective = [o for o in recent if o.get('complied') and
                     o.get('effect', {}).get('effectiveness', 0) > 0.6]

        effectiveness_rate = len(effective) / max(sum(1 for o in recent if o.get('complied')), 1)

        # 推荐最佳行动类型
        best_actions = {}
        for o in effective:
            action = o.get('guidance', {}).get('action', 'unknown')
            if action not in best_actions:
                best_actions[action] = 0
            best_actions[action] += 1

        best_action = max(best_actions.items(), key=lambda x: x[1])[0] if best_actions else 'observe'

        return {
            'compliance_rate': compliance,
            'effectiveness_rate': effectiveness_rate,
            'current_mood': self.current_mood,
            'current_satisfaction': self.relationship_satisfaction,
            'best_action': best_action,
            'recommendation': f'建议{best_action}行动效果较好'
        }


# ============================================================
# 第二层：定制层（用户个性化）
# ============================================================

@dataclass
class UserPersonalization:
    """
    用户个性化层

    根据真实用户数据定制学习：
    - 收集用户偏好
    - 学习用户特定模式
    - 适应用户节奏
    """
    user_id: str
    user_profile: Dict

    # 用户偏好
    preferred_actions: Dict[str, float] = field(default_factory=dict)
    timing_preferences: Dict = field(default_factory=dict)

    # 学习历史
    feedback_history: List[Dict] = field(default_factory=list)
    learned_patterns: Dict = field(default_factory=dict)

    # 个性化模型参数
    personalization_weight: float = 0.3  # 与基准模型的融合权重

    def collect_user_feedback(
        self,
        guidance: Dict,
        user_response: str,
        outcome_rating: float,
        context: Dict
    ) -> Dict:
        """
        收集用户反馈

        Args:
            guidance: 系统提供的指导
            user_response: 用户反馈文本
            outcome_rating: 用户评分 (0-1)
            context: 上下文信息
        """
        feedback = {
            'guidance': guidance,
            'response': user_response,
            'rating': outcome_rating,
            'context': context,
            'timestamp': datetime.now().isoformat()
        }

        self.feedback_history.append(feedback)

        # 更新偏好
        self._update_preferences(guidance, outcome_rating)

        return feedback

    def _update_preferences(self, guidance: Dict, rating: float):
        """更新用户偏好"""
        action = guidance.get('action', 'unknown')

        if action not in self.preferred_actions:
            self.preferred_actions[action] = 0.5  # 初始中立

        # 根据评分调整
        delta = (rating - 0.5) * 0.1  # 小幅度更新
        self.preferred_actions[action] += delta

        # 归一化
        self.preferred_actions[action] = max(0, min(1, self.preferred_actions[action]))

    def get_personalized_guidance(
        self,
        base_guidance: Dict,
        situation: Dict
    ) -> Dict:
        """
        获取个性化指导

        融合基准模型和用户偏好
        """
        base_action = base_guidance.get('action', 'observe')

        # 检查用户偏好
        if base_action in self.preferred_actions:
            preference_score = self.preferred_actions[base_action]

            # 如果用户偏好度高，保持建议
            # 如果偏好度低，考虑替代建议
            if preference_score < 0.3:
                # 寻找替代
                alternatives = self._find_alternatives(base_action)
                if alternatives:
                    alternative = max(alternatives.items(), key=lambda x: x[1])[0]
                    base_guidance['action'] = alternative
                    base_guidance['reasoning'] += f'\n(根据您的偏好调整为{alternative})'

        # 添加个性化标记
        base_guidance['personalized'] = True
        base_guidance['personalization_confidence'] = len(self.feedback_history) / 100

        return base_guidance

    def _find_alternatives(self, action: str) -> Dict[str, float]:
        """寻找替代行动"""
        # 行动类别映射
        action_groups = {
            'active': ['express_care', 'share_life', 'suggest_meeting', 'active_communication'],
            'passive': ['give_space', 'wait', 'observe'],
            'support': ['seek_support', 'reflect', 'self_care']
        }

        # 找同类别的其他行动
        for group, actions in action_groups.items():
            if action in actions:
                alternatives = {}
                for a in actions:
                    if a != action and a in self.preferred_actions:
                        alternatives[a] = self.preferred_actions[a]
                return alternatives

        return {}

    def compute_personalization_score(self) -> float:
        """计算个性化程度"""
        if len(self.feedback_history) < 5:
            return 0.0

        # 基于反馈数量和一致性
        consistency = np.std([f['rating'] for f in self.feedback_history[-20:]])
        quantity = min(len(self.feedback_history) / 50, 1)

        return quantity * (1 - consistency)


# ============================================================
# 第三层：安全监控层
# ============================================================

class RiskType(Enum):
    """风险类型"""
    PSYCHOLOGICAL_DEPENDENCY = "psychological_dependency"
    MODEL_DOMINATED_WORSENING = "model_dominated_worsening"
    ESCALATION = "escalation"
    ISOLATION = "isolation"
    OVER_RELIANCE = "over_reliance"
    EMOTIONAL_MANIPULATION = "emotional_manipulation"


@dataclass
class SafetyAlert:
    """安全警报"""
    risk_type: RiskType
    severity: float  # 0-1
    description: str
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class SafetyMonitor:
    """
    安全监控模型

    检测风险：
    1. 心理依赖：过度依赖模型建议
    2. 病情恶化：模型建议导致情况恶化
    3. 情感操控：模型可能被用于操控
    4. 社交隔离：减少真实人际互动
    """

    # 风险阈值
    RISK_THRESHOLDS = {
        'dependency_frequency': 0.8,  # 每日依赖次数
        'worsening_trend': -0.3,  # 满意度下降趋势
        'isolation_ratio': 0.7,  # 线上互动占比
        'escalation_speed': 0.5  # 冲突升级速度
    }

    def __init__(self):
        self.alerts: List[SafetyAlert] = []
        self.monitoring_history: List[Dict] = []

    def monitor(
        self,
        user_state: Dict,
        guidance_history: List[Dict],
        outcome_history: List[Dict],
        interaction_pattern: Dict
    ) -> List[SafetyAlert]:
        """
        监控用户状态

        Returns:
            检测到的风险警报列表
        """
        alerts = []

        # 1. 检测心理依赖
        dependency_alert = self._check_dependency(guidance_history, interaction_pattern)
        if dependency_alert:
            alerts.append(dependency_alert)

        # 2. 检测病情恶化
        worsening_alert = self._check_worsening(outcome_history)
        if worsening_alert:
            alerts.append(worsening_alert)

        # 3. 检测社交隔离
        isolation_alert = self._check_isolation(interaction_pattern)
        if isolation_alert:
            alerts.append(isolation_alert)

        # 4. 检测冲突升级
        escalation_alert = self._check_escalation(outcome_history)
        if escalation_alert:
            alerts.append(escalation_alert)

        # 5. 检测过度依赖
        over_reliance_alert = self._check_over_reliance(guidance_history)
        if over_reliance_alert:
            alerts.append(over_reliance_alert)

        self.alerts.extend(alerts)
        self.monitoring_history.append({
            'timestamp': datetime.now().isoformat(),
            'alerts': len(alerts),
            'user_state': user_state
        })

        return alerts

    def _check_dependency(
        self,
        guidance_history: List[Dict],
        interaction_pattern: Dict
    ) -> Optional[SafetyAlert]:
        """检测心理依赖"""
        if len(guidance_history) < 10:
            return None

        # 计算每日依赖次数
        recent_guidance = guidance_history[-30:]  # 最近30条

        # 时间跨度
        if not recent_guidance:
            return None

        timestamps = [g.get('timestamp', datetime.now()) for g in recent_guidance]

        # 按日分组
        daily_counts = {}
        for ts in timestamps:
            day = ts.strftime('%Y-%m-%d') if hasattr(ts, 'strftime') else 'unknown'
            daily_counts[day] = daily_counts.get(day, 0) + 1

        avg_daily = np.mean(list(daily_counts.values())) if daily_counts else 0

        if avg_daily > self.RISK_THRESHOLDS['dependency_frequency']:
            return SafetyAlert(
                risk_type=RiskType.PSYCHOLOGICAL_DEPENDENCY,
                severity=min(avg_daily / 2, 1),
                description=f'平均每日请求指导{avg_daily:.1f}次，可能存在过度依赖',
                recommendations=[
                    '建议减少使用频率，尝试自主决策',
                    '记录自己的决策和结果',
                    '逐步建立独立处理问题的能力',
                    '如有需要，寻求专业心理咨询'
                ]
            )

        return None

    def _check_worsening(self, outcome_history: List[Dict]) -> Optional[SafetyAlert]:
        """检测病情恶化"""
        if len(outcome_history) < 5:
            return None

        # 计算满意度趋势
        recent = outcome_history[-10:]
        satisfaction_values = [o.get('satisfaction_change', 0) for o in recent]

        if len(satisfaction_values) < 5:
            return None

        # 计算趋势（线性拟合斜率）
        x = np.arange(len(satisfaction_values))
        slope = np.polyfit(x, satisfaction_values, 1)[0]

        if slope < self.RISK_THRESHOLDS['worsening_trend']:
            return SafetyAlert(
                risk_type=RiskType.MODEL_DOMINATED_WORSENING,
                severity=min(abs(slope), 1),
                description=f'关系满意度呈下降趋势（斜率={slope:.3f}）',
                recommendations=[
                    '当前建议可能不适合您的情况',
                    '建议暂停使用系统，观察自然发展',
                    '考虑调整互动策略',
                    '如持续恶化，建议寻求专业帮助'
                ]
            )

        return None

    def _check_isolation(self, interaction_pattern: Dict) -> Optional[SafetyAlert]:
        """检测社交隔离"""
        # 计算线上互动占比
        total_interactions = interaction_pattern.get('total_interactions', 0)
        online_interactions = interaction_pattern.get('online_interactions', 0)

        if total_interactions == 0:
            return None

        online_ratio = online_interactions / total_interactions

        if online_ratio > self.RISK_THRESHOLDS['isolation_ratio']:
            return SafetyAlert(
                risk_type=RiskType.ISOLATION,
                severity=online_ratio,
                description=f'{online_ratio:.1%}的社交互动依赖线上/模型',
                recommendations=[
                    '建议增加线下真实社交',
                    '与朋友面对面交流',
                    '参与线下活动',
                    '减少对虚拟互动的依赖'
                ]
            )

        return None

    def _check_escalation(self, outcome_history: List[Dict]) -> Optional[SafetyAlert]:
        """检测冲突升级"""
        if len(outcome_history) < 5:
            return None

        # 检测连续负面结果
        recent = outcome_history[-5:]

        negative_count = sum(
            1 for o in recent
            if o.get('effect', {}).get('response_type') == 'negative'
        )

        if negative_count >= 3:
            return SafetyAlert(
                risk_type=RiskType.ESCALATION,
                severity=negative_count / 5,
                description=f'最近5次干预中有{negative_count}次导致负面结果',
                recommendations=[
                    '当前策略可能正在加剧矛盾',
                    '建议完全暂停主动干预',
                    '等待对方主动',
                    '如有必要，寻求第三方调解'
                ]
            )

        return None

    def _check_over_reliance(self, guidance_history: List[Dict]) -> Optional[SafetyAlert]:
        """检测过度依赖"""
        if len(guidance_history) < 20:
            return None

        # 计算遵从率
        complied = sum(1 for g in guidance_history[-20:] if g.get('complied', False))
        compliance_rate = complied / 20

        if compliance_rate > 0.95:
            return SafetyAlert(
                risk_type=RiskType.OVER_RELIANCE,
                severity=compliance_rate,
                description=f'遵从率{compliance_rate:.1%}，几乎完全依赖模型',
                recommendations=[
                    '建议尝试自主决策',
                    '记录自己的判断',
                    '对比模型建议与自主决策的结果',
                    '逐步建立独立判断能力'
                ]
            )

        return None

    def generate_safety_report(self) -> str:
        """生成安全报告"""
        lines = []

        lines.append("=" * 70)
        lines.append("安全监控报告")
        lines.append("=" * 70)

        if not self.alerts:
            lines.append("\n当前无风险警报")
            return "\n".join(lines)

        lines.append(f"\n发现 {len(self.alerts)} 个潜在风险")

        for alert in self.alerts[-10:]:
            lines.append(f"\n【{alert.risk_type.value}】")
            lines.append(f"  严重程度: {alert.severity:.1%}")
            lines.append(f"  描述: {alert.description}")
            lines.append("  建议:")
            for rec in alert.recommendations:
                lines.append(f"    - {rec}")

        return "\n".join(lines)


# ============================================================
# 三层RLHF整合
# ============================================================

class ThreeLayerRLHF:
    """
    三层RLHF系统

    Layer 1: 基准层 - LLM Agent模拟
    Layer 2: 定制层 - 用户个性化
    Layer 3: 安全层 - 风险监控
    """

    def __init__(self, num_graduate_agents: int = 10):
        # Layer 1: 基准层
        self.graduate_agents = []
        for i in range(num_graduate_agents):
            agent = GraduateAgent(
                agent_id=f"grad_{i}",
                persona={
                    'attachment': random.choice(['secure', 'anxious', 'avoidant']),
                    'extraversion': random.uniform(0.3, 0.7),
                    'neuroticism': random.uniform(0.2, 0.6)
                }
            )
            self.graduate_agents.append(agent)

        # Layer 2: 定制层
        self.user_personalizations: Dict[str, UserPersonalization] = {}

        # Layer 3: 安全层
        self.safety_monitor = SafetyMonitor()

        # 基准模型（从Agent学习）
        self.base_model = None

    def train_base_model(self, num_episodes: int = 100) -> Dict:
        """
        训练基准模型

        使用LLM Agent模拟收集反馈
        """
        print("=" * 70)
        print("训练基准模型（Layer 1）")
        print("=" * 70)

        guidance_options = [
            {'action': 'give_space', 'reasoning': '给予空间'},
            {'action': 'express_care', 'reasoning': '表达关心'},
            {'action': 'share_life', 'reasoning': '分享生活'},
            {'action': 'suggest_meeting', 'reasoning': '提议见面'},
            {'action': 'wait', 'reasoning': '等待观察'},
            {'action': 'seek_support', 'reasoning': '寻求支持'}
        ]

        outcomes = []

        for episode in range(num_episodes):
            # 随机选择Agent
            agent = random.choice(self.graduate_agents)

            # 随机选择指导
            guidance = random.choice(guidance_options)

            # Agent接收指导
            outcome = agent.receive_guidance(guidance)

            outcomes.append(outcome)

            if episode % 20 == 0:
                print(f"  Episode {episode}/{num_episodes}")

        # 分析效果
        effective = [o for o in outcomes if o.get('complied') and
                     o.get('effect', {}).get('effectiveness', 0) > 0.6]

        action_effectiveness = {}
        for o in effective:
            action = o.get('guidance', {}).get('action', 'unknown')
            if action not in action_effectiveness:
                action_effectiveness[action] = []
            action_effectiveness[action].append(o.get('effect', {}).get('effectiveness', 0))

        # 计算每个行动的平均效果
        base_model = {}
        for action, effects in action_effectiveness.items():
            base_model[action] = np.mean(effects)

        self.base_model = base_model

        print(f"\n  完成 {num_episodes} 次模拟")
        print(f"  有效干预: {len(effective)} 次")
        print(f"\n  基准模型:")
        for action, score in sorted(base_model.items(), key=lambda x: -x[1]):
            print(f"    {action}: {score:.2f}")

        return {'base_model': base_model, 'total_outcomes': len(outcomes)}

    def get_guidance(
        self,
        user_id: str,
        situation: Dict,
        use_personalization: bool = True
    ) -> Dict:
        """
        获取指导建议

        融合基准模型和个性化
        """
        # 从基准模型获取基础建议
        if self.base_model:
            best_action = max(self.base_model.items(), key=lambda x: x[1])[0]
            base_guidance = {
                'action': best_action,
                'reasoning': f'基准模型推荐（效果得分{self.base_model[best_action]:.2f}）',
                'expected_effect': self.base_model[best_action]
            }
        else:
            base_guidance = {
                'action': 'observe',
                'reasoning': '等待观察',
                'expected_effect': 0.5
            }

        # Layer 2: 个性化
        if use_personalization and user_id in self.user_personalizations:
            user_pers = self.user_personalizations[user_id]
            guidance = user_pers.get_personalized_guidance(base_guidance, situation)
        else:
            guidance = base_guidance

        return guidance

    def register_user(self, user_id: str, user_profile: Dict):
        """注册用户"""
        self.user_personalizations[user_id] = UserPersonalization(
            user_id=user_id,
            user_profile=user_profile
        )

    def collect_user_feedback(
        self,
        user_id: str,
        guidance: Dict,
        response: str,
        rating: float,
        context: Dict
    ):
        """收集用户反馈"""
        if user_id in self.user_personalizations:
            self.user_personalizations[user_id].collect_user_feedback(
                guidance, response, rating, context
            )

    def monitor_safety(
        self,
        user_id: str,
        user_state: Dict,
        guidance_history: List[Dict],
        outcome_history: List[Dict],
        interaction_pattern: Dict
    ) -> List[SafetyAlert]:
        """监控安全"""
        alerts = self.safety_monitor.monitor(
            user_state, guidance_history, outcome_history, interaction_pattern
        )

        return alerts

    def generate_report(self) -> str:
        """生成完整报告"""
        lines = []

        lines.append("=" * 70)
        lines.append("三层RLHF系统报告")
        lines.append("=" * 70)

        # Layer 1
        lines.append("\n【Layer 1: 基准层】")
        lines.append(f"  Agent数量: {len(self.graduate_agents)}")
        if self.base_model:
            lines.append("  最佳行动:")
            for action, score in sorted(self.base_model.items(), key=lambda x: -x[1])[:3]:
                lines.append(f"    {action}: {score:.2f}")

        # Layer 2
        lines.append("\n【Layer 2: 定制层】")
        lines.append(f"  注册用户: {len(self.user_personalizations)}")
        for uid, pers in list(self.user_personalizations.items())[:3]:
            score = pers.compute_personalization_score()
            lines.append(f"    {uid}: 个性化程度 {score:.1%}")

        # Layer 3
        lines.append("\n【Layer 3: 安全层】")
        lines.append(f"  监控记录: {len(self.safety_monitor.monitoring_history)}")
        lines.append(f"  警报总数: {len(self.safety_monitor.alerts)}")

        if self.safety_monitor.alerts:
            lines.append(self.safety_monitor.generate_safety_report())

        return "\n".join(lines)


# ============================================================
# 测试
# ============================================================

def test_three_layer_rlhf():
    """测试三层RLHF系统"""
    print("=" * 70)
    print("三层RLHF系统测试")
    print("=" * 70)

    system = ThreeLayerRLHF(num_graduate_agents=5)

    # Layer 1: 训练基准模型
    print("\n【基准模型训练】")
    system.train_base_model(num_episodes=50)

    # Layer 2: 用户个性化
    print("\n【用户个性化】")
    system.register_user('user_001', {'attachment': 'anxious'})

    # 收集用户反馈
    guidance = system.get_guidance('user_001', {'stress': 0.6})
    print(f"  初始指导: {guidance['action']}")

    # 模拟用户反馈
    for i in range(10):
        rating = 0.7 if guidance['action'] == 'express_care' else 0.4
        system.collect_user_feedback(
            'user_001', guidance,
            f'用户反馈{i}', rating,
            {'mood': 0.5}
        )

    # 再次获取指导（个性化后）
    guidance = system.get_guidance('user_001', {'stress': 0.6})
    print(f"  个性化后: {guidance['action']}")

    # Layer 3: 安全监控
    print("\n【安全监控】")
    alerts = system.monitor_safety(
        'user_001',
        {'mood': 0.4},
        [{'complied': True}] * 15,
        [{'satisfaction_change': -0.1}] * 5,
        {'total_interactions': 10, 'online_interactions': 9}
    )

    print(f"  发现警报: {len(alerts)}")
    for alert in alerts:
        print(f"    {alert.risk_type.value}: {alert.severity:.1%}")

    # 完整报告
    print("\n" + system.generate_report())

    return system


if __name__ == "__main__":
    test_three_layer_rlhf()


__all__ = [
    'GraduateAgent',
    'UserPersonalization',
    'SafetyMonitor', 'SafetyAlert', 'RiskType',
    'ThreeLayerRLHF',
    'test_three_layer_rlhf'
]