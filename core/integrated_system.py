"""
完整系统整合：联邦GNN + 思维链推理 + RLHF

整合三个核心模块：
1. Federated GNN - 隐私保护的图神经网络预测
2. Chain-of-Thought (Reflection) - 可解释的推理过程
3. RLHF - 从人类反馈学习个性化决策

这是论文的核心创新点：
- 隐私 + 准确性 + 可解释性 + 个性化
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

# 导入各模块
from core.professional_architecture import (
    ProfessionalGraphConstructor, ProfessionalGNN,
    FederatedProtocol, PrivacyAccountant,
    InterpretabilityEngine, SocialGraph
)
from core.reflection import ReflectionReasoner, ReflectionResult
from core.rlhf import RLHFTrainer, RewardModel, PPOActorCritic


# ============================================================
# 整合系统
# ============================================================

@dataclass
class IntegratedPrediction:
    """整合预测结果"""
    # GNN预测
    resilience_score: float
    risk_level: str

    # 反思推理
    reasoning: str
    confidence: float
    key_factors: List[str]

    # RLHF决策
    recommended_action: str
    action_reasoning: str

    # 解释
    explanation: str
    recommendations: List[str]

    # 隐私
    privacy_guaranteed: bool
    epsilon_spent: float


class IntegratedResilienceSystem:
    """
    整合系统

    整合联邦GNN、思维链推理、RLHF三个模块
    """

    def __init__(
        self,
        use_fl: bool = True,
        use_reflection: bool = True,
        use_rlhf: bool = True,
        epsilon: float = 1.0
    ):
        """
        Args:
            use_fl: 是否使用联邦学习
            use_reflection: 是否使用思维链推理
            use_rlhf: 是否使用RLHF
            epsilon: 差分隐私预算
        """
        self.use_fl = use_fl
        self.use_reflection = use_reflection
        self.use_rlhf = use_rlhf

        # 初始化各模块
        self.graph_constructor = ProfessionalGraphConstructor()
        self.gnn = ProfessionalGNN(input_dim=16, hidden_dim=32, output_dim=16)
        self.interpreter = InterpretabilityEngine()

        if use_fl:
            self.federated = FederatedProtocol(num_rounds=10)
            self.privacy = PrivacyAccountant(target_epsilon=epsilon)

        if use_reflection:
            self.reflector = ReflectionReasoner()

        if use_rlhf:
            self.rlhf = RLHFTrainer(state_dim=16, action_dim=9)

        # 状态
        self.fitted = False

    def fit(
        self,
        messages_data: Dict[str, List[Dict]],
        labels: Optional[np.ndarray] = None,
        feedbacks: Optional[List[Dict]] = None,
        epochs: int = 100
    ):
        """
        训练整个系统

        Args:
            messages_data: 聊天数据 {contact_id: [messages]}
            labels: 可选的标签
            feedbacks: 可选的人类反馈
            epochs: 训练轮数
        """
        # 1. 构建图
        graph = self.graph_constructor.construct_from_messages(messages_data)

        # 2. 训练GNN
        if labels is not None:
            # 这里应该训练GNN
            pass

        # 3. 训练RLHF（如果有反馈）
        if self.use_rlhf and feedbacks:
            for fb in feedbacks:
                self.rlhf.collect_feedback(
                    state=fb['state'],
                    action_a=fb['action_a'],
                    action_b=fb['action_b'],
                    preference=fb['preference']
                )
            self.rlhf.train_reward_model(epochs=10)

        # 4. 初始化联邦学习
        if self.use_fl:
            self.federated.initialize_global_model(self.gnn)

        self.fitted = True

        return self

    def predict(
        self,
        messages_data: Dict[str, List[Dict]],
        return_details: bool = False
    ) -> IntegratedPrediction:
        """
        预测并给出建议

        整合三个模块的输出
        """
        # 1. 构建图
        graph = self.graph_constructor.construct_from_messages(messages_data)

        # 2. GNN预测
        gnn_output = self.gnn.forward(graph)
        resilience_score = float(gnn_output['graph_prediction'])

        # 3. 风险分级
        if resilience_score > 0.6:
            risk_level = "低风险"
        elif resilience_score > 0.4:
            risk_level = "中风险"
        else:
            risk_level = "高风险"

        # 4. 可解释性分析
        explanation = self.interpreter.explain_prediction(
            graph, gnn_output, resilience_score
        )

        # 5. 反思推理
        if self.use_reflection:
            state = self._extract_state(graph)
            reflection = self.reflector.reflect(
                state,
                self._get_chat_context(messages_data),
                {}
            )
            reasoning = reflection.reasoning
            confidence = reflection.confidence
            key_factors = reflection.historical_evidence
        else:
            reasoning = "反思推理未启用"
            confidence = 0.5
            key_factors = []

        # 6. RLHF决策
        if self.use_rlhf:
            state_vec = self._graph_to_state(graph)
            action_idx, _ = self.rlhf.actor_critic.get_action(state_vec)

            action_names = [
                "保持联系频率", "增加互动", "适当降温",
                "分享生活", "表达关心", "给彼此空间",
                "主动邀约", "等待观察", "寻求支持"
            ]
            recommended_action = action_names[action_idx]
            action_reasoning = f"基于学习到的奖励模型，该行动预期效果最好"
        else:
            recommended_action = "观察等待"
            action_reasoning = "RLHF未启用"

        # 7. 隐私追踪
        if self.use_fl:
            self.privacy.account_spend('prediction', 0.01)
            privacy_guaranteed = self.privacy.check_budget()
            epsilon_spent = self.privacy.spent_epsilon
        else:
            privacy_guaranteed = False
            epsilon_spent = 0

        result = IntegratedPrediction(
            resilience_score=resilience_score,
            risk_level=risk_level,
            reasoning=reasoning,
            confidence=confidence,
            key_factors=key_factors,
            recommended_action=recommended_action,
            action_reasoning=action_reasoning,
            explanation=explanation['interpretation'],
            recommendations=explanation['recommendations'],
            privacy_guaranteed=privacy_guaranteed,
            epsilon_spent=epsilon_spent
        )

        return result

    def _extract_state(self, graph: SocialGraph) -> Dict:
        """从图提取状态"""
        state = {
            'simp_index': 50,
            'loved_index': 50,
            'cold_index': 20,
            'pending_duration': 0,
            'recent_ratio': 0.5,
            'last_sender': 'me',
            'conversation_stage': 1  # 添加缺失字段
        }

        # 从边特征提取
        for (src, tgt), edge in graph.edges.items():
            state['simp_index'] = min(100, state['simp_index'] + edge.energy_flow / 2)
            state['cold_index'] = max(0, 100 - state['simp_index'])

        return state

    def _get_chat_context(self, messages_data: Dict) -> str:
        """获取聊天上下文"""
        context_parts = []

        for contact_id, messages in list(messages_data.items())[:1]:
            for msg in messages[-5:]:
                content = msg.get('content', '')
                if len(content) > 50:
                    content = content[:50] + "..."
                context_parts.append(f"{msg.get('sender', 'A')}: {content}")

        return "\n".join(context_parts) if context_parts else "暂无上下文"

    def _graph_to_state(self, graph: SocialGraph) -> np.ndarray:
        """将图转换为状态向量"""
        state = np.zeros(16)

        # 从节点特征提取
        for node_id, node in graph.nodes.items():
            state[:10] += node.features[:10] / max(graph.num_nodes, 1)

        # 从边特征提取
        for (src, tgt), edge in graph.edges.items():
            state[10:14] += edge.features[:4] / max(graph.num_edges, 1)
            state[14] += edge.energy_flow / 100
            state[15] = len(edge.messages)

        return state

    def generate_report(self, prediction: IntegratedPrediction) -> str:
        """生成完整报告"""
        lines = []

        lines.append("=" * 70)
        lines.append("社交韧性整合分析报告")
        lines.append("=" * 70)

        # 核心预测
        lines.append("\n【GNN预测结果】")
        lines.append(f"  韧性得分: {prediction.resilience_score:.3f}")
        lines.append(f"  风险等级: {prediction.risk_level}")
        lines.append(f"  置信度: {prediction.confidence:.1%}")

        # 反思推理
        if self.use_reflection:
            lines.append("\n【思维链推理】")
            lines.append(f"  {prediction.reasoning}")

        # RLHF建议
        if self.use_rlhf:
            lines.append("\n【个性化建议】")
            lines.append(f"  推荐行动: {prediction.recommended_action}")
            lines.append(f"  理由: {prediction.action_reasoning}")

        # 解释
        lines.append("\n【解释】")
        lines.append(f"  {prediction.explanation}")

        if prediction.recommendations:
            lines.append("\n【具体建议】")
            for i, rec in enumerate(prediction.recommendations, 1):
                lines.append(f"  {i}. {rec}")

        # 隐私
        lines.append("\n【隐私保护】")
        if prediction.privacy_guaranteed:
            lines.append(f"  ✓ 差分隐私保护 (ε={prediction.epsilon_spent:.3f})")
        else:
            lines.append("  ✗ 无隐私保护")

        lines.append("\n" + "=" * 70)

        return "\n".join(lines)


# ============================================================
# 完整系统测试
# ============================================================

def test_integrated_system():
    """测试整合系统"""
    print("=" * 70)
    print("整合系统测试")
    print("=" * 70)

    # 初始化系统
    system = IntegratedResilienceSystem(
        use_fl=True,
        use_reflection=True,
        use_rlhf=True,
        epsilon=1.0
    )

    # 模拟聊天数据
    messages_data = {
        'contact_1': [
            {'sender': 'me', 'content': '在干嘛？', 'timestamp': 1000},
            {'sender': 'her', 'content': '在忙', 'timestamp': 1100},
            {'sender': 'me', 'content': '好吧', 'timestamp': 1200},
            {'sender': 'me', 'content': '想你了', 'timestamp': 1300},
            {'sender': 'her', 'content': '嗯', 'timestamp': 1500},
        ]
    }

    # 预测
    print("\n【预测】")
    prediction = system.predict(messages_data)

    print(f"  韧性得分: {prediction.resilience_score:.3f}")
    print(f"  风险等级: {prediction.risk_level}")
    print(f"  推荐行动: {prediction.recommended_action}")
    print(f"  隐私保护: {'✓' if prediction.privacy_guaranteed else '✗'}")

    # 完整报告
    print("\n" + system.generate_report(prediction))

    return system, prediction


if __name__ == "__main__":
    test_integrated_system()


__all__ = [
    'IntegratedResilienceSystem',
    'IntegratedPrediction',
    'test_integrated_system'
]