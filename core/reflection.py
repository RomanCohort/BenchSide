"""
反思推理器 - Reflection Reasoner

基于思维链激发深度思考的推理模块
"""
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np


@dataclass
class ReflectionResult:
    """反思推理结果"""
    recommended_action: str
    reasoning: str
    confidence: float
    alternative_actions: List[str] = field(default_factory=list)
    warning: str = ""
    risk_assessment: Dict[str, float] = field(default_factory=dict)
    historical_evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "recommended_action": self.recommended_action,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "alternative_actions": self.alternative_actions,
            "warning": self.warning,
            "risk_assessment": self.risk_assessment,
            "historical_evidence": self.historical_evidence
        }


@dataclass
class HistoricalCase:
    """历史案例"""
    timestamp: str
    state_summary: str
    action_taken: str
    outcome: str
    user_feedback: float  # -1 to 1
    effectiveness: float  # 0 to 1


class ReflectionReasoner:
    """
    反思推理器

    核心思想：
    1. 观察当前状态
    2. 回忆类似历史场景
    3. 分析行为后果
    4. 生成改进建议

    与RL Agent协同：
    - RL提供Q值估计
    - 反思提供语义推理
    - 两者融合做决策
    """

    REFLECTION_PROMPT = """
## 当前场景分析

### 对话上下文
{chat_context}

### 当前状态指标
- 主动指数: {simp_index}/100
- 被爱指数: {loved_index}/100
- 冷淡指数: {cold_index}/100
- 等待时长: {pending_duration}
- 近期发送比例: {recent_ratio}%

### 历史类似场景
{similar_cases}

### 对方行为模式
{their_pattern}

## 反思推理

### Step 1: 状态诊断
请分析当前关系状态：
1. 她现在的态度是什么？（基于回复速度、消息长度、冷淡词）
2. 你最近的行为模式是什么？（是否太主动/太被动）
3. 是否存在追逃循环？（你追她逃/她追你逃）

### Step 2: 历史对比
回顾类似场景，分析：
1. 之前类似情况下发生了什么？
2. 你的行为导致了什么结果？
3. 有什么可以改进的？

### Step 3: 行动规划
基于以上分析：
1. 现在应该做什么？
2. 不应该做什么？
3. 预期结果是什么？

### Step 4: 风险评估
1. 这个行动的风险是什么？
2. 最坏情况是什么？
3. 是否有更稳妥的方案？

## 最终建议

输出JSON格式：
{{
    "recommended_action": "具体行动",
    "reasoning": "推理过程（分步骤）",
    "confidence": 0.0-1.0,
    "alternative_actions": ["备选方案1", "备选方案2"],
    "warning": "注意事项",
    "risk_assessment": {{
        "rejection_risk": 0.0-1.0,
        "misunderstanding_risk": 0.0-1.0,
        "over_active_risk": 0.0-1.0
    }}
}}
"""

    # 反思规则库（硬编码规则，作为基础推理）
    REFLECTION_RULES = [
        {
            "condition": lambda s: s['simp_index'] > 70 and s['loved_index'] < 50,
            "diagnosis": "你太主动了，她可能感到压力",
            "suggestion": "给彼此空间，等待她主动",
            "action": "take_space",
            "confidence": 0.8
        },
        {
            "condition": lambda s: s['cold_index'] > 50 and s['pending_duration'] > 3600,
            "diagnosis": "她可能冷淡，需要测试她的态度",
            "suggestion": "不要追问，用轻松话题破冰",
            "action": "send_meme",
            "confidence": 0.7
        },
        {
            "condition": lambda s: s['loved_index'] > 60 and s['pending_duration'] < 300,
            "diagnosis": "关系良好，可以推进",
            "suggestion": "分享生活或约见面",
            "action": "share_life",
            "confidence": 0.75
        },
        {
            "condition": lambda s: s['recent_ratio'] > 80,
            "diagnosis": "你发言太多，需要平衡",
            "suggestion": "减少主动，让她多说",
            "action": "ask_question",
            "confidence": 0.7
        },
        {
            "condition": lambda s: s['pending_duration'] > 86400 and s['last_sender'] == 'me',
            "diagnosis": "超过24小时未回，可能有冷信号",
            "suggestion": "不要追加消息，等待她的响应",
            "action": "take_space",
            "confidence": 0.85
        },
        {
            "condition": lambda s: s['conversation_stage'] >= 3 and s['loved_index'] > 65,
            "diagnosis": "关系成熟，可以尝试推进",
            "suggestion": "约线下见面",
            "action": "suggest_meeting",
            "confidence": 0.8
        }
    ]

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: LLM客户端（可选，用于深度推理）
        """
        self.llm_client = llm_client
        self.history_buffer: List[HistoricalCase] = []

    def reflect(self,
                state: Dict,
                chat_context: str,
                stats: Dict) -> ReflectionResult:
        """
        执行反思推理

        Args:
            state: 当前状态字典
            chat_context: 聊天上下文摘要
            stats: 统计数据

        Returns:
            反思结果
        """
        # 1. 基于规则的快速推理
        rule_result = self._rule_based_reflection(state)

        # 2. 检索相似历史案例
        similar_cases = self._retrieve_similar_cases(state)

        # 3. 分析对方行为模式
        their_pattern = self._analyze_their_pattern(stats)

        # 4. 如果有LLM，进行深度推理
        if self.llm_client:
            llm_result = self._llm_reflection(
                state, chat_context, similar_cases, their_pattern
            )
            # 融合规则和LLM结果
            return self._merge_results(rule_result, llm_result)

        # 5. 无LLM时使用规则结果
        return self._build_result_from_rule(rule_result, similar_cases)

    def _rule_based_reflection(self, state: Dict) -> Dict:
        """基于规则的推理"""
        for rule in self.REFLECTION_RULES:
            if rule["condition"](state):
                return {
                    "action": rule["action"],
                    "diagnosis": rule["diagnosis"],
                    "suggestion": rule["suggestion"],
                    "confidence": rule["confidence"]
                }

        # 默认建议
        return {
            "action": "no_action",
            "diagnosis": "当前状态不明显，建议观察",
            "suggestion": "暂时不需要特别行动",
            "confidence": 0.5
        }

    def _retrieve_similar_cases(self, state: Dict, top_k: int = 3) -> List[HistoricalCase]:
        """检索相似历史案例"""
        if len(self.history_buffer) < 2:
            return []

        # 简单的相似度计算（状态向量距离）
        def similarity(case: HistoricalCase) -> float:
            # 基于状态的相似度
            score = 0.0
            if case.user_feedback > 0:
                score += 0.5
            if case.effectiveness > 0.5:
                score += 0.3
            return score

        # 排序并返回top_k
        sorted_cases = sorted(
            self.history_buffer,
            key=similarity,
            reverse=True
        )

        return sorted_cases[:top_k]

    def _analyze_their_pattern(self, stats: Dict) -> str:
        """分析对方行为模式"""
        patterns = []

        # 回复速度模式
        their_speed = stats.get('reply_speed', {}).get('their_avg_seconds', 0)
        if their_speed < 60:
            patterns.append("她回复很快，说明她在意")
        elif their_speed > 1800:
            patterns.append("她回复较慢，可能忙碌或不太在意")

        # 主动模式
        their_starts = stats.get('initiative', {}).get('their_starts', 0)
        total_starts = stats.get('initiative', {}).get('my_starts', 0) + their_starts
        if total_starts > 0:
            their_ratio = their_starts / total_starts
            if their_ratio > 0.4:
                patterns.append(f"她主动发起{their_ratio*100:.0f}%的对话，积极")
            elif their_ratio < 0.2:
                patterns.append(f"她几乎不主动发起对话（{their_ratio*100:.0f}%），被动")

        # 表情使用
        emoji = stats.get('message_types', {}).get('emoji', {})
        their_emoji = emoji.get('them', 0)
        if their_emoji > 10:
            patterns.append("她喜欢用表情包，性格活泼")

        return " | ".join(patterns) if patterns else "暂无明显模式"

    def _llm_reflection(self,
                        state: Dict,
                        chat_context: str,
                        similar_cases: List[HistoricalCase],
                        their_pattern: str) -> Dict:
        """使用LLM进行深度推理"""
        prompt = self.REFLECTION_PROMPT.format(
            chat_context=chat_context,
            simp_index=state.get('simp_index', 50),
            loved_index=state.get('loved_index', 50),
            cold_index=state.get('cold_index', 0),
            pending_duration=f"{state.get('pending_duration', 0)/60:.0f}分钟",
            recent_ratio=state.get('recent_ratio', 50) * 100,
            similar_cases=self._format_cases(similar_cases),
            their_pattern=their_pattern
        )

        # 调用LLM
        response = self.llm_client.generate(prompt)

        # 解析JSON
        try:
            result = json.loads(response)
            return result
        except:
            # 解析失败，返回默认
            return {
                "recommended_action": "observe",
                "reasoning": response,
                "confidence": 0.5
            }

    def _format_cases(self, cases: List[HistoricalCase]) -> str:
        """格式化历史案例"""
        if not cases:
            return "暂无类似历史案例"

        formatted = []
        for i, case in enumerate(cases, 1):
            formatted.append(
                f"案例{i}: {case.state_summary}\n"
                f"  行动: {case.action_taken}\n"
                f"  结果: {case.outcome}\n"
                f"  效果: {case.effectiveness*100:.0f}%"
            )

        return "\n\n".join(formatted)

    def _build_result_from_rule(self,
                                 rule_result: Dict,
                                 similar_cases: List[HistoricalCase]) -> ReflectionResult:
        """从规则结果构建ReflectionResult"""
        # 从历史案例提取证据
        evidence = []
        for case in similar_cases:
            if case.effectiveness > 0.5:
                evidence.append(f"历史经验: {case.action_taken} 效果良好")

        return ReflectionResult(
            recommended_action=rule_result["suggestion"],
            reasoning=f"诊断: {rule_result['diagnosis']}\n建议: {rule_result['suggestion']}",
            confidence=rule_result["confidence"],
            alternative_actions=["等待观察", "换种方式"],
            warning="此建议基于规则推理，建议结合实际情况判断",
            historical_evidence=evidence
        )

    def _merge_results(self,
                       rule_result: Dict,
                       llm_result: Dict) -> ReflectionResult:
        """融合规则和LLM结果"""
        # 如果LLM置信度高，优先使用LLM
        if llm_result.get('confidence', 0) > rule_result['confidence']:
            return ReflectionResult(
                recommended_action=llm_result.get('recommended_action', ''),
                reasoning=llm_result.get('reasoning', ''),
                confidence=llm_result.get('confidence', 0.5),
                alternative_actions=llm_result.get('alternative_actions', []),
                warning=llm_result.get('warning', ''),
                risk_assessment=llm_result.get('risk_assessment', {})
            )

        # 否则使用规则结果
        return self._build_result_from_rule(rule_result, [])

    def store_experience(self,
                         state: Dict,
                         action: str,
                         outcome: str,
                         user_feedback: float,
                         effectiveness: float):
        """存储经验到历史库"""
        case = HistoricalCase(
            timestamp=datetime.now().isoformat(),
            state_summary=f"主动{state.get('simp_index', 50)} 被爱{state.get('loved_index', 50)}",
            action_taken=action,
            outcome=outcome,
            user_feedback=user_feedback,
            effectiveness=effectiveness
        )
        self.history_buffer.append(case)

    def get_reflection_embedding(self, result: ReflectionResult) -> np.ndarray:
        """
        获取反思结果的嵌入向量

        用于DQN的反思注意力模块
        """
        # 简化：使用置信度和风险作为特征
        embedding = np.zeros(128, dtype=np.float32)
        embedding[0] = result.confidence
        embedding[1] = result.risk_assessment.get('rejection_risk', 0.5)
        embedding[2] = result.risk_assessment.get('misunderstanding_risk', 0.5)
        embedding[3] = result.risk_assessment.get('over_active_risk', 0.5)

        # 剩余维度用动作类型one-hot
        action_types = ['reply_now', 'wait', 'meme', 'question',
                        'share', 'meeting', 'supportive', 'space', 'none']
        action_idx = action_types.index(result.recommended_action.lower().split()[0]) \
            if result.recommended_action else 8
        embedding[4 + action_idx] = 1.0

        return embedding


__all__ = ['ReflectionReasoner', 'ReflectionResult', 'HistoricalCase']
