"""
场景分类器 + MoE路由系统

先分类关系场景，再路由到专属决策模型

场景分类：
1. 单相思 (舔狗) - 一方投入远超另一方
2. 暧昧期 - 双方有兴趣但未确定
3. 热恋期 - 高频互动，双向投入
4. 稳定期 - 平稳互动，有承诺
5. 冷淡期 - 双方互动减少
6. 分手边缘 - 高风险信号

MoE架构：
- 每个场景一个专家模型
- 门控网络决定专家权重
- 输出加权融合的决策
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np


class SceneType(Enum):
    """场景类型"""
    UNREQUITED = "单相思"      # 舔狗场景
    AMBIGUOUS = "暧昧期"       # 双向试探
    PASSIONATE = "热恋期"      # 高频双向
    STABLE = "稳定期"          # 平稳关系
    COLD = "冷淡期"            # 互动减少
    BREAKUP_EDGE = "分手边缘"  # 高风险


@dataclass
class SceneClassification:
    """场景分类结果"""
    primary_scene: SceneType
    confidence: float
    scene_scores: Dict[SceneType, float] = field(default_factory=dict)
    indicators: Dict[str, any] = field(default_factory=dict)


class SceneClassifier:
    """
    场景分类器

    根据互动模式、投入比例、心理学指标判断场景类型
    """

    def classify(self,
                 messages: List[Dict],
                 stats: Dict,
                 psych_profile: Dict) -> SceneClassification:
        """
        分类场景

        Args:
            messages: 消息列表
            stats: 统计数据
            psych_profile: 心理学分析结果

        Returns:
            场景分类结果
        """
        # 计算各场景得分
        scores = {}
        indicators = {}

        # ========== 基础指标 ==========
        me_msgs = [m for m in messages if m.get('sender') == 'me']
        them_msgs = [m for m in messages if m.get('sender') == 'them']

        # 发送比例
        me_ratio = len(me_msgs) / len(messages) if messages else 0.5
        indicators['me_ratio'] = me_ratio

        # 提问比例
        me_questions = sum(1 for m in me_msgs if '?' in m.get('content', '') or '？' in m.get('content', ''))
        them_questions = sum(1 for m in them_msgs if '?' in m.get('content', '') or '？' in m.get('content', ''))
        question_ratio = me_questions / (me_questions + them_questions) if (me_questions + them_questions) > 0 else 0.5
        indicators['question_ratio'] = question_ratio

        # 主动次数
        me_initiations = stats.get('initiative', {}).get('me_start_count', 0)
        them_initiations = stats.get('initiative', {}).get('them_start_count', 0)
        initiation_ratio = me_initiations / (me_initiations + them_initiations) if (me_initiations + them_initiations) > 0 else 0.5
        indicators['initiation_ratio'] = initiation_ratio

        # ROI
        roi = psych_profile.get('exchange', {}).get('roi', 1.0)
        indicators['roi'] = roi

        # 依恋类型
        my_attachment = psych_profile.get('attachment', {}).get('my_style', 'SECURE')
        their_attachment = psych_profile.get('attachment', {}).get('their_style', 'SECURE')
        indicators['my_attachment'] = my_attachment
        indicators['their_attachment'] = their_attachment

        # 四骑士
        horsemen = psych_profile.get('horsemen', {}).get('detected', [])
        indicators['horsemen_count'] = len(horsemen)

        # Sternberg三角
        intimacy = psych_profile.get('triangle', {}).get('intimacy', 50)
        passion = psych_profile.get('triangle', {}).get('passion', 50)
        commitment = psych_profile.get('triangle', {}).get('commitment', 50)
        indicators['intimacy'] = intimacy
        indicators['passion'] = passion
        indicators['commitment'] = commitment

        # ========== 场景评分 ==========

        # 1. 单相思 (舔狗)
        unrequited_score = 0
        if my_attachment == 'ANXIOUS' and their_attachment == 'AVOIDANT':
            unrequited_score += 40  # 焦虑-回避陷阱
        if roi < 0.5:
            unrequited_score += 30  # 投入回报严重失衡
        if initiation_ratio > 0.8:
            unrequited_score += 20  # 几乎全是你在主动
        if question_ratio > 0.7:
            unrequited_score += 10  # 你提问远多于对方
        scores[SceneType.UNREQUITED] = min(unrequited_score, 100)

        # 2. 暧昧期
        ambiguous_score = 0
        if 0.4 < me_ratio < 0.6:
            ambiguous_score += 20  # 发送相对平衡
        if 0.3 < initiation_ratio < 0.7:
            ambiguous_score += 20  # 双方都有主动
        if passion > 40 and commitment < 30:
            ambiguous_score += 30  # 有激情但没承诺
        if intimacy > 30:
            ambiguous_score += 20  # 有一定亲密度
        scores[SceneType.AMBIGUOUS] = min(ambiguous_score, 100)

        # 3. 热恋期
        passionate_score = 0
        if passion > 60:
            passionate_score += 30
        if intimacy > 50:
            passionate_score += 20
        avg_daily = stats.get('basic', {}).get('avg_daily', 0)
        if avg_daily > 20:
            passionate_score += 20  # 高频互动
        if 0.4 < me_ratio < 0.6:
            passionate_score += 15  # 双向平衡
        if them_initiations > 10:
            passionate_score += 15  # 对方也主动
        scores[SceneType.PASSIONATE] = min(passionate_score, 100)

        # 4. 稳定期
        stable_score = 0
        if commitment > 50:
            stable_score += 30
        if intimacy > 40:
            stable_score += 20
        if 0.4 < me_ratio < 0.6:
            stable_score += 20
        if len(horsemen) == 0:
            stable_score += 20  # 无危险信号
        if 5 < avg_daily < 20:
            stable_score += 10  # 适中频率
        scores[SceneType.STABLE] = min(stable_score, 100)

        # 5. 冷淡期
        cold_score = 0
        if avg_daily < 3:
            cold_score += 30  # 低频互动
        if intimacy < 20:
            cold_score += 20
        if passion < 20:
            cold_score += 20
        if 'STONEWALLING' in horsemen:
            cold_score += 20
        scores[SceneType.COLD] = min(cold_score, 100)

        # 6. 分手边缘
        breakup_score = 0
        if len(horsemen) >= 2:
            breakup_score += 40  # 多个危险信号
        if intimacy < 10 and passion < 10 and commitment < 10:
            breakup_score += 30  # Sternberg全低
        if roi < 0.3:
            breakup_score += 20
        if avg_daily < 1:
            breakup_score += 10  # 几乎不联系
        scores[SceneType.BREAKUP_EDGE] = min(breakup_score, 100)

        # ========== 确定主场景 ==========
        primary_scene = max(scores, key=scores.get)
        confidence = scores[primary_scene] / 100.0

        return SceneClassification(
            primary_scene=primary_scene,
            confidence=confidence,
            scene_scores=scores,
            indicators=indicators
        )


# ============================================================
# MoE 专家模型
# ============================================================

@dataclass
class ExpertDecision:
    """专家决策"""
    action: str
    confidence: float
    reasoning: str
    risk_level: str
    specific_advice: List[str] = field(default_factory=list)


class UnrequitedExpert:
    """
    单相思专家 (舔狗场景)

    核心策略：止损、撤退、自我保护
    """

    def decide(self, indicators: Dict, psych_profile: Dict) -> ExpertDecision:
        roi = indicators.get('roi', 1.0)
        initiation_ratio = indicators.get('initiation_ratio', 0.5)
        horsemen_count = indicators.get('horsemen_count', 0)

        # 决策逻辑
        if roi < 0.3:
            action = "STOP_IMMEDIATELY"
            reasoning = "ROI严重亏损，立即止损"
            risk_level = "critical"
            advice = [
                "立即停止所有主动行为",
                "对方不主动 = 明确拒绝",
                "删除聊天记录，物理隔离",
                "专注自我提升，转移注意力"
            ]
        elif roi < 0.5:
            action = "PULL_BACK"
            reasoning = "投入回报失衡，需要撤退"
            risk_level = "high"
            advice = [
                "大幅减少主动频率",
                "不追问、不期待、不投入",
                "观察对方是否会主动",
                "设定止损线：1周不主动联系"
            ]
        elif initiation_ratio > 0.8:
            action = "MATCH_ENERGY"
            reasoning = "你主动过多，需要匹配对方能量"
            risk_level = "medium"
            advice = [
                "将主动频率降到对方水平",
                "不要秒回，延迟回复",
                "停止讨好型表达",
                "建立自己的生活和兴趣"
            ]
        else:
            action = "OBSERVE"
            reasoning = "继续观察，暂不行动"
            risk_level = "low"
            advice = [
                "保持当前节奏",
                "注意对方回应质量",
                "不要增加投入"
            ]

        confidence = min(1.0, (1 - roi) * 1.5)

        return ExpertDecision(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            risk_level=risk_level,
            specific_advice=advice
        )


class AmbiguousExpert:
    """
    暧昧期专家

    核心策略：试探、推进、确认关系
    """

    def decide(self, indicators: Dict, psych_profile: Dict) -> ExpertDecision:
        intimacy = indicators.get('intimacy', 50)
        passion = indicators.get('passion', 50)
        them_initiations = indicators.get('them_initiations', 0)

        if passion > 60 and intimacy > 40:
            action = "CLARIFY_RELATION"
            reasoning = "暧昧程度高，可以确认关系"
            risk_level = "low"
            advice = [
                "找机会自然地确认关系",
                "观察对方是否提到'我们'",
                "增加见面频率",
                "适度表达你的想法"
            ]
        elif them_initiations > 5:
            action = "RESPOND_POSITIVELY"
            reasoning = "对方有主动，积极回应"
            risk_level = "low"
            advice = [
                "及时回应对方的主动",
                "分享更多个人信息",
                "创造见面机会",
                "适度升温关系"
            ]
        else:
            action = "TEST_INTEREST"
            reasoning = "试探对方兴趣程度"
            risk_level = "medium"
            advice = [
                "轻微减少主动，观察对方反应",
                "用轻松话题测试",
                "不要过度投入",
                "保持自己的生活节奏"
            ]

        return ExpertDecision(
            action=action,
            confidence=0.7,
            reasoning=reasoning,
            risk_level=risk_level,
            specific_advice=advice
        )


class PassionateExpert:
    """
    热恋期专家

    核心策略：维护、平衡、建立深度
    """

    def decide(self, indicators: Dict, psych_profile: Dict) -> ExpertDecision:
        commitment = indicators.get('commitment', 50)

        if commitment < 30:
            action = "BUILD_COMMITMENT"
            reasoning = "激情高但承诺低，需要建立承诺"
            risk_level = "medium"
            advice = [
                "自然地聊未来规划",
                "增加共同经历",
                "建立关系仪式感",
                "观察对方是否提到未来"
            ]
        else:
            action = "MAINTAIN_BALANCE"
            reasoning = "关系良好，保持平衡"
            risk_level = "low"
            advice = [
                "保持当前互动节奏",
                "注意给彼此空间",
                "持续建立深度连接",
                "享受关系但不依赖"
            ]

        return ExpertDecision(
            action=action,
            confidence=0.8,
            reasoning=reasoning,
            risk_level=risk_level,
            specific_advice=advice
        )


class StableExpert:
    """
    稳定期专家

    核心策略：维护、保鲜、深化
    """

    def decide(self, indicators: Dict, psych_profile: Dict) -> ExpertDecision:
        passion = indicators.get('passion', 50)

        if passion < 30:
            action = "REFRESH"
            reasoning = "激情下降，需要保鲜"
            risk_level = "medium"
            advice = [
                "创造新的共同体验",
                "保持个人魅力和成长",
                "增加惊喜元素",
                "保持适度的独立空间"
            ]
        else:
            action = "MAINTAIN"
            reasoning = "关系稳定，继续保持"
            risk_level = "low"
            advice = [
                "保持日常关心",
                "定期约会/见面",
                "持续沟通和分享",
                "共同规划未来"
            ]

        return ExpertDecision(
            action=action,
            confidence=0.8,
            reasoning=reasoning,
            risk_level=risk_level,
            specific_advice=advice
        )


class ColdExpert:
    """
    冷淡期专家

    核心策略：诊断、修复或接受
    """

    def decide(self, indicators: Dict, psych_profile: Dict) -> ExpertDecision:
        horsemen_count = indicators.get('horsemen_count', 0)

        if horsemen_count >= 2:
            action = "ASSESS_VIABILITY"
            reasoning = "多个危险信号，评估关系是否值得挽救"
            risk_level = "high"
            advice = [
                "冷静评估关系价值",
                "考虑是否值得继续",
                "如果决定挽救，需要双方努力",
                "如果对方不愿配合，考虑放手"
            ]
        else:
            action = "REPAIR"
            reasoning = "关系冷淡，尝试修复"
            risk_level = "medium"
            advice = [
                "主动沟通，了解原因",
                "增加高质量相处时间",
                "回忆美好时光",
                "解决潜在问题"
            ]

        return ExpertDecision(
            action=action,
            confidence=0.6,
            reasoning=reasoning,
            risk_level=risk_level,
            specific_advice=advice
        )


class BreakupEdgeExpert:
    """
    分手边缘专家

    核心策略：挽救或放手
    """

    def decide(self, indicators: Dict, psych_profile: Dict) -> ExpertDecision:
        roi = indicators.get('roi', 1.0)

        if roi < 0.2:
            action = "ACCEPT_END"
            reasoning = "关系已无法挽救，接受结束"
            risk_level = "critical"
            advice = [
                "接受关系已经结束",
                "做好分手准备",
                "寻求朋友/家人支持",
                "专注自我恢复"
            ]
        else:
            action = "LAST_RESORT"
            reasoning = "最后尝试挽救"
            risk_level = "high"
            advice = [
                "进行一次深度沟通",
                "明确表达你的感受",
                "询问对方的想法",
                "如果对方不愿努力，就放手"
            ]

        return ExpertDecision(
            action=action,
            confidence=0.7,
            reasoning=reasoning,
            risk_level=risk_level,
            specific_advice=advice
        )


# ============================================================
# MoE 路由器
# ============================================================

class MoERouter:
    """
    MoE路由器

    根据场景分类，路由到对应专家模型
    """

    def __init__(self):
        self.classifier = SceneClassifier()
        self.experts = {
            SceneType.UNREQUITED: UnrequitedExpert(),
            SceneType.AMBIGUOUS: AmbiguousExpert(),
            SceneType.PASSIONATE: PassionateExpert(),
            SceneType.STABLE: StableExpert(),
            SceneType.COLD: ColdExpert(),
            SceneType.BREAKUP_EDGE: BreakupEdgeExpert()
        }

    def route(self,
              messages: List[Dict],
              stats: Dict,
              psych_profile: Dict) -> Tuple[SceneClassification, ExpertDecision]:
        """
        路由决策

        Args:
            messages: 消息列表
            stats: 统计数据
            psych_profile: 心理学分析结果

        Returns:
            (场景分类, 专家决策)
        """
        # 1. 场景分类
        scene_class = self.classifier.classify(messages, stats, psych_profile)

        # 2. 获取对应专家
        expert = self.experts.get(scene_class.primary_scene)

        if expert is None:
            # 默认使用稳定期专家
            expert = self.experts[SceneType.STABLE]

        # 3. 专家决策
        decision = expert.decide(scene_class.indicators, psych_profile)

        return scene_class, decision


def create_moe_report(scene_class: SceneClassification, decision: ExpertDecision) -> str:
    """生成MoE决策报告"""
    lines = []
    lines.append("=" * 60)
    lines.append("MoE 决策报告")
    lines.append("=" * 60)

    lines.append("\n[场景分类]")
    lines.append(f"  主场景: {scene_class.primary_scene.value}")
    lines.append(f"  置信度: {scene_class.confidence:.1%}")
    lines.append("\n  各场景得分:")
    for scene, score in sorted(scene_class.scene_scores.items(), key=lambda x: -x[1]):
        marker = " <--" if scene == scene_class.primary_scene else ""
        lines.append(f"    {scene.value}: {score:.0f}{marker}")

    lines.append("\n[专家决策]")
    lines.append(f"  动作: {decision.action}")
    lines.append(f"  置信度: {decision.confidence:.1%}")
    lines.append(f"  风险等级: {decision.risk_level}")
    lines.append(f"  理由: {decision.reasoning}")

    lines.append("\n[具体建议]")
    for i, advice in enumerate(decision.specific_advice, 1):
        lines.append(f"  {i}. {advice}")

    return "\n".join(lines)


__all__ = [
    'SceneType', 'SceneClassification', 'SceneClassifier',
    'ExpertDecision', 'MoERouter', 'create_moe_report'
]