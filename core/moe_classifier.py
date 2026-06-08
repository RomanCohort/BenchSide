"""
多层 MoE 人格分类器

通过多个专家模型的门控融合，实现准确的人格分类

架构：
1. 底层特征提取器 - 从聊天数据提取多维特征
2. 中层专家网络 - 多个专家模型分别分析
3. 上层门控网络 - 动态加权专家意见
4. 输出分类结果 - 人格类型 + 置信度

人格分类体系：
- 依恋类型: 安全型/焦虑型/回避型/混乱型
- 大五人格: OCEAN 五维度
- 行为模式: 主动型/被动型/平衡型
- 社交风格: 外向/内向/混合
- 决策风格: 理性/感性/混合
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import numpy as np


# ============================================================
# 人格类型定义
# ============================================================

class AttachmentType(Enum):
    """依恋类型"""
    SECURE = "安全型"
    ANXIOUS = "焦虑型"
    AVOIDANT = "回避型"
    DISORGANIZED = "混乱型"


class BehaviorPattern(Enum):
    """行为模式"""
    ACTIVE = "主动型"      # 主动发起、高频互动
    PASSIVE = "被动型"     # 被动回应、低频互动
    BALANCED = "平衡型"    # 适度互动


class SocialStyle(Enum):
    """社交风格"""
    EXTROVERT = "外向型"
    INTROVERT = "内向型"
    AMBIVERT = "混合型"


class DecisionStyle(Enum):
    """决策风格"""
    RATIONAL = "理性型"
    EMOTIONAL = "感性型"
    MIXED = "混合型"


class RelationshipRole(Enum):
    """关系角色"""
    PURSUER = "追求者"       # 主动追求
    DISTANCER = "疏离者"     # 保持距离
    PARTNER = "伙伴"         # 平等互动
    CARETAKER = "照顾者"     # 过度付出
    AVOIDER = "回避者"       # 回避亲密


@dataclass
class PersonalityClassification:
    """人格分类结果"""
    # 依恋类型
    attachment: AttachmentType = AttachmentType.SECURE
    attachment_confidence: float = 0.5

    # 行为模式
    behavior: BehaviorPattern = BehaviorPattern.BALANCED
    behavior_confidence: float = 0.5

    # 社交风格
    social: SocialStyle = SocialStyle.AMBIVERT
    social_confidence: float = 0.5

    # 决策风格
    decision: DecisionStyle = DecisionStyle.MIXED
    decision_confidence: float = 0.5

    # 关系角色
    role: RelationshipRole = RelationshipRole.PARTNER
    role_confidence: float = 0.5

    # 大五人格简版
    big_five: Dict[str, float] = field(default_factory=dict)

    # 综合描述
    summary: str = ""
    key_traits: List[str] = field(default_factory=list)

    # 专家权重
    expert_weights: Dict[str, float] = field(default_factory=dict)


# ============================================================
# 底层特征提取
# ============================================================

class FeatureExtractor:
    """
    特征提取器

    从原始聊天数据提取多维特征
    """

    def extract(self, messages: List[Dict], stats: Dict, target: str = "me") -> Dict[str, Any]:
        """
        提取特征

        Args:
            messages: 消息列表
            stats: 统计数据
            target: 分析目标 ("me" 分析发送者='me'的人, "them" 分析发送者='them'的人)

        Returns:
            特征字典
        """
        features = {}

        # 确定分析对象
        if target == "me":
            # 分析发送者是'me'的人（用户自己）
            subject_msgs = [m for m in messages if m.get('sender') == 'me']
            partner_msgs = [m for m in messages if m.get('sender') == 'them']
            subject_key = 'me'
            partner_key = 'them'
        else:
            # 分析发送者是'them'的人（对方）
            subject_msgs = [m for m in messages if m.get('sender') == 'them']
            partner_msgs = [m for m in messages if m.get('sender') == 'me']
            subject_key = 'them'
            partner_key = 'me'

        subject_text = ' '.join(m.get('content', '') for m in subject_msgs)
        partner_text = ' '.join(m.get('content', '') for m in partner_msgs)

        # ========== 互动特征 ==========
        total = len(subject_msgs) + len(partner_msgs)
        features['subject_ratio'] = len(subject_msgs) / total if total > 0 else 0.5

        # 主动次数 - 需要从stats中提取
        if target == "me":
            subject_init = stats.get('initiative', {}).get('my_starts', 0)
            partner_init = stats.get('initiative', {}).get('their_starts', 0)
        else:
            subject_init = stats.get('initiative', {}).get('their_starts', 0)
            partner_init = stats.get('initiative', {}).get('my_starts', 0)

        features['initiation_ratio'] = subject_init / (subject_init + partner_init + 1)

        # 提问比例
        subject_q = sum(1 for m in subject_msgs if '?' in m.get('content', '') or '？' in m.get('content', ''))
        partner_q = sum(1 for m in partner_msgs if '?' in m.get('content', '') or '？' in m.get('content', ''))
        features['question_ratio'] = subject_q / (subject_q + partner_q + 1)

        # 连发次数
        consecutive = 0
        target_sender = 'me' if target == 'me' else 'them'
        for i in range(len(messages) - 1):
            if messages[i].get('sender') == target_sender and messages[i+1].get('sender') == target_sender:
                consecutive += 1
        features['consecutive_ratio'] = consecutive / (len(subject_msgs) + 1)

        # ========== 回复时间特征 ==========
        subject_reply_times = []
        partner_reply_times = []

        for i in range(1, len(messages)):
            prev = messages[i-1]
            curr = messages[i]
            if prev.get('sender') != curr.get('sender'):
                gap = curr.get('timestamp', 0) - prev.get('timestamp', 0)
                if 0 < gap < 86400:
                    if curr.get('sender') == target_sender:
                        subject_reply_times.append(gap)
                    else:
                        partner_reply_times.append(gap)

        features['subject_avg_reply'] = np.mean(subject_reply_times) if subject_reply_times else 3600
        features['partner_avg_reply'] = np.mean(partner_reply_times) if partner_reply_times else 3600
        features['reply_speed_ratio'] = features['subject_avg_reply'] / (features['partner_avg_reply'] + 1)

        # ========== 消息内容特征 ==========
        # 消息长度
        features['subject_avg_len'] = np.mean([len(m.get('content', '')) for m in subject_msgs]) if subject_msgs else 0
        features['partner_avg_len'] = np.mean([len(m.get('content', '')) for m in partner_msgs]) if partner_msgs else 0

        # 表情使用
        features['subject_emoji_ratio'] = subject_text.count('[') / (len(subject_text) + 1)
        features['partner_emoji_ratio'] = partner_text.count('[') / (len(partner_text) + 1)

        # ========== 情感特征 ==========
        # 正面情绪词
        positive_words = ['哈哈', '嘻嘻', '好呀', '可以', '喜欢', '开心', '谢谢']
        negative_words = ['烦', '累', '难过', '无语', '崩溃', '抑郁', '焦虑']

        features['subject_positive'] = sum(subject_text.count(w) for w in positive_words) / (len(subject_text) + 1)
        features['subject_negative'] = sum(subject_text.count(w) for w in negative_words) / (len(subject_text) + 1)
        features['partner_positive'] = sum(partner_text.count(w) for w in positive_words) / (len(partner_text) + 1)
        features['partner_negative'] = sum(partner_text.count(w) for w in negative_words) / (len(partner_text) + 1)

        # ========== 讨好特征 ==========
        pleasing_words = ['好吧', '没事', '哈哈', '可以的', '收到', 'okk']
        dismissive_words = ['哦', '嗯', '好的', '[表情]', '知道了']

        features['subject_pleasing'] = sum(subject_text.count(w) for w in pleasing_words) / (len(subject_msgs) + 1)
        features['partner_dismissive'] = sum(partner_text.count(w) for w in dismissive_words) / (len(partner_msgs) + 1)

        # ========== 承诺特征 ==========
        commitment_words = ['未来', '一起', '我们', '以后', '结婚']
        features['subject_commitment'] = sum(subject_text.count(w) for w in commitment_words)
        features['partner_commitment'] = sum(partner_text.count(w) for w in commitment_words)

        return features


# ============================================================
# 专家网络
# ============================================================

class AttachmentExpert:
    """依恋类型专家"""

    def predict(self, features: Dict) -> Tuple[AttachmentType, float, str]:
        """
        预测依恋类型

        Returns:
            (类型, 置信度, 理由)
        """
        score_anxious = 0
        score_avoidant = 0
        score_secure = 0

        # 焦虑型特征
        if features['initiation_ratio'] > 0.7:
            score_anxious += 30
        if features['question_ratio'] > 0.7:
            score_anxious += 20
        if features['consecutive_ratio'] > 0.3:
            score_anxious += 20
        if features['subject_pleasing'] > 0.3:
            score_anxious += 15
        if features['reply_speed_ratio'] < 0.5:  # 回复快
            score_anxious += 15

        # 回避型特征
        if features['initiation_ratio'] < 0.3:
            score_avoidant += 30
        if features['partner_dismissive'] > 0.5:
            score_avoidant += 25
        if features['subject_avg_len'] < 5:
            score_avoidant += 20

        # 安全型特征
        if 0.4 < features['initiation_ratio'] < 0.6:
            score_secure += 25
        if features['subject_positive'] > features['subject_negative'] * 2:
            score_secure += 20

        # 选择最高分
        scores = {
            AttachmentType.ANXIOUS: score_anxious,
            AttachmentType.AVOIDANT: score_avoidant,
            AttachmentType.SECURE: score_secure
        }

        best_type = max(scores, key=scores.get)
        confidence = min(scores[best_type] / 100, 1.0)

        # 理由
        reasons = []
        if best_type == AttachmentType.ANXIOUS:
            if features['initiation_ratio'] > 0.7:
                reasons.append("主动发起比例高")
            if features['consecutive_ratio'] > 0.3:
                reasons.append("频繁连发消息")
            if features['subject_pleasing'] > 0.3:
                reasons.append("讨好型表达多")
        elif best_type == AttachmentType.AVOIDANT:
            if features['initiation_ratio'] < 0.3:
                reasons.append("几乎不主动")
            if features['partner_dismissive'] > 0.5:
                reasons.append("敷衍回复多")
        else:
            reasons.append("互动平衡")

        return best_type, confidence, "; ".join(reasons) if reasons else "特征均衡"


class BehaviorExpert:
    """行为模式专家"""

    def predict(self, features: Dict) -> Tuple[BehaviorPattern, float, str]:
        """预测行为模式"""
        init_ratio = features['initiation_ratio']

        if init_ratio > 0.7:
            return BehaviorPattern.ACTIVE, 0.8, "主动发起率{:.0%}".format(init_ratio)
        elif init_ratio < 0.3:
            return BehaviorPattern.PASSIVE, 0.8, "主动发起率仅{:.0%}".format(init_ratio)
        else:
            return BehaviorPattern.BALANCED, 0.7, "主动发起率{:.0%}，较平衡".format(init_ratio)


class SocialExpert:
    """社交风格专家"""

    def predict(self, features: Dict) -> Tuple[SocialStyle, float, str]:
        """预测社交风格"""
        # 基于表情使用和消息长度
        emoji_score = features['subject_emoji_ratio']
        len_score = features['subject_avg_len']
        positive_score = features['subject_positive']

        # 外向性得分
        extraversion = 0
        if emoji_score > 0.03:
            extraversion += 25
        if positive_score > 0.01:
            extraversion += 25
        if len_score > 15:
            extraversion += 25

        if extraversion > 50:
            return SocialStyle.EXTROVERT, 0.7, "表达活跃、情绪外露"
        elif extraversion < 25:
            return SocialStyle.INTROVERT, 0.7, "表达克制、情绪内敛"
        else:
            return SocialStyle.AMBIVERT, 0.6, "表达适中"


class DecisionExpert:
    """决策风格专家"""

    def predict(self, features: Dict) -> Tuple[DecisionStyle, float, str]:
        """预测决策风格"""
        # 基于提问方式、承诺表达等
        q_ratio = features['question_ratio']
        commitment = features['subject_commitment']

        # 理性: 提问多、承诺讨论
        # 感性: 表情多、情绪词多

        rational = q_ratio * 50 + min(commitment, 20)
        emotional = features['subject_emoji_ratio'] * 500 + features['subject_positive'] * 100

        if rational > emotional * 1.5:
            return DecisionStyle.RATIONAL, 0.65, "理性思考、注重逻辑"
        elif emotional > rational * 1.5:
            return DecisionStyle.EMOTIONAL, 0.65, "情感驱动、注重感受"
        else:
            return DecisionStyle.MIXED, 0.6, "理感混合"


class RoleExpert:
    """关系角色专家"""

    def predict(self, features: Dict) -> Tuple[RelationshipRole, float, str]:
        """预测关系角色"""
        init_ratio = features['initiation_ratio']
        q_ratio = features['question_ratio']
        pleasing = features['subject_pleasing']
        dismissive = features['partner_dismissive']

        # 追求者特征
        if init_ratio > 0.7 and q_ratio > 0.6:
            return RelationshipRole.PURSUER, 0.8, "高主动、高频提问"

        # 疏离者/回避者
        if init_ratio < 0.3:
            if dismissive > 0.5:
                return RelationshipRole.AVOIDER, 0.75, "低主动、敷衍回复"
            else:
                return RelationshipRole.DISTANCER, 0.7, "低主动、保持距离"

        # 照顾者
        if pleasing > 0.4 and init_ratio > 0.5:
            return RelationshipRole.CARETAKER, 0.7, "讨好型表达多、过度付出"

        # 伙伴
        return RelationshipRole.PARTNER, 0.65, "互动相对平衡"


# ============================================================
# 门控网络
# ============================================================

class GatingNetwork:
    """
    门控网络

    根据输入特征动态调整各专家的权重
    """

    def __init__(self):
        # 专家默认权重
        self.base_weights = {
            'attachment': 0.30,   # 依恋类型最重要
            'behavior': 0.20,
            'social': 0.15,
            'decision': 0.15,
            'role': 0.20         # 关系角色也很重要
        }

    def compute_weights(self, features: Dict) -> Dict[str, float]:
        """
        计算动态权重

        根据数据特征调整专家权重
        """
        weights = self.base_weights.copy()

        # 如果互动数据充足，提高行为专家权重
        if features.get('initiation_ratio', 0.5) not in [0, 1]:
            weights['behavior'] *= 1.2

        # 如果有明显的讨好/敷衍特征，提高依恋专家权重
        if features.get('me_pleasing', 0) > 0.3 or features.get('them_dismissive', 0) > 0.5:
            weights['attachment'] *= 1.3

        # 归一化
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}


# ============================================================
# MoE 分类器
# ============================================================

class MoEClassifier:
    """
    多层 MoE 人格分类器

    架构：
    1. 特征提取层
    2. 专家网络层
    3. 门控网络层
    4. 融合输出层
    """

    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.experts = {
            'attachment': AttachmentExpert(),
            'behavior': BehaviorExpert(),
            'social': SocialExpert(),
            'decision': DecisionExpert(),
            'role': RoleExpert()
        }
        self.gating = GatingNetwork()

    def classify(self, messages: List[Dict], stats: Dict, target: str = "me") -> PersonalityClassification:
        """
        执行人格分类

        Args:
            messages: 消息列表
            stats: 统计数据
            target: 分析目标 ("me" 或 "them")

        Returns:
            人格分类结果
        """
        # 1. 特征提取
        features = self.feature_extractor.extract(messages, stats, target)

        # 2. 各专家预测
        predictions = {}
        for name, expert in self.experts.items():
            predictions[name] = expert.predict(features)

        # 3. 门控权重
        weights = self.gating.compute_weights(features)

        # 4. 构建结果
        result = PersonalityClassification()

        # 依恋类型
        result.attachment = predictions['attachment'][0]
        result.attachment_confidence = predictions['attachment'][1] * weights['attachment']

        # 行为模式
        result.behavior = predictions['behavior'][0]
        result.behavior_confidence = predictions['behavior'][1] * weights['behavior']

        # 社交风格
        result.social = predictions['social'][0]
        result.social_confidence = predictions['social'][1] * weights['social']

        # 决策风格
        result.decision = predictions['decision'][0]
        result.decision_confidence = predictions['decision'][1] * weights['decision']

        # 关系角色
        result.role = predictions['role'][0]
        result.role_confidence = predictions['role'][1] * weights['role']

        # 专家权重
        result.expert_weights = weights

        # 关键特征
        result.key_traits = self._extract_key_traits(features, predictions)

        # 综合描述
        result.summary = self._generate_summary(result, predictions)

        return result

    def _extract_key_traits(self, features: Dict, predictions: Dict) -> List[str]:
        """提取关键特征"""
        traits = []

        # 高主动
        if features['initiation_ratio'] > 0.7:
            traits.append("高主动发起")
        elif features['initiation_ratio'] < 0.3:
            traits.append("低主动发起")

        # 连发消息
        if features['consecutive_ratio'] > 0.3:
            traits.append("消息轰炸")

        # 讨好表达
        if features['subject_pleasing'] > 0.3:
            traits.append("讨好型沟通")

        # 敷衍回复
        if features['partner_dismissive'] > 0.5:
            traits.append("对方敷衍")

        # 回复速度差异
        if features['reply_speed_ratio'] < 0.3:
            traits.append("秒回")

        return traits

    def _generate_summary(self, result: PersonalityClassification, predictions: Dict) -> str:
        """生成综合描述"""
        parts = []

        # 依恋类型
        parts.append(f"依恋类型: {result.attachment.value}")
        if result.attachment == AttachmentType.ANXIOUS:
            parts.append("（过度关注对方回应，容易焦虑）")
        elif result.attachment == AttachmentType.AVOIDANT:
            parts.append("（保持距离，不愿深入）")

        # 行为模式
        parts.append(f"\n行为模式: {result.behavior.value}")

        # 关系角色
        parts.append(f"\n关系角色: {result.role.value}")

        # 专家理由
        parts.append("\n\n专家分析:")
        for name, (etype, conf, reason) in predictions.items():
            parts.append(f"\n  [{name}] {etype.value} ({conf:.0%}) - {reason}")

        return "".join(parts)


def create_classification_report(result: PersonalityClassification) -> str:
    """生成分类报告"""
    lines = []
    lines.append("=" * 60)
    lines.append("MoE 人格分类报告")
    lines.append("=" * 60)

    lines.append("\n[分类结果]")
    lines.append(f"  依恋类型: {result.attachment.value} (置信度: {result.attachment_confidence:.1%})")
    lines.append(f"  行为模式: {result.behavior.value} (置信度: {result.behavior_confidence:.1%})")
    lines.append(f"  社交风格: {result.social.value} (置信度: {result.social_confidence:.1%})")
    lines.append(f"  决策风格: {result.decision.value} (置信度: {result.decision_confidence:.1%})")
    lines.append(f"  关系角色: {result.role.value} (置信度: {result.role_confidence:.1%})")

    lines.append("\n[专家权重]")
    for expert, weight in result.expert_weights.items():
        lines.append(f"  {expert}: {weight:.1%}")

    lines.append("\n[关键特征]")
    for trait in result.key_traits:
        lines.append(f"  - {trait}")

    lines.append(f"\n[综合描述]\n{result.summary}")

    return "\n".join(lines)


__all__ = [
    'AttachmentType', 'BehaviorPattern', 'SocialStyle',
    'DecisionStyle', 'RelationshipRole',
    'PersonalityClassification', 'FeatureExtractor',
    'MoEClassifier', 'create_classification_report'
]