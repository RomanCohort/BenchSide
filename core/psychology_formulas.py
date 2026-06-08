"""
心理学量化公式模块

所有量化指标基于已建立的心理科学理论和测量工具

理论基础：
1. Rusbult's Investment Model (1980)
   Commitment = Satisfaction + Investments - Quality of Alternatives

2. Social Exchange Theory (Thibaut & Kelley, 1959)
   Outcome = Rewards - Costs
   Comparison Level (CL) = 预期的合理回报
   Comparison Level for Alternatives (CLalt) = 替代选项的价值

3. Attachment Theory (Bowlby, Ainsworth)
   ECR Scale: Anxiety dimension + Avoidance dimension
   焦虑维度: 害怕被拒绝/抛弃
   回避维度: 回避亲密/依赖

4. Gottman's Four Horsemen (1995)
   稳定关系的正向互动比例应 ≥ 5:1
   四骑士: 批评、蔑视、防御、筑墙

5. Sternberg's Triangular Theory (1986)
   Love = Intimacy + Passion + Commitment

6. MSPSS - Multidimensional Scale of Perceived Social Support (Zimet, 1988)
   Family + Friends + Significant Other

参考文献：
- Rusbult, C. E. (1980). Commitment and satisfaction in romantic associations.
- Thibaut, J. & Kelley, H. (1959). The social psychology of groups.
- Brennan, K. A., Clark, C. L., & Shaver, P. R. (1998). Self-report measurement of adult attachment.
- Gottman, J. (1995). Why marriages succeed or fail.
- Sternberg, R. J. (1986). A triangular theory of love.
- Zimet, G. D. et al. (1988). The Multidimensional Scale of Perceived Social Support.
"""
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


# ============================================================
# 1. Rusbult's Investment Model（投入模型）
# ============================================================

@dataclass
class InvestmentModelScores:
    """
    Rusbult's Investment Model 评分

    Commitment = Satisfaction + Investments - Quality of Alternatives

    这是预测关系稳定性最有效的模型之一
    """
    satisfaction: float = 0      # 满意度 (rewards - costs)
    investments: float = 0       # 投入量
    alternatives: float = 0      # 替代选项质量
    commitment: float = 0        # 承诺度 = S + I - A


class InvestmentModelCalculator:
    """
    投入模型计算器

    基于Rusbult (1980, 1983)的理论
    """

    def calculate(
        self,
        rewards: float,
        costs: float,
        investments: float,
        alternatives: float
    ) -> InvestmentModelScores:
        """
        计算投入模型各维度

        Args:
            rewards: 关系中的奖励（正向体验）
            costs: 关系中的成本（负向体验）
            investments: 投入资源（时间、情感、物质）
            alternatives: 替代选项的质量（单身或其他人）

        Returns:
            InvestmentModelScores
        """
        # Satisfaction = Rewards - Costs (Social Exchange Theory)
        satisfaction = rewards - costs

        # Commitment Formula (Rusbult, 1980)
        # C = S + I - CLalt
        commitment = satisfaction + investments - alternatives

        return InvestmentModelScores(
            satisfaction=satisfaction,
            investments=investments,
            alternatives=alternatives,
            commitment=commitment
        )

    def predict_stability(self, scores: InvestmentModelScores) -> Dict:
        """
        预测关系稳定性

        基于meta-analysis (Le & Agnew, 2003):
        Commitment是预测关系持续的最佳指标
        """
        stability = {
            'probability': 0,
            'risk_level': 'unknown',
            'prediction': ''
        }

        # Commitment越高，稳定性越高
        if scores.commitment > 50:
            stability['probability'] = 0.85
            stability['risk_level'] = 'low'
            stability['prediction'] = '关系稳定，继续投入合理'
        elif scores.commitment > 20:
            stability['probability'] = 0.60
            stability['risk_level'] = 'medium'
            stability['prediction'] = '关系存在风险，需关注'
        elif scores.commitment > 0:
            stability['probability'] = 0.30
            stability['risk_level'] = 'high'
            stability['prediction'] = '关系不稳定，可能结束'
        else:
            stability['probability'] = 0.10
            stability['risk_level'] = 'critical'
            stability['prediction'] = '建议考虑结束关系'

        return stability


# ============================================================
# 2. Social Exchange Theory（社会交换理论）
# ============================================================

@dataclass
class SocialExchangeScores:
    """
    Social Exchange Theory 评分

    Outcome = Rewards - Costs
    CL (Comparison Level) = 期望的合理回报水平
    CLalt (Comparison Level for Alternatives) = 替代选项的比较水平

    决策规则：
    - Outcome > CL → 满意
    - Outcome > CLalt → 维持关系
    - Outcome < CLalt → 离开关系
    """
    rewards: float = 0          # 奖励总和
    costs: float = 0            # 成本总和
    outcome: float = 0          # Outcome = R - C
    comparison_level: float = 0  # CL (期望水平)
    comparison_alt: float = 0    # CLalt (替代水平)
    satisfaction_index: float = 0  # Outcome - CL
    dependence_index: float = 0    # Outcome - CLalt


class SocialExchangeCalculator:
    """
    社会交换计算器

    基于Thibaut & Kelley (1959)
    """

    # 标准化的奖励/成本权重
    REWARD_TYPES = {
        'affection': 10,       # 情感表达
        'support': 8,          # 支持
        'companionship': 7,    #陪伴
        'intimacy': 9,         # 亲密
        'validation': 6,       # 肯定/认可
        'fun': 5,              # 愉悦
        'status': 4,           # 社会地位
        'security': 8,         # 安全感
    }

    COST_TYPES = {
        'conflict': 10,        # 冲突
        'effort': 5,           # 努力投入
        'time': 3,             # 时间消耗
        'opportunity': 4,      # 机会成本
        'emotional': 8,        # 情绪消耗
        'uncertainty': 6,      # 不确定性
        'dependency': 5,       # 依赖风险
    }

    def calculate_outcome(
        self,
        reward_counts: Dict[str, float],
        cost_counts: Dict[str, float]
    ) -> SocialExchangeScores:
        """
        计算社会交换结果

        Args:
            reward_counts: 各类奖励的频率/强度
            cost_counts: 各类成本的频率/强度

        Returns:
            SocialExchangeScores
        """
        # 计算加权奖励
        rewards = sum(
            self.REWARD_TYPES.get(r, 5) * reward_counts.get(r, 0)
            for r in reward_counts
        )

        # 计算加权成本
        costs = sum(
            self.COST_TYPES.get(c, 5) * cost_counts.get(c, 0)
            for c in cost_counts
        )

        # Outcome = Rewards - Costs
        outcome = rewards - costs

        # CL (Comparison Level) - 基于过往经验和社会期望
        # 假设CL = 50 (中等期望)
        comparison_level = 50

        # CLalt - 基于替代选项的价值
        # 这里需要外部输入
        comparison_alt = 0

        # 满意度指数
        satisfaction_index = outcome - comparison_level

        # 依赖度指数
        dependence_index = outcome - comparison_alt

        return SocialExchangeScores(
            rewards=rewards,
            costs=costs,
            outcome=outcome,
            comparison_level=comparison_level,
            comparison_alt=comparison_alt,
            satisfaction_index=satisfaction_index,
            dependence_index=dependence_index
        )


# ============================================================
# 3. Attachment Theory - ECR Scale（依恋理论）
# ============================================================

class AttachmentDimension(Enum):
    """依恋维度"""
    SECURE = "secure"          # 安全型: 低焦虑 + 低回避
    PREOCCUPIED = "anxious"    # 焦虑型: 高焦虑 + 低回避
    DISMISSING = "avoidant"    # 回避型: 低焦虑 + 高回避
    FEARFUL = "fearful"        # 恐惧型: 高焦虑 + 高回避


@dataclass
class AttachmentScores:
    """
    ECR (Experiences in Close Relationships) Scale 评分

    Brennan, Clark, & Shaver (1998)

    两个维度：
    - Anxiety: 害怕被拒绝，担心不被爱
    - Avoidance: 回避亲密，不喜欢依赖他人

    测量方式：36个项目，Likert 7点量表
    """
    anxiety: float = 0         # 焦虑维度 (1-7)
    avoidance: float = 0       # 回避维度 (1-7)
    attachment_type: AttachmentDimension = AttachmentDimension.SECURE

    # 子维度
    proximity_seek: float = 0  # 寻求亲近
    support_seek: float = 0    # 寻求支持
    separation_distress: float = 0  # 分离焦虑
    emotional_unavailability: float = 0  # 情感回避


class AttachmentCalculator:
    """
    依恋风格计算器

    基于ECR Scale (Brennan et al., 1998)
    """

    def calculate_from_behavior(
        self,
        initiative_ratio: float,
        reply_delay_avg: float,
        consecutive_messages: float,
        emotional_expression: float,
        conflict_response: float
    ) -> AttachmentScores:
        """
        从行为指标推算依恋维度

        注意：这不是标准ECR问卷测量，而是从聊天行为推断

        ECR问卷的项目示例：
        焦虑维度：
        - "I worry about being abandoned"
        - "I need a lot of reassurance"

        回避维度：
        - "I prefer not to show how I feel"
        - "I don't feel comfortable getting close"

        Args:
            initiative_ratio: 主动发起比例 (0-1)
            reply_delay_avg: 平均回复延迟 (秒)
            consecutive_messages: 连发消息比例
            emotional_expression: 情感表达程度
            conflict_response: 冲突时的回应方式 (0=回避, 1=直接)

        Returns:
            AttachmentScores
        """
        # 焦虑维度推断
        # 高主动 + 连发 + 快回复 → 高焦虑
        anxiety = 0

        # 主动程度 → 焦虑（过度主动可能是焦虑）
        if initiative_ratio > 0.7:
            anxiety += 1.5  # 高主动增加焦虑

        # 连发消息 → 焦虑（等待焦虑）
        if consecutive_messages > 3:
            anxiety += 1.0

        # 快速回复期待 → 焦虑
        # 如果对方回复慢但自己快，增加焦虑
        anxiety += min(reply_delay_avg / 3600, 2)  # 等待时间增加焦虑

        # 回避维度推断
        # 低主动 + 慢回复 + 低情感表达 → 高回避
        avoidance = 0

        if initiative_ratio < 0.3:
            avoidance += 1.5  # 低主动增加回避

        if emotional_expression < 0.3:
            avoidance += 1.0  # 低情感表达

        if conflict_response < 0.5:
            avoidance += 1.0  # 冲突回避

        # 归一化到1-7量表
        anxiety = min(max(anxiety + 2, 1), 7)  # 基准2
        avoidance = min(max(avoidance + 2, 1), 7)  # 基准2

        # 确定依恋类型
        attachment_type = self._determine_type(anxiety, avoidance)

        return AttachmentScores(
            anxiety=anxiety,
            avoidance=avoidance,
            attachment_type=attachment_type
        )

    def _determine_type(self, anxiety: float, avoidance: float) -> AttachmentDimension:
        """
        根据焦虑/回避维度确定依恋类型

        标准分类（Brennan et al., 1998）:
        - Secure: Anxiety < 3, Avoidance < 3
        - Anxious: Anxiety > 3, Avoidance < 3
        - Avoidant: Anxiety < 3, Avoidance > 3
        - Fearful: Anxiety > 3, Avoidance > 3
        """
        anxious_threshold = 3.5  # ECR常模中值约为3.5
        avoid_threshold = 3.0

        if anxiety < anxious_threshold and avoidance < avoid_threshold:
            return AttachmentDimension.SECURE
        elif anxiety >= anxious_threshold and avoidance < avoid_threshold:
            return AttachmentDimension.PREOCCUPIED
        elif anxiety < anxious_threshold and avoidance >= avoid_threshold:
            return AttachmentDimension.DISMISSING
        else:
            return AttachmentDimension.FEARFUL


# ============================================================
# 4. Gottman's Four Horsemen Ratio（四骑士比例）
# ============================================================

@dataclass
class FourHorsemenScores:
    """
    Gottman's Four Horsemen 评分

    John Gottman (1995) 发现：
    - 稳定关系的正向互动比例应 ≥ 5:1
    - 四骑士（负面互动模式）会导致关系失败:
      1. Criticism (批评)
      2. Contempt (蔑视)
      3. Defensiveness (防御)
      4. Stonewalling (筑墙/回避)

    预测准确率: 91% (Gottman, 1995)
    """
    positive_ratio: float = 0   # 正向互动比例
    criticism: float = 0        # 批评频率
    contempt: float = 0         # 蔑视频率
    defensiveness: float = 0    # 防御频率
    stonewalling: float = 0     # 筑墙频率
    stability_prediction: float = 0  # 关系稳定性预测


class FourHorsemenCalculator:
    """
    四骑士计算器

    基于Gottman (1995)的研究
    """

    # 5:1 比例阈值
    STABLE_THRESHOLD = 5.0  # 正向:负向 ≥ 5:1
    WARNING_THRESHOLD = 1.0  # 正向:负向 < 1:1

    def calculate(
        self,
        positive_interactions: int,
        negative_interactions: int,
        criticism_count: int = 0,
        contempt_count: int = 0,
        defensiveness_count: int = 0,
        stonewalling_count: int = 0
    ) -> FourHorsemenScores:
        """
        计算四骑士评分

        Args:
            positive_interactions: 正向互动次数
            negative_interactions: 负向互动次数
            criticism_count: 批评次数
            contempt_count: 蔑视次数
            defensiveness_count: 防御次数
            stonewalling_count: 筑墙次数

        Returns:
            FourHorsemenScores
        """
        # 正向比例
        if negative_interactions > 0:
            positive_ratio = positive_interactions / negative_interactions
        else:
            positive_ratio = positive_interactions  # 只有正向

        # 稳定性预测
        if positive_ratio >= self.STABLE_THRESHOLD:
            stability_prediction = 0.91  # Gottman的准确率
        elif positive_ratio >= 1:
            stability_prediction = 0.50
        else:
            stability_prediction = 0.10

        # 归一化四骑士计数
        total = max(negative_interactions, 1)

        return FourHorsemenScores(
            positive_ratio=positive_ratio,
            criticism=criticism_count / total,
            contempt=contempt_count / total,
            defensiveness=defensiveness_count / total,
            stonewalling=stonewalling_count / total,
            stability_prediction=stability_prediction
        )

    def assess_risk(self, scores: FourHorsemenScores) -> Dict:
        """
        评估关系风险

        Gottman发现:
        - 蔑视是最具破坏性的
        - 四骑士累积会加速关系恶化
        """
        risk = {
            'level': 'unknown',
            'primary_issue': '',
            'recommendations': []
        }

        # 风险等级
        if scores.positive_ratio < 1:
            risk['level'] = 'critical'
            risk['recommendations'].append('关系严重危机，建议寻求专业帮助')
        elif scores.positive_ratio < 5:
            risk['level'] = 'warning'
            risk['recommendations'].append('正向互动不足，需要增加')
        else:
            risk['level'] = 'healthy'

        # 主要问题识别
        horsemen_values = {
            'criticism': scores.criticism,
            'contempt': scores.contempt,
            'defensiveness': scores.defensiveness,
            'stonewalling': scores.stonewalling
        }

        max_horseman = max(horsemen_values, key=horsemen_values.get)
        if horsemen_values[max_horseman] > 0.1:
            risk['primary_issue'] = max_horseman

            if max_horseman == 'contempt':
                risk['recommendations'].append('蔑视是最大杀手，建议培养欣赏')
            elif max_horseman == 'criticism':
                risk['recommendations'].append('批评过多，建议用温和抱怨替代')
            elif max_horseman == 'defensiveness':
                risk['recommendations'].append('防御反应过多，建议承担责任')
            elif max_horseman == 'stonewalling':
                risk['recommendations'].append('筑墙行为，建议表达需要暂停')

        return risk


# ============================================================
# 5. Sternberg's Triangular Theory（爱情三角理论）
# ============================================================

@dataclass
class TriangularLoveScores:
    """
    Sternberg's Triangular Theory of Love (1986)

    Love = Intimacy + Passion + Commitment

    三种成分组合出8种爱情类型：
    - Non-love: 三者都低
    - Liking: 只有Intimacy
    - Infatuated love: 只有Passion
    - Empty love: 只有Commitment
    - Romantic love: Intimacy + Passion
    - Companionate love: Intimacy + Commitment
    - Fatuous love: Passion + Commitment
    - Consummate love: 三者都有（理想）
    """
    intimacy: float = 0      # 亲密度 (0-100)
    passion: float = 0       # 激情 (0-100)
    commitment: float = 0    # 承诺 (0-100)
    love_type: str = ""      # 爱情类型


class TriangularLoveCalculator:
    """
    爱情三角理论计算器

    基于Sternberg (1986)
    """

    def calculate(
        self,
        intimacy_indicators: Dict,
        passion_indicators: Dict,
        commitment_indicators: Dict
    ) -> TriangularLoveScores:
        """
        计算爱情三角

        Intimacy indicators:
        - 深度交流
        - 情感支持
        - 互相理解
        - 共享秘密

        Passion indicators:
        - 性吸引
        - 浪漫行为
        - 思念频率
        - 激情表达

        Commitment indicators:
        - 未来规划
        - 长期投入
        - 关系定义
        - 牺牲意愿
        """
        # 计算各维度分数
        intimacy = self._compute_dimension(intimacy_indicators)
        passion = self._compute_dimension(passion_indicators)
        commitment = self._compute_dimension(commitment_indicators)

        # 确定爱情类型
        love_type = self._determine_type(intimacy, passion, commitment)

        return TriangularLoveScores(
            intimacy=intimacy,
            passion=passion,
            commitment=commitment,
            love_type=love_type
        )

    def _compute_dimension(self, indicators: Dict) -> float:
        """计算维度分数"""
        if not indicators:
            return 50

        total = sum(indicators.values())
        count = len(indicators)

        return min(total / count, 100)

    def _determine_type(self, i: float, p: float, c: float) -> str:
        """
        确定爱情类型

        阈值: 30为低，30-60为中，60+为高
        """
        threshold = 30

        i_high = i >= threshold
        p_high = p >= threshold
        c_high = c >= threshold

        if not i_high and not p_high and not c_high:
            return "Non-love"
        elif i_high and not p_high and not c_high:
            return "Liking (友谊)"
        elif not i_high and p_high and not c_high:
            return "Infatuated love (迷恋)"
        elif not i_high and not p_high and c_high:
            return "Empty love (空洞)"
        elif i_high and p_high and not c_high:
            return "Romantic love (浪漫)"
        elif i_high and not p_high and c_high:
            return "Companionate love (伴侣)"
        elif not i_high and p_high and c_high:
            return "Fatuous love (愚昧)"
        else:
            return "Consummate love (完美)"


# ============================================================
# 6. MSPSS - Social Support Scale（社会支持量表）
# ============================================================

@dataclass
class SocialSupportScores:
    """
    MSPSS (Multidimensional Scale of Perceived Social Support)

    Zimet, Dahlem, Zimet & Farley (1988)

    12个项目，三个维度：
    - Family Support
    - Friend Support
    - Significant Other Support

    Likert 7点量表
    """
    family_support: float = 0
    friend_support: float = 0
    significant_other: float = 0
    total_support: float = 0


class SocialSupportCalculator:
    """
    社会支持计算器

    基于MSPSS (Zimet et al., 1988)
    """

    def calculate(
        self,
        family_interactions: float,
        friend_interactions: float,
        romantic_interactions: float
    ) -> SocialSupportScores:
        """
        计算社会支持

        从互动数据推断支持水平

        Args:
            family_interactions: 家庭互动能量净值
            friend_interactions: 朋友互动能量净值
            romantic_interactions: 情感关系能量净值

        Returns:
            SocialSupportScores (归一化到1-7量表)
        """
        # 归一化到1-7量表
        def normalize(x):
            return min(max((x / 100) * 3 + 4, 1), 7)  # 正向能量=高支持

        family_support = normalize(family_interactions)
        friend_support = normalize(friend_interactions)
        significant_other = normalize(romantic_interactions)

        total_support = (family_support + friend_support + significant_other) / 3

        return SocialSupportScores(
            family_support=family_support,
            friend_support=friend_support,
            significant_other=significant_other,
            total_support=total_support
        )

    def assess_support_system(self, scores: SocialSupportScores) -> Dict:
        """
        评估支持系统

        MSPSS临床应用：
        - 低支持 (<4): 需要干预
        - 中等支持 (4-5): 正常
        - 高支持 (>5): 良好保护因素
        """
        assessment = {
            'overall': 'unknown',
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }

        # 整体评估
        if scores.total_support >= 5:
            assessment['overall'] = 'strong'
        elif scores.total_support >= 4:
            assessment['overall'] = 'moderate'
        else:
            assessment['overall'] = 'weak'
            assessment['recommendations'].append('支持系统不足，建议主动建立联系')

        # 各维度分析
        dimensions = {
            'family': scores.family_support,
            'friends': scores.friend_support,
            'partner': scores.significant_other
        }

        for dim, score in dimensions.items():
            if score >= 5:
                assessment['strengths'].append(f'{dim}支持良好')
            elif score < 4:
                assessment['weaknesses'].append(f'{dim}支持不足')

        return assessment


# ============================================================
# 综合评估报告
# ============================================================

def create_science_based_report(
    investment: InvestmentModelScores,
    exchange: SocialExchangeScores,
    attachment: AttachmentScores,
    horsemen: FourHorsemenScores,
    love: TriangularLoveScores,
    support: SocialSupportScores
) -> str:
    """生成基于科学理论的报告"""
    lines = []

    lines.append("=" * 60)
    lines.append("心理学量化分析报告（科学依据版）")
    lines.append("=" * 60)
    lines.append("\n注：本报告使用已验证的心理学理论框架")

    # 1. Investment Model
    lines.append("\n【投入模型分析】(Rusbult, 1980)")
    lines.append("  公式: Commitment = Satisfaction + Investments - Alternatives")
    lines.append(f"  满意度: {investment.satisfaction:.1f}")
    lines.append(f"  投入量: {investment.investments:.1f}")
    lines.append(f"  替代选项: {investment.alternatives:.1f}")
    lines.append(f"  → 承诺度: {investment.commitment:.1f}")

    # 2. Social Exchange
    lines.append("\n【社会交换分析】(Thibaut & Kelley, 1959)")
    lines.append("  公式: Outcome = Rewards - Costs")
    lines.append(f"  奖励: {exchange.rewards:.1f}")
    lines.append(f"  成本: {exchange.costs:.1f}")
    lines.append(f"  → 结果: {exchange.outcome:.1f}")

    # 3. Attachment
    lines.append("\n【依恋风格分析】(ECR Scale, Brennan et al., 1998)")
    lines.append(f"  焦虑维度: {attachment.anxiety:.1f}/7")
    lines.append(f"  回避维度: {attachment.avoidance:.1f}/7")
    lines.append(f"  → 依恋类型: {attachment.attachment_type.value}")

    # 4. Four Horsemen
    lines.append("\n【四骑士比例分析】(Gottman, 1995)")
    lines.append(f"  正向:负向比例: {horsemen.positive_ratio:.1f}:1")
    lines.append(f"  (健康阈值: ≥5:1)")
    lines.append(f"  → 稳定性预测: {horsemen.stability_prediction:.0%}")

    # 5. Love Triangle
    lines.append("\n【爱情三角分析】(Sternberg, 1986)")
    lines.append("  公式: Love = Intimacy + Passion + Commitment")
    lines.append(f"  亲密: {love.intimacy:.1f}")
    lines.append(f"  激情: {love.passion:.1f}")
    lines.append(f"  承诺: {love.commitment:.1f}")
    lines.append(f"  → 爱情类型: {love.love_type}")

    # 6. Social Support
    lines.append("\n【社会支持分析】(MSPSS, Zimet et al., 1988)")
    lines.append(f"  家庭支持: {support.family_support:.1f}/7")
    lines.append(f"  朋友支持: {support.friend_support:.1f}/7")
    lines.append(f"  情感支持: {support.significant_other:.1f}/7")
    lines.append(f"  → 总支持: {support.total_support:.1f}/7")

    return "\n".join(lines)


__all__ = [
    # Investment Model
    'InvestmentModelScores', 'InvestmentModelCalculator',
    # Social Exchange
    'SocialExchangeScores', 'SocialExchangeCalculator',
    # Attachment
    'AttachmentDimension', 'AttachmentScores', 'AttachmentCalculator',
    # Four Horsemen
    'FourHorsemenScores', 'FourHorsemenCalculator',
    # Triangular Love
    'TriangularLoveScores', 'TriangularLoveCalculator',
    # Social Support
    'SocialSupportScores', 'SocialSupportCalculator',
    # Report
    'create_science_based_report'
]