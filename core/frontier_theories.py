"""
前沿社交与精神病学理论整合模块

融入的理论：
1. 社会神经科学 (Social Neuroscience)
   - Cacioppo的社会大脑理论
   - 社会孤立对大脑的影响机制

2. 社会基线理论 (Social Baseline Theory)
   - Coan & Beckes, 2013
   - 人类默认状态是"社会性"的

3. 人际神经生物学 (Interpersonal Neurobiology)
   - Siegel, 2012
   - 调谐(Attunement)与同步(Synchrony)

4. 网络精神病学 (Network Psychiatry)
   - Borsboom, 2017
   - 症状网络模型替代潜变量模型

5. 社会决定因素 (Social Determinants of Mental Health)
   - 流行病学精神病学框架

参考文献：
- Cacioppo, J. T., & Cacioppo, S. (2018). The growing problem of loneliness.
- Coan, J. A., & Beckes, L. (2013). Social Baseline Theory.
- Siegel, D. J. (2012). The developing mind.
- Borsboom, D. (2017). A network theory of mental disorders.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


# ============================================================
# 1. 社会神经科学 - 社会孤立神经机制
# ============================================================

@dataclass
class SocialIsolationIndicators:
    """
    社会孤立神经指标

    基于Cacioppo的社会神经科学研究：
    - 社会孤立影响特定脑区
    - 增加抑郁、焦虑风险
    - 与吸烟相当的健康风险
    """
    # 行为指标（可从聊天数据推断）
    social_interaction_frequency: float = 0    # 社交频率
    social_diversity: float = 0                 # 社交多样性
    reciprocity_rate: float = 0                 # 互惠率

    # 神经风险指标（推断）
    hpa_axis_dysregulation: float = 0          # HPA轴失调风险
    neuroinflammation_risk: float = 0          # 神经炎症风险
    reward_system_blunting: float = 0          # 奖赏系统钝化

    # 综合风险
    isolation_risk: float = 0                   # 孤立风险 0-100


class SocialNeuroscienceAnalyzer:
    """
    社会神经科学分析器

    基于Cacioppo & Cacioppo的研究
    """

    def analyze_isolation(self, social_network: Dict) -> SocialIsolationIndicators:
        """
        分析社会孤立风险

        基于神经科学机制：
        1. 社交频率 → HPA轴激活频率
        2. 社交多样性 → 社会认知维持
        3. 互惠率 → 催产素系统激活
        """
        indicators = SocialIsolationIndicators()

        # 1. 计算社交频率
        # 从网络数据提取
        total_interactions = social_network.get('total_interactions', 0)
        time_span_days = social_network.get('time_span_days', 30)

        daily_interactions = total_interactions / max(time_span_days, 1)

        # 神经科学基准：
        # 健康成年人每天需要 3-5 次有意义的社交互动
        indicators.social_interaction_frequency = min(daily_interactions / 5 * 100, 100)

        # 2. 计算社交多样性
        # 不同类型关系的数量
        relation_types = social_network.get('relation_types', {})
        indicators.social_diversity = min(len(relation_types) / 4 * 100, 100)

        # 3. 计算互惠率
        # 双向互动 vs 单向互动
        my_init = social_network.get('my_initiations', 0)
        their_init = social_network.get('their_initiations', 0)
        total_init = my_init + their_init

        if total_init > 0:
            # 理想互惠：50-50
            my_ratio = my_init / total_init
            indicators.reciprocity_rate = 100 - abs(my_ratio - 0.5) * 100

        # 4. 推断神经风险
        # 基于Cacioppo的机制研究

        # HPA轴失调：低社交频率 + 高压力
        # 孤立 → 慢性压力 → HPA轴过度激活
        if indicators.social_interaction_frequency < 30:
            indicators.hpa_axis_dysregulation = 70
        elif indicators.social_interaction_frequency < 50:
            indicators.hpa_axis_dysregulation = 40
        else:
            indicators.hpa_axis_dysregulation = 15

        # 神经炎症风险
        # 孤立 → 炎症标志物升高 (CRP, IL-6)
        if indicators.social_diversity < 25:
            indicators.neuroinflammation_risk = 60
        else:
            indicators.neuroinflammation_risk = 20

        # 奖赏系统钝化
        # 孤立 → 多巴胺系统对社交奖赏反应降低
        if indicators.reciprocity_rate < 30:
            indicators.reward_system_blunting = 50
        else:
            indicators.reward_system_blunting = 15

        # 5. 综合孤立风险
        indicators.isolation_risk = (
            (100 - indicators.social_interaction_frequency) * 0.4 +
            (100 - indicators.social_diversity) * 0.25 +
            (100 - indicators.reciprocity_rate) * 0.15 +
            indicators.hpa_axis_dysregulation * 0.1 +
            indicators.neuroinflammation_risk * 0.1
        )

        return indicators

    def assess_neuro_risk(self, indicators: SocialIsolationIndicators) -> Dict:
        """
        评估神经风险

        基于Cacioppo的研究：孤立与多种神经精神疾病相关
        """
        risk = {
            'overall_risk': indicators.isolation_risk,
            'brain_regions_affected': [],
            'neurotransmitter_systems': [],
            'recommendations': []
        }

        # 受影响的脑区（基于fMRI研究）
        if indicators.isolation_risk > 50:
            risk['brain_regions_affected'] = [
                '前额叶皮层 - 社会认知',
                '前扣带回 - 社会痛苦处理',
                '杏仁核 - 威胁敏感性升高',
                '腹侧纹状体 - 社交奖赏反应降低'
            ]

            risk['neurotransmitter_systems'] = [
                '多巴胺系统 - 奖赏钝化',
                '催产素系统 - 社会联结减弱',
                '血清素系统 - 情绪调节受损'
            ]

        # 建议（基于神经可塑性研究）
        if indicators.social_interaction_frequency < 40:
            risk['recommendations'].append(
                '社交频率低：建议每天至少3次有意义的社交互动'
            )

        if indicators.social_diversity < 30:
            risk['recommendations'].append(
                '社交多样性低：建议拓展不同类型的关系（家人、朋友、同事）'
            )

        if indicators.reciprocity_rate < 40:
            risk['recommendations'].append(
                '互惠不平衡：建议等待对方主动，培养双向关系'
            )

        return risk


# ============================================================
# 2. 社会基线理论 (Social Baseline Theory)
# ============================================================

@dataclass
class SocialBaselineMetrics:
    """
    社会基线理论指标

    Coan & Beckes (2013):
    - 人类大脑默认假设"他人存在"
    - 孤立状态是"异常"的，需要额外认知资源
    - 社会支持降低能量消耗
    """
    # 基线偏离度
    baseline_deviation: float = 0              # 偏离社会基线的程度

    # 资源消耗
    cognitive_load: float = 0                  # 认知负荷（孤立增加）
    energy_expenditure: float = 0              # 能量消耗

    # 社会资源利用
    proximity_to_baseline: float = 0           # 接近基线程度
    social_resource_utilization: float = 0     # 社会资源利用效率


class SocialBaselineCalculator:
    """
    社会基线计算器

    基于Coan & Beckes (2013) 的理论
    """

    # 社会基线参数
    BASELINE_PARAMS = {
        'proximity_contacts': 3,       # 近距离联系人数量
        'daily_interactions': 5,        # 每日有意义互动
        'support_availability': 0.7,   # 支持可获得性
        'emotional_sharing': 0.5       # 情感分享频率
    }

    def calculate_baseline_deviation(
        self,
        social_network: Dict
    ) -> SocialBaselineMetrics:
        """
        计算社会基线偏离

        偏离基线 = 需要更多认知资源来处理环境
        """
        metrics = SocialBaselineMetrics()

        # 1. 计算各维度偏离
        deviations = {}

        # 联系人数量偏离
        num_contacts = social_network.get('num_contacts', 0)
        expected_contacts = self.BASELINE_PARAMS['proximity_contacts']
        deviations['contacts'] = abs(num_contacts - expected_contacts) / expected_contacts

        # 互动频率偏离
        daily_interactions = social_network.get('daily_interactions', 0)
        expected_interactions = self.BASELINE_PARAMS['daily_interactions']
        deviations['interactions'] = abs(daily_interactions - expected_interactions) / expected_interactions

        # 支持可获得性偏离
        support_score = social_network.get('support_score', 0) / 100
        expected_support = self.BASELINE_PARAMS['support_availability']
        deviations['support'] = max(0, expected_support - support_score)

        # 情感分享偏离
        emotional_sharing = social_network.get('emotional_sharing', 0)
        expected_sharing = self.BASELINE_PARAMS['emotional_sharing']
        deviations['sharing'] = max(0, expected_sharing - emotional_sharing)

        # 2. 计算基线偏离度
        metrics.baseline_deviation = np.mean(list(deviations.values())) * 100

        # 3. 计算认知负荷
        # 偏离基线 = 需要更多自我调节
        metrics.cognitive_load = metrics.baseline_deviation * 0.8

        # 4. 计算能量消耗
        # 孤立状态消耗更多代谢能量
        metrics.energy_expenditure = metrics.baseline_deviation * 1.2

        # 5. 计算社会资源利用
        metrics.proximity_to_baseline = 100 - metrics.baseline_deviation
        metrics.social_resource_utilization = (
            support_score * 100 * 0.5 +
            (1 - deviations['contacts']) * 50
        )

        return metrics

    def predict_cognitive_burden(self, metrics: SocialBaselineMetrics) -> Dict:
        """
        预测认知负担

        基于社会基线理论：孤立状态增加认知负荷
        """
        burden = {
            'level': 'normal',
            'affected_functions': [],
            'recovery_suggestions': []
        }

        if metrics.baseline_deviation > 60:
            burden['level'] = 'severe'
            burden['affected_functions'] = [
                '自我调节能力下降',
                '注意力分散',
                '决策质量下降',
                '情绪调节困难'
            ]
        elif metrics.baseline_deviation > 40:
            burden['level'] = 'moderate'
            burden['affected_functions'] = [
                '需要更多努力维持日常功能',
                '应对压力能力下降'
            ]
        elif metrics.baseline_deviation > 20:
            burden['level'] = 'mild'

        if burden['level'] != 'normal':
            burden['recovery_suggestions'] = [
                '重新建立近距离社会联系',
                '增加日常社交互动频率',
                '寻求可用的社会支持资源'
            ]

        return burden


# ============================================================
# 3. 人际神经生物学 - 调谐与同步
# ============================================================

@dataclass
class InterpersonalSynchronyMetrics:
    """
    人际同步指标

    Siegel (2012) - 人际神经生物学：
    - 调谐(Attunement)：感知并回应他人内部状态
    - 同步(Synchrony)：生理和行为协调
    - 共鸣(Resonance)：情感状态的相互影响

    研究发现：
    - 心率同步预测关系质量
    - 行为同步预测信任
    - 情感同步预测亲密感
    """
    # 行为同步
    response_synchrony: float = 0           # 回复同步度
    timing_coordination: float = 0          # 时间协调度
    emotional_attunement: float = 0         # 情感调谐

    # 互动模式
    mirroring_score: float = 0              # 镜像行为得分
    turn_taking_quality: float = 0          # 轮流质量

    # 关系质量预测
    connection_prediction: float = 0        # 连接感预测
    trust_prediction: float = 0             # 信任预测


class InterpersonalNeurobiologyAnalyzer:
    """
    人际神经生物学分析器

    基于Siegel和Berscheid的研究
    """

    def analyze_synchrony(
        self,
        interaction_data: Dict
    ) -> InterpersonalSynchronyMetrics:
        """
        分析人际同步

        从聊天数据推断同步程度
        """
        metrics = InterpersonalSynchronyMetrics()

        # 1. 回复同步度
        # 双方回复时间的一致性
        my_delays = interaction_data.get('my_reply_delays', [])
        their_delays = interaction_data.get('their_reply_delays', [])

        if my_delays and their_delays:
            # 计算时间模式的相似性
            my_pattern = np.array(my_delays[-10:]) if len(my_delays) >= 10 else np.array(my_delays)
            their_pattern = np.array(their_delays[-10:]) if len(their_delays) >= 10 else np.array(their_delays)

            # 相关系数作为同步度量
            if len(my_pattern) > 1 and len(their_pattern) > 1:
                min_len = min(len(my_pattern), len(their_pattern))
                if min_len > 1:
                    corr = np.corrcoef(my_pattern[:min_len], their_pattern[:min_len])[0, 1]
                    metrics.response_synchrony = max((corr + 1) / 2 * 100, 0)  # 归一化到0-100

        # 2. 时间协调度
        # 互动时间的重叠
        my_active_hours = interaction_data.get('my_active_hours', set())
        their_active_hours = interaction_data.get('their_active_hours', set())

        if my_active_hours and their_active_hours:
            overlap = len(my_active_hours & their_active_hours)
            union = len(my_active_hours | their_active_hours)
            metrics.timing_coordination = overlap / union * 100 if union > 0 else 50

        # 3. 情感调谐
        # 情感表达的一致性
        my_emotions = interaction_data.get('my_emotional_pattern', [])
        their_emotions = interaction_data.get('their_emotional_pattern', [])

        if my_emotions and their_emotions:
            # 简化：检查情感是否匹配
            # 实际应该用更复杂的NLP
            metrics.emotional_attunement = 50  # 中性默认

        # 4. 镜像行为
        # 消息长度、表情使用的相似性
        my_msg_lengths = interaction_data.get('my_msg_lengths', [])
        their_msg_lengths = interaction_data.get('their_msg_lengths', [])

        if my_msg_lengths and their_msg_lengths:
            my_avg = np.mean(my_msg_lengths[-20:])
            their_avg = np.mean(their_msg_lengths[-20:])
            ratio = min(my_avg, their_avg) / max(my_avg, their_avg)
            metrics.mirroring_score = ratio * 100

        # 5. 轮流质量
        # 对话轮次的平衡
        my_turns = interaction_data.get('my_turns', 0)
        their_turns = interaction_data.get('their_turns', 0)

        if my_turns + their_turns > 0:
            balance = min(my_turns, their_turns) / max(my_turns, their_turns)
            metrics.turn_taking_quality = balance * 100

        # 6. 预测关系质量
        # 基于同步研究
        metrics.connection_prediction = (
            metrics.response_synchrony * 0.3 +
            metrics.timing_coordination * 0.25 +
            metrics.emotional_attunement * 0.25 +
            metrics.mirroring_score * 0.2
        )

        metrics.trust_prediction = (
            metrics.turn_taking_quality * 0.4 +
            metrics.response_synchrony * 0.3 +
            metrics.mirroring_score * 0.3
        )

        return metrics


# ============================================================
# 4. 网络精神病学 - 症状网络模型
# ============================================================

@dataclass
class SymptomNetworkNode:
    """症状网络节点"""
    symptom: str
    severity: float = 0
    centrality: float = 0          # 中心度（影响其他症状的程度）
    activation: float = 0          # 激活水平


@dataclass
class SymptomNetworkMetrics:
    """
    症状网络指标

    Borsboom (2017) - 网络精神病学：
    - 心理障碍是症状网络的涌现属性
    - 症状之间相互激活形成反馈循环
    - 治疗目标是打破核心症状的循环
    """
    nodes: Dict[str, SymptomNetworkNode] = field(default_factory=dict)

    # 网络指标
    density: float = 0                 # 网络密度
    average_strength: float = 0        # 平均连接强度
    clustering: float = 0              # 聚类系数

    # 关键症状
    core_symptoms: List[str] = field(default_factory=list)   # 核心症状
    bridge_symptoms: List[str] = field(default_factory=list) # 桥接症状


class NetworkPsychiatryAnalyzer:
    """
    网络精神病学分析器

    基于Borsboom的网络理论
    """

    # 症状节点定义
    SYMPTOM_NODES = {
        # 抑郁症状
        'low_mood': '低落情绪',
        'anhedonia': '快感缺失',
        'fatigue': '疲劳',
        'sleep_disturbance': '睡眠障碍',
        'worthlessness': '无价值感',
        'concentration_problems': '注意力问题',

        # 焦虑症状
        'worry': '担忧',
        'tension': '紧张',
        'restlessness': '坐立不安',
        'irritability': '易激惹',

        # 社交症状
        'social_withdrawal': '社交退缩',
        'loneliness': '孤独感',
        'fear_of_rejection': '害怕被拒绝',
        'difficulty_trusting': '信任困难'
    }

    # 症状之间的激活关系（基于文献）
    SYMPTOM_EDGES = {
        ('loneliness', 'low_mood'): 0.7,
        ('loneliness', 'social_withdrawal'): 0.6,
        ('social_withdrawal', 'loneliness'): 0.8,
        ('low_mood', 'anhedonia'): 0.7,
        ('low_mood', 'fatigue'): 0.6,
        ('anhedonia', 'social_withdrawal'): 0.5,
        ('worry', 'tension'): 0.6,
        ('worry', 'sleep_disturbance'): 0.5,
        ('fear_of_rejection', 'worry'): 0.6,
        ('fear_of_rejection', 'social_withdrawal'): 0.7,
        ('difficulty_trusting', 'loneliness'): 0.5,
        ('fatigue', 'concentration_problems'): 0.6,
        ('worthlessness', 'low_mood'): 0.7,
        ('low_mood', 'worthlessness'): 0.6,
    }

    def build_symptom_network(
        self,
        social_indicators: Dict
    ) -> SymptomNetworkMetrics:
        """
        构建症状网络

        从社交指标推断症状激活
        """
        network = SymptomNetworkMetrics()

        # 1. 创建症状节点
        for symptom_id, symptom_name in self.SYMPTOM_NODES.items():
            node = SymptomNetworkNode(
                symptom=symptom_name,
                severity=self._infer_severity(symptom_id, social_indicators),
                centrality=0
            )
            network.nodes[symptom_id] = node

        # 2. 计算激活传播
        # 症状之间相互激活
        for _ in range(5):  # 迭代传播
            for (src, tgt), strength in self.SYMPTOM_EDGES.items():
                if src in network.nodes and tgt in network.nodes:
                    # 激活传播
                    activation = network.nodes[src].severity * strength * 0.3
                    network.nodes[tgt].activation += activation

        # 3. 计算中心度
        for node_id, node in network.nodes.items():
            # 出度 + 入度
            out_strength = sum(
                s for (s, t), s in self.SYMPTOM_EDGES.items() if s == node_id
            )
            in_strength = sum(
                s for (s, t), s in self.SYMPTOM_EDGES.items() if t == node_id
            )
            node.centrality = out_strength + in_strength

        # 4. 识别核心症状
        # 高严重度 + 高中心度 = 核心症状
        sorted_nodes = sorted(
            network.nodes.items(),
            key=lambda x: x[1].severity * x[1].centrality,
            reverse=True
        )
        network.core_symptoms = [
            n[0] for n in sorted_nodes[:3]
            if n[1].severity > 30
        ]

        # 5. 计算网络密度
        possible_edges = len(network.nodes) * (len(network.nodes) - 1)
        actual_edges = len(self.SYMPTOM_EDGES)
        network.density = actual_edges / possible_edges if possible_edges > 0 else 0

        # 6. 计算平均强度
        network.average_strength = np.mean([
            n.severity for n in network.nodes.values()
        ])

        return network

    def _infer_severity(self, symptom_id: str, indicators: Dict) -> float:
        """
        从社交指标推断症状严重度

        基于流行病学和精神病学研究
        """
        severity = 0

        isolation_risk = indicators.get('isolation_risk', 0)
        reciprocity = indicators.get('reciprocity_rate', 50)
        diversity = indicators.get('social_diversity', 50)
        support = indicators.get('support_score', 50)

        # 根据症状类型映射
        if symptom_id == 'loneliness':
            severity = isolation_risk * 0.8 + (100 - diversity) * 0.2

        elif symptom_id == 'social_withdrawal':
            severity = (100 - diversity) * 0.6 + isolation_risk * 0.4

        elif symptom_id == 'low_mood':
            severity = isolation_risk * 0.5 + (100 - support) * 0.3 + (100 - reciprocity) * 0.2

        elif symptom_id == 'fear_of_rejection':
            severity = (100 - reciprocity) * 0.7 + isolation_risk * 0.3

        elif symptom_id == 'worry':
            severity = (100 - support) * 0.5 + (100 - reciprocity) * 0.5

        elif symptom_id == 'worthlessness':
            severity = isolation_risk * 0.6 + (100 - support) * 0.4

        else:
            severity = isolation_risk * 0.3 + (100 - support) * 0.3 + (100 - diversity) * 0.2 + (100 - reciprocity) * 0.2

        return min(severity, 100)

    def identify_intervention_targets(self, network: SymptomNetworkMetrics) -> Dict:
        """
        识别干预目标

        网络理论：针对核心症状干预效果最大
        """
        targets = {
            'primary_targets': [],
            'secondary_targets': [],
            'intervention_strategies': []
        }

        # 核心症状作为主要目标
        targets['primary_targets'] = network.core_symptoms

        # 生成干预策略
        for symptom_id in network.core_symptoms[:2]:
            symptom_name = self.SYMPTOM_NODES.get(symptom_id, symptom_id)

            if symptom_id == 'loneliness':
                targets['intervention_strategies'].append(
                    f'针对{symptom_name}：建立至少2个正向社交关系'
                )
            elif symptom_id == 'social_withdrawal':
                targets['intervention_strategies'].append(
                    f'针对{symptom_name}：渐进式社交暴露，从小规模互动开始'
                )
            elif symptom_id == 'low_mood':
                targets['intervention_strategies'].append(
                    f'针对{symptom_name}：行为激活 + 社交活动安排'
                )
            elif symptom_id == 'fear_of_rejection':
                targets['intervention_strategies'].append(
                    f'针对{symptom_name}：认知重构 + 安全关系建立'
                )
            else:
                targets['intervention_strategies'].append(
                    f'针对{symptom_name}：专业心理评估建议'
                )

        return targets


# ============================================================
# 5. 综合报告生成
# ============================================================

def create_frontier_theory_report(
    isolation_indicators: SocialIsolationIndicators,
    baseline_metrics: SocialBaselineMetrics,
    synchrony_metrics: InterpersonalSynchronyMetrics,
    symptom_network: SymptomNetworkMetrics
) -> str:
    """生成融合前沿理论的报告"""
    lines = []

    lines.append("=" * 60)
    lines.append("前沿社交与精神病学综合分析报告")
    lines.append("=" * 60)

    # 社会神经科学
    lines.append("\n【社会神经科学分析】(Cacioppo & Cacioppo)")
    lines.append("  孤立风险与神经机制:")
    lines.append(f"    社交频率: {isolation_indicators.social_interaction_frequency:.0f}/100")
    lines.append(f"    社交多样性: {isolation_indicators.social_diversity:.0f}/100")
    lines.append(f"    互惠率: {isolation_indicators.reciprocity_rate:.0f}/100")
    lines.append(f"    → 孤立风险: {isolation_indicators.isolation_risk:.0f}/100")

    if isolation_indicators.hpa_axis_dysregulation > 50:
        lines.append(f"    ⚠️ HPA轴失调风险: {isolation_indicators.hpa_axis_dysregulation:.0f}")

    # 社会基线理论
    lines.append("\n【社会基线理论分析】(Coan & Beckes, 2013)")
    lines.append("  偏离社会基线程度:")
    lines.append(f"    基线偏离: {baseline_metrics.baseline_deviation:.0f}%")
    lines.append(f"    认知负荷: {baseline_metrics.cognitive_load:.0f}")
    lines.append(f"    社会资源利用: {baseline_metrics.social_resource_utilization:.0f}")

    if baseline_metrics.baseline_deviation > 40:
        lines.append("    ⚠️ 显著偏离社会基线，需要额外认知资源")

    # 人际神经生物学
    lines.append("\n【人际神经生物学分析】(Siegel, 2012)")
    lines.append("  人际同步与调谐:")
    lines.append(f"    回复同步度: {synchrony_metrics.response_synchrony:.0f}/100")
    lines.append(f"    时间协调: {synchrony_metrics.timing_coordination:.0f}/100")
    lines.append(f"    镜像行为: {synchrony_metrics.mirroring_score:.0f}/100")
    lines.append(f"    → 连接感预测: {synchrony_metrics.connection_prediction:.0f}/100")

    # 网络精神病学
    lines.append("\n【网络精神病学分析】(Borsboom, 2017)")
    lines.append("  症状网络激活:")

    # 显示激活最高的症状
    sorted_symptoms = sorted(
        symptom_network.nodes.items(),
        key=lambda x: x[1].severity + x[1].activation,
        reverse=True
    )[:5]

    for symptom_id, node in sorted_symptoms:
        if node.severity > 20:
            lines.append(f"    {node.symptom}: 严重度{node.severity:.0f}, 激活{node.activation:.1f}")

    if symptom_network.core_symptoms:
        lines.append(f"  核心症状: {', '.join([symptom_network.nodes[s].symptom for s in symptom_network.core_symptoms])}")

    # 综合评估
    lines.append("\n【综合评估】")
    overall_risk = (
        isolation_indicators.isolation_risk * 0.3 +
        baseline_metrics.baseline_deviation * 0.25 +
        symptom_network.average_strength * 0.25 +
        (100 - synchrony_metrics.connection_prediction) * 0.2
    )
    lines.append(f"  综合风险指数: {overall_risk:.0f}/100")

    if overall_risk > 60:
        lines.append("  ⚠️ 高风险：建议专业心理评估")
    elif overall_risk > 40:
        lines.append("  中等风险：建议主动改善社交状况")
    else:
        lines.append("  风险较低：继续保持")

    return "\n".join(lines)


__all__ = [
    # Social Neuroscience
    'SocialIsolationIndicators', 'SocialNeuroscienceAnalyzer',
    # Social Baseline Theory
    'SocialBaselineMetrics', 'SocialBaselineCalculator',
    # Interpersonal Neurobiology
    'InterpersonalSynchronyMetrics', 'InterpersonalNeurobiologyAnalyzer',
    # Network Psychiatry
    'SymptomNetworkNode', 'SymptomNetworkMetrics', 'NetworkPsychiatryAnalyzer',
    # Report
    'create_frontier_theory_report'
]