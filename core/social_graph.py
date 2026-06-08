"""
关系图谱与时序分析系统

功能：
1. 构建个人社交关系图谱
2. 时序追踪关系变化
3. 早期心理风险预警
4. 可视化展示

核心洞察：
- 社交关系的变化是精神疾病的"房间里的大象"
- 关系突然恶化、孤立化、能量亏损都是早期信号
- 通过量化追踪可以提前发现风险

应用场景：
- 抑郁症早期预警
- 焦虑症监测
- 社交孤立检测
- 情感支持系统评估
"""
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import math


# ============================================================
# 数据结构
# ============================================================

@dataclass
class RelationNode:
    """关系节点"""
    contact_id: str
    name: str
    category: str = ""           # romantic, family, friend, work, etc.
    current_score: float = 50    # 当前关系分数 0-100

    # 时序数据
    score_history: List[Tuple[datetime, float]] = field(default_factory=list)

    # 风险标记
    risk_flags: List[str] = field(default_factory=list)

    # 能量指标
    energy_net: float = 0
    investment: float = 0
    return_received: float = 0


@dataclass
class SocialGraph:
    """社交图谱"""
    nodes: Dict[str, RelationNode] = field(default_factory=dict)

    # 整体指标
    total_energy: float = 0
    diversity_score: float = 0
    support_score: float = 0
    isolation_risk: float = 0

    # 时序
    snapshot_history: List[Tuple[datetime, Dict]] = field(default_factory=list)


@dataclass
class MentalHealthRisk:
    """心理健康风险"""
    overall_risk: float = 0      # 0-100
    risk_factors: List[str] = field(default_factory=list)

    # 具体风险
    depression_risk: float = 0
    anxiety_risk: float = 0
    isolation_risk: float = 0
    burnout_risk: float = 0

    # 预警等级
    alert_level: str = "normal"  # normal, watch, warning, critical

    # 建议
    recommendations: List[str] = field(default_factory=list)


# ============================================================
# 时序分析
# ============================================================

class TemporalAnalyzer:
    """时序分析器"""

    def __init__(self):
        self.snapshots = []  # 历史快照

    def analyze_trends(self, graph: SocialGraph, window_days: int = 30) -> Dict:
        """
        分析关系变化趋势

        Returns:
            趋势分析结果
        """
        trends = {}

        for node_id, node in graph.nodes.items():
            if len(node.score_history) < 2:
                continue

            # 提取窗口内的分数
            now = datetime.now()
            recent_scores = [
                (t, s) for t, s in node.score_history
                if (now - t).days <= window_days
            ]

            if len(recent_scores) < 2:
                continue

            # 计算趋势
            times = [t.timestamp() for t, _ in recent_scores]
            scores = [s for _, s in recent_scores]

            # 线性回归
            if len(scores) >= 3:
                slope = np.polyfit(times, scores, 1)[0]
                # 转换为每天的变化
                daily_change = slope * 86400

                if daily_change < -0.5:
                    trend = "declining"
                    severity = min(abs(daily_change) * 10, 100)
                elif daily_change > 0.5:
                    trend = "improving"
                    severity = 0
                else:
                    trend = "stable"
                    severity = 0

                trends[node_id] = {
                    'trend': trend,
                    'daily_change': daily_change,
                    'severity': severity,
                    'recent_score': scores[-1],
                    'score_change': scores[-1] - scores[0]
                }

        return trends

    def detect_sudden_changes(self, graph: SocialGraph) -> List[Dict]:
        """
        检测突然变化

        突然变化可能是风险信号
        """
        alerts = []

        for node_id, node in graph.nodes.items():
            if len(node.score_history) < 3:
                continue

            # 最近3个分数
            recent = node.score_history[-3:]
            scores = [s for _, s in recent]

            # 突然下降
            if scores[-1] - scores[-2] < -10:
                alerts.append({
                    'type': 'sudden_drop',
                    'contact': node.name,
                    'change': scores[-1] - scores[-2],
                    'message': f"与{node.name}的关系突然恶化"
                })

            # 持续下降
            if all(scores[i] > scores[i+1] for i in range(len(scores)-1)):
                alerts.append({
                    'type': 'continuous_decline',
                    'contact': node.name,
                    'message': f"与{node.name}的关系持续恶化"
                })

        return alerts


# ============================================================
# 心理健康风险评估
# ============================================================

class MentalHealthAssessor:
    """心理健康风险评估器"""

    # 风险因素定义
    RISK_FACTORS = {
        # 孤立风险
        'social_isolation': {
            'description': '社交孤立',
            'indicators': ['总联系人少于3人', '无亲密关系', '无情感支持'],
            'weight': 1.5
        },
        'relationship_collapse': {
            'description': '关系崩塌',
            'indicators': ['多个关系同时恶化', '核心支持者消失'],
            'weight': 2.0
        },
        'energy_depletion': {
            'description': '能量耗竭',
            'indicators': ['整体能量净值为负', '高投入低回报关系多'],
            'weight': 1.8
        },
        'attachment_anxiety': {
            'description': '依恋焦虑',
            'indicators': ['过度主动', '高焦虑行为模式', '低回复率'],
            'weight': 1.3
        },
        'support_lack': {
            'description': '支持缺失',
            'indicators': ['无核心支持者', '无互惠关系'],
            'weight': 1.6
        }
    }

    def assess(self, graph: SocialGraph, trends: Dict) -> MentalHealthRisk:
        """
        评估心理健康风险
        """
        risk = MentalHealthRisk()

        # 1. 计算各项风险

        # 孤立风险
        isolation_risk = self._compute_isolation_risk(graph)
        risk.isolation_risk = isolation_risk

        # 能量耗竭风险
        burnout_risk = self._compute_burnout_risk(graph)
        risk.burnout_risk = burnout_risk

        # 抑郁风险（基于孤立+能量亏损+关系恶化）
        depression_risk = self._compute_depression_risk(graph, trends)
        risk.depression_risk = depression_risk

        # 焦虑风险（基于依恋模式+不确定性）
        anxiety_risk = self._compute_anxiety_risk(graph, trends)
        risk.anxiety_risk = anxiety_risk

        # 2. 计算综合风险
        risk.overall_risk = (
            isolation_risk * 0.2 +
            burnout_risk * 0.25 +
            depression_risk * 0.3 +
            anxiety_risk * 0.25
        )

        # 3. 确定预警等级
        if risk.overall_risk > 70:
            risk.alert_level = "critical"
        elif risk.overall_risk > 50:
            risk.alert_level = "warning"
        elif risk.overall_risk > 30:
            risk.alert_level = "watch"
        else:
            risk.alert_level = "normal"

        # 4. 生成风险因素列表
        risk.risk_factors = self._identify_risk_factors(graph, trends)

        # 5. 生成建议
        risk.recommendations = self._generate_recommendations(risk, graph)

        return risk

    def _compute_isolation_risk(self, graph: SocialGraph) -> float:
        """计算孤立风险"""
        risk = 0

        # 联系人数量
        num_contacts = len(graph.nodes)
        if num_contacts == 0:
            return 100
        elif num_contacts < 3:
            risk += 40
        elif num_contacts < 5:
            risk += 20

        # 亲密关系
        intimate = sum(1 for n in graph.nodes.values() if n.category in ['romantic', 'family'])
        if intimate == 0:
            risk += 30

        # 支持系统
        if graph.support_score < 20:
            risk += 30

        return min(risk, 100)

    def _compute_burnout_risk(self, graph: SocialGraph) -> float:
        """计算耗竭风险"""
        risk = 0

        # 整体能量
        if graph.total_energy < -500:
            risk += 40
        elif graph.total_energy < -200:
            risk += 25
        elif graph.total_energy < 0:
            risk += 10

        # 高亏损关系数量
        losing_relations = sum(
            1 for n in graph.nodes.values()
            if n.energy_net < -100
        )
        if losing_relations >= 3:
            risk += 30
        elif losing_relations >= 2:
            risk += 20

        # 无正向关系
        positive_relations = sum(
            1 for n in graph.nodes.values()
            if n.energy_net > 50
        )
        if positive_relations == 0:
            risk += 20

        return min(risk, 100)

    def _compute_depression_risk(self, graph: SocialGraph, trends: Dict) -> float:
        """计算抑郁风险"""
        risk = 0

        # 关系恶化趋势
        declining = sum(1 for t in trends.values() if t.get('trend') == 'declining')
        if declining >= 3:
            risk += 35
        elif declining >= 2:
            risk += 25

        # 整体分数下降
        avg_score = np.mean([n.current_score for n in graph.nodes.values()]) if graph.nodes else 50
        if avg_score < 40:
            risk += 30

        # 孤立+能量亏损组合
        if graph.isolation_risk > 50 and graph.total_energy < 0:
            risk += 25

        return min(risk, 100)

    def _compute_anxiety_risk(self, graph: SocialGraph, trends: Dict) -> float:
        """计算焦虑风险"""
        risk = 0

        # 高主动低回报
        high_initiative_low_return = sum(
            1 for n in graph.nodes.values()
            if n.investment > 70 and n.return_received < 30
        )
        if high_initiative_low_return >= 2:
            risk += 35

        # 关系不确定性（波动大）
        volatile_relations = 0
        for node in graph.nodes.values():
            if len(node.score_history) >= 5:
                scores = [s for _, s in node.score_history[-5:]]
                std = np.std(scores)
                if std > 15:  # 高波动
                    volatile_relations += 1

        if volatile_relations >= 2:
            risk += 25

        # 整体不稳定性
        if graph.diversity_score < 30:
            risk += 20  # 社交圈单一增加焦虑

        return min(risk, 100)

    def _identify_risk_factors(self, graph: SocialGraph, trends: Dict) -> List[str]:
        """识别风险因素"""
        factors = []

        # 孤立
        if len(graph.nodes) < 3:
            factors.append("社交圈过小")

        # 无支持
        if graph.support_score < 30:
            factors.append("缺乏情感支持系统")

        # 能量亏损
        if graph.total_energy < -200:
            factors.append("整体情感能量亏损")

        # 关系恶化
        declining = [n.name for n in graph.nodes.values()
                     if any(t.get('trend') == 'declining' for tid, t in trends.items() if tid == n.contact_id)]
        if declining:
            factors.append(f"关系恶化: {', '.join(declining[:3])}")

        # 高焦虑模式
        anxious_relations = sum(1 for n in graph.nodes.values() if n.investment > 70)
        if anxious_relations > len(graph.nodes) * 0.5:
            factors.append("过度投入模式")

        return factors

    def _generate_recommendations(self, risk: MentalHealthRisk, graph: SocialGraph) -> List[str]:
        """生成建议"""
        recommendations = []

        if risk.alert_level == "critical":
            recommendations.append("⚠️ 建议寻求专业心理支持")
            recommendations.append("考虑联系信任的朋友或家人")

        if risk.isolation_risk > 50:
            recommendations.append("建议主动拓展社交圈")
            recommendations.append("参加兴趣小组或社区活动")

        if risk.burnout_risk > 50:
            recommendations.append("减少对亏损关系的投入")
            recommendations.append("将能量集中在正向关系上")

        if risk.depression_risk > 40:
            recommendations.append("关注自身情绪状态")
            recommendations.append("保持规律作息和运动")

        if risk.anxiety_risk > 50:
            recommendations.append("学习设定关系边界")
            recommendations.append("练习降低对他人回应的依赖")

        # 积极建议
        positive = [n.name for n in graph.nodes.values() if n.energy_net > 50]
        if positive:
            recommendations.append(f"珍惜与{', '.join(positive[:2])}的正向关系")

        return recommendations[:6]  # 最多6条


# ============================================================
# 关系图谱构建
# ============================================================

class SocialGraphBuilder:
    """社交图谱构建器"""

    def __init__(self):
        self.temporal_analyzer = TemporalAnalyzer()
        self.health_assessor = MentalHealthAssessor()

    def build_graph(self, classifications: Dict, quantifications: Dict) -> SocialGraph:
        """
        构建社交图谱

        Args:
            classifications: 分类结果
            quantifications: 量化结果
        """
        graph = SocialGraph()

        # 创建节点
        for contact_id, class_data in classifications.items():
            if not contact_id.endswith('_me'):
                continue

            node = RelationNode(
                contact_id=contact_id,
                name=class_data.get('name', contact_id),
                category=class_data.get('relation_type', 'unknown')
            )

            # 添加量化数据
            if contact_id in quantifications:
                quant = quantifications[contact_id]
                node.current_score = quant.get('health_score', 50)
                node.energy_net = quant.get('energy_net', 0)
                node.investment = quant.get('investment_score', 50)
                node.return_received = quant.get('response_score', 50)

            graph.nodes[contact_id] = node

        # 计算整体指标
        graph.total_energy = sum(n.energy_net for n in graph.nodes.values())
        graph.diversity_score = self._compute_diversity(graph)
        graph.support_score = self._compute_support(graph)
        graph.isolation_risk = self._compute_isolation_risk(graph)

        return graph

    def _compute_diversity(self, graph: SocialGraph) -> float:
        """计算多样性"""
        if not graph.nodes:
            return 0

        categories = [n.category for n in graph.nodes.values()]
        unique = len(set(categories))
        # 理想情况下有5种关系类型
        return min(unique / 5 * 100, 100)

    def _compute_support(self, graph: SocialGraph) -> float:
        """计算支持系统强度"""
        score = 0

        # 有亲密关系
        intimate = sum(1 for n in graph.nodes.values()
                       if n.category in ['romantic', 'family'] and n.energy_net > 0)
        score += intimate * 25

        # 有正向朋友
        positive_friends = sum(1 for n in graph.nodes.values()
                               if n.category == 'friend' and n.energy_net > 30)
        score += positive_friends * 15

        return min(score, 100)

    def _compute_isolation_risk(self, graph: SocialGraph) -> float:
        """计算孤立风险"""
        if len(graph.nodes) == 0:
            return 100
        if len(graph.nodes) < 3:
            return 70
        if graph.support_score < 20:
            return 50
        return max(100 - graph.support_score - len(graph.nodes) * 5, 0)

    def analyze(self, graph: SocialGraph) -> Dict:
        """
        完整分析
        """
        # 趋势分析
        trends = self.temporal_analyzer.analyze_trends(graph)

        # 突然变化
        sudden_changes = self.temporal_analyzer.detect_sudden_changes(graph)

        # 心理健康评估
        risk = self.health_assessor.assess(graph, trends)

        return {
            'graph': graph,
            'trends': trends,
            'sudden_changes': sudden_changes,
            'mental_health_risk': risk
        }


# ============================================================
# 可视化数据生成
# ============================================================

def generate_visualization_data(graph: SocialGraph, analysis: Dict) -> Dict:
    """
    生成可视化数据

    用于前端渲染
    """
    viz_data = {
        'nodes': [],
        'edges': [],
        'metrics': {},
        'timeline': [],
        'alerts': []
    }

    # 节点
    for node_id, node in graph.nodes.items():
        viz_data['nodes'].append({
            'id': node_id,
            'name': node.name,
            'category': node.category,
            'score': node.current_score,
            'energy': node.energy_net,
            'size': 20 + abs(node.energy_net) / 10,  # 节点大小
            'color': _score_to_color(node.current_score)
        })

    # 中心节点（用户自己）
    viz_data['nodes'].append({
        'id': 'self',
        'name': '我',
        'category': 'self',
        'score': 100,
        'size': 40,
        'color': '#4CAF50'
    })

    # 边（从用户到各联系人）
    for node_id, node in graph.nodes.items():
        viz_data['edges'].append({
            'source': 'self',
            'target': node_id,
            'weight': min(1 + abs(node.energy_net) / 100, 3),
            'color': '#4CAF50' if node.energy_net > 0 else '#f44336',
            'type': 'solid' if node.energy_net > 0 else 'dashed'
        })

    # 整体指标
    viz_data['metrics'] = {
        'total_energy': graph.total_energy,
        'diversity_score': graph.diversity_score,
        'support_score': graph.support_score,
        'isolation_risk': graph.isolation_risk,
        'num_contacts': len(graph.nodes)
    }

    # 时间线
    for node_id, node in graph.nodes.items():
        for timestamp, score in node.score_history:
            viz_data['timeline'].append({
                'date': timestamp.strftime('%Y-%m-%d'),
                'contact': node.name,
                'score': score
            })

    # 警报
    risk = analysis.get('mental_health_risk')
    if risk:
        viz_data['alerts'] = [{
            'level': risk.alert_level,
            'score': risk.overall_risk,
            'factors': risk.risk_factors
        }]
        viz_data['alerts'].extend([
            {'type': change['type'], 'message': change['message']}
            for change in analysis.get('sudden_changes', [])
        ])

    return viz_data


def _score_to_color(score: float) -> str:
    """分数转颜色"""
    if score >= 70:
        return '#4CAF50'  # 绿色
    elif score >= 50:
        return '#FFC107'  # 黄色
    elif score >= 30:
        return '#FF9800'  # 橙色
    else:
        return '#f44336'  # 红色


# ============================================================
# 报告生成
# ============================================================

def create_mental_health_report(risk: MentalHealthRisk, graph: SocialGraph) -> str:
    """生成心理健康报告"""
    lines = []

    lines.append("=" * 60)
    lines.append("心理健康风险评估报告")
    lines.append("=" * 60)

    # 预警等级
    level_display = {
        'normal': '✅ 正常',
        'watch': '👀 关注',
        'warning': '⚠️ 警告',
        'critical': '🚨 紧急'
    }
    lines.append(f"\n【预警等级】{level_display.get(risk.alert_level, '未知')}")
    lines.append(f"综合风险分数: {risk.overall_risk:.1f}/100")

    # 风险分解
    lines.append("\n【风险分解】")
    lines.append(f"  抑郁风险: {risk.depression_risk:.1f}/100")
    lines.append(f"  焦虑风险: {risk.anxiety_risk:.1f}/100")
    lines.append(f"  孤立风险: {risk.isolation_risk:.1f}/100")
    lines.append(f"  耗竭风险: {risk.burnout_risk:.1f}/100")

    # 风险因素
    if risk.risk_factors:
        lines.append("\n【识别到的风险因素】")
        for factor in risk.risk_factors:
            lines.append(f"  • {factor}")

    # 社交图谱概览
    lines.append("\n【社交图谱概览】")
    lines.append(f"  联系人数量: {len(graph.nodes)}")
    lines.append(f"  关系多样性: {graph.diversity_score:.1f}/100")
    lines.append(f"  支持系统强度: {graph.support_score:.1f}/100")
    lines.append(f"  整体能量净值: {graph.total_energy:+.1f}")

    # 建议
    if risk.recommendations:
        lines.append("\n【行动建议】")
        for i, rec in enumerate(risk.recommendations, 1):
            lines.append(f"  {i}. {rec}")

    return "\n".join(lines)


__all__ = [
    'RelationNode', 'SocialGraph', 'MentalHealthRisk',
    'TemporalAnalyzer', 'MentalHealthAssessor', 'SocialGraphBuilder',
    'generate_visualization_data', 'create_mental_health_report'
]