"""
社交网络分析与建议系统

分析用户的完整社交网络，提供全局视角的关系建议

功能：
1. 构建社交网络图
2. 识别关键关系节点
3. 分析关系网络健康度
4. 发现潜在问题关系
5. 提供社交策略建议

分析维度：
- 关系多样性：是否有多种类型的关系
- 关系平衡：付出/回报是否平衡
- 关系密度：社交圈的紧密程度
- 支持系统：是否有足够的支持网络
- 风险关系：识别可能有害的关系
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import json
from pathlib import Path
from collections import defaultdict


class RelationHealth(Enum):
    """关系健康度"""
    HEALTHY = "健康"
    ATTENTION_NEEDED = "需要关注"
    WARNING = "警示"
    CRITICAL = "危险"


class SocialRole(Enum):
    """社交角色"""
    ANCHOR = "核心支持者"      # 提供情感支持的人
    BRIDGE = "桥梁"           # 连接不同社交圈的人
    DRAINER = "能量消耗者"    # 消耗你能量的人
    GIVER = "给予者"          # 总是付出的人
    MENTOR = "导师"           # 提供指导的人
    MENTEE = "学徒"           # 你指导的人
    PEER = "同辈"             # 平等关系
    ACQUAINTANCE = "泛交"     # 浅层关系


@dataclass
class RelationNode:
    """关系节点"""
    contact_id: str
    name: str

    # 关系类型
    category: str = ""
    specific_type: str = ""

    # 人格特征
    attachment: str = ""
    behavior_pattern: str = ""
    role_in_relation: str = ""

    # 互动数据
    interaction_frequency: float = 0  # 日均消息
    initiative_ratio: float = 0.5     # 你的主动比例
    last_interaction: str = ""        # 最后互动时间

    # 评估指标
    health_score: float = 50          # 健康度 0-100
    investment_score: float = 50      # 你投入程度
    return_score: float = 50          # 回报程度
    satisfaction_score: float = 50    # 满意度

    # 社交角色
    social_role: SocialRole = SocialRole.PEER

    # 风险标记
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class SocialNetworkAnalysis:
    """社交网络分析结果"""
    # 网络结构
    total_contacts: int = 0
    relation_distribution: Dict[str, int] = field(default_factory=dict)

    # 核心指标
    diversity_score: float = 0       # 关系多样性
    balance_score: float = 0         # 关系平衡度
    support_score: float = 0         # 支持系统强度
    health_score: float = 0          # 整体健康度

    # 关键节点
    anchor_contacts: List[str] = field(default_factory=list)  # 核心支持者
    drainer_contacts: List[str] = field(default_factory=list) # 能量消耗者
    risk_contacts: List[str] = field(default_factory=list)    # 风险关系

    # 建议
    strategic_recommendations: List[str] = field(default_factory=list)


class SocialNetworkAnalyzer:
    """
    社交网络分析器

    分析用户的完整社交网络
    """

    def __init__(self):
        pass

    def analyze(self, classifications: Dict[str, dict]) -> SocialNetworkAnalysis:
        """
        分析社交网络

        Args:
            classifications: 分类结果字典

        Returns:
            社交网络分析结果
        """
        result = SocialNetworkAnalysis()
        nodes = self._build_nodes(classifications)

        result.total_contacts = len(nodes)

        # 1. 关系分布
        result.relation_distribution = self._compute_distribution(nodes)

        # 2. 多样性评分
        result.diversity_score = self._compute_diversity(nodes)

        # 3. 平衡度评分
        result.balance_score = self._compute_balance(nodes)

        # 4. 支持系统强度
        result.support_score = self._compute_support(nodes)

        # 5. 整体健康度
        result.health_score = self._compute_health(nodes)

        # 6. 识别关键节点
        result.anchor_contacts = [n.contact_id for n in nodes
                                  if n.social_role == SocialRole.ANCHOR]
        result.drainer_contacts = [n.contact_id for n in nodes
                                   if n.social_role == SocialRole.DRAINER]
        result.risk_contacts = [n.contact_id for n in nodes
                                if n.health_score < 40]

        # 7. 生成建议
        result.strategic_recommendations = self._generate_recommendations(result, nodes)

        return result

    def _build_nodes(self, classifications: Dict[str, dict]) -> List[RelationNode]:
        """构建关系节点"""
        nodes = []

        for contact_id, data in classifications.items():
            # 只分析用户自己（_me 结尾的）
            if not contact_id.endswith('_me'):
                continue

            node = RelationNode(
                contact_id=contact_id,
                name=data.get('name', contact_id),
                category=data.get('moe', {}).get('attachment', 'UNKNOWN'),
                behavior_pattern=data.get('moe', {}).get('behavior', 'BALANCED'),
                initiative_ratio=data.get('moe', {}).get('initiative_ratio', 0.5)
            )

            # 计算健康度和社交角色
            self._evaluate_node(node, data)
            nodes.append(node)

        return nodes

    def _evaluate_node(self, node: RelationNode, data: dict):
        """评估单个节点"""
        moe = data.get('moe', {})
        big_five = data.get('big_five', {})

        # 计算投入分数（基于主动程度）
        init_ratio = moe.get('initiative_ratio', 0.5)
        node.investment_score = init_ratio * 100

        # 计算回报分数（反向）
        node.return_score = (1 - init_ratio) * 100

        # 计算健康度
        # 高主动 + 低回报 = 不健康
        if init_ratio > 0.8:
            node.health_score = 30 + (1 - init_ratio) * 40
            node.warnings.append("高投入低回报")
            node.social_role = SocialRole.DRAINER
        elif init_ratio > 0.6:
            node.health_score = 50
            node.warnings.append("投入略高于回报")
        elif init_ratio < 0.4:
            node.health_score = 60 + init_ratio * 30
            node.social_role = SocialRole.ANCHOR
        else:
            node.health_score = 70
            node.social_role = SocialRole.PEER

        # 根据关系类型调整
        relation_type = data.get('relation_type', 'ACQUAINTANCE')
        if relation_type == 'ROMANTIC':
            if init_ratio > 0.7:
                node.health_score -= 15
                node.warnings.append("恋爱关系失衡")
        elif relation_type == 'FRIENDSHIP':
            node.health_score += 5

        # 根据依恋类型调整
        attachment = moe.get('attachment', 'SECURE')
        if attachment == 'ANXIOUS':
            node.warnings.append("焦虑依恋模式")
            node.health_score -= 10

        # 满意度
        node.satisfaction_score = node.health_score

        # 生成个人建议
        if node.health_score < 40:
            node.recommendations.append("考虑减少投入或调整期望")
        elif node.health_score < 60:
            node.recommendations.append("关注互动平衡")

    def _compute_distribution(self, nodes: List[RelationNode]) -> Dict[str, int]:
        """计算关系分布"""
        dist = defaultdict(int)
        for node in nodes:
            dist[node.category] += 1
        return dict(dist)

    def _compute_diversity(self, nodes: List[RelationNode]) -> float:
        """
        计算关系多样性

        多样性 = 不同关系类型的熵
        """
        if not nodes:
            return 0

        dist = self._compute_distribution(nodes)
        total = sum(dist.values())

        if total == 0:
            return 0

        # 计算熵
        import math
        entropy = 0
        for count in dist.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        # 归一化到 0-100
        max_entropy = math.log2(len(dist)) if len(dist) > 1 else 1
        return min(entropy / max_entropy * 100, 100) if max_entropy > 0 else 0

    def _compute_balance(self, nodes: List[RelationNode]) -> float:
        """计算整体平衡度"""
        if not nodes:
            return 50

        balances = []
        for node in nodes:
            balance = 100 - abs(node.investment_score - node.return_score)
            balances.append(balance)

        return sum(balances) / len(balances)

    def _compute_support(self, nodes: List[RelationNode]) -> float:
        """计算支持系统强度"""
        if not nodes:
            return 0

        # 核心支持者数量
        anchor_count = sum(1 for n in nodes if n.social_role == SocialRole.ANCHOR)
        healthy_count = sum(1 for n in nodes if n.health_score >= 60)

        # 支持分数
        score = min(anchor_count * 20 + healthy_count * 10, 100)
        return score

    def _compute_health(self, nodes: List[RelationNode]) -> float:
        """计算整体健康度"""
        if not nodes:
            return 50

        scores = [n.health_score for n in nodes]
        return sum(scores) / len(scores)

    def _generate_recommendations(self,
                                   result: SocialNetworkAnalysis,
                                   nodes: List[RelationNode]) -> List[str]:
        """生成策略建议"""
        recommendations = []

        # 1. 多样性建议
        if result.diversity_score < 30:
            recommendations.append("📉 社交圈单一，建议拓展不同类型的关系")
        elif result.diversity_score > 70:
            recommendations.append("✅ 社交圈多样，关系类型丰富")

        # 2. 平衡性建议
        if result.balance_score < 50:
            recommendations.append("⚖️ 整体投入不平衡，考虑调整对某些关系的投入")
        else:
            recommendations.append("✅ 整体关系平衡良好")

        # 3. 支持系统建议
        if result.support_score < 30:
            recommendations.append("🆘 支持系统薄弱，需要建立更多支持性关系")
        elif result.support_score < 60:
            recommendations.append("💡 支持系统一般，可以主动维护核心关系")

        # 4. 风险关系处理
        if result.risk_contacts:
            recommendations.append(f"⚠️ 发现 {len(result.risk_contacts)} 个风险关系，建议评估是否值得继续投入")

        # 5. 能量消耗者处理
        if result.drainer_contacts:
            recommendations.append(f"🔋 {len(result.drainer_contacts)} 个关系可能消耗你的能量，考虑设定边界")

        # 6. 核心支持者
        if result.anchor_contacts:
            recommendations.append(f"❤️ 珍惜你的 {len(result.anchor_contacts)} 个核心支持者")

        return recommendations


def create_social_network_report(analysis: SocialNetworkAnalysis) -> str:
    """生成社交网络报告"""
    lines = []
    lines.append("=" * 60)
    lines.append("社交网络分析报告")
    lines.append("=" * 60)

    lines.append(f"\n[概览]")
    lines.append(f"  联系人总数: {analysis.total_contacts}")

    lines.append(f"\n[关系分布]")
    for rel_type, count in analysis.relation_distribution.items():
        lines.append(f"  {rel_type}: {count}人")

    lines.append(f"\n[核心指标]")
    lines.append(f"  多样性: {analysis.diversity_score:.0f}/100")
    lines.append(f"  平衡度: {analysis.balance_score:.0f}/100")
    lines.append(f"  支持系统: {analysis.support_score:.0f}/100")
    lines.append(f"  整体健康: {analysis.health_score:.0f}/100")

    if analysis.anchor_contacts:
        lines.append(f"\n[核心支持者]")
        for contact_id in analysis.anchor_contacts:
            lines.append(f"  ❤️ {contact_id}")

    if analysis.drainer_contacts:
        lines.append(f"\n[能量消耗者]")
        for contact_id in analysis.drainer_contacts:
            lines.append(f"  🔋 {contact_id}")

    if analysis.risk_contacts:
        lines.append(f"\n[风险关系]")
        for contact_id in analysis.risk_contacts:
            lines.append(f"  ⚠️ {contact_id}")

    lines.append(f"\n[策略建议]")
    for rec in analysis.strategic_recommendations:
        lines.append(f"  {rec}")

    return "\n".join(lines)


__all__ = [
    'RelationHealth', 'SocialRole', 'RelationNode',
    'SocialNetworkAnalysis', 'SocialNetworkAnalyzer',
    'create_social_network_report'
]