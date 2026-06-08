"""
动态图神经网络社交圈分析系统

核心思想：
- 每个人是一个社交圈的锚点
- 社交圈是动态演化的时序图
- 用 GNN 学习关系模式
- 预测关系变化和风险

架构：
1. 节点：每个人（用户 + 联系人）
2. 边：关系（带权重和属性）
3. 时序：边权重随时间变化
4. 学习：GNN 学习节点表示和边预测

应用：
- 关系预测：这段关系会如何发展？
- 风险预警：谁的社交圈正在恶化？
- 社区发现：识别支持系统/有毒圈子
- 动态追踪：关系网络的演化趋势
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import json
from pathlib import Path
import pickle


# ============================================================
# 图结构定义
# ============================================================

@dataclass
class Node:
    """图节点 - 代表一个人"""
    node_id: str
    name: str
    node_type: str = "contact"  # self, contact, group

    # 节点属性
    features: np.ndarray = field(default_factory=lambda: np.zeros(16))

    # 时序特征
    feature_history: List[Tuple[datetime, np.ndarray]] = field(default_factory=list)

    # 社区归属
    community_id: int = -1


@dataclass
class Edge:
    """图边 - 代表关系"""
    source: str  # 源节点
    target: str  # 目标节点

    # 边属性
    weight: float = 1.0
    edge_type: str = "general"  # romantic, family, friend, work, etc.

    # 能量属性
    energy_flow: float = 0.0  # 正=流入，负=流出
    investment: float = 0.0
    return_received: float = 0.0

    # 时序
    weight_history: List[Tuple[datetime, float]] = field(default_factory=list)
    energy_history: List[Tuple[datetime, float]] = field(default_factory=list)

    # 关系状态
    status: str = "active"  # active, declining, stable, growing


@dataclass
class DynamicGraph:
    """动态图 - 时序社交网络"""
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: Dict[Tuple[str, str], Edge] = field(default_factory=dict)

    # 图级特征
    num_nodes: int = 0
    num_edges: int = 0

    # 时序快照
    snapshots: List[Tuple[datetime, Dict]] = field(default_factory=list)

    # 全局指标
    total_energy: float = 0.0
    density: float = 0.0
    clustering_coefficient: float = 0.0


# ============================================================
# 图构建器
# ============================================================

class SocialGraphBuilder:
    """社交图谱构建器"""

    def build_from_data(
        self,
        contacts_data: Dict[str, Dict],
        quantifications: Dict[str, Dict],
        self_id: str = "self"
    ) -> DynamicGraph:
        """
        从联系人数据构建动态图

        Args:
            contacts_data: 联系人统计数据
            quantifications: 量化结果
            self_id: 用户自身ID
        """
        graph = DynamicGraph()

        # 1. 添加自身节点（社交圈锚点）
        self_node = Node(
            node_id=self_id,
            name="我",
            node_type="self",
            features=self._compute_self_features(quantifications)
        )
        graph.nodes[self_id] = self_node

        # 2. 添加联系人节点
        for contact_id, stats in contacts_data.items():
            node = Node(
                node_id=contact_id,
                name=stats.get('name', contact_id),
                node_type="contact",
                features=self._compute_node_features(stats)
            )
            graph.nodes[contact_id] = node

        # 3. 添加边（关系）
        for contact_id, quant in quantifications.items():
            edge = Edge(
                source=self_id,
                target=contact_id,
                weight=self._compute_edge_weight(quant),
                edge_type=quant.get('relation_type', 'general'),
                energy_flow=quant.get('energy_net', 0),
                investment=quant.get('investment_score', 50),
                return_received=quant.get('response_score', 50)
            )
            graph.edges[(self_id, contact_id)] = edge

        # 4. 计算图级特征
        graph.num_nodes = len(graph.nodes)
        graph.num_edges = len(graph.edges)
        graph.total_energy = sum(e.energy_flow for e in graph.edges.values())
        graph.density = self._compute_density(graph)

        return graph

    def _compute_self_features(self, quantifications: Dict) -> np.ndarray:
        """计算自身节点特征"""
        features = np.zeros(16)

        if not quantifications:
            return features

        # 汇总所有关系
        energies = [q.get('energy_net', 0) for q in quantifications.values()]
        investments = [q.get('investment_score', 50) for q in quantifications.values()]
        returns = [q.get('response_score', 50) for q in quantifications.values()]

        features[0] = len(quantifications)  # 联系人数量
        features[1] = np.mean(energies) if energies else 0
        features[2] = np.std(energies) if len(energies) > 1 else 0
        features[3] = np.mean(investments)
        features[4] = np.mean(returns)
        features[5] = sum(1 for e in energies if e > 0)  # 正向关系数
        features[6] = sum(1 for e in energies if e < 0)  # 负向关系数

        return features

    def _compute_node_features(self, stats: Dict) -> np.ndarray:
        """计算联系人节点特征"""
        features = np.zeros(16)

        basic = stats.get('basic', {})
        initiative = stats.get('initiative', {})
        reply = stats.get('reply_speed', {})

        features[0] = basic.get('total_messages', 0)
        features[1] = basic.get('my_ratio', 0.5)
        features[2] = initiative.get('my_start_ratio', 0.5)
        features[3] = reply.get('their_avg_seconds', 3600) / 3600  # 小时
        features[4] = stats.get('scores', {}).get('simp_index', 50)
        features[5] = stats.get('scores', {}).get('loved_index', 50)

        return features

    def _compute_edge_weight(self, quant: Dict) -> float:
        """计算边权重"""
        health = quant.get('health_score', 50)
        energy = quant.get('energy_net', 0)

        # 权重 = 健康度 + 能量归一化
        weight = health / 100 + np.tanh(energy / 500)

        return max(weight, 0.1)

    def _compute_density(self, graph: DynamicGraph) -> float:
        """计算图密度"""
        n = graph.num_nodes
        if n < 2:
            return 0
        max_edges = n * (n - 1) / 2
        return graph.num_edges / max_edges


# ============================================================
# 简化版 GNN（无深度学习框架依赖）
# ============================================================

class SimpleGNN:
    """
    简化版图神经网络

    实现消息传递机制，无需 PyTorch/TF
    用于学习和推理
    """

    def __init__(self, hidden_dim: int = 32):
        self.hidden_dim = hidden_dim
        self.weights = None
        self.trained = False

    def message_passing(
        self,
        graph: DynamicGraph,
        num_hops: int = 2
    ) -> Dict[str, np.ndarray]:
        """
        消息传递

        每个节点聚合邻居信息
        """
        node_embeddings = {}

        # 初始化嵌入
        for node_id, node in graph.nodes.items():
            node_embeddings[node_id] = node.features.copy()

        # 多跳消息传递
        for hop in range(num_hops):
            new_embeddings = {}

            for node_id, node in graph.nodes.items():
                # 聚合邻居
                neighbor_msgs = []

                for (src, tgt), edge in graph.edges.items():
                    if src == node_id:
                        # 出边：消息发给对方
                        msg = node_embeddings[tgt] * edge.weight
                        neighbor_msgs.append(msg)
                    elif tgt == node_id:
                        # 入边：接收消息
                        msg = node_embeddings[src] * edge.weight
                        neighbor_msgs.append(msg)

                if neighbor_msgs:
                    # 平均聚合
                    aggregated = np.mean(neighbor_msgs, axis=0)
                else:
                    aggregated = np.zeros_like(node_embeddings[node_id])

                # 更新嵌入
                new_embeddings[node_id] = node_embeddings[node_id] + aggregated

            node_embeddings = new_embeddings

        return node_embeddings

    def compute_node_importance(
        self,
        graph: DynamicGraph,
        node_embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        计算节点重要性（PageRank 风格）
        """
        importance = {}

        # 基于度中心性
        for node_id in graph.nodes:
            degree = sum(
                1 for (s, t) in graph.edges
                if s == node_id or t == node_id
            )
            # 加上能量流
            energy = sum(
                e.energy_flow for (s, t), e in graph.edges.items()
                if t == node_id
            )
            importance[node_id] = degree + max(energy / 100, 0)

        # 归一化
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}

        return importance

    def predict_edge_evolution(
        self,
        graph: DynamicGraph,
        edge: Edge
    ) -> Dict:
        """
        预测边的演化

        基于历史趋势预测关系走向
        """
        prediction = {
            'trend': 'stable',
            'confidence': 0.5,
            'risk': 0.0,
            'timeline': 'unknown'
        }

        # 基于能量历史
        if len(edge.energy_history) >= 3:
            energies = [e for _, e in edge.energy_history[-5:]]
            slope = np.polyfit(range(len(energies)), energies, 1)[0]

            if slope < -10:
                prediction['trend'] = 'declining'
                prediction['risk'] = min(abs(slope) / 10, 100)
            elif slope > 10:
                prediction['trend'] = 'growing'
                prediction['risk'] = 0
            else:
                prediction['trend'] = 'stable'

        # 基于当前状态
        if edge.energy_flow < -200:
            prediction['risk'] = max(prediction['risk'], 60)
            prediction['timeline'] = '需要尽快调整'
        elif edge.energy_flow < -100:
            prediction['risk'] = max(prediction['risk'], 40)

        return prediction


# ============================================================
# 动态演化追踪
# ============================================================

class DynamicEvolutionTracker:
    """动态演化追踪器"""

    def __init__(self):
        self.history = []

    def add_snapshot(
        self,
        graph: DynamicGraph,
        timestamp: datetime = None
    ):
        """添加时间快照"""
        if timestamp is None:
            timestamp = datetime.now()

        snapshot = {
            'timestamp': timestamp,
            'num_nodes': graph.num_nodes,
            'num_edges': graph.num_edges,
            'total_energy': graph.total_energy,
            'density': graph.density,
            'node_energies': {
                tgt: e.energy_flow
                for (src, tgt), e in graph.edges.items()
            }
        }

        self.history.append(snapshot)
        graph.snapshots.append((timestamp, snapshot))

    def compute_evolution_metrics(self) -> Dict:
        """计算演化指标"""
        if len(self.history) < 2:
            return {'status': 'insufficient_data'}

        recent = self.history[-1]
        previous = self.history[-2]

        metrics = {
            'energy_change': recent['total_energy'] - previous['total_energy'],
            'energy_trend': 'increasing' if recent['total_energy'] > previous['total_energy'] else 'decreasing',
            'density_change': recent['density'] - previous['density'],
            'node_changes': {}
        }

        # 每个节点的能量变化
        for node_id in recent['node_energies']:
            prev_energy = previous['node_energies'].get(node_id, 0)
            curr_energy = recent['node_energies'][node_id]
            change = curr_energy - prev_energy

            if abs(change) > 50:  # 显著变化
                metrics['node_changes'][node_id] = {
                    'change': change,
                    'trend': 'improving' if change > 0 else 'declining'
                }

        return metrics

    def detect_critical_changes(self) -> List[Dict]:
        """检测关键变化"""
        alerts = []

        if len(self.history) < 2:
            return alerts

        recent = self.history[-1]
        previous = self.history[-2]

        # 1. 能量骤降
        energy_drop = previous['total_energy'] - recent['total_energy']
        if energy_drop > 200:
            alerts.append({
                'type': 'energy_crisis',
                'severity': 'high',
                'message': f'社交能量骤降 {energy_drop:.0f}，可能面临关系危机'
            })

        # 2. 节点消失（关系断开）
        prev_nodes = set(previous['node_energies'].keys())
        curr_nodes = set(recent['node_energies'].keys())
        lost_nodes = prev_nodes - curr_nodes

        if lost_nodes:
            alerts.append({
                'type': 'relationship_loss',
                'severity': 'medium',
                'message': f'失去与 {len(lost_nodes)} 人的联系'
            })

        # 3. 整体恶化
        declining_count = sum(
            1 for nid, change in self.compute_evolution_metrics().get('node_changes', {}).items()
            if change['trend'] == 'declining'
        )
        if declining_count >= 3:
            alerts.append({
                'type': 'network_collapse',
                'severity': 'critical',
                'message': f'{declining_count} 段关系同时恶化，需关注心理健康'
            })

        return alerts


# ============================================================
# 社区发现
# ============================================================

class CommunityDetector:
    """社区发现器"""

    def detect_communities(self, graph: DynamicGraph) -> Dict[int, List[str]]:
        """
        检测社交圈内的社区

        返回：社区ID -> 成员列表
        """
        communities = defaultdict(list)

        # 简单的标签传播算法
        labels = {node_id: i for i, node_id in enumerate(graph.nodes)}

        # 迭代传播
        for _ in range(10):
            new_labels = {}

            for node_id in graph.nodes:
                # 收集邻居标签
                neighbor_labels = []

                for (src, tgt), edge in graph.edges.items():
                    if src == node_id:
                        neighbor_labels.append(labels[tgt])
                    elif tgt == node_id:
                        neighbor_labels.append(labels[src])

                if neighbor_labels:
                    # 选择最常见的标签
                    from collections import Counter
                    most_common = Counter(neighbor_labels).most_common(1)[0][0]
                    new_labels[node_id] = most_common
                else:
                    new_labels[node_id] = labels[node_id]

            labels = new_labels

        # 按社区分组
        for node_id, label in labels.items():
            communities[label].append(node_id)
            graph.nodes[node_id].community_id = label

        return dict(communities)

    def identify_support_system(self, graph: DynamicGraph) -> Dict:
        """识别支持系统"""
        support = {
            'core': [],      # 核心支持（高能量流入）
            'peripheral': [], # 边缘支持
            'draining': []   # 消耗型关系
        }

        for (src, tgt), edge in graph.edges.items():
            if src == 'self':
                if edge.energy_flow > 100:
                    support['core'].append({
                        'contact': tgt,
                        'energy': edge.energy_flow,
                        'type': edge.edge_type
                    })
                elif edge.energy_flow > 0:
                    support['peripheral'].append({
                        'contact': tgt,
                        'energy': edge.energy_flow
                    })
                else:
                    support['draining'].append({
                        'contact': tgt,
                        'energy': edge.energy_flow,
                        'risk': abs(edge.energy_flow) / 10
                    })

        return support


# ============================================================
# 主分析器
# ============================================================

class DynamicSocialAnalyzer:
    """动态社交网络分析器"""

    def __init__(self):
        self.graph_builder = SocialGraphBuilder()
        self.gnn = SimpleGNN()
        self.evolution_tracker = DynamicEvolutionTracker()
        self.community_detector = CommunityDetector()

    def analyze(
        self,
        contacts_data: Dict[str, Dict],
        quantifications: Dict[str, Dict],
        add_snapshot: bool = True
    ) -> Dict:
        """
        完整分析
        """
        # 1. 构建图
        graph = self.graph_builder.build_from_data(
            contacts_data, quantifications
        )

        # 2. 添加快照
        if add_snapshot:
            self.evolution_tracker.add_snapshot(graph)

        # 3. GNN 消息传递
        node_embeddings = self.gnn.message_passing(graph)

        # 4. 节点重要性
        importance = self.gnn.compute_node_importance(graph, node_embeddings)

        # 5. 边演化预测
        edge_predictions = {}
        for edge_key, edge in graph.edges.items():
            edge_predictions[edge_key] = self.gnn.predict_edge_evolution(graph, edge)

        # 6. 社区发现
        communities = self.community_detector.detect_communities(graph)

        # 7. 支持系统识别
        support_system = self.community_detector.identify_support_system(graph)

        # 8. 演化指标
        evolution = self.evolution_tracker.compute_evolution_metrics()

        # 9. 关键变化警报
        alerts = self.evolution_tracker.detect_critical_changes()

        return {
            'graph': graph,
            'node_embeddings': node_embeddings,
            'importance': importance,
            'edge_predictions': edge_predictions,
            'communities': communities,
            'support_system': support_system,
            'evolution': evolution,
            'alerts': alerts
        }

    def predict_network_trajectory(self, days_ahead: int = 30) -> Dict:
        """
        预测网络轨迹
        """
        if len(self.evolution_tracker.history) < 2:
            return {'status': 'insufficient_data'}

        # 基于历史预测未来
        energies = [h['total_energy'] for h in self.evolution_tracker.history]
        times = list(range(len(energies)))

        # 线性外推
        slope, intercept = np.polyfit(times, energies, 1)
        future_energy = intercept + slope * (len(times) + days_ahead / 7)

        prediction = {
            'current_energy': energies[-1],
            'predicted_energy': future_energy,
            'trend': 'improving' if slope > 0 else 'declining' if slope < 0 else 'stable',
            'slope_per_week': slope,
            'days_ahead': days_ahead
        }

        # 风险评估
        if prediction['trend'] == 'declining' and abs(slope) > 50:
            prediction['risk_level'] = 'high'
            prediction['recommendation'] = '需要主动干预：增加正向社交，减少消耗型关系'
        elif prediction['trend'] == 'declining':
            prediction['risk_level'] = 'medium'
            prediction['recommendation'] = '关注趋势，考虑调整社交策略'
        else:
            prediction['risk_level'] = 'low'
            prediction['recommendation'] = '继续保持'

        return prediction


# ============================================================
# 可视化数据生成
# ============================================================

def generate_gnn_visualization(analysis: Dict) -> Dict:
    """生成 GNN 可视化数据"""
    graph = analysis['graph']

    viz_data = {
        'nodes': [],
        'links': [],
        'metrics': {},
        'communities': {},
        'predictions': []
    }

    # 节点
    for node_id, node in graph.nodes.items():
        viz_data['nodes'].append({
            'id': node_id,
            'name': node.name,
            'type': node.node_type,
            'community': node.community_id,
            'importance': analysis['importance'].get(node_id, 0),
            'size': 10 + analysis['importance'].get(node_id, 0) * 100
        })

    # 边
    for (src, tgt), edge in graph.edges.items():
        pred = analysis['edge_predictions'].get((src, tgt), {})

        viz_data['links'].append({
            'source': src,
            'target': tgt,
            'weight': edge.weight,
            'energy': edge.energy_flow,
            'type': edge.edge_type,
            'trend': pred.get('trend', 'stable'),
            'risk': pred.get('risk', 0),
            'color': '#4CAF50' if edge.energy_flow > 0 else '#f44336'
        })

    # 指标
    viz_data['metrics'] = {
        'total_energy': graph.total_energy,
        'density': graph.density,
        'num_nodes': graph.num_nodes,
        'num_edges': graph.num_edges
    }

    # 社区
    for comm_id, members in analysis['communities'].items():
        viz_data['communities'][comm_id] = members

    # 预测
    if 'trajectory' in analysis:
        viz_data['predictions'].append(analysis['trajectory'])

    return viz_data


def create_gnn_report(analysis: Dict) -> str:
    """生成 GNN 分析报告"""
    lines = []

    lines.append("=" * 60)
    lines.append("动态社交网络分析报告")
    lines.append("=" * 60)

    # 图概览
    graph = analysis['graph']
    lines.append(f"\n【网络概览】")
    lines.append(f"  节点数: {graph.num_nodes}")
    lines.append(f"  边数: {graph.num_edges}")
    lines.append(f"  密度: {graph.density:.2f}")
    lines.append(f"  总能量: {graph.total_energy:+.0f}")

    # 支持系统
    support = analysis['support_system']
    lines.append(f"\n【支持系统分析】")
    lines.append(f"  核心支持: {len(support['core'])}人")
    for s in support['core']:
        lines.append(f"    • {s['contact']}: {s['energy']:+.0f} ({s['type']})")

    lines.append(f"  边缘支持: {len(support['peripheral'])}人")
    lines.append(f"  消耗型关系: {len(support['draining'])}人")
    for s in support['draining'][:3]:
        lines.append(f"    ⚠️ {s['contact']}: {s['energy']:+.0f} (风险: {s['risk']:.0f})")

    # 社区
    communities = analysis['communities']
    lines.append(f"\n【社交圈结构】")
    lines.append(f"  检测到 {len(communities)} 个社交圈")
    for comm_id, members in communities.items():
        lines.append(f"  圈子 {comm_id + 1}: {len(members)}人")

    # 演化趋势
    evolution = analysis['evolution']
    lines.append(f"\n【演化趋势】")
    lines.append(f"  能量变化: {evolution.get('energy_change', 0):+.0f}")
    lines.append(f"  趋势方向: {evolution.get('energy_trend', 'unknown')}")

    # 警报
    alerts = analysis['alerts']
    if alerts:
        lines.append(f"\n【⚠️ 关键警报】")
        for alert in alerts:
            lines.append(f"  [{alert['severity']}] {alert['message']}")

    return "\n".join(lines)


__all__ = [
    'Node', 'Edge', 'DynamicGraph',
    'SocialGraphBuilder', 'SimpleGNN',
    'DynamicEvolutionTracker', 'CommunityDetector',
    'DynamicSocialAnalyzer',
    'generate_gnn_visualization', 'create_gnn_report'
]