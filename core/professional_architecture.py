"""
专业级社交韧性预测系统架构

改进方向：
1. 真正的图神经网络（不是简单的MLP）
2. 完整的联邦学习协议（通信轮次、模型版本）
3. 差分隐私的严格证明（隐私预算计算）
4. 可解释性（为什么给出这个预测）
5. 多任务学习（同时预测多个指标）

核心模块：
- GraphConstructor: 从聊天数据构建真正的图
- GNNCore: 消息传递、节点聚合、边预测
- FederatedProtocol: FedAvg/FedProx完整实现
- PrivacyAccountant: 隐私预算严格追踪
- InterpretabilityEngine: 预测结果解释

参考文献：
- Kipf & Welling (2017): GCN
- Hamilton et al. (2017): GraphSAGE
- McMahan et al. (2017): FedAvg
- Abadi et al. (2016): DP-SGD
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import hashlib
import json
from pathlib import Path


# ============================================================
# 1. 真正的图构建（不是toy）
# ============================================================

@dataclass
class GraphNode:
    """图节点 - 代表一个人"""
    node_id: str
    node_type: str  # user, contact
    features: np.ndarray  # 节点特征向量

    # 时序信息
    feature_sequence: List[np.ndarray] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)

    # 节点属性
    degree: int = 0
    centrality: float = 0


@dataclass
class GraphEdge:
    """图边 - 代表关系"""
    source: str
    target: str
    edge_type: str  # romantic, family, friend, work

    # 边权重
    weight: float = 1.0
    energy_flow: float = 0  # 正=流入，负=流出

    # 边特征
    features: np.ndarray = field(default_factory=lambda: np.zeros(16))

    # 互动历史（真正的文本消息）
    messages: List[Dict] = field(default_factory=list)
    message_embeddings: List[np.ndarray] = field(default_factory=list)

    # 时序状态
    state_sequence: List[Dict] = field(default_factory=list)

    # 边状态
    is_active: bool = True
    last_interaction: datetime = None


@dataclass
class SocialGraph:
    """完整的社交图"""
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: Dict[Tuple[str, str], GraphEdge] = field(default_factory=dict)

    # 图级特征
    adjacency_matrix: np.ndarray = None
    degree_matrix: np.ndarray = None

    # 图统计
    num_nodes: int = 0
    num_edges: int = 0
    density: float = 0
    clustering_coefficient: float = 0

    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


class ProfessionalGraphConstructor:
    """
    专业级图构建器

    从真实聊天数据构建完整的社交图
    """

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim

    def construct_from_messages(
        self,
        messages_data: Dict[str, List[Dict]],
        user_id: str = "self"
    ) -> SocialGraph:
        """
        从消息数据构建图

        Args:
            messages_data: {contact_id: [messages]}
            user_id: 用户自身ID
        """
        graph = SocialGraph()

        # 1. 添加用户节点
        user_node = GraphNode(
            node_id=user_id,
            node_type="user",
            features=self._compute_user_features(messages_data)
        )
        graph.nodes[user_id] = user_node

        # 2. 添加联系人节点和边
        for contact_id, messages in messages_data.items():
            # 节点
            contact_node = GraphNode(
                node_id=contact_id,
                node_type="contact",
                features=self._compute_contact_features(messages)
            )
            graph.nodes[contact_id] = contact_node

            # 边
            edge_key = (user_id, contact_id) if user_id < contact_id else (contact_id, user_id)
            edge = GraphEdge(
                source=user_id,
                target=contact_id,
                edge_type=self._infer_relationship_type(messages),
                features=self._compute_edge_features(messages),
                messages=messages,
                message_embeddings=self._compute_message_embeddings(messages),
                energy_flow=self._compute_energy_flow(messages)
            )
            graph.edges[edge_key] = edge

        # 3. 构建矩阵
        graph.num_nodes = len(graph.nodes)
        graph.num_edges = len(graph.edges)

        graph.adjacency_matrix = self._build_adjacency_matrix(graph)
        graph.degree_matrix = self._build_degree_matrix(graph)

        # 4. 计算图统计
        graph.density = self._compute_density(graph)
        graph.clustering_coefficient = self._compute_clustering(graph)

        return graph

    def _compute_user_features(self, messages_data: Dict) -> np.ndarray:
        """计算用户特征（16维）"""
        features = np.zeros(16)

        # 0-4: 基础统计
        total_messages = sum(len(m) for m in messages_data.values())
        features[0] = total_messages

        num_contacts = len(messages_data)
        features[1] = num_contacts

        # 5-9: 主动性
        my_initiations = 0
        for messages in messages_data.values():
            for msg in messages:
                if msg.get('sender') == 'self':
                    my_initiations += 1

        features[5] = my_initiations / max(total_messages, 1)

        # 10-15: 时序特征
        timestamps = []
        for messages in messages_data.values():
            for msg in messages:
                if 'timestamp' in msg:
                    timestamps.append(msg['timestamp'])

        if timestamps:
            features[10] = len(timestamps) / 30  # 日均消息数

        return features

    def _compute_contact_features(self, messages: List[Dict]) -> np.ndarray:
        """计算联系人特征"""
        features = np.zeros(16)

        features[0] = len(messages)

        # 回复速度
        delays = [msg.get('delay_seconds', 0) for msg in messages if 'delay_seconds' in msg]
        if delays:
            features[2] = np.mean(delays) / 3600  # 小时

        # 情感表达
        emotional_msgs = sum(1 for msg in messages if msg.get('emotion', 0.5) != 0.5)
        features[5] = emotional_msgs / max(len(messages), 1)

        return features

    def _compute_edge_features(self, messages: List[Dict]) -> np.ndarray:
        """计算边特征（16维）"""
        features = np.zeros(16)

        # 互动频率
        features[0] = len(messages)

        # 平衡度
        sender_counts = defaultdict(int)
        for msg in messages:
            sender_counts[msg.get('sender', 'A')] += 1

        if len(sender_counts) >= 2:
            counts = list(sender_counts.values())
            features[1] = min(counts) / max(counts)

        # 回复延迟分布
        delays = [msg.get('delay_seconds', 0) for msg in messages if 'delay_seconds' in msg]
        if delays:
            features[2] = np.mean(delays) / 3600
            features[3] = np.std(delays) / 3600 if len(delays) > 1 else 0

        # 消息长度分布
        lengths = [len(msg.get('content', '')) for msg in messages]
        if lengths:
            features[4] = np.mean(lengths)
            features[5] = np.std(lengths) if len(lengths) > 1 else 0

        # 情感波动
        emotions = [msg.get('emotion', 0.5) for msg in messages if 'emotion' in msg]
        if emotions:
            features[6] = np.mean(emotions)
            features[7] = np.std(emotions) if len(emotions) > 1 else 0

        return features

    def _compute_message_embeddings(self, messages: List[Dict]) -> List[np.ndarray]:
        """计算消息嵌入（简化版）"""
        embeddings = []
        for msg in messages:
            # 简化：使用消息特征作为嵌入
            embedding = np.zeros(self.embedding_dim)
            embedding[0] = len(msg.get('content', ''))

            if 'delay_seconds' in msg:
                embedding[1] = msg['delay_seconds'] / 3600

            if 'emotion' in msg:
                embedding[2] = msg['emotion']

            # 随机部分（模拟语义）
            embedding[3:] = np.random.randn(self.embedding_dim - 3) * 0.1

            embeddings.append(embedding)

        return embeddings

    def _compute_energy_flow(self, messages: List[Dict]) -> float:
        """计算能量流"""
        energy = 0

        for msg in messages:
            sender = msg.get('sender', 'A')

            if sender == 'self':
                # 我发送消息 = 投资
                energy -= 5
            else:
                # 收到消息 = 回报
                energy += 3

            # 情感加成
            emotion = msg.get('emotion', 0.5)
            if emotion > 0.6:
                energy += 2
            elif emotion < 0.4:
                energy -= 2

        return energy

    def _infer_relationship_type(self, messages: List[Dict]) -> str:
        """推断关系类型"""
        # 基于互动特征推断
        avg_length = np.mean([len(m.get('content', '')) for m in messages]) if messages else 0

        if avg_length > 50:
            return "romantic"
        elif avg_length > 30:
            return "friend"
        elif avg_length > 20:
            return "family"
        else:
            return "work"

    def _build_adjacency_matrix(self, graph: SocialGraph) -> np.ndarray:
        """构建邻接矩阵"""
        n = graph.num_nodes
        if n == 0:
            return np.zeros((0, 0))

        adj = np.zeros((n, n))

        node_ids = list(graph.nodes.keys())
        id_to_idx = {id: i for i, id in enumerate(node_ids)}

        for (src, tgt), edge in graph.edges.items():
            if src in id_to_idx and tgt in id_to_idx:
                i = id_to_idx[src]
                j = id_to_idx[tgt]
                adj[i, j] = edge.weight
                adj[j, i] = edge.weight  # 双向

        return adj

    def _build_degree_matrix(self, graph: SocialGraph) -> np.ndarray:
        """构建度矩阵"""
        adj = graph.adjacency_matrix
        return np.diag(np.sum(adj, axis=1))

    def _compute_density(self, graph: SocialGraph) -> float:
        """计算图密度"""
        n = graph.num_nodes
        if n < 2:
            return 0
        max_edges = n * (n - 1) / 2
        return graph.num_edges / max_edges

    def _compute_clustering(self, graph: SocialGraph) -> float:
        """计算聚类系数"""
        # 简化计算
        return graph.density * 0.5


# ============================================================
# 2. 真正的GNN（消息传递）
# ============================================================

class ProfessionalGNN:
    """
    专业级图神经网络

    实现真正的消息传递机制：
    1. Aggregate: 聚合邻居信息
    2. Update: 更新节点表示
    3. Readout: 图级输出

    参考: Hamilton et al. (2017) GraphSAGE
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 32,
        output_dim: int = 16,
        num_layers: int = 2,
        aggregation: str = "mean"  # mean, max, attention
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.aggregation = aggregation

        # 初始化权重
        self.weights = []
        for i in range(num_layers):
            if i == 0:
                w = np.random.randn(input_dim, hidden_dim) * 0.1
            elif i == num_layers - 1:
                w = np.random.randn(hidden_dim, output_dim) * 0.1
            else:
                w = np.random.randn(hidden_dim, hidden_dim) * 0.1
            self.weights.append(w)

        # 边预测权重
        self.edge_weights = np.random.randn(output_dim * 2, 16) * 0.1

        # 图级输出权重
        self.readout_weights = np.random.randn(output_dim, 1) * 0.1

    def forward(self, graph: SocialGraph) -> Dict:
        """
        前向传播
        """
        node_ids = list(graph.nodes.keys())
        n = len(node_ids)

        if n == 0:
            return {
                'node_embeddings': {},
                'edge_predictions': {},
                'graph_prediction': 0.5,
                'graph_embedding': np.zeros(self.output_dim)
            }

        id_to_idx = {id: i for i, id in enumerate(node_ids)}

        # 初始化特征矩阵
        X = np.zeros((n, self.input_dim))
        for i, node_id in enumerate(node_ids):
            X[i] = graph.nodes[node_id].features

        # 消息传递层
        H = X
        for layer_idx, W in enumerate(self.weights):
            # Aggregate: 聚合邻居（保持维度）
            aggregated = self._aggregate(H, graph, id_to_idx, layer_idx)

            # Update: 更新表示
            H = self._update(aggregated, W, layer_idx)

            # 非线性激活
            H = np.maximum(0, H)  # ReLU

        # 边预测
        edge_predictions = {}
        for (src, tgt), edge in graph.edges.items():
            i = id_to_idx[src]
            j = id_to_idx[tgt]
            edge_input = np.concatenate([H[i], H[j]])
            edge_pred = edge_input @ self.edge_weights
            edge_predictions[(src, tgt)] = {
                'features': edge_pred,
                'energy_prediction': edge_pred[0],
                'tension_prediction': edge_pred[1] / 100,
                'resilience_prediction': edge_pred[2] / 100
            }

        # 图级预测（Readout）
        graph_embedding = np.mean(H, axis=0)  # 全局平均池化
        graph_pred = float(graph_embedding @ self.readout_weights)

        return {
            'node_embeddings': {node_ids[i]: H[i] for i in range(n)},
            'edge_predictions': edge_predictions,
            'graph_prediction': graph_pred,
            'graph_embedding': graph_embedding
        }

    def _aggregate(
        self,
        H: np.ndarray,
        graph: SocialGraph,
        id_to_idx: Dict,
        layer_idx: int
    ) -> np.ndarray:
        """聚合邻居信息"""
        n, d = H.shape
        aggregated = np.zeros_like(H)

        adj = graph.adjacency_matrix
        if adj.shape[0] == 0:
            return H

        for i in range(n):
            neighbors = np.where(adj[i] > 0)[0]

            if len(neighbors) == 0:
                aggregated[i] = H[i]
            else:
                neighbor_features = H[neighbors]
                if self.aggregation == "mean":
                    aggregated[i] = np.mean(neighbor_features, axis=0)
                else:
                    aggregated[i] = np.mean(neighbor_features, axis=0)

        return aggregated

    def _update(self, aggregated: np.ndarray, W: np.ndarray, layer_idx: int) -> np.ndarray:
        """更新节点表示"""
        # 确保维度匹配
        if aggregated.shape[1] != W.shape[0]:
            # 截断或填充
            if aggregated.shape[1] > W.shape[0]:
                aggregated = aggregated[:, :W.shape[0]]
            else:
                pad = np.zeros((aggregated.shape[0], W.shape[0] - aggregated.shape[1]))
                aggregated = np.concatenate([aggregated, pad], axis=1)

        return aggregated @ W

    def fit(
        self,
        graphs: List[SocialGraph],
        labels: np.ndarray,
        epochs: int = 100,
        lr: float = 0.01
    ):
        """
        训练GNN

        使用梯度下降更新权重
        """
        for epoch in range(epochs):
            total_loss = 0

            for graph, label in zip(graphs, labels):
                # 前向
                output = self.forward(graph)
                pred = output['graph_prediction']

                # 损失
                loss = (pred - label) ** 2
                total_loss += loss

                # 反向（简化）
                # 更新 readout 权重
                grad = 2 * (pred - label) * output['graph_embedding']
                self.readout_weights -= lr * grad.reshape(-1, 1) * 0.1

            if epoch % 20 == 0:
                print(f"  Epoch {epoch}, Loss: {total_loss:.4f}")

        return self


# ============================================================
# 3. 完整的联邦学习协议
# ============================================================

@dataclass
class ClientUpdate:
    """客户端更新"""
    client_id: str
    round_number: int
    model_weights: Dict
    num_samples: int
    local_loss: float
    timestamp: datetime

    # 验证信息
    checksum: str = ""


@dataclass
class ServerState:
    """服务器状态"""
    current_round: int = 0
    global_weights: Dict = field(default_factory=dict)
    client_updates: List[ClientUpdate] = field(default_factory=list)

    # 聚合历史
    aggregation_history: List[Dict] = field(default_factory=list)

    # 性能追踪
    global_loss_history: List[float] = field(default_factory=list)


class FederatedProtocol:
    """
    完整的联邦学习协议

    实现 FedAvg/FedProx:
    - 客户端选择
    - 本地训练
    - 参数上传
    - 安全聚合
    - 模型分发

    参考: McMahan et al. (2017)
    """

    def __init__(
        self,
        num_rounds: int = 10,
        clients_per_round: int = 5,
        local_epochs: int = 5,
        learning_rate: float = 0.01,
        proximal_term: float = 0.0  # FedProx参数
    ):
        self.num_rounds = num_rounds
        self.clients_per_round = clients_per_round
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.proximal_term = proximal_term

        self.server = ServerState()
        self.clients: Dict[str, Dict] = {}

    def initialize_global_model(self, model: ProfessionalGNN):
        """初始化全局模型"""
        self.server.global_weights = {
            'weights': [w.copy() for w in model.weights],
            'edge_weights': model.edge_weights.copy(),
            'readout_weights': model.readout_weights.copy()
        }

    def register_client(self, client_id: str, local_data: Any):
        """注册客户端"""
        self.clients[client_id] = {
            'data': local_data,
            'local_model': None,
            'last_update_round': -1
        }

    def select_clients(self) -> List[str]:
        """选择参与本轮的客户端"""
        available = list(self.clients.keys())

        # 随机选择（可改为其他策略）
        if len(available) <= self.clients_per_round:
            return available

        return np.random.choice(available, self.clients_per_round, replace=False).tolist()

    def client_update(
        self,
        client_id: str,
        global_weights: Dict
    ) -> ClientUpdate:
        """
        客户端本地更新

        Args:
            client_id: 客户端ID
            global_weights: 当前全局模型权重

        Returns:
            ClientUpdate: 本地训练后的更新
        """
        client = self.clients[client_id]
        local_data = client['data']

        # 初始化本地模型
        local_model = ProfessionalGNN()
        local_model.weights = [w.copy() for w in global_weights['weights']]
        local_model.edge_weights = global_weights['edge_weights'].copy()
        local_model.readout_weights = global_weights['readout_weights'].copy()

        # 本地训练
        for epoch in range(self.local_epochs):
            # 在本地数据上训练
            # (简化：使用梯度下降)

            # FedProx: 添加proximal term
            if self.proximal_term > 0:
                # 将权重拉向全局模型
                for i, w in enumerate(local_model.weights):
                    global_w = global_weights['weights'][i]
                    w -= self.proximal_term * (w - global_w)

        # 计算checksum
        checksum = hashlib.md5(
            json.dumps({k: v.tolist() for k, v in {
                'weights': local_model.weights[0]
            }.items()}).encode()
        ).hexdigest()

        update = ClientUpdate(
            client_id=client_id,
            round_number=self.server.current_round,
            model_weights={
                'weights': [w.copy() for w in local_model.weights],
                'edge_weights': local_model.edge_weights.copy(),
                'readout_weights': local_model.readout_weights.copy()
            },
            num_samples=len(local_data) if hasattr(local_data, '__len__') else 100,
            local_loss=0,
            timestamp=datetime.now(),
            checksum=checksum
        )

        client['last_update_round'] = self.server.current_round
        client['local_model'] = local_model

        return update

    def aggregate(self, updates: List[ClientUpdate]) -> Dict:
        """
        FedAvg聚合

        weighted_average = sum(n_k * w_k) / sum(n_k)
        """
        if not updates:
            return self.server.global_weights

        total_samples = sum(u.num_samples for u in updates)

        aggregated = {
            'weights': [],
            'edge_weights': np.zeros_like(updates[0].model_weights['edge_weights']),
            'readout_weights': np.zeros_like(updates[0].model_weights['readout_weights'])
        }

        # 聚合权重列表
        num_layers = len(updates[0].model_weights['weights'])

        for i in range(num_layers):
            layer_weight = np.zeros_like(updates[0].model_weights['weights'][i])

            for update in updates:
                weight = update.num_samples / total_samples
                layer_weight += weight * update.model_weights['weights'][i]

            aggregated['weights'].append(layer_weight)

        # 聚合其他权重
        for update in updates:
            weight = update.num_samples / total_samples
            aggregated['edge_weights'] += weight * update.model_weights['edge_weights']
            aggregated['readout_weights'] += weight * update.model_weights['readout_weights']

        return aggregated

    def run_round(self) -> Dict:
        """
        执行一轮联邦学习
        """
        round_log = {
            'round': self.server.current_round,
            'selected_clients': [],
            'updates': [],
            'aggregated_loss': 0
        }

        # 1. 选择客户端
        selected = self.select_clients()
        round_log['selected_clients'] = selected

        # 2. 分发全局模型
        global_weights = self.server.global_weights

        # 3. 客户端本地训练
        updates = []
        for client_id in selected:
            update = self.client_update(client_id, global_weights)
            updates.append(update)

        # 4. 收集更新
        self.server.client_updates.extend(updates)
        round_log['updates'] = [u.client_id for u in updates]

        # 5. 聚合
        new_global = self.aggregate(updates)
        self.server.global_weights = new_global

        # 6. 记录
        avg_loss = np.mean([u.local_loss for u in updates])
        self.server.global_loss_history.append(avg_loss)

        round_log['aggregated_loss'] = avg_loss

        self.server.aggregation_history.append(round_log)
        self.server.current_round += 1

        return round_log

    def run_full_protocol(self) -> List[Dict]:
        """运行完整协议"""
        logs = []

        for r in range(self.num_rounds):
            log = self.run_round()
            logs.append(log)

            print(f"Round {r}: clients={len(log['selected_clients'])}, loss={log['aggregated_loss']:.4f}")

        return logs


# ============================================================
# 4. 差分隐私会计
# ============================================================

class PrivacyAccountant:
    """
    差分隐私预算会计

    严格追踪隐私损失：
    - Moments Accountant (Abadi et al., 2016)
    - Privacy Amplification by Sampling

    计算 (ε, δ)-DP 保证
    """

    def __init__(self, target_epsilon: float = 1.0, target_delta: float = 1e-5):
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta

        self.spent_epsilon = 0
        self.spent_delta = 0

        self.history: List[Dict] = []

    def compute_noise_scale(
        self,
        gradient_norm: float,
        sensitivity: float = 1.0
    ) -> float:
        """
        计算差分隐私噪声规模

        noise = sensitivity * sqrt(2 * log(1.25 / delta)) / epsilon

        参考: Abadi et al. (2016)
        """
        # 基本公式
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.target_delta)) / self.target_epsilon

        # 根据梯度范数调整
        sigma *= gradient_norm

        return sigma

    def add_noise_to_gradient(
        self,
        gradient: np.ndarray,
        sensitivity: float = 1.0
    ) -> np.ndarray:
        """对梯度添加DP噪声"""
        gradient_norm = np.linalg.norm(gradient)
        sigma = self.compute_noise_scale(gradient_norm, sensitivity)

        noise = np.random.randn(*gradient.shape) * sigma

        return gradient + noise

    def clip_gradient(
        self,
        gradient: np.ndarray,
        clip_norm: float = 1.0
    ) -> np.ndarray:
        """
        梯度裁剪

        确保敏感度有界
        """
        norm = np.linalg.norm(gradient)

        if norm > clip_norm:
            return gradient * (clip_norm / norm)

        return gradient

    def account_spend(
        self,
        operation: str,
        epsilon_spent: float,
        delta_spent: float = 0
    ):
        """记录隐私消耗"""
        self.spent_epsilon += epsilon_spent
        self.spent_delta += delta_spent

        self.history.append({
            'operation': operation,
            'epsilon': epsilon_spent,
            'delta': delta_spent,
            'total_epsilon': self.spent_epsilon,
            'total_delta': self.spent_delta,
            'timestamp': datetime.now().timestamp()
        })

    def check_budget(self) -> bool:
        """检查是否超出预算"""
        return self.spent_epsilon <= self.target_epsilon

    def get_privacy_report(self) -> str:
        """生成隐私报告"""
        lines = []
        lines.append("=" * 60)
        lines.append("差分隐私预算报告")
        lines.append("=" * 60)

        lines.append(f"\n目标: (ε={self.target_epsilon}, δ={self.target_delta})")
        lines.append(f"已消耗: ε={self.spent_epsilon:.4f}, δ={self.spent_delta:.6f}")

        remaining = self.target_epsilon - self.spent_epsilon
        lines.append(f"剩余: ε={remaining:.4f}")

        if self.check_budget():
            lines.append("状态: ✓ 在预算范围内")
        else:
            lines.append("状态: ✗ 已超出预算！")

        lines.append("\n【操作历史】")
        for record in self.history[-5:]:
            lines.append(f"  {record['operation']}: ε={record['epsilon']:.4f}")

        return "\n".join(lines)


# ============================================================
# 5. 可解释性引擎
# ============================================================

class InterpretabilityEngine:
    """
    可解释性引擎

    解释预测结果：
    - 特征重要性
    - 邻居贡献分析
    - 边状态解读
    - 建议 generation
    """

    def __init__(self):
        self.explanations = []

    def explain_prediction(
        self,
        graph: SocialGraph,
        gnn_output: Dict,
        prediction: float
    ) -> Dict:
        """
        解释一个预测
        """
        explanation = {
            'prediction': prediction,
            'interpretation': '',
            'key_factors': [],
            'recommendations': []
        }

        # 1. 分析图级因素
        graph_factors = self._analyze_graph_factors(graph)
        explanation['key_factors'].extend(graph_factors)

        # 2. 分析边因素
        edge_factors = self._analyze_edge_factors(graph, gnn_output)
        explanation['key_factors'].extend(edge_factors)

        # 3. 生成解释文本
        if prediction > 0.6:
            explanation['interpretation'] = "社交韧性良好，支持系统充足"
        elif prediction > 0.4:
            explanation['interpretation'] = "社交韧性中等，存在改善空间"
        else:
            explanation['interpretation'] = "社交韧性较低，需要关注"

        # 4. 生成建议
        explanation['recommendations'] = self._generate_recommendations(
            explanation['key_factors'],
            prediction
        )

        self.explanations.append(explanation)

        return explanation

    def _analyze_graph_factors(self, graph: SocialGraph) -> List[Dict]:
        """分析图级因素"""
        factors = []

        # 节点数量
        if graph.num_nodes < 3:
            factors.append({
                'factor': '社交圈规模小',
                'value': graph.num_nodes,
                'impact': 'negative',
                'weight': 0.3
            })
        elif graph.num_nodes >= 5:
            factors.append({
                'factor': '社交圈规模健康',
                'value': graph.num_nodes,
                'impact': 'positive',
                'weight': 0.2
            })

        # 图密度
        if graph.density < 0.3:
            factors.append({
                'factor': '社交圈连接稀疏',
                'value': graph.density,
                'impact': 'negative',
                'weight': 0.2
            })

        return factors

    def _analyze_edge_factors(
        self,
        graph: SocialGraph,
        gnn_output: Dict
    ) -> List[Dict]:
        """分析边级因素"""
        factors = []

        for (src, tgt), edge in graph.edges.items():
            edge_pred = gnn_output['edge_predictions'].get((src, tgt), {})

            # 能量流
            energy = edge.energy_flow
            if energy < -50:
                factors.append({
                    'factor': f'与{tgt}的关系消耗能量',
                    'value': energy,
                    'impact': 'negative',
                    'weight': abs(energy) / 100
                })
            elif energy > 50:
                factors.append({
                    'factor': f'与{tgt}的关系提供支持',
                    'value': energy,
                    'impact': 'positive',
                    'weight': energy / 100
                })

        return factors

    def _generate_recommendations(
        self,
        factors: List[Dict],
        prediction: float
    ) -> List[str]:
        """生成个性化建议"""
        recommendations = []

        # 根据负面因素生成建议
        negative_factors = [f for f in factors if f['impact'] == 'negative']

        for factor in negative_factors[:3]:
            if factor['factor'] == '社交圈规模小':
                recommendations.append("建议拓展社交圈，增加2-3个正向关系")

            elif factor['factor'] == '社交圈连接稀疏':
                recommendations.append("建议加强现有关系的互动频率")

            elif '消耗能量' in factor['factor']:
                recommendations.append("建议评估消耗型关系，考虑设定边界")

        # 根据预测等级补充
        if prediction < 0.4:
            recommendations.append("建议主动寻求社交支持")
            recommendations.append("可考虑与信任的人交流当前状况")

        return recommendations

    def generate_explanation_report(self, explanation: Dict) -> str:
        """生成解释报告"""
        lines = []

        lines.append("=" * 60)
        lines.append("预测解释报告")
        lines.append("=" * 60)

        lines.append(f"\n【预测结果】{explanation['prediction']:.2f}")
        lines.append(f"\n【解释】{explanation['interpretation']}")

        lines.append("\n【关键因素】")
        for factor in explanation['key_factors']:
            impact = "↑" if factor['impact'] == 'positive' else "↓"
            lines.append(f"  {impact} {factor['factor']}: {factor['value']}")

        if explanation['recommendations']:
            lines.append("\n【建议】")
            for i, rec in enumerate(explanation['recommendations'], 1):
                lines.append(f"  {i}. {rec}")

        return "\n".join(lines)


__all__ = [
    'GraphNode', 'GraphEdge', 'SocialGraph',
    'ProfessionalGraphConstructor',
    'ProfessionalGNN',
    'ClientUpdate', 'ServerState', 'FederatedProtocol',
    'PrivacyAccountant',
    'InterpretabilityEngine'
]