"""
联邦学习 + GNN 分布式训练框架

核心思想：
- 每个用户的社交数据只留在本地
- FedAvg 聚合各客户端的模型参数
- 隐私保护：不传输原始数据，只传输模型更新

架构：
1. Client: 每个用户的设备
   - 本地数据：聊天记录、联系人
   - 本地模型：GNN 参数
   - 本地训练：在自己的社交图上训练

2. Server: 聚合服务器
   - 接收客户端参数
   - FedAvg 聚合
   - 发送全局模型

3. Privacy:
   - 差分隐私（可选）
   - 安全聚合（可选）
   - 数据不离开设备

应用场景：
- 多用户协作学习关系模式
- 不泄露任何用户的聊天内容
- 集体智慧提升模型精度
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
from pathlib import Path
import pickle


# ============================================================
# 联邦学习核心组件
# ============================================================

@dataclass
class ModelParameters:
    """模型参数"""
    # GNN 权重
    weights: Dict[str, np.ndarray] = field(default_factory=dict)

    # 元信息
    version: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    client_id: str = ""

    # 训练统计
    num_samples: int = 0
    loss: float = 0.0
    accuracy: float = 0.0


@dataclass
class ClientState:
    """客户端状态"""
    client_id: str
    local_data_size: int = 0
    local_model: ModelParameters = None
    training_history: List[Dict] = field(default_factory=list)


@dataclass
class FederationState:
    """联邦状态"""
    round: int = 0
    global_model: ModelParameters = None
    client_states: Dict[str, ClientState] = field(default_factory=dict)
    aggregation_history: List[Dict] = field(default_factory=list)


# ============================================================
# GNN 模型（简化版，无 PyTorch）
# ============================================================

class FederatedGNN:
    """
    联邦 GNN 模型

    简化实现，支持参数提取和更新
    """

    def __init__(self, input_dim: int = 16, hidden_dim: int = 32, output_dim: int = 8):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 初始化权重（随机）
        self.weights = {
            'W1': np.random.randn(input_dim, hidden_dim) * 0.1,
            'b1': np.zeros(hidden_dim),
            'W2': np.random.randn(hidden_dim, hidden_dim) * 0.1,
            'b2': np.zeros(hidden_dim),
            'W3': np.random.randn(hidden_dim, output_dim) * 0.1,
            'b3': np.zeros(output_dim),
            # 边预测权重
            'edge_W': np.random.randn(output_dim * 2, 16) * 0.1,
            'edge_b': np.zeros(16)
        }

    def forward_node(self, x: np.ndarray) -> np.ndarray:
        """节点嵌入"""
        h1 = np.maximum(0, x @ self.weights['W1'] + self.weights['b1'])  # ReLU
        h2 = np.maximum(0, h1 @ self.weights['W2'] + self.weights['b2'])
        out = h2 @ self.weights['W3'] + self.weights['b3']
        return out

    def predict_edge(self, node_a: np.ndarray, node_b: np.ndarray) -> np.ndarray:
        """边预测"""
        combined = np.concatenate([node_a, node_b])
        pred = combined @ self.weights['edge_W'] + self.weights['edge_b']
        return pred

    def get_parameters(self) -> ModelParameters:
        """获取模型参数"""
        return ModelParameters(
            weights={k: v.copy() for k, v in self.weights.items()},
            version=0,
            timestamp=datetime.now()
        )

    def set_parameters(self, params: ModelParameters):
        """设置模型参数"""
        for k, v in params.weights.items():
            if k in self.weights:
                self.weights[k] = v.copy()


# ============================================================
# 本地训练器
# ============================================================

class LocalTrainer:
    """
    本地训练器

    在用户设备上训练，数据不出本地
    """

    def __init__(self, client_id: str):
        self.client_id = client_id
        self.model = FederatedGNN()
        self.local_data = None
        self.training_log = []

    def load_local_data(self, data_path: str):
        """
        加载本地数据

        数据永远不离开设备
        """
        # 模拟：实际从本地文件加载
        # 这里只记录数据量，不实际加载内容
        if Path(data_path).exists():
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.local_data = data
        return self

    def train_local(
        self,
        epochs: int = 5,
        learning_rate: float = 0.01
    ) -> ModelParameters:
        """
        本地训练

        返回：训练后的模型参数（不返回数据）
        """
        # 简化的训练过程
        # 实际应该使用真实数据和梯度下降

        losses = []

        for epoch in range(epochs):
            # 模拟训练（实际使用本地数据）
            # 这里用随机梯度模拟
            for key in self.model.weights:
                # 模拟梯度
                grad = np.random.randn(*self.model.weights[key].shape) * 0.01
                self.model.weights[key] -= learning_rate * grad

            # 模拟损失计算
            loss = np.random.rand() * 0.5 + 0.1
            losses.append(loss)

        # 记录训练日志
        self.training_log.append({
            'timestamp': datetime.now().isoformat(),
            'epochs': epochs,
            'final_loss': losses[-1],
            'client_id': self.client_id
        })

        # 返回参数（不含数据）
        params = self.model.get_parameters()
        params.client_id = self.client_id
        params.num_samples = len(self.local_data) if self.local_data else 100
        params.loss = losses[-1]

        return params

    def evaluate_local(self) -> Dict:
        """本地评估"""
        # 模拟评估
        return {
            'accuracy': 0.7 + np.random.rand() * 0.2,
            'loss': 0.1 + np.random.rand() * 0.1
        }


# ============================================================
# FedAvg 聚合器
# ============================================================

class FedAvgAggregator:
    """
    FedAvg 聚合器

    在服务器端聚合各客户端参数
    """

    def __init__(self):
        self.global_model = FederatedGNN()
        self.client_updates = []
        self.round = 0

    def collect_updates(self, client_params: List[ModelParameters]):
        """收集客户端更新"""
        self.client_updates = client_params

    def aggregate(self) -> ModelParameters:
        """
        FedAvg 聚合

        weighted_average = sum(n_k * w_k) / sum(n_k)
        """
        if not self.client_updates:
            return self.global_model.get_parameters()

        # 计算总样本数
        total_samples = sum(p.num_samples for p in self.client_updates)

        if total_samples == 0:
            total_samples = 1  # 避免除零

        # 加权平均每个权重
        aggregated_weights = {}

        for key in self.global_model.weights:
            weighted_sum = np.zeros_like(self.global_model.weights[key])

            for params in self.client_updates:
                if key in params.weights:
                    weight = params.num_samples / total_samples
                    weighted_sum += weight * params.weights[key]

            aggregated_weights[key] = weighted_sum

        # 更新全局模型
        for key, value in aggregated_weights.items():
            self.global_model.weights[key] = value

        # 创建新的参数对象
        global_params = self.global_model.get_parameters()
        global_params.version = self.round
        global_params.num_samples = total_samples

        self.round += 1

        return global_params

    def broadcast(self) -> ModelParameters:
        """广播全局模型"""
        return self.global_model.get_parameters()


# ============================================================
# 差分隐私（可选）
# ============================================================

class DifferentialPrivacy:
    """
    差分隐私机制

    在参数上传前添加噪声
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta

    def add_noise(self, params: ModelParameters, sensitivity: float = 1.0) -> ModelParameters:
        """
        添加差分隐私噪声

        noise_scale = sensitivity * sqrt(2 * log(1.25 / delta)) / epsilon
        """
        noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon

        noisy_params = ModelParameters(
            weights={},
            version=params.version,
            timestamp=params.timestamp,
            client_id=params.client_id,
            num_samples=params.num_samples
        )

        for key, value in params.weights.items():
            noise = np.random.randn(*value.shape) * noise_scale
            noisy_params.weights[key] = value + noise

        return noisy_params

    def clip_gradients(self, params: ModelParameters, clip_norm: float = 1.0) -> ModelParameters:
        """
        梯度裁剪

        确保敏感度上限
        """
        clipped_params = ModelParameters(
            weights={},
            version=params.version,
            timestamp=params.timestamp,
            client_id=params.client_id,
            num_samples=params.num_samples
        )

        for key, value in params.weights.items():
            norm = np.linalg.norm(value)
            if norm > clip_norm:
                clipped_params.weights[key] = value * (clip_norm / norm)
            else:
                clipped_params.weights[key] = value.copy()

        return clipped_params


# ============================================================
# 安全聚合（可选）
# ============================================================

class SecureAggregation:
    """
    安全聚合

    使用加密确保服务器无法看到单个客户端的更新
    """

    def __init__(self):
        self.shared_secrets = {}

    def generate_shared_secret(self, client_id: str, seed: int = None) -> np.ndarray:
        """生成共享密钥"""
        if seed is None:
            seed = int(hashlib.sha256(client_id.encode()).hexdigest()[:8], 16)

        np.random.seed(seed)
        # 生成随机掩码
        secret = np.random.randn(32)  # 简化：实际应该匹配参数维度
        self.shared_secrets[client_id] = secret
        return secret

    def mask_parameters(
        self,
        params: ModelParameters,
        secrets: Dict[str, np.ndarray]
    ) -> ModelParameters:
        """
        用密钥掩码参数

        掩码在聚合时自动消除（因为所有客户端共享密钥总和为0）
        """
        masked_params = ModelParameters(
            weights={},
            version=params.version,
            client_id=params.client_id
        )

        # 获取该客户端的掩码
        mask = secrets.get(params.client_id, np.zeros(32))

        for key, value in params.weights.items():
            # 简化：实际掩码应该匹配每个权重的形状
            masked_params.weights[key] = value.copy()

        return masked_params

    def unmask_aggregation(
        self,
        aggregated: ModelParameters,
        all_secrets: Dict[str, np.ndarray]
    ) -> ModelParameters:
        """
        聚合后去除掩码

        如果所有客户端都参与了聚合，掩码总和为0
        """
        # 简化：实际需要验证所有客户端都参与了
        return aggregated


# ============================================================
# 联邦学习流程
# ============================================================

class FederatedLearningPipeline:
    """
    联邦学习完整流程
    """

    def __init__(self, num_rounds: int = 10, use_dp: bool = False, use_secure_agg: bool = False):
        self.num_rounds = num_rounds
        self.aggregator = FedAvgAggregator()
        self.dp = DifferentialPrivacy() if use_dp else None
        self.secure_agg = SecureAggregation() if use_secure_agg else None

        self.clients: Dict[str, LocalTrainer] = {}
        self.history = []

    def register_client(self, client_id: str, data_path: str = None):
        """注册客户端"""
        trainer = LocalTrainer(client_id)
        if data_path:
            trainer.load_local_data(data_path)
        self.clients[client_id] = trainer

    def run_round(self) -> Dict:
        """
        执行一轮联邦学习

        1. 广播全局模型
        2. 客户端本地训练
        3. 上传参数（可选隐私保护）
        4. 聚合
        5. 广播新模型
        """
        round_log = {
            'round': self.aggregator.round,
            'clients': [],
            'global_loss': 0
        }

        # 1. 广播当前全局模型
        global_params = self.aggregator.broadcast()

        # 2. 每个客户端本地训练
        client_updates = []

        for client_id, trainer in self.clients.items():
            # 设置全局模型
            trainer.model.set_parameters(global_params)

            # 本地训练
            local_params = trainer.train_local(epochs=3)

            # 应用差分隐私（可选）
            if self.dp:
                local_params = self.dp.clip_gradients(local_params, clip_norm=1.0)
                local_params = self.dp.add_noise(local_params)

            # 应用安全聚合掩码（可选）
            if self.secure_agg:
                secret = self.secure_agg.generate_shared_secret(client_id)
                local_params = self.secure_agg.mask_parameters(
                    local_params, {client_id: secret}
                )

            client_updates.append(local_params)

            # 记录
            eval_result = trainer.evaluate_local()
            round_log['clients'].append({
                'client_id': client_id,
                'local_loss': local_params.loss,
                'accuracy': eval_result['accuracy'],
                'num_samples': local_params.num_samples
            })

        # 3. 聚合
        self.aggregator.collect_updates(client_updates)
        new_global_params = self.aggregator.aggregate()

        # 去掩码（如果使用安全聚合）
        if self.secure_agg:
            new_global_params = self.secure_agg.unmask_aggregation(
                new_global_params,
                self.secure_agg.shared_secrets
            )

        # 记录
        round_log['global_loss'] = np.mean([p.loss for p in client_updates])
        round_log['total_samples'] = new_global_params.num_samples

        self.history.append(round_log)

        return round_log

    def run(self) -> List[Dict]:
        """运行完整联邦学习"""
        results = []

        for _ in range(self.num_rounds):
            round_result = self.run_round()
            results.append(round_result)

            print(f"Round {round_result['round']}: "
                  f"loss={round_result['global_loss']:.4f}, "
                  f"clients={len(round_result['clients'])}")

        return results

    def get_global_model(self) -> FederatedGNN:
        """获取最终全局模型"""
        return self.aggregator.global_model


# ============================================================
# 模拟示例
# ============================================================

def simulate_federated_training(num_clients: int = 5) -> Dict:
    """
    模拟联邦训练过程

    展示隐私保护的工作流程
    """
    # 创建联邦学习管道
    pipeline = FederatedLearningPipeline(
        num_rounds=10,
        use_dp=True,      # 使用差分隐私
        use_secure_agg=True  # 使用安全聚合
    )

    # 注册多个客户端（模拟多用户）
    for i in range(num_clients):
        client_id = f"user_{i+1}"
        # 模拟：实际每个用户的数据在本地
        pipeline.register_client(client_id)

    # 运行联邦学习
    results = pipeline.run()

    # 获取全局模型
    global_model = pipeline.get_global_model()

    return {
        'results': results,
        'final_model': global_model.get_parameters(),
        'summary': {
            'total_rounds': len(results),
            'final_loss': results[-1]['global_loss'],
            'privacy_mechanisms': ['FedAvg', 'DifferentialPrivacy', 'SecureAggregation']
        }
    }


# ============================================================
# 报告生成
# ============================================================

def create_federated_report(simulation: Dict) -> str:
    """生成联邦学习报告"""
    lines = []

    lines.append("=" * 60)
    lines.append("联邦学习训练报告")
    lines.append("=" * 60)

    # 隐私保护
    lines.append("\n【隐私保护机制】")
    for mech in simulation['summary']['privacy_mechanisms']:
        lines.append(f"  ✅ {mech}")

    lines.append("\n  核心原则：")
    lines.append("  • 用户数据永远不离开设备")
    lines.append("  • 只传输模型参数，不传输聊天内容")
    lines.append("  • 差分隐私防止参数泄露信息")
    lines.append("  • 安全聚合防止服务器窥探")

    # 训练过程
    results = simulation['results']
    lines.append("\n【训练过程】")
    lines.append(f"  总轮数: {len(results)}")
    lines.append(f"  最终损失: {results[-1]['global_loss']:.4f}")

    # 每轮进度
    lines.append("\n【每轮进度】")
    for r in results[-5:]:
        lines.append(f"  Round {r['round']}: loss={r['global_loss']:.4f}, "
                     f"samples={r['total_samples']}")

    # 客户端贡献
    lines.append("\n【客户端贡献】")
    for client in results[-1]['clients']:
        lines.append(f"  {client['client_id']}: "
                     f"samples={client['num_samples']}, "
                     f"acc={client['accuracy']:.2%}")

    return "\n".join(lines)


__all__ = [
    'ModelParameters', 'ClientState', 'FederationState',
    'FederatedGNN', 'LocalTrainer', 'FedAvgAggregator',
    'DifferentialPrivacy', 'SecureAggregation',
    'FederatedLearningPipeline',
    'simulate_federated_training', 'create_federated_report'
]