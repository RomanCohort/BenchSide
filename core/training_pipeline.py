"""
大规模训练管道

使用 DeepSeek 生成的合成数据进行模型训练

数据来源：
- data/training_large/checkpoint_100.json (500+ 样本)
- 包含多种场景：unrequited_love, passionate_dating, ambiguous 等

训练目标：
1. 学习韧性因子的权重（替代硬编码）
2. 学习场景识别
3. 学习风险预测
4. 验证模型在合成数据上的表现
"""
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import pickle


# ============================================================
# 数据加载与处理
# ============================================================

@dataclass
class TrainingSample:
    """训练样本"""
    sample_id: str
    scenario: str
    messages: List[Dict]

    # 提取的特征
    features: np.ndarray = None

    # 标签
    resilience_label: float = 0
    risk_label: float = 0
    attachment_label: str = ""


class TrainingDataLoader:
    """训练数据加载器"""

    SCENARIO_LABELS = {
        'unrequited_love': {'resilience': 0.3, 'risk': 0.7, 'attachment': 'anxious'},
        'passionate_dating': {'resilience': 0.7, 'risk': 0.3, 'attachment': 'secure'},
        'ambiguous': {'resilience': 0.5, 'risk': 0.5, 'attachment': 'mixed'},
        'conflict_escalation': {'resilience': 0.2, 'risk': 0.8, 'attachment': 'unstable'},
        'recovery_process': {'resilience': 0.6, 'risk': 0.4, 'attachment': 'improving'},
        'gradual_drift': {'resilience': 0.4, 'risk': 0.6, 'attachment': 'avoidant'},
        'single_pursuit': {'resilience': 0.35, 'risk': 0.65, 'attachment': 'anxious'},
        'reconnection': {'resilience': 0.65, 'risk': 0.35, 'attachment': 'secure'},
        'normal_interaction': {'resilience': 0.7, 'risk': 0.3, 'attachment': 'secure'},
        'mutual_support': {'resilience': 0.8, 'risk': 0.2, 'attachment': 'secure'}
    }

    def load_from_json(self, json_path: str) -> List[TrainingSample]:
        """从 JSON 文件加载"""
        samples = []

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            sample = TrainingSample(
                sample_id=item.get('id', ''),
                scenario=item.get('scenario_name', ''),
                messages=item.get('messages', [])
            )

            # 添加标签
            labels = self.SCENARIO_LABELS.get(sample.scenario, {
                'resilience': 0.5, 'risk': 0.5, 'attachment': 'unknown'
            })
            sample.resilience_label = labels['resilience']
            sample.risk_label = labels['risk']
            sample.attachment_label = labels['attachment']

            samples.append(sample)

        return samples

    def extract_features(self, sample: TrainingSample) -> np.ndarray:
        """
        从消息中提取特征

        特征向量 (10维):
        - 消息数量
        - 消息平衡度
        - 平均回复延迟
        - 延迟标准差
        - 情感表达频率
        - 问题频率
        - 简短回复比例
        - 主动发起比例
        - 消息长度变化
        - 情绪波动
        """
        messages = sample.messages

        if not messages:
            return np.zeros(10)

        features = np.zeros(10)

        # 1. 消息数量
        features[0] = len(messages)

        # 统计
        sender_counts = {}
        delays = []
        lengths = []
        emotions = []
        questions = 0
        short_replies = 0

        for msg in messages:
            sender = msg.get('sender', 'A')
            sender_counts[sender] = sender_counts.get(sender, 0) + 1

            if 'delay_seconds' in msg:
                delays.append(msg['delay_seconds'])

            length = len(msg.get('content', ''))
            lengths.append(length)

            if msg.get('emotion'):
                emotions.append(msg['emotion'])

            if '?' in msg.get('content', '') or '吗' in msg.get('content', ''):
                questions += 1

            if length < 10:
                short_replies += 1

        # 2. 消息平衡度
        if len(sender_counts) >= 2:
            counts = list(sender_counts.values())
            features[1] = min(counts) / max(counts)
        else:
            features[1] = 0

        # 3. 平均回复延迟
        if delays:
            features[2] = np.mean(delays) / 3600  # 小时

        # 4. 延迟标准差
        if len(delays) > 1:
            features[3] = np.std(delays) / 3600

        # 5. 情感表达频率
        if emotions:
            features[4] = len([e for e in emotions if e != 0.5]) / len(emotions)

        # 6. 问题频率
        features[5] = questions / len(messages) if messages else 0

        # 7. 简短回复比例
        features[6] = short_replies / len(messages) if messages else 0

        # 8. 主动发起比例（假设第一条消息是主动）
        if sender_counts:
            first_sender = messages[0].get('sender', 'A')
            features[7] = sender_counts.get(first_sender, 0) / len(messages)

        # 9. 消息长度变化
        if len(lengths) > 1:
            features[8] = np.std(lengths) / (np.mean(lengths) + 1)

        # 10. 情绪波动
        if len(emotions) > 1:
            features[9] = np.std(emotions)

        sample.features = features

        return features

    def prepare_dataset(
        self,
        json_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        准备训练数据集

        Returns:
            X: 特征矩阵
            y_resilience: 韧性标签
            y_risk: 风险标签
        """
        samples = self.load_from_json(json_path)

        X = []
        y_resilience = []
        y_risk = []

        for sample in samples:
            features = self.extract_features(sample)
            X.append(features)
            y_resilience.append(sample.resilience_label)
            y_risk.append(sample.risk_label)

        return np.array(X), np.array(y_resilience), np.array(y_risk)


# ============================================================
# 模型训练
# ============================================================

class ResilienceModelTrainer:
    """韧性模型训练器"""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 16):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 初始化权重
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * 0.1
        self.b2 = np.zeros(1)

        # 训练历史
        self.training_history = []
        self.fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 200,
        lr: float = 0.01,
        batch_size: int = 32,
        verbose: bool = True
    ) -> 'ResilienceModelTrainer':
        """
        训练模型
        """
        n_samples = X.shape[0]

        for epoch in range(epochs):
            # 随机打乱
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0
            n_batches = 0

            # 批量训练
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size].reshape(-1, 1)

                # 前向传播
                h = np.maximum(0, X_batch @ self.W1 + self.b1)
                pred = 1 / (1 + np.exp(-(h @ self.W2 + self.b2)))

                # 损失
                loss = -np.mean(
                    y_batch * np.log(pred + 1e-8) +
                    (1 - y_batch) * np.log(1 - pred + 1e-8)
                )
                epoch_loss += loss
                n_batches += 1

                # 反向传播
                d_pred = pred - y_batch
                d_W2 = h.T @ d_pred / len(X_batch)
                d_b2 = np.mean(d_pred)

                d_h = d_pred @ self.W2.T
                d_h[h <= 0] = 0

                d_W1 = X_batch.T @ d_h / len(X_batch)
                d_b1 = np.mean(d_h, axis=0)

                # 更新
                self.W1 -= lr * d_W1
                self.b1 -= lr * d_b1
                self.W2 -= lr * d_W2
                self.b2 -= lr * d_b2

            avg_loss = epoch_loss / n_batches
            self.training_history.append(avg_loss)

            if verbose and (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        self.fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.fitted:
            raise ValueError("Model not fitted")

        h = np.maximum(0, X @ self.W1 + self.b1)
        pred = 1 / (1 + np.exp(-(h @ self.W2 + self.b2)))

        return pred.flatten()

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """评估"""
        pred = self.predict(X)

        mse = np.mean((pred - y) ** 2)
        mae = np.mean(np.abs(pred - y))

        # 相关系数
        if np.std(pred) > 0 and np.std(y) > 0:
            corr = np.corrcoef(pred, y)[0, 1]
        else:
            corr = 0

        return {
            'mse': mse,
            'mae': mae,
            'correlation': corr,
            'predictions': pred,
            'ground_truth': y
        }

    def get_feature_importance(self) -> Dict:
        """获取特征重要性"""
        # 通过第一层权重的绝对值
        importance = np.abs(self.W1).mean(axis=1)

        feature_names = [
            '消息数量', '消息平衡度', '平均延迟', '延迟波动',
            '情感表达', '问题频率', '简短回复', '主动比例',
            '长度变化', '情绪波动'
        ]

        return {name: float(imp) for name, imp in zip(feature_names, importance)}


# ============================================================
# 完整训练管道
# ============================================================

class TrainingPipeline:
    """完整训练管道"""

    def __init__(self):
        self.loader = TrainingDataLoader()
        self.resilience_model = None
        self.risk_model = None

    def run(
        self,
        data_path: str,
        test_ratio: float = 0.2,
        epochs: int = 200,
        verbose: bool = True
    ) -> Dict:
        """
        运行训练管道
        """
        results = {
            'data_stats': {},
            'training_stats': {},
            'evaluation': {},
            'feature_importance': {}
        }

        # 1. 加载数据
        if verbose:
            print("=" * 60)
            print("大规模训练管道")
            print("=" * 60)
            print(f"\n【加载数据】{data_path}")

        X, y_resilience, y_risk = self.loader.prepare_dataset(data_path)

        results['data_stats'] = {
            'total_samples': len(X),
            'feature_dim': X.shape[1],
            'resilience_mean': float(np.mean(y_resilience)),
            'risk_mean': float(np.mean(y_risk))
        }

        if verbose:
            print(f"  样本数: {len(X)}")
            print(f"  特征维度: {X.shape[1]}")
            print(f"  韧性标签均值: {np.mean(y_resilience):.3f}")
            print(f"  风险标签均值: {np.mean(y_risk):.3f}")

        # 2. 划分训练/测试
        n_samples = len(X)
        n_test = int(n_samples * test_ratio)
        indices = np.random.permutation(n_samples)

        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        X_train, X_test = X[train_idx], X[test_idx]
        y_res_train, y_res_test = y_resilience[train_idx], y_resilience[test_idx]
        y_risk_train, y_risk_test = y_risk[train_idx], y_risk[test_idx]

        # 3. 训练韧性模型
        if verbose:
            print(f"\n【训练韧性模型】")

        self.resilience_model = ResilienceModelTrainer()
        self.resilience_model.fit(X_train, y_res_train, epochs=epochs, verbose=verbose)

        # 4. 训练风险模型
        if verbose:
            print(f"\n【训练风险模型】")

        self.risk_model = ResilienceModelTrainer()
        self.risk_model.fit(X_train, y_risk_train, epochs=epochs, verbose=verbose)

        # 5. 评估
        if verbose:
            print(f"\n【评估】")

        res_eval = self.resilience_model.evaluate(X_test, y_res_test)
        risk_eval = self.risk_model.evaluate(X_test, y_risk_test)

        results['evaluation'] = {
            'resilience': {
                'mse': res_eval['mse'],
                'mae': res_eval['mae'],
                'correlation': res_eval['correlation']
            },
            'risk': {
                'mse': risk_eval['mse'],
                'mae': risk_eval['mae'],
                'correlation': risk_eval['correlation']
            }
        }

        if verbose:
            print(f"  韧性模型: MAE={res_eval['mae']:.4f}, Corr={res_eval['correlation']:.3f}")
            print(f"  风险模型: MAE={risk_eval['mae']:.4f}, Corr={risk_eval['correlation']:.3f}")

        # 6. 特征重要性
        results['feature_importance'] = self.resilience_model.get_feature_importance()

        if verbose:
            print(f"\n【特征重要性】")
            for name, imp in sorted(results['feature_importance'].items(), key=lambda x: -x[1])[:5]:
                print(f"  {name}: {imp:.4f}")

        return results

    def save_models(self, save_dir: str):
        """保存模型"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        if self.resilience_model:
            with open(save_path / 'resilience_model.pkl', 'wb') as f:
                pickle.dump({
                    'W1': self.resilience_model.W1,
                    'b1': self.resilience_model.b1,
                    'W2': self.resilience_model.W2,
                    'b2': self.resilience_model.b2
                }, f)

        if self.risk_model:
            with open(save_path / 'risk_model.pkl', 'wb') as f:
                pickle.dump({
                    'W1': self.risk_model.W1,
                    'b1': self.risk_model.b1,
                    'W2': self.risk_model.W2,
                    'b2': self.risk_model.b2
                }, f)

    def predict_sample(self, messages: List[Dict]) -> Dict:
        """预测单个样本"""
        # 创建临时样本
        sample = TrainingSample(
            sample_id='temp',
            scenario='unknown',
            messages=messages
        )

        # 提取特征
        features = self.loader.extract_features(sample).reshape(1, -1)

        # 预测
        if self.resilience_model and self.risk_model:
            resilience = float(self.resilience_model.predict(features)[0])
            risk = float(self.risk_model.predict(features)[0])
        else:
            resilience = 0.5
            risk = 0.5

        return {
            'resilience_prediction': resilience,
            'risk_prediction': risk,
            'features': features.flatten().tolist()
        }


# ============================================================
# 主程序
# ============================================================

def main():
    """主程序"""
    # 数据路径
    data_path = "data/training_large/checkpoint_100.json"

    if not Path(data_path).exists():
        print(f"数据文件不存在: {data_path}")
        return

    # 创建管道
    pipeline = TrainingPipeline()

    # 运行训练
    results = pipeline.run(data_path, epochs=200, verbose=True)

    # 保存模型
    pipeline.save_models("models/trained")

    print("\n" + "=" * 60)
    print("训练完成！模型已保存到 models/trained/")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()


__all__ = [
    'TrainingSample', 'TrainingDataLoader',
    'ResilienceModelTrainer', 'TrainingPipeline',
    'main'
]