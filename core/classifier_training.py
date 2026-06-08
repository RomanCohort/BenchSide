"""
关系分类器训练模块

使用生成的合成数据训练神经网络分类器

分类任务：
1. 关系类型分类 (10类)
2. 依恋类型分类 (4类)
3. 行为模式分类 (3类)
4. 关系角色分类 (5类)

模型架构：
- 输入: 对话特征向量
- 输出: 多标签分类

特征工程：
- 统计特征: 消息数量、主动比例、回复速度
- 语言特征: 消息长度、情感词汇、提问频率
- 互动特征: 连发比例、回复延迟、称呼类型
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import pickle

# ML 库
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier


# ============================================================
# 特征提取
# ============================================================

@dataclass
class DialogueFeatures:
    """对话特征"""
    # 基础统计
    total_messages: float = 0
    a_message_count: float = 0
    b_message_count: float = 0
    a_ratio: float = 0.5

    # 时间特征
    avg_response_delay_a: float = 0
    avg_response_delay_b: float = 0
    response_delay_ratio: float = 1

    # 消息特征
    avg_msg_length_a: float = 0
    avg_msg_length_b: float = 0
    msg_length_ratio: float = 1

    # 情感特征
    emotional_words_a: float = 0
    emotional_words_b: float = 0
    question_count_a: float = 0
    question_count_b: float = 0

    # 连发特征
    consecutive_a: float = 0
    consecutive_b: float = 0

    # 关系动态
    power_balance: float = 50
    emotional_intensity: float = 50
    conflict_level: float = 0


class FeatureExtractor:
    """对话特征提取器"""

    # 情感词汇表
    EMOTIONAL_WORDS = {
        'positive': ['爱', '喜欢', '想', '开心', '高兴', '快乐', '幸福', '亲爱的', '宝贝', '想你'],
        'negative': ['烦', '累', '难过', '伤心', '失望', '生气', '讨厌', '对不起', '抱歉'],
        'anxious': ['担心', '焦虑', '紧张', '害怕', '不安', '纠结', '犹豫'],
        'avoidant': ['忙', '累', '再说', '以后', '不急', '随便', '无所谓']
    }

    def extract_features(self, messages: List[Dict]) -> DialogueFeatures:
        """
        从对话中提取特征

        Args:
            messages: 消息列表

        Returns:
            特征对象
        """
        features = DialogueFeatures()

        if not messages:
            return features

        # 分离双方消息
        a_msgs = [m for m in messages if m['sender'] == 'A']
        b_msgs = [m for m in messages if m['sender'] == 'B']

        # 基础统计
        features.total_messages = len(messages)
        features.a_message_count = len(a_msgs)
        features.b_message_count = len(b_msgs)
        features.a_ratio = len(a_msgs) / len(messages) if messages else 0.5

        # 时间特征
        delays_a = self._compute_response_delays(messages, 'A')
        delays_b = self._compute_response_delays(messages, 'B')

        features.avg_response_delay_a = np.mean(delays_a) if delays_a else 0
        features.avg_response_delay_b = np.mean(delays_b) if delays_b else 0
        features.response_delay_ratio = (
            features.avg_response_delay_a / features.avg_response_delay_b
            if features.avg_response_delay_b > 0 else 1
        )

        # 消息长度
        lengths_a = [len(m.get('content', '')) for m in a_msgs]
        lengths_b = [len(m.get('content', '')) for m in b_msgs]

        features.avg_msg_length_a = np.mean(lengths_a) if lengths_a else 0
        features.avg_msg_length_b = np.mean(lengths_b) if lengths_b else 0
        features.msg_length_ratio = (
            features.avg_msg_length_a / features.avg_msg_length_b
            if features.avg_msg_length_b > 0 else 1
        )

        # 情感词汇统计
        all_text_a = ' '.join(m.get('content', '') for m in a_msgs)
        all_text_b = ' '.join(m.get('content', '') for m in b_msgs)

        features.emotional_words_a = self._count_emotional_words(all_text_a)
        features.emotional_words_b = self._count_emotional_words(all_text_b)

        # 提问统计
        features.question_count_a = sum(
            1 for m in a_msgs if '?' in m.get('content', '') or '吗' in m.get('content', '')
        )
        features.question_count_b = sum(
            1 for m in b_msgs if '?' in m.get('content', '') or '吗' in m.get('content', '')
        )

        # 连发统计
        features.consecutive_a = self._count_consecutive(messages, 'A')
        features.consecutive_b = self._count_consecutive(messages, 'B')

        return features

    def _compute_response_delays(self, messages: List[Dict], sender: str) -> List[float]:
        """计算回复延迟"""
        delays = []
        prev_time = None
        prev_sender = None

        for m in messages:
            curr_time = m.get('timestamp', 0)
            curr_sender = m.get('sender', '')

            if curr_sender == sender and prev_sender != sender and prev_time:
                delay = curr_time - prev_time
                if delay > 0:
                    delays.append(delay)

            prev_time = curr_time
            prev_sender = curr_sender

        return delays

    def _count_emotional_words(self, text: str) -> int:
        """统计情感词汇数量"""
        count = 0
        for category, words in self.EMOTIONAL_WORDS.items():
            for word in words:
                if word in text:
                    count += text.count(word)
        return count

    def _count_consecutive(self, messages: List[Dict], sender: str) -> int:
        """统计连续发言次数"""
        consecutive_count = 0
        prev_sender = None

        for m in messages:
            curr_sender = m.get('sender', '')

            if curr_sender == sender and prev_sender == sender:
                consecutive_count += 1

            prev_sender = curr_sender

        return consecutive_count

    def to_vector(self, features: DialogueFeatures) -> np.ndarray:
        """将特征转换为向量"""
        return np.array([
            features.total_messages,
            features.a_message_count,
            features.b_message_count,
            features.a_ratio,
            features.avg_response_delay_a,
            features.avg_response_delay_b,
            features.response_delay_ratio,
            features.avg_msg_length_a,
            features.avg_msg_length_b,
            features.msg_length_ratio,
            features.emotional_words_a,
            features.emotional_words_b,
            features.question_count_a,
            features.question_count_b,
            features.consecutive_a,
            features.consecutive_b
        ])


# ============================================================
# 数据加载与预处理
# ============================================================

class DataLoader:
    """训练数据加载器"""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.feature_extractor = FeatureExtractor()

        # 标签编码器
        self.relation_encoder = LabelEncoder()
        self.attachment_encoder = LabelEncoder()
        self.behavior_encoder = LabelEncoder()

    def load_dataset(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        加载并预处理数据集

        Returns:
            (特征矩阵, 标签字典)
        """
        # 加载 JSON
        with open(self.data_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        # 提取特征和标签
        features_list = []
        labels_dict = {
            'relation': [],
            'a_attachment': [],
            'b_attachment': [],
            'a_ratio': [],
            'power_balance': []
        }

        for sample in dataset:
            messages = sample['messages']
            labels = sample['labels']

            # 提取特征
            features = self.feature_extractor.extract_features(messages)
            feature_vec = self.feature_extractor.to_vector(features)
            features_list.append(feature_vec)

            # 收集标签
            labels_dict['relation'].append(labels['relation_type'])
            labels_dict['a_attachment'].append(labels['a_attachment'])
            labels_dict['b_attachment'].append(labels['b_attachment'])
            labels_dict['a_ratio'].append(labels['a_ratio'])
            labels_dict['power_balance'].append(labels['power_balance'])

        # 转换为 numpy
        X = np.array(features_list)

        # 标签编码
        # 清理标签值
        cleaned_relation = [self._clean_label(r) for r in labels_dict['relation']]
        cleaned_a_attach = [self._clean_label(a) for a in labels_dict['a_attachment']]
        cleaned_b_attach = [self._clean_label(a) for a in labels_dict['b_attachment']]

        y_relation = self.relation_encoder.fit_transform(cleaned_relation)

        # 合并所有 attachment 类型来拟合编码器
        all_attachments = cleaned_a_attach + cleaned_b_attach
        self.attachment_encoder.fit(all_attachments)

        y_a_attachment = self.attachment_encoder.transform(cleaned_a_attach)
        y_b_attachment = self.attachment_encoder.transform(cleaned_b_attach)

        # 数值标签
        y_ratio = np.array(labels_dict['a_ratio'])
        y_power = np.array(labels_dict['power_balance'])

        labels_encoded = {
            'relation': y_relation,
            'a_attachment': y_a_attachment,
            'b_attachment': y_b_attachment,
            'a_ratio': y_ratio,
            'power_balance': y_power
        }

        return X, labels_encoded, dataset

    def _clean_label(self, label: str) -> str:
        """清理标签值"""
        label = str(label).lower().strip()
        # 修正常见的拼写错误
        corrections = {
            'avoidan': 'avoidant',
            'anxius': 'anxious',
            'secur': 'secure',
            'romanti': 'romantic'
        }
        return corrections.get(label, label)

    def split_data(self, X: np.ndarray, y: Dict, test_size: float = 0.2) -> Tuple:
        """分割训练和测试数据"""
        splits = {}

        for label_name, y_values in y.items():
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_values, test_size=test_size, random_state=42,
                    stratify=y_values if len(set(y_values)) > 1 and min(np.bincount(y_values.astype(int) if y_values.dtype == float else y_values)) >= 2 else None
                )
            except:
                # 如果分割失败，使用简单分割
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_values, test_size=test_size, random_state=42
                )

            splits[label_name] = {
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test
            }

        return splits


# ============================================================
# 模型训练
# ============================================================

class RelationClassifier:
    """关系分类器"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_extractor = FeatureExtractor()

    def train(self, X: np.ndarray, y: Dict[str, np.ndarray]):
        """
        训练多个分类任务

        Args:
            X: 特征矩阵
            y: 标签字典
        """
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)

        # 训练关系类型分类器
        print("Training relation type classifier...")
        self.models['relation'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.models['relation'].fit(X_scaled, y['relation'])

        # 训练依恋类型分类器 (A)
        print("Training attachment classifier (A)...")
        self.models['a_attachment'] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.models['a_attachment'].fit(X_scaled, y['a_attachment'])

        # 训练依恋类型分类器 (B)
        print("Training attachment classifier (B)...")
        self.models['b_attachment'] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.models['b_attachment'].fit(X_scaled, y['b_attachment'])

        # 训练主动比例回归器
        print("Training initiative ratio regressor...")
        from sklearn.ensemble import GradientBoostingRegressor
        self.models['a_ratio'] = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.models['a_ratio'].fit(X_scaled, y['a_ratio'])

        # 训练权力平衡回归器
        print("Training power balance regressor...")
        self.models['power_balance'] = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.models['power_balance'].fit(X_scaled, y['power_balance'])

        print("Training complete!")

    def predict(self, messages: List[Dict]) -> Dict:
        """
        预测关系特征

        Args:
            messages: 消息列表

        Returns:
            预测结果
        """
        # 提取特征
        features = self.feature_extractor.extract_features(messages)
        X = self.feature_extractor.to_vector(features)
        X_scaled = self.scaler.transform(X.reshape(1, -1))

        # 预测
        results = {}

        # 关系类型
        relation_pred = self.models['relation'].predict(X_scaled)[0]
        relation_proba = self.models['relation'].predict_proba(X_scaled)[0]
        results['relation_type'] = {
            'predicted': relation_pred,
            'confidence': max(relation_proba)
        }

        # 依恋类型
        a_attach_pred = self.models['a_attachment'].predict(X_scaled)[0]
        b_attach_pred = self.models['b_attachment'].predict(X_scaled)[0]
        results['a_attachment'] = a_attach_pred
        results['b_attachment'] = b_attach_pred

        # 主动比例
        ratio_pred = self.models['a_ratio'].predict(X_scaled)[0]
        results['a_ratio'] = ratio_pred

        # 权力平衡
        power_pred = self.models['power_balance'].predict(X_scaled)[0]
        results['power_balance'] = power_pred

        return results

    def evaluate(self, X_test: np.ndarray, y_test: Dict) -> Dict:
        """
        评估模型性能

        Returns:
            评估结果
        """
        X_scaled = self.scaler.transform(X_test)

        metrics = {}

        # 分类任务评估
        for task in ['relation', 'a_attachment', 'b_attachment']:
            y_pred = self.models[task].predict(X_scaled)
            report = classification_report(y_test[task], y_pred, output_dict=True)
            metrics[task] = {
                'accuracy': report['accuracy'],
                'f1_macro': report['macro avg']['f1-score']
            }

        # 回归任务评估
        for task in ['a_ratio', 'power_balance']:
            y_pred = self.models[task].predict(X_scaled)
            mse = np.mean((y_pred - y_test[task]) ** 2)
            mae = np.mean(np.abs(y_pred - y_test[task]))
            r2 = 1 - mse / np.var(y_test[task])
            metrics[task] = {
                'mse': mse,
                'mae': mae,
                'r2': r2
            }

        return metrics

    def save(self, path: str):
        """保存模型"""
        model_data = {
            'scaler': self.scaler,
            'models': self.models,
            'feature_extractor': self.feature_extractor
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

    def load(self, path: str):
        """加载模型"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        self.scaler = model_data['scaler']
        self.models = model_data['models']
        self.feature_extractor = model_data['feature_extractor']


# ============================================================
# 训练流程
# ============================================================

def train_classifier(data_path: str, model_path: str = None) -> RelationClassifier:
    """
    完整训练流程

    Args:
        data_path: 数据文件路径
        model_path: 模型保存路径

    Returns:
        训练好的分类器
    """
    # 加载数据
    print("Loading dataset...")
    loader = DataLoader(data_path)
    X, y, dataset = loader.load_dataset()

    print(f"Dataset size: {len(X)} samples")
    print(f"Feature dimension: {X.shape[1]}")

    # 分割数据
    print("\nSplitting data...")
    splits = loader.split_data(X, y, test_size=0.2)

    # 创建分类器
    classifier = RelationClassifier()

    # 训练
    print("\nTraining models...")
    # 使用全部数据训练
    classifier.train(X, y)

    # 评估
    print("\nEvaluating...")
    for task, split in splits.items():
        if task in classifier.models:
            X_test = split['X_test']
            y_test = split['y_test']
            X_scaled = classifier.scaler.transform(X_test)

            if task in ['relation', 'a_attachment', 'b_attachment']:
                y_pred = classifier.models[task].predict(X_scaled)
                acc = np.mean(y_pred == y_test)
                print(f"{task}: accuracy = {acc:.2%}")
            else:
                y_pred = classifier.models[task].predict(X_scaled)
                mae = np.mean(np.abs(y_pred - y_test))
                print(f"{task}: MAE = {mae:.2f}")

    # 保存
    if model_path:
        print(f"\nSaving model to {model_path}")
        classifier.save(model_path)

    return classifier


__all__ = [
    'DialogueFeatures', 'FeatureExtractor',
    'DataLoader', 'RelationClassifier',
    'train_classifier'
]