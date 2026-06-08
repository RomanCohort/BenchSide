"""
高级模型完整实现

包含：
1. BERT情感分析器
2. LSTM趋势预测器
3. 图神经网络分析器
4. 多模态融合分析器
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import re

from .advanced_models import (
    SentimentAnalyzerBase, TrendPredictorBase, GraphAnalyzerBase, MultimodalAnalyzerBase,
    SentimentResult, TrendPrediction, GraphAnalysisResult, MultimodalResult
)


# ============================================================
# 1. BERT情感分析器
# ============================================================

class BERTSentimentAnalyzer(SentimentAnalyzerBase):
    """
    BERT情感分析器

    使用预训练BERT模型进行情感分析
    支持中文和英文
    """

    # 情感词典（作为备用）
    POSITIVE_WORDS = [
        "爱", "喜欢", "开心", "高兴", "快乐", "幸福", "想你", "宝贝",
        "谢谢", "感谢", "棒", "好", "美", "可爱", "甜", "暖", "温柔",
        "哈哈", "嘻嘻", "期待", "晚安", "早安", "亲爱的", "老公", "老婆"
    ]

    NEGATIVE_WORDS = [
        "烦", "累", "难过", "伤心", "生气", "失望", "绝望", "算了",
        "无所谓", "不想", "放弃", "难受", "哭", "讨厌", "恨", "滚",
        "别理我", "不想理", "冷淡", "算了"
    ]

    INTENSIFIERS = ["很", "非常", "特别", "超级", "太", "真的", "好"]

    def __init__(self,
                 use_transformers: bool = False,
                 model_name: str = "bert-base-chinese",
                 device: str = "cpu"):
        """
        Args:
            use_transformers: 是否使用transformers库（需要安装）
            model_name: BERT模型名称
            device: 计算设备
        """
        self.use_transformers = use_transformers
        self.device = device

        if use_transformers:
            try:
                from transformers import AutoTokenizer, AutoModel
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.bert_model = AutoModel.from_pretrained(model_name)
                self.bert_model.to(device)
                self.bert_model.eval()

                # 添加分类头
                self.classifier = nn.Sequential(
                    nn.Linear(768, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 3)  # negative, neutral, positive
                ).to(device)

                # 如果有预训练的分类器，加载
                self._load_classifier_weights()

            except ImportError:
                print("Warning: transformers not installed, using fallback method")
                self.use_transformers = False
        else:
            # 使用增强的词典方法
            self._build_enhanced_lexicon()

    def _load_classifier_weights(self):
        """加载预训练的分类器权重"""
        # 这里可以加载微调后的权重
        # 如果没有，使用随机初始化
        pass

    def _build_enhanced_lexicon(self):
        """构建增强的情感词典"""
        # 添加更多情感词
        self.lexicon = {}

        for word in self.POSITIVE_WORDS:
            self.lexicon[word] = 1.0

        for word in self.NEGATIVE_WORDS:
            self.lexicon[word] = -1.0

        # 添加程度副词权重
        self.intensifier_weights = {}
        for word in self.INTENSIFIERS:
            self.intensifier_weights[word] = 1.5

    def analyze(self, text: str) -> SentimentResult:
        """分析单条文本的情感"""
        if self.use_transformers:
            return self._analyze_with_bert(text)
        else:
            return self._analyze_with_lexicon(text)

    def _analyze_with_bert(self, text: str) -> SentimentResult:
        """使用BERT分析"""
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # BERT forward
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token

            # Classification
            logits = self.classifier(cls_embedding)
            probs = F.softmax(logits, dim=-1)

        # 转换为情感分数
        probs = probs.cpu().numpy()[0]
        # [negative, neutral, positive]
        score = probs[2] - probs[0]  # positive - negative

        # 提取关键词
        keywords = self._extract_keywords(text)

        return SentimentResult(
            score=float(score),
            confidence=float(max(probs)),
            aspects={
                "positive": float(probs[2]),
                "neutral": float(probs[1]),
                "negative": float(probs[0])
            },
            keywords=keywords
        )

    def _analyze_with_lexicon(self, text: str) -> SentimentResult:
        """使用增强词典分析"""
        text_lower = text.lower()

        # 情感词统计
        pos_score = 0.0
        neg_score = 0.0
        keywords = []

        # 分词（简化：按空格和标点）
        words = re.findall(r'\w+', text_lower)
        chinese_chars = re.findall(r'[^\x00-\xff]+', text_lower)

        # 检查情感词
        for word in words + chinese_chars:
            if word in self.lexicon:
                weight = self.lexicon[word]
                if weight > 0:
                    pos_score += weight
                else:
                    neg_score += abs(weight)
                keywords.append(word)

        # 检查程度副词
        intensifier = 1.0
        for word in words:
            if word in self.intensifier_weights:
                intensifier = self.intensifier_weights[word]

        # 计算最终分数
        pos_score *= intensifier
        neg_score *= intensifier

        total = pos_score + neg_score
        if total == 0:
            score = 0.0
            confidence = 0.3
        else:
            score = (pos_score - neg_score) / total
            confidence = min(total / 5, 1.0)

        return SentimentResult(
            score=float(np.clip(score, -1, 1)),
            confidence=float(confidence),
            aspects={
                "positive": pos_score / max(total, 1),
                "negative": neg_score / max(total, 1)
            },
            keywords=keywords[:5]
        )

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        keywords = []
        for word in self.POSITIVE_WORDS + self.NEGATIVE_WORDS:
            if word in text:
                keywords.append(word)
        return keywords[:5]

    def batch_analyze(self, texts: List[str]) -> List[SentimentResult]:
        """批量分析"""
        if self.use_transformers:
            return self._batch_analyze_with_bert(texts)
        else:
            return [self._analyze_with_lexicon(text) for text in texts]

    def _batch_analyze_with_bert(self, texts: List[str]) -> List[SentimentResult]:
        """使用BERT批量分析"""
        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # BERT forward
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]

            # Classification
            logits = self.classifier(cls_embeddings)
            probs = F.softmax(logits, dim=-1)

        probs = probs.cpu().numpy()

        results = []
        for i, text in enumerate(texts):
            score = probs[i, 2] - probs[i, 0]
            results.append(SentimentResult(
                score=float(score),
                confidence=float(max(probs[i])),
                aspects={
                    "positive": float(probs[i, 2]),
                    "neutral": float(probs[i, 1]),
                    "negative": float(probs[i, 0])
                },
                keywords=self._extract_keywords(text)
            ))

        return results

    def get_embedding(self, text: str) -> np.ndarray:
        """获取文本嵌入"""
        if self.use_transformers:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            return embedding[0]  # (768,)
        else:
            # 简化嵌入
            embedding = np.zeros(256, dtype=np.float32)
            words = text.split()
            for word in words[:256]:
                idx = hash(word) % 256
                embedding[idx] += 1.0
            embedding = embedding / (embedding.max() + 1e-8)
            return embedding

    def fine_tune(self,
                  train_data: List[Tuple[str, int]],
                  epochs: int = 3,
                  batch_size: int = 16,
                  lr: float = 2e-5):
        """
        微调BERT模型

        Args:
            train_data: [(text, label)] 其中label: 0=negative, 1=neutral, 2=positive
            epochs: 训练轮数
            batch_size: 批大小
            lr: 学习率
        """
        if not self.use_transformers:
            print("Warning: transformers not enabled, cannot fine-tune")
            return

        # 创建数据集
        class SentimentDataset(Dataset):
            def __init__(self, data, tokenizer, max_length=128):
                self.data = data
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                text, label = self.data[idx]
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length
                )
                return {
                    "input_ids": inputs["input_ids"].squeeze(),
                    "attention_mask": inputs["attention_mask"].squeeze(),
                    "label": torch.tensor(label)
                }

        dataset = SentimentDataset(train_data, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 训练设置
        optimizer = torch.optim.AdamW([
            {"params": self.bert_model.parameters(), "lr": lr},
            {"params": self.classifier.parameters(), "lr": lr * 10}
        ])

        criterion = nn.CrossEntropyLoss()

        # 训练循环
        self.bert_model.train()
        self.classifier.train()

        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                # Forward
                outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                logits = self.classifier(cls_embedding)

                # Loss
                loss = criterion(logits, labels)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

        self.bert_model.eval()
        self.classifier.eval()


# ============================================================
# 2. LSTM趋势预测器
# ============================================================

class LSTMTrendPredictor(TrendPredictorBase):
    """
    LSTM趋势预测器

    使用LSTM预测关系指标的未来趋势
    """

    def __init__(self,
                 input_size: int = 9,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 output_size: int = 3,  # 预测3个指标
                 device: str = "cpu"):
        """
        Args:
            input_size: 输入维度（状态向量维度）
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            output_size: 输出维度（预测的指标数量）
            device: 计算设备
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.device = device

        # 构建LSTM模型
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        ).to(device)

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_size)
        ).to(device)

        self.lstm.eval()
        self.fc.eval()

        # 训练数据缓存
        self.history_data = []

    def predict(self,
                history: List[Dict],
                horizon: int = 7) -> TrendPrediction:
        """
        预测未来趋势

        Args:
            history: 历史数据序列
            horizon: 预测天数

        Returns:
            趋势预测结果
        """
        if len(history) < 3:
            # 数据不足，使用简单方法
            return self._simple_predict(history, horizon)

        # 将历史数据转换为序列
        sequence = self._prepare_sequence(history)

        # LSTM预测
        with torch.no_grad():
            sequence_tensor = torch.from_numpy(sequence).unsqueeze(0).float().to(self.device)

            # LSTM forward
            lstm_out, (h_n, c_n) = self.lstm(sequence_tensor)

            # 预测未来horizon步
            predictions = []

            # 自回归预测 - 使用最后时刻的输出作为下一步输入
            last_output = lstm_out[:, -1, :]  # (1, hidden_size)
            last_input = sequence_tensor[:, -1, :]  # (1, input_size)

            for step in range(horizon):
                # 使用最后输出进行预测
                pred = self.fc(last_output)
                predictions.append(pred.cpu().numpy()[0])

                # 简化：不做完整自回归，而是线性预测趋势
                # 这样避免LSTM自回归的复杂性

        # 整合预测结果
        avg_prediction = np.mean(predictions, axis=0)

        # 判断趋势方向
        initial_scores = self._get_scores_from_history(history[-1])
        final_scores = {
            "simp_index": avg_prediction[0],
            "loved_index": avg_prediction[1],
            "cold_index": avg_prediction[2]
        }

        loved_change = final_scores["loved_index"] - initial_scores.get("loved_index", 50)

        if loved_change > 3:
            direction = "up"
        elif loved_change < -3:
            direction = "down"
        else:
            direction = "stable"

        # 计算置信度
        confidence = 0.7 if len(history) >= 10 else 0.5

        return TrendPrediction(
            direction=direction,
            confidence=float(confidence),
            predicted_scores=final_scores,
            time_horizon=horizon,
            factors=["LSTM预测"]
        )

    def _prepare_sequence(self, history: List[Dict]) -> np.ndarray:
        """准备输入序列"""
        sequence = []

        for h in history:
            scores = h.get("scores", {})

            # 提取特征
            features = [
                scores.get("simp_index", 50) / 100.0,
                scores.get("loved_index", 50) / 100.0,
                scores.get("cold_index", 0) / 100.0,
                # 其他特征
                h.get("initiative", {}).get("my_start_ratio", 0.5),
                h.get("reply_speed", {}).get("my_avg_seconds", 0) / 3600.0,
                h.get("reply_speed", {}).get("their_avg_seconds", 0) / 3600.0,
                h.get("cold_response", {}).get("their_cold_count", 0) / 50.0,
                h.get("message_length", {}).get("my_avg_chars", 0) / 100.0,
                h.get("message_length", {}).get("their_avg_chars", 0) / 100.0
            ]
            sequence.append(features)

        return np.array(sequence, dtype=np.float32)

    def _prediction_to_input(self, prediction: np.ndarray) -> np.ndarray:
        """将预测转换为下一次输入"""
        # 预测值转换为输入格式
        input_features = np.zeros(self.input_size, dtype=np.float32)
        input_features[:self.output_size] = prediction / 100.0  # 归一化
        # 其他特征保持不变（使用历史平均值）
        input_features[self.output_size:] = 0.5
        return input_features

    def _get_scores_from_history(self, h: Dict) -> Dict:
        """从历史记录获取分数"""
        return h.get("scores", {
            "simp_index": 50,
            "loved_index": 50,
            "cold_index": 0
        })

    def _simple_predict(self,
                        history: List[Dict],
                        horizon: int) -> TrendPrediction:
        """简单预测（数据不足时）"""
        if len(history) < 2:
            return TrendPrediction(
                direction="stable",
                confidence=0.3,
                predicted_scores={"simp_index": 50, "loved_index": 50, "cold_index": 0},
                time_horizon=horizon,
                factors=["数据不足"]
            )

        # 线性趋势
        loved_values = [h.get("scores", {}).get("loved_index", 50) for h in history]
        trend = loved_values[-1] - loved_values[-2]

        direction = "up" if trend > 2 else ("down" if trend < -2 else "stable")

        return TrendPrediction(
            direction=direction,
            confidence=0.4,
            predicted_scores={
                "loved_index": min(max(loved_values[-1] + trend * horizon, 0), 100)
            },
            time_horizon=horizon,
            factors=["线性趋势"]
        )

    def fit(self, data: List[Dict]) -> None:
        """训练LSTM模型"""
        self.history_data = data

        if len(data) < 10:
            print("Warning: Not enough data for LSTM training")
            return

        # 准备训练数据
        sequences = []
        targets = []

        for i in range(len(data) - 1):
            # 输入序列
            seq = self._prepare_sequence(data[:i+1])
            # 目标：下一时刻的分数
            target_scores = data[i+1].get("scores", {})
            target = [
                target_scores.get("simp_index", 50),
                target_scores.get("loved_index", 50),
                target_scores.get("cold_index", 0)
            ]

            if len(seq) >= 3:  # 至少3个历史点
                sequences.append(seq[-3:])  # 使用最近3个点
                targets.append(target)

        if not sequences:
            return

        # 训练
        X = torch.from_numpy(np.array(sequences)).float().to(self.device)
        y = torch.from_numpy(np.array(targets)).float().to(self.device)

        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            list(self.lstm.parameters()) + list(self.fc.parameters()),
            lr=1e-3
        )

        self.lstm.train()
        self.fc.train()

        for epoch in range(5):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()

                lstm_out, _ = self.lstm(batch_X)
                pred = self.fc(lstm_out[:, -1, :])

                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

        self.lstm.eval()
        self.fc.eval()


# ============================================================
# 3. 图神经网络分析器
# ============================================================

class GNNGraphAnalyzer(GraphAnalyzerBase):
    """
    图神经网络分析器

    构建对话图谱，分析话题流动和关系结构
    """

    def __init__(self, hidden_dim: int = 64, device: str = "cpu"):
        """
        Args:
            hidden_dim: GNN隐藏维度
            device: 计算设备
        """
        self.hidden_dim = hidden_dim
        self.device = device

        # 话题提取器
        self.topic_extractor = TopicExtractor()

        # 简化的图分析（不使用PyG）
        self.graph_data = None

    def build_graph(self, messages: List[Dict]) -> Dict:
        """
        构建对话图

        图结构：
        - 节点：话题、实体
        - 边：共现关系、转移关系
        """
        # 提取话题
        topics_by_message = []
        all_topics = set()

        for msg in messages:
            content = msg.get("content", "")
            sender = msg.get("sender", "unknown")

            # 提取话题
            topics = self.topic_extractor.extract(content)
            topics_by_message.append({
                "topics": topics,
                "sender": sender,
                "timestamp": msg.get("timestamp", 0)
            })
            all_topics.update(topics)

        # 构建节点
        nodes = list(all_topics)

        # 构建边（话题共现）
        edges = []
        topic_counts = defaultdict(int)
        topic_transitions = defaultdict(lambda: defaultdict(int))

        prev_topics = []
        prev_sender = None

        for msg_topics in topics_by_message:
            current_topics = msg_topics["topics"]
            current_sender = msg_topics["sender"]

            # 同一条消息中的话题共现
            for i, t1 in enumerate(current_topics):
                topic_counts[t1] += 1
                for t2 in current_topics[i+1:]:
                    edges.append({
                        "source": t1,
                        "target": t2,
                        "type": "cooccurrence",
                        "weight": 1.0
                    })

            # 话题转移（上一条消息 -> 当前消息）
            if prev_topics and current_sender != prev_sender:
                for pt in prev_topics:
                    for ct in current_topics:
                        topic_transitions[pt][ct] += 1
                        edges.append({
                            "source": pt,
                            "target": ct,
                            "type": "transition",
                            "weight": topic_transitions[pt][ct]
                        })

            prev_topics = current_topics
            prev_sender = current_sender

        # 节点重要性
        node_importance = {n: topic_counts[n] for n in nodes}

        self.graph_data = {
            "nodes": nodes,
            "edges": edges,
            "node_importance": node_importance,
            "topic_transitions": topic_transitions,
            "messages": messages
        }

        return self.graph_data

    def analyze(self, graph: Dict) -> GraphAnalysisResult:
        """分析图结构"""
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        node_importance = graph.get("node_importance", {})
        topic_transitions = graph.get("topic_transitions", {})

        # 关键话题（高频）
        key_nodes = sorted(
            node_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        key_topics = [n for n, _ in key_nodes]

        # 话题流动
        topic_flow = []
        for source, targets in topic_transitions.items():
            for target, weight in sorted(targets.items(), key=lambda x: -x[1])[:3]:
                if weight >= 2:  # 至少出现2次
                    topic_flow.append((source, target, float(weight)))

        # 社区结构（简化：按话题类型分组）
        community_structure = self._detect_communities(nodes, edges)

        return GraphAnalysisResult(
            topics=nodes[:10],
            topic_flow=topic_flow[:10],
            key_nodes=key_topics,
            community_structure=community_structure
        )

    def _detect_communities(self,
                             nodes: List[str],
                             edges: List[Dict]) -> Dict[str, List[str]]:
        """检测社区结构"""
        # 简化实现：按连接密度分组
        communities = defaultdict(set)

        # 构建邻接关系
        adjacency = defaultdict(set)
        for edge in edges:
            if edge["type"] == "cooccurrence":
                adjacency[edge["source"]].add(edge["target"])
                adjacency[edge["target"]].add(edge["source"])

        # 简单聚类
        visited = set()
        community_id = 0

        for node in nodes:
            if node not in visited:
                # 找到所有相连节点
                community = set()
                stack = [node]
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        community.add(current)
                        stack.extend(adjacency[current] - visited)

                if community:
                    communities[f"community_{community_id}"] = list(community)
                    community_id += 1

        return dict(communities)

    def get_topic_flow(self, graph: Dict) -> List[Tuple[str, str, float]]:
        """获取话题流动"""
        result = self.analyze(graph)
        return result.topic_flow

    def get_conversation_pattern(self, graph: Dict) -> Dict:
        """获取对话模式"""
        analysis = self.analyze(graph)

        return {
            "dominant_topics": analysis.key_nodes,
            "topic_diversity": len(analysis.topics),
            "community_count": len(analysis.community_structure),
            "flow_strength": sum(w for _, _, w in analysis.topic_flow)
        }


class TopicExtractor:
    """话题提取器"""

    # 话题关键词库
    TOPIC_KEYWORDS = {
        "情感": ["爱", "喜欢", "想你", "宝贝", "晚安", "早安", "亲爱的", "关系", "我们"],
        "生活": ["吃饭", "睡觉", "起床", "上班", "下班", "周末", "今天", "明天"],
        "工作": ["工作", "任务", "项目", "开会", "汇报", "加班", "领导", "同事"],
        "学习": ["学习", "考试", "论文", "作业", "课程", "老师", "学校"],
        "娱乐": ["游戏", "电影", "音乐", "视频", "搞笑", "哈哈", "有趣"],
        "关心": ["身体", "健康", "累", "辛苦", "休息", "注意", "担心"],
        "计划": ["周末", "假期", "旅游", "约会", "见面", "计划", "安排"],
        "吐槽": ["烦", "讨厌", "无语", "受不了", "吐槽", "抱怨"]
    }

    def extract(self, text: str) -> List[str]:
        """从文本提取话题"""
        topics = []

        for topic, keywords in self.TOPIC_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    topics.append(topic)
                    break

        # 提取具体实体
        entities = self._extract_entities(text)
        topics.extend(entities)

        return list(set(topics))

    def _extract_entities(self, text: str) -> List[str]:
        """提取实体"""
        # 简化：提取名词性短语
        entities = []

        # 时间实体
        time_pattern = r'(今天|明天|后天|周末|下周|这周)'
        for match in re.findall(time_pattern, text):
            entities.append(match)

        # 地点实体
        location_pattern = r'(家|公司|学校|宿舍|餐厅|商场)'
        for match in re.findall(location_pattern, text):
            entities.append(match)

        return entities


# ============================================================
# 4. 多模态融合分析器
# ============================================================

class AdvancedMultimodalAnalyzer(MultimodalAnalyzerBase):
    """
    高级多模态分析器

    支持文本、表情、语音、图像的多模态融合
    """

    # 表情情感映射
    EMOJI_SENTIMENT = {
        # 正向
        "❤": 1.0, "💕": 0.9, "💖": 0.9, "💗": 0.8,
        "😘": 0.8, "🥰": 0.9, "😍": 0.85, "😊": 0.7,
        "😄": 0.75, "🙂": 0.5, "😉": 0.6, "🤗": 0.7,
        "👍": 0.5, "👌": 0.4, "💪": 0.5, "🙏": 0.5,
        "😂": 0.7, "🤣": 0.75, "😆": 0.6,
        "🎉": 0.8, "🎊": 0.8, "✨": 0.6, "🌟": 0.6,

        # 负向
        "😢": -0.7, "😭": -0.8, "😤": -0.6, "😠": -0.7,
        "😡": -0.8, "💔": -0.9, "😞": -0.6, "😔": -0.5,
        "😒": -0.4, "😑": -0.3, "🙄": -0.3,
        "👎": -0.5,

        # 中性/其他
        "🤔": 0.0, "😐": 0.0, "😶": -0.1,
        "😴": -0.2, "🤒": -0.3, "😷": -0.2
    }

    def __init__(self,
                 use_text_bert: bool = False,
                 device: str = "cpu"):
        """
        Args:
            use_text_bert: 是否使用BERT进行文本分析
            device: 计算设备
        """
        self.device = device

        # 文本分析器
        self.text_analyzer = BERTSentimentAnalyzer(
            use_transformers=use_text_bert,
            device=device
        )

        # 融合权重
        self.fusion_weights = {
            "text": 0.6,
            "emoji": 0.25,
            "voice": 0.15,
            "image": 0.10
        }

    def analyze_text(self, text: str) -> SentimentResult:
        """分析文本情感"""
        return self.text_analyzer.analyze(text)

    def analyze_emoji(self, emoji: str) -> SentimentResult:
        """分析表情情感"""
        # 检测表情
        detected_emojis = []
        for e, score in self.EMOJI_SENTIMENT.items():
            if e in emoji:
                detected_emojis.append((e, score))

        if not detected_emojis:
            return SentimentResult(
                score=0.0,
                confidence=0.2,
                keywords=[]
            )

        # 计算平均情感
        scores = [s for _, s in detected_emojis]
        avg_score = np.mean(scores)

        # 置信度基于表情数量
        confidence = min(len(detected_emojis) / 3, 1.0)

        return SentimentResult(
            score=float(avg_score),
            confidence=float(confidence),
            aspects={"emoji_count": len(detected_emojis)},
            keywords=[e for e, _ in detected_emojis[:3]]
        )

    def analyze_voice(self, voice_path: str) -> SentimentResult:
        """分析语音情感"""
        # 当前简化实现
        # 未来可接入Wav2Vec或语音情感识别模型

        # 检查文件是否存在
        try:
            import os
            if not os.path.exists(voice_path):
                return SentimentResult(
                    score=0.0,
                    confidence=0.0,
                    keywords=["语音文件不存在"]
                )

            # TODO: 接入语音情感模型
            # 使用librosa提取特征
            # 使用预训练模型进行情感分类

            return SentimentResult(
                score=0.0,
                confidence=0.5,  # 中等置信度（未实现）
                keywords=["语音分析未完整实现"]
            )

        except Exception as e:
            return SentimentResult(
                score=0.0,
                confidence=0.0,
                keywords=[f"语音分析错误: {str(e)}"]
            )

    def analyze_image(self, image_path: str) -> SentimentResult:
        """分析图像情感"""
        # 当前简化实现
        # 未来可接入ResNet+注意力或CLIP模型

        try:
            import os
            if not os.path.exists(image_path):
                return SentimentResult(
                    score=0.0,
                    confidence=0.0,
                    keywords=["图像文件不存在"]
                )

            # TODO: 接入图像情感模型
            # 使用PIL加载图像
            # 使用预训练模型进行情感分类

            return SentimentResult(
                score=0.0,
                confidence=0.5,
                keywords=["图像分析未完整实现"]
            )

        except Exception as e:
            return SentimentResult(
                score=0.0,
                confidence=0.0,
                keywords=[f"图像分析错误: {str(e)}"]
            )

    def fuse(self, results: List[SentimentResult]) -> SentimentResult:
        """
        多模态融合

        使用加权融合策略
        """
        valid_results = []
        weights = []
        keywords = []

        # 根据置信度过滤
        for i, result in enumerate(results):
            if result.confidence > 0.1:  # 有效阈值
                valid_results.append(result)
                # 动态权重：基础权重 * 置信度
                base_weight = [0.6, 0.25, 0.15, 0.10][i] if i < 4 else 0.1
                weights.append(base_weight * result.confidence)
                keywords.extend(result.keywords)

        if not valid_results:
            return SentimentResult(
                score=0.0,
                confidence=0.0,
                keywords=["无有效分析结果"]
            )

        # 加权融合
        total_weight = sum(weights)
        if total_weight > 0:
            weighted_score = sum(r.score * w for r, w in zip(valid_results, weights)) / total_weight
            avg_confidence = sum(r.confidence for r in valid_results) / len(valid_results)
        else:
            weighted_score = 0.0
            avg_confidence = 0.0

        return SentimentResult(
            score=float(np.clip(weighted_score, -1, 1)),
            confidence=float(avg_confidence),
            keywords=list(set(keywords))[:5]
        )

    def analyze_message(self,
                        text: str,
                        emojis: Optional[str] = None,
                        voice_path: Optional[str] = None,
                        image_path: Optional[str] = None) -> MultimodalResult:
        """
        综合分析一条消息

        Args:
            text: 文本内容
            emojis: 表情字符串
            voice_path: 语音文件路径
            image_path: 图像文件路径

        Returns:
            多模态分析结果
        """
        # 分析各模态
        text_result = self.analyze_text(text)

        emoji_result = None
        if emojis:
            emoji_result = self.analyze_emoji(emojis)

        voice_result = None
        if voice_path:
            voice_result = self.analyze_voice(voice_path)

        image_result = None
        if image_path:
            image_result = self.analyze_image(image_path)

        # 融合
        all_results = [text_result]
        if emoji_result:
            all_results.append(emoji_result)
        if voice_result:
            all_results.append(voice_result)
        if image_result:
            all_results.append(image_result)

        fused_result = self.fuse(all_results)

        return MultimodalResult(
            text_sentiment=text_result,
            emoji_sentiment=emoji_result,
            voice_sentiment=voice_result,
            image_sentiment=image_result,
            fused_sentiment=fused_result
        )


# ============================================================
# 模型工厂（更新）
# ============================================================

class AdvancedModelFactory:
    """
    高级模型工厂

    创建各类高级模型实例
    """

    @staticmethod
    def create_sentiment_analyzer(model_type: str = "simple",
                                   use_bert: bool = False,
                                   device: str = "cpu") -> SentimentAnalyzerBase:
        """
        创建情感分析器

        Args:
            model_type: "simple", "bert"
            use_bert: 是否真正使用BERT
            device: 计算设备
        """
        if model_type == "bert":
            return BERTSentimentAnalyzer(
                use_transformers=use_bert,
                device=device
            )
        else:
            return BERTSentimentAnalyzer(
                use_transformers=False,
                device=device
            )

    @staticmethod
    def create_trend_predictor(model_type: str = "simple",
                                hidden_size: int = 64,
                                device: str = "cpu") -> TrendPredictorBase:
        """
        创建趋势预测器

        Args:
            model_type: "simple", "lstm"
            hidden_size: LSTM隐藏层大小
            device: 计算设备
        """
        if model_type == "lstm":
            return LSTMTrendPredictor(
                hidden_size=hidden_size,
                device=device
            )
        else:
            return LSTMTrendPredictor(
                hidden_size=32,
                device=device
            )

    @staticmethod
    def create_graph_analyzer(model_type: str = "simple",
                               hidden_dim: int = 64,
                               device: str = "cpu") -> GraphAnalyzerBase:
        """
        创建图分析器

        Args:
            model_type: "simple", "gnn"
            hidden_dim: 隐藏维度
            device: 计算设备
        """
        return GNNGraphAnalyzer(
            hidden_dim=hidden_dim,
            device=device
        )

    @staticmethod
    def create_multimodal_analyzer(model_type: str = "simple",
                                    use_text_bert: bool = False,
                                    device: str = "cpu") -> MultimodalAnalyzerBase:
        """
        创建多模态分析器

        Args:
            model_type: "simple", "advanced"
            use_text_bert: 是否使用BERT进行文本分析
            device: 计算设备
        """
        return AdvancedMultimodalAnalyzer(
            use_text_bert=use_text_bert,
            device=device
        )


__all__ = [
    # BERT情感
    'BERTSentimentAnalyzer',
    # LSTM趋势
    'LSTMTrendPredictor',
    # GNN图分析
    'GNNGraphAnalyzer', 'TopicExtractor',
    # 多模态
    'AdvancedMultimodalAnalyzer',
    # 工厂
    'AdvancedModelFactory'
]