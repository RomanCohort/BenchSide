"""
高级模型接口定义

预留接口，支持未来接入：
- BERT情感分析
- LSTM/Transformer时序预测
- 图神经网络分析
- 多模态分析

当前使用简化实现，便于后续替换
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


# ============================================================
# 基础数据结构
# ============================================================

@dataclass
class SentimentResult:
    """情感分析结果"""
    score: float              # 情感分数 (-1到+1)
    confidence: float         # 置信度
    aspects: Dict[str, float] = field(default_factory=dict)  # 各维度分数
    # 例如: {"positive": 0.8, "negative": 0.1, "neutral": 0.1}
    keywords: List[str] = field(default_factory=list)  # 关键情感词


@dataclass
class TrendPrediction:
    """趋势预测结果"""
    direction: str            # "up", "down", "stable"
    confidence: float
    predicted_scores: Dict[str, float]  # 预测的未来指标
    time_horizon: int         # 预测时间范围（天）
    factors: List[str]        # 影响因素


@dataclass
class GraphAnalysisResult:
    """图分析结果"""
    topics: List[str]         # 识别的话题
    topic_flow: List[Tuple[str, str, float]]  # 话题转移 (from, to, weight)
    key_nodes: List[str]      # 关键节点
    community_structure: Dict[str, List[str]]  # 社区结构


@dataclass
class MultimodalResult:
    """多模态分析结果"""
    text_sentiment: Optional[SentimentResult] = None
    emoji_sentiment: Optional[SentimentResult] = None
    voice_sentiment: Optional[SentimentResult] = None
    image_sentiment: Optional[SentimentResult] = None
    fused_sentiment: Optional[SentimentResult] = None  # 融合结果


# ============================================================
# 抽象接口定义
# ============================================================

class SentimentAnalyzerBase(ABC):
    """
    情感分析器基类

    未来可替换为：
    - BERT-based模型
    - RoBERTa
    - 情感词典方法
    """

    @abstractmethod
    def analyze(self, text: str) -> SentimentResult:
        """分析单条文本的情感"""
        pass

    @abstractmethod
    def batch_analyze(self, texts: List[str]) -> List[SentimentResult]:
        """批量分析"""
        pass

    @abstractmethod
    def get_embedding(self, text: str) -> np.ndarray:
        """获取文本嵌入向量"""
        pass


class TrendPredictorBase(ABC):
    """
    趋势预测器基类

    未来可替换为：
    - LSTM
    - Transformer
    - Prophet
    """

    @abstractmethod
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
        pass

    @abstractmethod
    def fit(self, data: List[Dict]) -> None:
        """训练/拟合模型"""
        pass


class GraphAnalyzerBase(ABC):
    """
    图分析器基类

    未来可替换为：
    - Graph Neural Networks
    - NetworkX分析
    - 话题模型
    """

    @abstractmethod
    def build_graph(self, messages: List[Dict]) -> Any:
        """构建对话图"""
        pass

    @abstractmethod
    def analyze(self, graph: Any) -> GraphAnalysisResult:
        """分析图结构"""
        pass

    @abstractmethod
    def get_topic_flow(self, graph: Any) -> List[Tuple[str, str, float]]:
        """获取话题流动"""
        pass


class MultimodalAnalyzerBase(ABC):
    """
    多模态分析器基类

    未来可扩展：
    - 表情包情感识别
    - 语音情感识别
    - 图像情感识别
    """

    @abstractmethod
    def analyze_text(self, text: str) -> SentimentResult:
        """分析文本"""
        pass

    @abstractmethod
    def analyze_emoji(self, emoji: str) -> SentimentResult:
        """分析表情"""
        pass

    @abstractmethod
    def analyze_voice(self, voice_path: str) -> SentimentResult:
        """分析语音"""
        pass

    @abstractmethod
    def analyze_image(self, image_path: str) -> SentimentResult:
        """分析图像"""
        pass

    @abstractmethod
    def fuse(self, results: List[SentimentResult]) -> SentimentResult:
        """多模态融合"""
        pass


# ============================================================
# 简化实现（当前使用）
# ============================================================

class SimpleSentimentAnalyzer(SentimentAnalyzerBase):
    """
    简化情感分析器

    基于词典的简单实现，未来可替换为BERT
    """

    # 情感词典
    POSITIVE_WORDS = [
        "爱", "喜欢", "开心", "高兴", "快乐", "幸福", "想你", "宝贝",
        "谢谢", "感谢", "棒", "好", "美", "可爱", "甜", "暖", "温柔",
        "哈哈", "嘻嘻", "期待", "晚安", "早安"
    ]

    NEGATIVE_WORDS = [
        "烦", "累", "难过", "伤心", "生气", "失望", "算了", "无所谓",
        "不想", "放弃", "难受", "哭", "讨厌", "恨", "滚", "别理我"
    ]

    INTENSIFIERS = ["很", "非常", "特别", "超级", "太", "真的"]

    def analyze(self, text: str) -> SentimentResult:
        """简化情感分析"""
        text_lower = text.lower()

        # 统计情感词
        pos_count = sum(1 for w in self.POSITIVE_WORDS if w in text_lower)
        neg_count = sum(1 for w in self.NEGATIVE_WORDS if w in text_lower)

        # 计算情感分数
        total = pos_count + neg_count
        if total == 0:
            score = 0.0
            confidence = 0.3
        else:
            score = (pos_count - neg_count) / total
            confidence = min(total / 5, 1.0)

        # 提取关键词
        keywords = []
        for w in self.POSITIVE_WORDS + self.NEGATIVE_WORDS:
            if w in text_lower:
                keywords.append(w)

        return SentimentResult(
            score=score,
            confidence=confidence,
            aspects={
                "positive": pos_count / max(total, 1),
                "negative": neg_count / max(total, 1)
            },
            keywords=keywords[:5]
        )

    def batch_analyze(self, texts: List[str]) -> List[SentimentResult]:
        """批量分析"""
        return [self.analyze(text) for text in texts]

    def get_embedding(self, text: str) -> np.ndarray:
        """简化嵌入（未来替换为BERT embedding）"""
        # 当前使用简单的词袋表示
        embedding = np.zeros(100, dtype=np.float32)
        words = text.split()
        for i, word in enumerate(words[:100]):
            embedding[i % 100] = hash(word) % 100 / 100.0
        return embedding


class SimpleTrendPredictor(TrendPredictorBase):
    """
    简化趋势预测器

    基于线性回归的简单实现，未来可替换为LSTM
    """

    def __init__(self):
        self.history_data = []

    def predict(self,
                history: List[Dict],
                horizon: int = 7) -> TrendPrediction:
        """简化趋势预测"""
        if len(history) < 2:
            return TrendPrediction(
                direction="stable",
                confidence=0.3,
                predicted_scores={},
                time_horizon=horizon,
                factors=["数据不足"]
            )

        # 提取loved_index趋势
        loved_values = [h.get("scores", {}).get("loved_index", 50) for h in history]

        # 简单线性趋势
        if len(loved_values) >= 2:
            trend = loved_values[-1] - loved_values[-2]
            if trend > 2:
                direction = "up"
            elif trend < -2:
                direction = "down"
            else:
                direction = "stable"
        else:
            direction = "stable"

        # 预测未来值
        predicted = loved_values[-1] + trend * horizon if loved_values else 50

        return TrendPrediction(
            direction=direction,
            confidence=0.5,
            predicted_scores={
                "loved_index": min(max(predicted, 0), 100)
            },
            time_horizon=horizon,
            factors=["线性趋势"]
        )

    def fit(self, data: List[Dict]) -> None:
        """拟合（简化版本不做训练）"""
        self.history_data = data


class SimpleGraphAnalyzer(GraphAnalyzerBase):
    """
    简化图分析器

    基于简单规则的实现，未来可替换为GNN
    """

    def build_graph(self, messages: List[Dict]) -> Dict:
        """构建简化对话图"""
        # 提取话题（简化：使用消息中的关键词）
        topics = set()
        edges = []

        for msg in messages:
            content = msg.get("content", "")
            # 简单提取话题
            words = content.split()
            for word in words:
                if len(word) >= 2:
                    topics.add(word)

        return {
            "nodes": list(topics),
            "edges": edges,
            "messages": messages
        }

    def analyze(self, graph: Dict) -> GraphAnalysisResult:
        """分析图结构"""
        return GraphAnalysisResult(
            topics=graph["nodes"][:10],
            topic_flow=[],
            key_nodes=graph["nodes"][:5],
            community_structure={}
        )

    def get_topic_flow(self, graph: Dict) -> List[Tuple[str, str, float]]:
        """获取话题流动"""
        # 简化：返回空
        return []


class SimpleMultimodalAnalyzer(MultimodalAnalyzerBase):
    """
    简化多模态分析器

    当前只支持文本，未来扩展表情/语音/图像
    """

    def __init__(self):
        self.text_analyzer = SimpleSentimentAnalyzer()

    def analyze_text(self, text: str) -> SentimentResult:
        """分析文本"""
        return self.text_analyzer.analyze(text)

    def analyze_emoji(self, emoji: str) -> SentimentResult:
        """分析表情（简化实现）"""
        # 表情情感映射
        positive_emojis = ["❤", "💕", "💖", "😘", "😊", "😄", "🥰"]
        negative_emojis = ["😢", "😭", "😤", "😠", "💔"]

        if any(e in emoji for e in positive_emojis):
            return SentimentResult(score=0.8, confidence=0.7)
        elif any(e in emoji for e in negative_emojis):
            return SentimentResult(score=-0.7, confidence=0.7)
        else:
            return SentimentResult(score=0.0, confidence=0.3)

    def analyze_voice(self, voice_path: str) -> SentimentResult:
        """分析语音（未实现）"""
        # TODO: 接入语音情感识别模型
        return SentimentResult(
            score=0.0,
            confidence=0.0,
            keywords=["语音分析未实现"]
        )

    def analyze_image(self, image_path: str) -> SentimentResult:
        """分析图像（未实现）"""
        # TODO: 接入图像情感识别模型
        return SentimentResult(
            score=0.0,
            confidence=0.0,
            keywords=["图像分析未实现"]
        )

    def fuse(self, results: List[SentimentResult]) -> SentimentResult:
        """多模态融合"""
        valid_results = [r for r in results if r.confidence > 0]

        if not valid_results:
            return SentimentResult(score=0.0, confidence=0.0)

        # 加权平均
        total_weight = sum(r.confidence for r in valid_results)
        weighted_score = sum(r.score * r.confidence for r in valid_results) / total_weight

        return SentimentResult(
            score=weighted_score,
            confidence=min(total_weight / len(valid_results), 1.0)
        )


# ============================================================
# 模型工厂
# ============================================================

class ModelFactory:
    """
    模型工厂

    统一创建各类模型，便于未来替换
    """

    @staticmethod
    def create_sentiment_analyzer(model_type: str = "simple") -> SentimentAnalyzerBase:
        """
        创建情感分析器

        Args:
            model_type: "simple", "bert", "roberta"
        """
        if model_type == "simple":
            return SimpleSentimentAnalyzer()
        elif model_type == "bert":
            # TODO: 实现BERT版本
            raise NotImplementedError("BERT模型尚未实现，请使用simple")
        else:
            return SimpleSentimentAnalyzer()

    @staticmethod
    def create_trend_predictor(model_type: str = "simple") -> TrendPredictorBase:
        """
        创建趋势预测器

        Args:
            model_type: "simple", "lstm", "transformer"
        """
        if model_type == "simple":
            return SimpleTrendPredictor()
        elif model_type == "lstm":
            # TODO: 实现LSTM版本
            raise NotImplementedError("LSTM模型尚未实现，请使用simple")
        else:
            return SimpleTrendPredictor()

    @staticmethod
    def create_graph_analyzer(model_type: str = "simple") -> GraphAnalyzerBase:
        """创建图分析器"""
        return SimpleGraphAnalyzer()

    @staticmethod
    def create_multimodal_analyzer(model_type: str = "simple") -> MultimodalAnalyzerBase:
        """创建多模态分析器"""
        return SimpleMultimodalAnalyzer()


__all__ = [
    # 数据结构
    'SentimentResult', 'TrendPrediction', 'GraphAnalysisResult', 'MultimodalResult',
    # 抽象基类
    'SentimentAnalyzerBase', 'TrendPredictorBase', 'GraphAnalyzerBase', 'MultimodalAnalyzerBase',
    # 简化实现
    'SimpleSentimentAnalyzer', 'SimpleTrendPredictor', 'SimpleGraphAnalyzer', 'SimpleMultimodalAnalyzer',
    # 工厂
    'ModelFactory'
]