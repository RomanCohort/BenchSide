# Advanced Models Integration Guide

本文档说明如何在未来接入高级模型。

---

## 📊 当前架构

```
┌─────────────────────────────────────────────────────┐
│                  ModelFactory                        │
│  ┌───────────────────────────────────────────────┐  │
│  │ create_sentiment_analyzer(model_type)          │  │
│  │ create_trend_predictor(model_type)             │  │
│  │ create_graph_analyzer(model_type)              │  │
│  │ create_multimodal_analyzer(model_type)          │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────┐
│              Abstract Base Classes                   │
│  ┌─────────────────────────────────────────────┐    │
│  │ SentimentAnalyzerBase                       │    │
│  │ TrendPredictorBase                          │    │
│  │ GraphAnalyzerBase                           │    │
│  │ MultimodalAnalyzerBase                      │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────┐
│              Current Implementations                 │
│  ┌─────────────────────────────────────────────┐    │
│  │ SimpleSentimentAnalyzer (词典方法)           │    │
│  │ SimpleTrendPredictor (线性回归)              │    │
│  │ SimpleGraphAnalyzer (规则方法)               │    │
│  │ SimpleMultimodalAnalyzer (仅文本)           │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

---

## 🔮 未来接入方案

### 1️⃣ BERT情感分析

```python
# core/advanced_models.py

class BERTSentimentAnalyzer(SentimentAnalyzerBase):
    """BERT情感分析器"""

    def __init__(self, model_name: str = "bert-base-chinese"):
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        # 加载微调后的分类头
        self.classifier = self._load_classifier()

    def analyze(self, text: str) -> SentimentResult:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        # 使用CLS token进行分类
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        probs = torch.softmax(logits, dim=-1)
        return SentimentResult(
            score=(probs[0, 2] - probs[0, 0]).item(),  # positive - negative
            confidence=probs.max().item()
        )

# 使用
analyzer = ModelFactory.create_sentiment_analyzer("bert")
```

### 2️⃣ LSTM时序预测

```python
class LSTMTrendPredictor(TrendPredictorBase):
    """LSTM趋势预测器"""

    def __init__(self, hidden_size: int = 64, num_layers: int = 2):
        self.model = nn.LSTM(
            input_size=9,  # 状态维度
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def predict(self, history: List[Dict], horizon: int = 7) -> TrendPrediction:
        # 将历史数据转换为序列
        sequence = self._prepare_sequence(history)
        with torch.no_grad():
            output, _ = self.model(sequence)
            prediction = self.fc(output[-1])
        # ... 返回预测结果

# 使用
predictor = ModelFactory.create_trend_predictor("lstm")
```

### 3️⃣ 图神经网络分析

```python
class GNNGraphAnalyzer(GraphAnalyzerBase):
    """图神经网络分析器"""

    def __init__(self, hidden_dim: int = 64):
        import torch_geometric
        self.conv1 = torch_geometric.nn.GCNConv(100, hidden_dim)
        self.conv2 = torch_geometric.nn.GCNConv(hidden_dim, hidden_dim)

    def build_graph(self, messages: List[Dict]) -> Data:
        # 构建PyG Data对象
        # 节点：话题/实体
        # 边：共现关系
        pass

    def analyze(self, graph: Data) -> GraphAnalysisResult:
        # GNN前向传播
        # 提取关键节点和社区
        pass

# 使用
analyzer = ModelFactory.create_graph_analyzer("gnn")
```

### 4️⃣ 多模态融合

```python
class AdvancedMultimodalAnalyzer(MultimodalAnalyzerBase):
    """高级多模态分析器"""

    def __init__(self):
        # 文本模型
        self.text_model = BERTSentimentAnalyzer()
        # 表情模型（预训练）
        self.emoji_model = self._load_emoji_model()
        # 语音模型（Wav2Vec）
        self.voice_model = self._load_voice_model()
        # 图像模型（ResNet+注意力）
        self.image_model = self._load_image_model()
        # 融合层
        self.fusion = nn.MultiheadAttention(embed_dim=256, num_heads=4)

    def analyze_voice(self, voice_path: str) -> SentimentResult:
        # 加载音频
        waveform = torchaudio.load(voice_path)
        # Wav2Vec特征提取
        features = self.voice_model(waveform)
        # 情感分类
        pass
```

---

## 📋 接入清单

| 模型 | 优先级 | 依赖 | 预期收益 |
|------|--------|------|----------|
| **BERT情感** | ⭐⭐⭐⭐ | transformers | 情感分析准确率+20% |
| **LSTM趋势** | ⭐⭐⭐ | pytorch | 预测准确率+15% |
| **GNN图分析** | ⭐⭐ | torch_geometric | 话题流动理解 |
| **多模态** | ⭐⭐ | torchaudio, torchvision | 表情包/语音情感 |

---

## 🚀 接入步骤

### Step 1: 准备数据

```python
# 训练数据格式
training_data = [
    {
        "text": "今天很开心",
        "label": 1,  # positive
        "context": {...}
    },
    ...
]
```

### Step 2: 训练模型

```bash
# 训练BERT情感分类器
python training/train_bert.py \
    --data data/sentiment_train.json \
    --output models/bert_sentiment.pth \
    --epochs 10
```

### Step 3: 注册到工厂

```python
# core/advanced_models.py

@staticmethod
def create_sentiment_analyzer(model_type: str = "simple"):
    if model_type == "bert":
        return BERTSentimentAnalyzer()
    elif model_type == "simple":
        return SimpleSentimentAnalyzer()
```

### Step 4: 更新配置

```yaml
# config/hyperparams.yaml

models:
  sentiment:
    type: bert  # simple | bert
    model_name: bert-base-chinese

  trend:
    type: lstm  # simple | lstm
    hidden_size: 64

  graph:
    type: simple  # simple | gnn

  multimodal:
    type: simple  # simple | advanced
```

---

## ⚠️ 注意事项

1. **依赖管理**: 高级模型需要额外依赖，建议使用 `requirements-advanced.txt`
2. **GPU支持**: BERT/LSTM建议GPU加速
3. **模型大小**: BERT模型约400MB，需提前下载
4. **推理速度**: GPU推理约10ms/条，CPU约100ms/条

---

## 📦 依赖文件

```txt
# requirements-advanced.txt

# BERT情感分析
transformers>=4.20.0
torch>=1.12.0

# LSTM时序预测
torch>=1.12.0

# 图神经网络
torch-geometric>=2.1.0

# 多模态分析
torchaudio>=0.12.0
torchvision>=0.13.0
Pillow>=9.0.0
```

---

**当前状态**: 所有接口已预留，使用简化实现，便于后续替换。