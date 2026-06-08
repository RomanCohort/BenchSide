"""
可学习的韧性因子模型

核心改进：
1. 不使用硬编码权重
2. 使用因子分析（FA）自然提取韧性因子
3. 使用监督学习学习权重
4. 输出分级建议而非定性标签

方法：
- 无监督：PCA/FA 提取因子，权重是载荷
- 半监督：用 PHQ-9/GAD-7 切点作为标签学习
- 输出：风险分级建议，不输出诊断标签
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pickle
import warnings

# 抑制警告
warnings.filterwarnings('ignore')


# ============================================================
# 数据结构定义
# ============================================================

class RiskLevel(Enum):
    """风险等级（非诊断标签）"""
    LOW = "低风险"
    MODERATE = "中风险"
    ELEVATED = "高风险"
    HIGH = "需关注"

    @property
    def recommendation(self) -> str:
        """分级建议"""
        recommendations = {
            RiskLevel.LOW: "继续保持当前社交状态",
            RiskLevel.MODERATE: "建议适当增加社交互动",
            RiskLevel.ELEVATED: "建议主动寻求社交支持，考虑与信任的人交流",
            RiskLevel.HIGH: "建议寻求专业心理支持资源"
        }
        return recommendations[self]


@dataclass
class ResilienceState:
    """
    韧性状态表征（非分数）

    R 是一个状态表征，不是"有病/没病"的标签
    """
    # 原始维度值
    dimensions: Dict[str, float] = field(default_factory=dict)

    # 因子得分（由FA计算，权重自然产生）
    factor_scores: Dict[str, float] = field(default_factory=dict)

    # 主韧性因子得分
    primary_factor_score: float = 0

    # 风险等级（分级建议）
    risk_level: RiskLevel = RiskLevel.LOW

    # 置信区间（不确定性量化）
    confidence_interval: Tuple[float, float] = (0, 0)


@dataclass
class LabeledSample:
    """
    带标签的样本

    使用客观事件作为标签，而非主观诊断
    """
    # 特征向量
    features: np.ndarray

    # 软标签：PHQ-9/GAD-7 得分
    phq9_score: Optional[float] = None
    gad7_score: Optional[float] = None

    # 硬标签：客观事件
    sought_counseling: Optional[bool] = None  # 是否求助咨询
    took_leave: Optional[bool] = None         # 是否休学
    referred_treatment: Optional[bool] = None # 是否转院治疗


# ============================================================
# 无监督因子分析
# ============================================================

class FactorAnalysisResilience:
    """
    因子分析韧性模型

    使用 PCA/FA 自然提取韧性因子
    权重由数据决定，不是手填
    """

    def __init__(self, n_factors: int = 2):
        self.n_factors = n_factors
        self.factor_loadings = None  # 因子载荷（自然权重）
        self.explained_variance = None
        self.mean = None
        self.std = None
        self.fitted = False

    def fit(self, data: np.ndarray) -> 'FactorAnalysisResilience':
        """
        拟合因子分析模型

        Args:
            data: 形状为 (n_samples, n_features) 的数据
                  features = [recovery_speed, recovery_depth, adaptability,
                             redundancy, diversity]
        """
        n_samples, n_features = data.shape

        # 标准化
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.std[self.std == 0] = 1  # 避免除零

        data_normalized = (data - self.mean) / self.std

        # PCA 提取主成分
        # 协方差矩阵
        cov_matrix = np.cov(data_normalized.T)

        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 按特征值降序排列
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 保留前 n_factors 个因子
        self.factor_loadings = eigenvectors[:, :self.n_factors]
        self.explained_variance = eigenvalues / np.sum(eigenvalues)

        self.fitted = True

        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        转换为因子得分

        Args:
            data: 原始数据

        Returns:
            因子得分
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # 标准化
        data_normalized = (data - self.mean) / self.std

        # 计算因子得分
        factor_scores = data_normalized @ self.factor_loadings

        return factor_scores

    def get_factor_interpretation(self) -> Dict:
        """
        解释因子含义

        基于载荷矩阵解释每个因子
        """
        if not self.fitted:
            return {}

        dimension_names = [
            'recovery_speed', 'recovery_depth', 'adaptability',
            'redundancy', 'diversity'
        ]

        interpretation = {}

        for i in range(self.n_factors):
            # 该因子的载荷
            loadings = self.factor_loadings[:, i]

            # 找出主要贡献维度
            contributions = {}
            for j, name in enumerate(dimension_names):
                contributions[name] = loadings[j]

            # 按贡献排序
            sorted_contributions = sorted(
                contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )

            interpretation[f'factor_{i+1}'] = {
                'loadings': dict(sorted_contributions),
                'explained_variance': self.explained_variance[i],
                'interpretation': self._interpret_factor(loadings)
            }

        return interpretation

    def _interpret_factor(self, loadings: np.ndarray) -> str:
        """解释单个因子"""
        # 根据载荷大小解释
        abs_loadings = np.abs(loadings)
        dominant_idx = np.argmax(abs_loadings)

        interpretations = [
            ('恢复速度主导因子', '主要反映扰动后的恢复速度'),
            ('恢复深度主导因子', '主要反映恢复的完整性'),
            ('适应性主导因子', '主要反映策略调整能力'),
            ('冗余度主导因子', '主要反映备用支持系统'),
            ('多样性主导因子', '主要反映社交圈多样性')
        ]

        return interpretations[dominant_idx][1]


# ============================================================
# 半监督学习模型
# ============================================================

class SemiSupervisedResilienceLearner:
    """
    半监督韧性学习器

    使用少量标签学习权重
    标签来源：
    - PHQ-9/GAD-7 切点（软标签）
    - 客观事件（硬标签）
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 8):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 初始化权重（随机，后续学习）
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * 0.1
        self.b2 = np.zeros(1)

        self.fitted = False
        self.training_history = []

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        hard_labels: Optional[np.ndarray] = None,
        epochs: int = 100,
        lr: float = 0.01
    ) -> 'SemiSupervisedResilienceLearner':
        """
        拟合半监督模型

        Args:
            X: 特征矩阵 (n_samples, 5)
            y: 软标签（PHQ-9/GAD-7 得分，归一化到0-1）
            hard_labels: 硬标签（是否求助咨询等）
            epochs: 训练轮数
            lr: 学习率
        """
        n_samples = X.shape[0]

        # 准备目标
        if hard_labels is not None:
            # 使用硬标签
            target = hard_labels.astype(float).reshape(-1, 1)
        elif y is not None:
            # 使用软标签
            target = y.reshape(-1, 1)
        else:
            # 无监督：使用自编码目标
            target = np.mean(X, axis=1, keepdims=True)

        # 训练
        for epoch in range(epochs):
            # 前向传播
            h = np.maximum(0, X @ self.W1 + self.b1)  # ReLU
            pred = 1 / (1 + np.exp(-(h @ self.W2 + self.b2)))  # Sigmoid

            # 计算损失
            loss = -np.mean(
                target * np.log(pred + 1e-8) +
                (1 - target) * np.log(1 - pred + 1e-8)
            )

            # 反向传播
            d_pred = pred - target
            d_W2 = h.T @ d_pred / n_samples
            d_b2 = np.mean(d_pred)

            d_h = d_pred @ self.W2.T
            d_h[h <= 0] = 0  # ReLU梯度

            d_W1 = X.T @ d_h / n_samples
            d_b1 = np.mean(d_h, axis=0)

            # 更新权重
            self.W1 -= lr * d_W1
            self.b1 -= lr * d_b1
            self.W2 -= lr * d_W2
            self.b2 -= lr * d_b2

            self.training_history.append(loss)

        self.fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测风险得分

        Args:
            X: 特征矩阵

        Returns:
            风险得分 0-1
        """
        if not self.fitted:
            raise ValueError("Model not fitted")

        h = np.maximum(0, X @ self.W1 + self.b1)
        pred = 1 / (1 + np.exp(-(h @ self.W2 + self.b2)))

        return pred.flatten()

    def get_learned_weights(self) -> Dict:
        """
        获取学习到的权重

        不是手填，而是从数据学习
        """
        if not self.fitted:
            return {}

        dimension_names = [
            'recovery_speed', 'recovery_depth', 'adaptability',
            'redundancy', 'diversity'
        ]

        # 计算每个输入维度对输出的贡献
        # 通过第一层权重的绝对值
        contributions = np.abs(self.W1)

        # 对隐藏层求和，乘以输出权重
        output_weights = np.abs(self.W2).flatten()
        total_contributions = contributions @ output_weights

        # 归一化
        weights = total_contributions / np.sum(total_contributions)

        return {name: weight for name, weight in zip(dimension_names, weights)}


# ============================================================
# 完整的韧性分析管道
# ============================================================

class LearnableResilienceAnalyzer:
    """
    可学习的韧性分析器

    整合因子分析和半监督学习
    """

    def __init__(self):
        self.fa_model = FactorAnalysisResilience(n_factors=2)
        self.ss_model = SemiSupervisedResilienceLearner()
        self.fitted = False

        # 阈值（用于分级，不是诊断）
        self.thresholds = {
            'low': 0.25,
            'moderate': 0.5,
            'elevated': 0.75
        }

    def fit(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None,
        hard_events: Optional[np.ndarray] = None
    ) -> 'LearnableResilienceAnalyzer':
        """
        拟合模型

        Args:
            features: 特征矩阵 (n_samples, 5)
            labels: PHQ-9/GAD-7 得分（可选）
            hard_events: 客观事件标签（可选）
        """
        # 1. 因子分析（无监督）
        self.fa_model.fit(features)

        # 2. 半监督学习（如果有标签）
        if labels is not None or hard_events is not None:
            self.ss_model.fit(features, labels, hard_events)

        self.fitted = True

        return self

    def analyze(self, features: np.ndarray) -> ResilienceState:
        """
        分析韧性状态

        Args:
            features: 5维特征向量

        Returns:
            ResilienceState
        """
        if not self.fitted:
            # 如果未拟合，使用默认方法
            return self._default_analyze(features)

        features = np.array(features).reshape(1, -1)

        dimension_names = [
            'recovery_speed', 'recovery_depth', 'adaptability',
            'redundancy', 'diversity'
        ]

        state = ResilienceState()

        # 原始维度
        state.dimensions = {
            name: float(features[0, i])
            for i, name in enumerate(dimension_names)
        }

        # 因子得分
        factor_scores = self.fa_model.transform(features)
        state.factor_scores = {
            f'factor_{i+1}': float(factor_scores[0, i])
            for i in range(factor_scores.shape[1])
        }

        # 主因子得分
        state.primary_factor_score = float(factor_scores[0, 0])

        # 风险预测（半监督）
        if self.ss_model.fitted:
            risk_score = float(self.ss_model.predict(features)[0])
        else:
            # 使用因子得分的负值作为风险指标
            risk_score = float(1 - (state.primary_factor_score + 1) / 2)

        # 分级（非诊断）
        state.risk_level = self._determine_risk_level(risk_score)

        # 置信区间（不确定性量化）
        state.confidence_interval = (max(0, risk_score - 0.1), min(1, risk_score + 0.1))

        return state

    def _default_analyze(self, features: np.ndarray) -> ResilienceState:
        """默认分析（无训练数据时）"""
        # 使用简单平均，但明确说明权重未学习
        dimension_names = [
            'recovery_speed', 'recovery_depth', 'adaptability',
            'redundancy', 'diversity'
        ]

        state = ResilienceState()

        state.dimensions = {
            name: float(features[i])
            for i, name in enumerate(dimension_names)
        }

        # 简单平均（注意：这是临时方法）
        state.primary_factor_score = float(np.mean(features))

        state.risk_level = self._determine_risk_level(1 - state.primary_factor_score / 100)

        return state

    def _determine_risk_level(self, score: float) -> RiskLevel:
        """
        确定风险等级

        输出分级建议，不输出诊断标签
        """
        if score < self.thresholds['low']:
            return RiskLevel.LOW
        elif score < self.thresholds['moderate']:
            return RiskLevel.MODERATE
        elif score < self.thresholds['elevated']:
            return RiskLevel.ELEVATED
        else:
            return RiskLevel.HIGH

    def get_weights_report(self) -> str:
        """
        生成权重报告

        说明权重来源（学习得到，不是手填）
        """
        lines = []

        lines.append("=" * 60)
        lines.append("韧性因子权重分析报告")
        lines.append("=" * 60)

        # 因子解释
        if self.fa_model.fitted:
            lines.append("\n【因子分析结果】")
            interpretation = self.fa_model.get_factor_interpretation()

            for factor_name, info in interpretation.items():
                lines.append(f"\n{factor_name}:")
                lines.append(f"  解释方差: {info['explained_variance']:.1%}")
                lines.append(f"  含义: {info['interpretation']}")
                lines.append("  载荷（自然权重）:")
                for dim, loading in info['loadings'].items():
                    lines.append(f"    {dim}: {loading:.3f}")

        # 学习权重
        if self.ss_model.fitted:
            lines.append("\n【学习到的权重】")
            lines.append("（从 PHQ-9/GAD-7 或客观事件学习）")
            weights = self.ss_model.get_learned_weights()
            for dim, weight in weights.items():
                lines.append(f"  {dim}: {weight:.3f}")

        lines.append("\n【说明】")
        lines.append("  以上权重均由数据自然产生，非人工设定")
        lines.append("  因子载荷来自 PCA/FA 分析")
        lines.append("  学习权重来自半监督训练")

        return "\n".join(lines)


# ============================================================
# 报告生成（分级建议，非诊断）
# ============================================================

def create_graded_report(state: ResilienceState) -> str:
    """
    生成分级建议报告

    重要：不输出诊断标签，只输出分级建议
    """
    lines = []

    lines.append("=" * 60)
    lines.append("社交韧性状态分析报告")
    lines.append("=" * 60)

    lines.append("\n【重要声明】")
    lines.append("  本报告仅提供分级建议，不构成医疗诊断")
    lines.append("  如有心理困扰，建议寻求专业支持")

    # 维度分析
    lines.append("\n【维度分析】")
    for dim, value in state.dimensions.items():
        lines.append(f"  {dim}: {value:.1f}")

    # 因子得分
    if state.factor_scores:
        lines.append("\n【因子得分】")
        for factor, score in state.factor_scores.items():
            lines.append(f"  {factor}: {score:.3f}")

    # 风险等级（分级建议）
    lines.append("\n【风险分级】")
    lines.append(f"  等级: {state.risk_level.value}")
    lines.append(f"  建议: {state.risk_level.recommendation}")

    # 置信区间
    if state.confidence_interval != (0, 0):
        lines.append(f"\n  置信区间: [{state.confidence_interval[0]:.2f}, {state.confidence_interval[1]:.2f}]")
        lines.append("  （表示评估的不确定性范围）")

    # 分级建议（详细）
    lines.append("\n【分级建议详情】")
    if state.risk_level == RiskLevel.LOW:
        lines.append("  ✓ 当前社交状态良好")
        lines.append("  ✓ 继续保持现有社交模式")
        lines.append("  ✓ 可考虑拓展社交圈")
    elif state.risk_level == RiskLevel.MODERATE:
        lines.append("  ○ 社交状态需要关注")
        lines.append("  ○ 建议适当增加社交互动")
        lines.append("  ○ 可尝试新的社交活动")
    elif state.risk_level == RiskLevel.ELEVATED:
        lines.append("  △ 社交状态需要改善")
        lines.append("  △ 建议主动寻求社交支持")
        lines.append("  △ 与信任的人交流当前状况")
        lines.append("  △ 考虑参加支持性活动")
    else:  # HIGH
        lines.append("  ⚠️ 需要额外支持")
        lines.append("  ⚠️ 建议寻求专业心理支持资源")
        lines.append("  ⚠️ 学校心理咨询中心可提供帮助")
        lines.append("  ⚠️ 与导师/辅导员沟通可能有助于获得支持")

    return "\n".join(lines)


__all__ = [
    'RiskLevel', 'ResilienceState', 'LabeledSample',
    'FactorAnalysisResilience', 'SemiSupervisedResilienceLearner',
    'LearnableResilienceAnalyzer',
    'create_graded_report'
]