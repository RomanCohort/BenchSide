"""
社交韧性与噪声过滤的动态图算法改进

新增指标：
1. 社交韧性 (Social Resilience)
   - 定义：社交网络在扰动后恢复的能力
   - 基于复杂网络理论和心理韧性研究

2. 噪声过滤
   - 区分"暂时性变化"和"结构性变化"
   - 识别"心情不佳"、"忙碌"等噪声因素

理论基础：
- Network Resilience (复杂网络韧性理论)
- Psychological Resilience (心理韧性量表)
- Signal Processing (信号处理中的噪声滤波)
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings


# ============================================================
# 噪声类型定义
# ============================================================

class NoiseType(Enum):
    """噪声类型"""
    MOOD_SWING = "心情波动"      # 情绪暂时性变化
    BUSY_PERIOD = "忙碌期"       # 工作/学业繁忙
    LIFE_EVENT = "生活事件"      # 短期事件影响
    HEALTH_ISSUE = "健康问题"    # 身体不适
    SEASONAL = "季节性"          # 周期性波动
    RANDOM = "随机噪声"          # 无规律波动


@dataclass
class NoiseSignal:
    """噪声信号"""
    noise_type: NoiseType
    start_time: datetime
    duration: timedelta
    intensity: float           # 噪声强度 0-100
    affected_contacts: List[str] = field(default_factory=list)

    # 噪声特征
    features: Dict = field(default_factory=dict)


# ============================================================
# 噪声检测器
# ============================================================

class NoiseDetector:
    """
    噪声检测器

    区分暂时性变化和结构性变化
    使用多维度信号处理方法
    """

    # 噪声模式特征
    NOISE_PATTERNS = {
        NoiseType.MOOD_SWING: {
            'duration_range': (0.5, 3),  # 小时
            'recovery_rate': 0.8,        # 快速恢复
            'affects_multiple': False,   # 通常只影响一个关系
            'pattern': 'spike'           # 突然变化
        },
        NoiseType.BUSY_PERIOD: {
            'duration_range': (1, 7),    # 天
            'recovery_rate': 0.6,
            'affects_multiple': True,
            'pattern': 'plateau'
        },
        NoiseType.LIFE_EVENT: {
            'duration_range': (1, 14),   # 天
            'recovery_rate': 0.4,
            'affects_multiple': True,
            'pattern': 'step'
        },
        NoiseType.HEALTH_ISSUE: {
            'duration_range': (1, 7),
            'recovery_rate': 0.5,
            'affects_multiple': True,
            'pattern': 'decline'
        },
        NoiseType.SEASONAL: {
            'duration_range': (7, 30),
            'recovery_rate': 0.7,
            'affects_multiple': False,
            'pattern': 'periodic'
        },
        NoiseType.RANDOM: {
            'duration_range': (0.1, 2),
            'recovery_rate': 0.9,
            'affects_multiple': False,
            'pattern': 'noise'
        }
    }

    def detect_noise(
        self,
        signal_history: List[Tuple[datetime, float]],
        baseline: float,
        threshold: float = 20
    ) -> List[NoiseSignal]:
        """
        检测噪声信号

        Args:
            signal_history: 时间序列信号
            baseline: 基线值
            threshold: 噪声检测阈值

        Returns:
            检测到的噪声信号列表
        """
        noises = []

        if len(signal_history) < 5:
            return noises

        # 1. 检测偏离基线的信号段
        deviations = []
        for timestamp, value in signal_history:
            deviation = abs(value - baseline)
            if deviation > threshold:
                deviations.append((timestamp, value, deviation))

        if not deviations:
            return noises

        # 2. 聚合连续偏离为噪声段
        noise_segments = self._aggregate_segments(deviations)

        # 3. 分类噪声类型
        for segment in noise_segments:
            noise_type = self._classify_noise(segment, baseline)
            noise_signal = NoiseSignal(
                noise_type=noise_type,
                start_time=segment[0][0],
                duration=segment[-1][0] - segment[0][0],
                intensity=np.mean([d[2] for d in segment]),
                features=self._extract_features(segment)
            )
            noises.append(noise_signal)

        return noises

    def _aggregate_segments(
        self,
        deviations: List[Tuple[datetime, float, float]]
    ) -> List[List]:
        """聚合连续偏离为段"""
        segments = []
        current_segment = [deviations[0]]

        for i in range(1, len(deviations)):
            # 检查时间间隔
            time_gap = deviations[i][0] - deviations[i-1][0]

            if time_gap < timedelta(hours=24):
                current_segment.append(deviations[i])
            else:
                segments.append(current_segment)
                current_segment = [deviations[i]]

        if current_segment:
            segments.append(current_segment)

        return segments

    def _classify_noise(
        self,
        segment: List,
        baseline: float
    ) -> NoiseType:
        """分类噪声类型"""
        duration = segment[-1][0] - segment[0][0]
        duration_hours = duration.total_seconds() / 3600

        # 计算恢复率
        start_dev = abs(segment[0][1] - baseline)
        end_dev = abs(segment[-1][1] - baseline)
        recovery_rate = 1 - (end_dev / start_dev) if start_dev > 0 else 0

        # 计算变化模式
        values = [s[1] for s in segment]
        if len(values) > 1:
            # 突然变化
            max_change = max(abs(values[i+1] - values[i]) for i in range(len(values)-1))
            avg_change = np.mean([abs(values[i+1] - values[i]) for i in range(len(values)-1)])

            if max_change > avg_change * 2:
                pattern = 'spike'
            elif np.std(values) < avg_change * 0.5:
                pattern = 'plateau'
            else:
                pattern = 'step'
        else:
            pattern = 'unknown'

        # 匹配噪声类型
        best_match = NoiseType.RANDOM
        best_score = 0

        for noise_type, features in self.NOISE_PATTERNS.items():
            score = 0

            # 持续时间匹配
            dur_range = features['duration_range']
            if dur_range[0] <= duration_hours <= dur_range[1]:
                score += 2

            # 恢复率匹配
            if abs(features['recovery_rate'] - recovery_rate) < 0.2:
                score += 2

            # 模式匹配
            if features['pattern'] == pattern:
                score += 1

            if score > best_score:
                best_score = score
                best_match = noise_type

        return best_match

    def _extract_features(self, segment: List) -> Dict:
        """提取噪声特征"""
        values = [s[1] for s in segment]

        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'trend': np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0
        }

    def filter_signal(
        self,
        signal_history: List[Tuple[datetime, float]],
        noises: List[NoiseSignal]
    ) -> List[Tuple[datetime, float]]:
        """
        过滤噪声后的信号

        使用自适应滤波器
        """
        filtered = []

        for timestamp, value in signal_history:
            # 检查是否在噪声期间
            in_noise = False
            correction = 0

            for noise in noises:
                if noise.start_time <= timestamp <= noise.start_time + noise.duration:
                    in_noise = True

                    # 根据噪声类型调整
                    if noise.noise_type == NoiseType.MOOD_SWING:
                        # 心情波动：恢复后恢复正常
                        correction = -noise.intensity * 0.3
                    elif noise.noise_type == NoiseType.BUSY_PERIOD:
                        # 忙碌：部分恢复
                        correction = -noise.intensity * 0.2
                    elif noise.noise_type == NoiseType.RANDOM:
                        # 随机：忽略
                        correction = -noise.intensity * 0.5
                    else:
                        # 其他：保守处理
                        correction = -noise.intensity * 0.1

                    break

            corrected_value = value + correction if in_noise else value
            filtered.append((timestamp, corrected_value))

        return filtered


# ============================================================
# 社交韧性指标
# ============================================================

@dataclass
class SocialResilienceMetrics:
    """
    社交韧性指标

    定义：社交网络在扰动后恢复稳定的能力

    基于复杂网络韧性理论 + 心理韧性量表
    """
    # 韧性分数 0-100
    resilience_score: float = 50

    # 子维度
    recovery_speed: float = 0        # 恢复速度（扰动后）
    recovery_depth: float = 0        # 恢复深度（是否完全恢复）
    adaptability: float = 0          # 适应性（调整策略）
    redundancy: float = 0            # 冗余度（备用支持）
    diversity: float = 0             # 多样性（不依赖单一关系）

    # 风险阈值
    critical_threshold: float = 30   # 低于此值为高风险
    healthy_threshold: float = 70    # 高于此值为健康

    # 韧性特征
    resilience_type: str = ""        # robust, resilient, brittle


class SocialResilienceCalculator:
    """
    社交韧性计算器

    改进的动态图算法，加入韧性度量
    """

    def __init__(self):
        self.noise_detector = NoiseDetector()

    def calculate_resilience(
        self,
        network_history: List[Dict],
        perturbations: List[Dict] = None
    ) -> SocialResilienceMetrics:
        """
        计算社交韧性

        Args:
            network_history: 网络历史快照
            perturbations: 已识别的扰动事件

        Returns:
            SocialResilienceMetrics
        """
        metrics = SocialResilienceMetrics()

        if len(network_history) < 3:
            metrics.resilience_type = "insufficient_data"
            return metrics

        # 1. 计算恢复速度
        metrics.recovery_speed = self._compute_recovery_speed(network_history)

        # 2. 计算恢复深度
        metrics.recovery_depth = self._compute_recovery_depth(network_history)

        # 3. 计算适应性
        metrics.adaptability = self._compute_adaptability(network_history)

        # 4. 计算冗余度
        metrics.redundancy = self._compute_redundancy(network_history)

        # 5. 计算多样性
        metrics.diversity = self._compute_diversity(network_history)

        # 6. 综合韧性分数
        metrics.resilience_score = self._compute_overall_resilience(metrics)

        # 7. 确定韧性类型
        metrics.resilience_type = self._determine_resilience_type(metrics)

        return metrics

    def _compute_recovery_speed(self, history: List[Dict]) -> float:
        """
        恢复速度

        扰动后多久恢复到基线的80%
        基于复杂网络的恢复动力学
        """
        if len(history) < 3:
            return 50

        # 寻找扰动（能量下降）
        baseline = history[0].get('total_energy', 0)
        perturbation_idx = None
        perturbation_value = baseline

        for i, snapshot in enumerate(history):
            energy = snapshot.get('total_energy', 0)
            if energy < baseline * 0.7:  # 30%下降
                perturbation_idx = i
                perturbation_value = energy
                break

        if perturbation_idx is None:
            return 100  # 无扰动，高韧性

        # 寻找恢复点
        recovery_threshold = baseline * 0.8
        recovery_idx = None

        for i in range(perturbation_idx + 1, len(history)):
            if history[i].get('total_energy', 0) >= recovery_threshold:
                recovery_idx = i
                break

        if recovery_idx is None:
            return 10  # 未恢复，低韧性

        # 计算恢复时间（归一化）
        recovery_time = recovery_idx - perturbation_idx
        total_time = len(history)

        # 更快恢复 = 更高韧性
        speed_score = 100 * (1 - recovery_time / total_time)

        return max(speed_score, 10)

    def _compute_recovery_depth(self, history: List[Dict]) -> float:
        """
        恢复深度

        是否恢复到原始水平，还是只部分恢复
        """
        if len(history) < 3:
            return 50

        baseline = history[0].get('total_energy', 0)
        final = history[-1].get('total_energy', 0)

        if baseline == 0:
            return 50

        # 恢复深度比例
        depth_ratio = final / baseline

        # 完全恢复 = 100, 过度恢复 = >100, 恢复不足 = <100
        depth_score = min(depth_ratio * 100, 120)

        return depth_score

    def _compute_adaptability(self, history: List[Dict]) -> float:
        """
        适应性

        面对扰动时是否能调整策略
        基于结构变化分析
        """
        if len(history) < 4:
            return 50

        # 计算结构变化
        # 添加新关系 = 适应性
        # 减少消耗关系 = 适应性

        initial_nodes = history[0].get('num_nodes', 0)
        final_nodes = history[-1].get('num_nodes', 0)

        initial_energy = history[0].get('total_energy', 0)
        final_energy = history[-1].get('total_energy', 0)

        # 如果能量下降但节点增加（寻求新支持）= 高适应性
        if final_energy < initial_energy and final_nodes > initial_nodes:
            return 70  # 主动寻求新连接

        # 如果能量下降且节点减少 = 低适应性
        if final_energy < initial_energy and final_nodes < initial_nodes:
            return 30  # 被动收缩

        # 稳定 = 中等
        return 50

    def _compute_redundancy(self, history: List[Dict]) -> float:
        """
        冗余度

        备用支持系统的存在
        基于网络冗余理论
        """
        if len(history) < 2:
            return 50

        # 计算正向关系数量
        last_snapshot = history[-1]
        node_energies = last_snapshot.get('node_energies', {})

        # 正向关系 = 冗余支持
        positive_count = sum(1 for e in node_energies.values() if e > 50)

        # 3+ 正向关系 = 高冗余
        if positive_count >= 3:
            return 80
        elif positive_count >= 2:
            return 60
        elif positive_count >= 1:
            return 40
        else:
            return 20

    def _compute_diversity(self, history: List[Dict]) -> float:
        """
        多样性

        不依赖单一关系类型
        基于生态多样性理论
        """
        if len(history) < 2:
            return 50

        last_snapshot = history[-1]

        # 统计关系类型
        edge_types = last_snapshot.get('edge_types', {})
        if not edge_types:
            return 50

        # Shannon多样性指数
        total = sum(edge_types.values())
        if total == 0:
            return 50

        diversity = 0
        for count in edge_types.values():
            p = count / total
            if p > 0:
                diversity -= p * np.log2(p)

        # 归一化到0-100
        max_diversity = np.log2(len(edge_types)) if len(edge_types) > 1 else 1
        normalized = diversity / max_diversity * 100

        return normalized

    def _compute_overall_resilience(self, metrics: SocialResilienceMetrics) -> float:
        """
        综合韧性分数

        加权平均各维度
        """
        # 权重分配（基于文献）
        weights = {
            'recovery_speed': 0.25,
            'recovery_depth': 0.20,
            'adaptability': 0.15,
            'redundancy': 0.20,
            'diversity': 0.20
        }

        overall = (
            metrics.recovery_speed * weights['recovery_speed'] +
            metrics.recovery_depth * weights['recovery_depth'] +
            metrics.adaptability * weights['adaptability'] +
            metrics.redundancy * weights['redundancy'] +
            metrics.diversity * weights['diversity']
        )

        return overall

    def _determine_resilience_type(self, metrics: SocialResilienceMetrics) -> str:
        """
        确定韧性类型

        Robust: 高韧性，扰动后完全恢复
        Resilient: 中等韧性，扰动后部分恢复
        Brittle: 低韧性，扰动后崩溃
        """
        if metrics.resilience_score >= metrics.healthy_threshold:
            return "robust"
        elif metrics.resilience_score >= metrics.critical_threshold:
            return "resilient"
        else:
            return "brittle"

    def predict_resilience_trajectory(
        self,
        metrics: SocialResilienceMetrics,
        future_perturbation: Dict = None
    ) -> Dict:
        """
        预测韧性轨迹

        预测在潜在扰动下的表现
        """
        prediction = {
            'expected_recovery_time': 0,
            'survival_probability': 0,
            'recommendations': []
        }

        # 基于韧性类型预测
        if metrics.resilience_type == "robust":
            prediction['survival_probability'] = 0.90
            prediction['expected_recovery_time'] = 7  # 天
            prediction['recommendations'].append('韧性良好，保持当前社交策略')

        elif metrics.resilience_type == "resilient":
            prediction['survival_probability'] = 0.65
            prediction['expected_recovery_time'] = 14
            prediction['recommendations'].append('建议增强冗余度和多样性')

        else:  # brittle
            prediction['survival_probability'] = 0.30
            prediction['expected_recovery_time'] = 30
            prediction['recommendations'].append('⚠️ 韧性不足，需要建立备用支持系统')
            prediction['recommendations'].append('建议：拓展社交圈，减少对单一关系的依赖')

        # 考虑具体弱点
        if metrics.redundancy < 40:
            prediction['recommendations'].append('冗余度低：建议培养2+正向关系')

        if metrics.diversity < 40:
            prediction['recommendations'].append('多样性低：建议建立不同类型的关系')

        if metrics.adaptability < 40:
            prediction['recommendations'].append('适应性低：建议学习调整社交策略')

        return prediction


# ============================================================
# 噪声感知的动态图分析
# ============================================================

class NoiseAwareDynamicAnalyzer:
    """
    噪声感知的动态图分析器

    改进的动态图算法：
    1. 过滤噪声后的真实信号分析
    2. 区分暂时性变化和结构性变化
    3. 加入韧性指标
    """

    def __init__(self):
        self.noise_detector = NoiseDetector()
        self.resilience_calc = SocialResilienceCalculator()
        self.history = []
        self.filtered_history = []

    def add_snapshot(self, snapshot: Dict, timestamp: datetime = None):
        """
        添加快照

        自动检测和过滤噪声
        """
        if timestamp is None:
            timestamp = datetime.now()

        self.history.append((timestamp, snapshot))

        # 检测噪声
        if len(self.history) >= 5:
            # 提取能量信号
            signal = [(t, s.get('total_energy', 0)) for t, s in self.history]
            baseline = self.history[0][1].get('total_energy', 0)

            noises = self.noise_detector.detect_noise(signal, baseline)

            # 过滤噪声
            filtered_signal = self.noise_detector.filter_signal(signal, noises)

            # 更新过滤后的历史
            self.filtered_history = [
                (t, {**s, 'filtered_energy': e})
                for (t, s), (t2, e) in zip(self.history, filtered_signal)
            ]

    def analyze_with_noise_filtering(self) -> Dict:
        """
        噪声过滤后的分析

        区分：
        - 真实结构性变化
        - 暂时性噪声影响
        """
        analysis = {
            'raw_trend': None,
            'filtered_trend': None,
            'noise_events': [],
            'structural_changes': [],
            'resilience': None
        }

        if len(self.history) < 3:
            return analysis

        # 1. 原始趋势
        raw_energies = [s.get('total_energy', 0) for _, s in self.history]
        analysis['raw_trend'] = {
            'values': raw_energies,
            'slope': np.polyfit(range(len(raw_energies)), raw_energies, 1)[0]
        }

        # 2. 过滤后趋势
        if self.filtered_history:
            filtered_energies = [s.get('filtered_energy', s.get('total_energy', 0))
                                for _, s in self.filtered_history]
            analysis['filtered_trend'] = {
                'values': filtered_energies,
                'slope': np.polyfit(range(len(filtered_energies)), filtered_energies, 1)[0]
            }

        # 3. 噪声事件
        signal = [(t, s.get('total_energy', 0)) for t, s in self.history]
        baseline = self.history[0][1].get('total_energy', 0)
        noises = self.noise_detector.detect_noise(signal, baseline)

        analysis['noise_events'] = [
            {
                'type': n.noise_type.value,
                'start': n.start_time.strftime('%Y-%m-%d'),
                'duration_hours': n.duration.total_seconds() / 3600,
                'intensity': n.intensity
            }
            for n in noises
        ]

        # 4. 结构性变化（非噪声）
        # 比较原始趋势和过滤后趋势的差异
        if analysis['raw_trend'] and analysis['filtered_trend']:
            raw_slope = analysis['raw_trend']['slope']
            filtered_slope = analysis['filtered_trend']['slope']

            # 如果过滤后趋势明显不同 = 有结构性变化
            if abs(raw_slope - filtered_slope) > 1:
                analysis['structural_changes'].append({
                    'type': 'divergence',
                    'raw_slope': raw_slope,
                    'filtered_slope': filtered_slope,
                    'interpretation': '存在结构性恶化（不是暂时性噪声）'
                })
            else:
                analysis['structural_changes'].append({
                    'type': 'noise_dominant',
                    'interpretation': '变化主要由暂时性因素造成'
                })

        # 5. 韧性分析
        network_history = [s for _, s in self.filtered_history if s]
        if len(network_history) >= 3:
            analysis['resilience'] = self.resilience_calc.calculate_resilience(
                network_history
            )

        return analysis

    def predict_with_resilience(self, days_ahead: int = 30) -> Dict:
        """
        带韧性因素的预测
        """
        analysis = self.analyze_with_noise_filtering()

        prediction = {
            'raw_prediction': None,
            'resilience_adjusted': None,
            'confidence': 0,
            'warnings': []
        }

        if not analysis.get('resilience'):
            return prediction

        resilience = analysis['resilience']

        # 基础预测
        if analysis.get('filtered_trend'):
            slope = analysis['filtered_trend']['slope']
            current = self.filtered_history[-1][1].get('filtered_energy', 0)
            raw_prediction = current + slope * days_ahead / 7
            prediction['raw_prediction'] = raw_prediction

        # 韧性调整
        # 高韧性 = 可能恢复，低韧性 = 可能恶化
        resilience_factor = resilience.resilience_score / 100

        if resilience.resilience_type == "robust":
            # 高韧性：乐观预测
            prediction['resilience_adjusted'] = (
                prediction['raw_prediction'] * 0.8 +  # 保留部分恶化
                resilience.recovery_depth  # 但加入恢复因素
            )
            prediction['confidence'] = 0.85
        elif resilience.resilience_type == "resilient":
            prediction['resilience_adjusted'] = prediction['raw_prediction']
            prediction['confidence'] = 0.65
        else:  # brittle
            # 低韧性：悲观预测
            prediction['resilience_adjusted'] = prediction['raw_prediction'] * 1.2
            prediction['confidence'] = 0.30
            prediction['warnings'].append('韧性不足，恶化可能加速')

        return prediction


# ============================================================
# 报告生成
# ============================================================

def create_resilience_report(
    resilience: SocialResilienceMetrics,
    noise_analysis: Dict,
    prediction: Dict = None
) -> str:
    """生成韧性分析报告"""
    lines = []

    lines.append("=" * 60)
    lines.append("社交韧性与噪声过滤分析报告")
    lines.append("=" * 60)

    # 韧性总览
    type_display = {
        'robust': '💪 强韧',
        'resilient': '🔄 有韧性',
        'brittle': '⚠️ 脆弱',
        'insufficient_data': '📊 数据不足'
    }
    lines.append(f"\n【韧性总览】{type_display.get(resilience.resilience_type, '未知')}")
    lines.append(f"  韧性分数: {resilience.resilience_score:.1f}/100")
    lines.append(f"  健康阈值: {resilience.healthy_threshold}")
    lines.append(f"  风险阈值: {resilience.critical_threshold}")

    # 韧性维度
    lines.append("\n【韧性维度分析】")
    lines.append(f"  恢复速度: {resilience.recovery_speed:.1f}")
    lines.append(f"  恢复深度: {resilience.recovery_depth:.1f}")
    lines.append(f"  适应性: {resilience.adaptability:.1f}")
    lines.append(f"  冗余度: {resilience.redundancy:.1f}")
    lines.append(f"  多样性: {resilience.diversity:.1f}")

    # 噪声分析
    if noise_analysis.get('noise_events'):
        lines.append("\n【噪声事件检测】")
        lines.append("  区分暂时性变化和结构性变化:")
        for event in noise_analysis['noise_events'][:5]:
            lines.append(f"    • {event['type']}: "
                        f"{event['duration_hours']:.1f}小时, "
                        f"强度={event['intensity']:.1f}")
        lines.append("\n  注: 这些变化可能由心情/忙碌等暂时性因素造成")

    # 结构性变化
    if noise_analysis.get('structural_changes'):
        lines.append("\n【结构性变化分析】")
        for change in noise_analysis['structural_changes']:
            lines.append(f"  {change['interpretation']}")

    # 预测
    if prediction:
        lines.append("\n【韧性预测】")
        if prediction.get('raw_prediction'):
            lines.append(f"  原始预测: {prediction['raw_prediction']:.1f}")
        if prediction.get('resilience_adjusted'):
            lines.append(f"  韧性调整后: {prediction['resilience_adjusted']:.1f}")
        if prediction.get('confidence'):
            lines.append(f"  置信度: {prediction['confidence']:.0%}")

        if prediction.get('warnings'):
            for warning in prediction['warnings']:
                lines.append(f"  ⚠️ {warning}")

    # 建议
    if resilience.resilience_type == "brittle":
        lines.append("\n【紧急建议】")
        lines.append("  韧性不足需要立即行动：")
        lines.append("  1. 建立至少2个正向支持关系")
        lines.append("  2. 减少对单一关系的依赖")
        lines.append("  3. 增加社交圈多样性")

    return "\n".join(lines)


__all__ = [
    'NoiseType', 'NoiseSignal', 'NoiseDetector',
    'SocialResilienceMetrics', 'SocialResilienceCalculator',
    'NoiseAwareDynamicAnalyzer',
    'create_resilience_report'
]