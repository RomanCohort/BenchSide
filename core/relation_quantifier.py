"""
关系量化系统

不只是分类，而是量化关系的各个维度

量化指标：
1. 情感投入指数 (投入度)
2. 对方回应指数 (回报度)
3. 能量净值 (投入-回报)
4. 关系ROI (长期收益)
5. 健康度评分
6. 风险指数
7. 满意度预测
8. 最佳行动时机

应用场景：
- "我该继续投入吗？" → 数据支持决策
- "这段关系值得吗？" → ROI分析
- "什么时候该止损？" → 预警系统
- "我需要改变什么？" → 具体建议
"""
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json


# ============================================================
# 量化指标定义
# ============================================================

@dataclass
class RelationMetrics:
    """关系量化指标"""
    # 核心指标 (0-100)
    investment_score: float = 50      # 你的投入度
    response_score: float = 50         # 对方回应度
    energy_net: float = 0              # 能量净值 (response - investment)
    roi: float = 0                     # 关系投资回报率

    # 健康指标
    health_score: float = 50           # 整体健康度
    balance_score: float = 50          # 平衡度
    stability_score: float = 50        # 稳定性

    # 风险指标
    risk_level: float = 0              # 风险等级 0-100
    burnout_risk: float = 0            # 情感耗竭风险
    rejection_risk: float = 0          # 被拒绝风险

    # 预测指标
    satisfaction_predicted: float = 50 # 预测满意度
    longevity_predicted: float = 50    # 预测关系寿命(月)
    improvement_potential: float = 50  # 改善潜力

    # 行动建议
    action_recommendation: str = ""
    urgency_level: str = "normal"      # low, normal, high, critical
    optimal_timing: str = ""           # 最佳行动时机


@dataclass
class EnergyTransaction:
    """能量交易记录"""
    timestamp: datetime
    action: str           # "invest" 或 "receive"
    amount: float         # 能量值
    category: str         # 情感/时间/金钱/精力
    description: str = ""


@dataclass
class RelationLedger:
    """关系账本 - 记录所有能量交易"""
    transactions: List[EnergyTransaction] = field(default_factory=list)

    def add_investment(self, amount: float, category: str, desc: str = ""):
        """记录投入"""
        self.transactions.append(EnergyTransaction(
            timestamp=datetime.now(),
            action="invest",
            amount=amount,
            category=category,
            description=desc
        ))

    def add_return(self, amount: float, category: str, desc: str = ""):
        """记录回报"""
        self.transactions.append(EnergyTransaction(
            timestamp=datetime.now(),
            action="receive",
            amount=amount,
            category=category,
            description=desc
        ))

    def compute_balance(self) -> float:
        """计算能量净值"""
        total_invest = sum(t.amount for t in self.transactions if t.action == "invest")
        total_return = sum(t.amount for t in self.transactions if t.action == "receive")
        return total_return - total_invest

    def compute_roi(self) -> float:
        """计算ROI"""
        total_invest = sum(t.amount for t in self.transactions if t.action == "invest")
        total_return = sum(t.amount for t in self.transactions if t.action == "receive")
        if total_invest == 0:
            return 0
        return (total_return - total_invest) / total_invest


# ============================================================
# 量化引擎
# ============================================================

class RelationQuantifier:
    """关系量化引擎"""

    # 能量换算基准
    ENERGY_RATES = {
        # 投入类
        'message_sent': 2,          # 每条消息投入2能量
        'long_message': 5,          # 长消息5能量
        'question': 3,              # 提问3能量(显示关注)
        'emoji': 1,                 # 表情1能量
        'initiative': 10,           # 主动发起话题10能量
        'waiting': 15,              # 等待回复的心理消耗15能量/小时
        'planning': 20,             # 规划约会/见面20能量
        'gift': 30,                 # 送礼物30能量
        'emotional_support': 25,    # 情感支持25能量
        'compromise': 15,           # 妥协/迁就15能量

        # 回报类
        'message_received': 3,      # 收到消息回报3能量
        'prompt_reply': 8,          # 快速回复额外回报8能量
        'affirmation': 15,          # 肯定/表扬回报15能量
        'affection': 20,            # 表达感情回报20能量
        'initiative_from_other': 15,# 对方主动15能量
        'quality_time': 25,         # 高质量相处时间25能量/小时
        'gift_received': 30,        # 收到礼物30能量
        'support_received': 25,     # 获得支持25能量
    }

    def quantify(self, messages: List[Dict], stats: Dict) -> RelationMetrics:
        """
        从对话数据量化关系

        Args:
            messages: 消息列表
            stats: 统计数据

        Returns:
            量化指标
        """
        metrics = RelationMetrics()

        # 1. 计算能量账本
        ledger = self._build_ledger(messages, stats)

        # 2. 核心指标
        metrics.investment_score = self._compute_investment(ledger, stats)
        metrics.response_score = self._compute_response(ledger, stats)
        metrics.energy_net = ledger.compute_balance()
        metrics.roi = ledger.compute_roi()

        # 3. 健康指标
        metrics.health_score = self._compute_health(metrics, stats)
        metrics.balance_score = self._compute_balance(metrics)
        metrics.stability_score = self._compute_stability(messages, stats)

        # 4. 风险指标
        metrics.risk_level = self._compute_risk(metrics)
        metrics.burnout_risk = self._compute_burnout_risk(metrics, stats)
        metrics.rejection_risk = self._compute_rejection_risk(metrics, stats)

        # 5. 预测指标
        metrics.satisfaction_predicted = self._predict_satisfaction(metrics)
        metrics.longevity_predicted = self._predict_longevity(metrics, stats)
        metrics.improvement_potential = self._compute_improvement_potential(metrics)

        # 6. 行动建议
        metrics.action_recommendation = self._generate_recommendation(metrics)
        metrics.urgency_level = self._determine_urgency(metrics)
        metrics.optimal_timing = self._suggest_timing(metrics, stats)

        return metrics

    def _build_ledger(self, messages: List[Dict], stats: Dict) -> RelationLedger:
        """构建能量账本"""
        ledger = RelationLedger()

        # ========== 关键：使用统计数据而非消息遍历 ==========
        # 这样更准确

        # 1. 从统计数据提取
        basic = stats.get('basic', {})
        initiative = stats.get('initiative', {})
        reply_speed = stats.get('reply_speed', {})
        bombing = stats.get('bombing', {})

        total_msgs = basic.get('total_messages', 0)
        my_msgs = basic.get('my_messages', 0)
        their_msgs = basic.get('their_messages', 0)
        my_start_ratio = initiative.get('my_start_ratio', 0.5)

        # 2. 投入计算（用户A/我）
        # 每条消息基础能量
        ledger.add_investment(my_msgs * self.ENERGY_RATES['message_sent'], 'message')

        # 主动发起的能量消耗（高主动=高投入）
        my_starts = initiative.get('my_starts', 0)
        for _ in range(my_starts):
            ledger.add_investment(self.ENERGY_RATES['initiative'], 'initiative')

        # 消息轰炸能量（连发=高焦虑）
        my_bombs = bombing.get('my_bomb_count', 0)
        ledger.add_investment(my_bombs * 5, 'anxiety')  # 每次轰炸额外5能量

        # 等待消耗
        my_avg_wait = reply_speed.get('my_avg_seconds', 3600)
        wait_hours = their_msgs / max(my_msgs, 1) * my_avg_wait / 3600
        ledger.add_investment(self.ENERGY_RATES['waiting'] * wait_hours * 0.1, 'waiting')

        # 3. 回报计算（对方B）
        # 每条消息回报
        ledger.add_return(their_msgs * self.ENERGY_RATES['message_received'], 'message')

        # 对方主动发起（如果有的话）
        their_starts = initiative.get('their_starts', 0)
        for _ in range(their_starts):
            ledger.add_return(self.ENERGY_RATES['initiative_from_other'], 'initiative')

        # 快速回复回报（对方回复快=好信号）
        their_reply_time = reply_speed.get('their_avg_seconds', 3600)
        if their_reply_time < 300:  # 5分钟内
            ledger.add_return(their_msgs * self.ENERGY_RATES['prompt_reply'] * 0.1, 'responsiveness')

        # 表情和情感表达
        msg_types = stats.get('message_types', {})
        their_emojis = msg_types.get('emoji', {}).get('them', 0)
        ledger.add_return(their_emojis * 2, 'affection')

        # 情感词汇
        linguistic = stats.get('linguistic', {})
        their_positive = linguistic.get('positive_emotion_count', {}).get('them', 0)
        ledger.add_return(their_positive * 3, 'emotional')

        # 4. 冷回复惩罚（降低回报）
        cold = stats.get('cold_response', {})
        their_cold = cold.get('their_cold_count', 0)
        ledger.add_investment(their_cold * 3, 'cold_response')  # 冷回复消耗我能量

        # 5. 未回复惩罚
        unanswered = stats.get('unanswered', {})
        my_unanswered = unanswered.get('my_unanswered', 0)
        ledger.add_investment(my_unanswered * 20, 'rejection')

        return ledger

    def _compute_investment(self, ledger: RelationLedger, stats: Dict) -> float:
        """计算投入分数"""
        total_invest = sum(t.amount for t in ledger.transactions if t.action == "invest")

        # 基准化到0-100
        # 假设"正常"投入是每天50能量，一个月1500能量为基准
        # 高投入(单相思)可能达到3000+

        baseline = 1500  # 月基准
        score = min(total_invest / baseline * 50, 100)

        return score

    def _compute_response(self, ledger: RelationLedger, stats: Dict) -> float:
        """计算回报分数"""
        total_return = sum(t.amount for t in ledger.transactions if t.action == "receive")

        baseline = 1500
        score = min(total_return / baseline * 50, 100)

        return score

    def _compute_health(self, metrics: RelationMetrics, stats: Dict) -> float:
        """计算健康度"""
        # 基于平衡和ROI
        balance_factor = metrics.balance_score / 100
        roi_factor = max(metrics.roi, 0) if metrics.roi > 0 else 0

        # 负ROI严重扣分
        if metrics.roi < -0.5:
            health = 20
        elif metrics.roi < -0.3:
            health = 40
        elif metrics.roi < 0:
            health = 60
        else:
            health = 80 + roi_factor * 20

        # 平衡度调整
        health = health * (0.5 + balance_factor * 0.5)

        return max(min(health, 100), 0)

    def _compute_balance(self, metrics: RelationMetrics) -> float:
        """计算平衡度"""
        # 投入和回报的差距
        diff = abs(metrics.investment_score - metrics.response_score)

        # 完全平衡=100, 完全不平衡=0
        balance = 100 - diff

        return max(balance, 0)

    def _compute_stability(self, messages: List[Dict], stats: Dict) -> float:
        """计算稳定性"""
        # 基于互动频率的稳定性
        # 稳定的关系有规律的互动

        # 简化：基于回复延迟的一致性
        delays = []
        prev_time = None
        for m in messages:
            if prev_time:
                delay = m.get('timestamp', 0) - prev_time
                delays.append(delay)
            prev_time = m.get('timestamp', 0)

        if not delays:
            return 50

        # 延迟的标准差越小，稳定性越高
        std = np.std(delays)
        mean = np.mean(delays)

        if mean == 0:
            return 50

        variability = std / mean
        stability = max(100 - variability * 100, 0)

        return stability

    def _compute_risk(self, metrics: RelationMetrics) -> float:
        """计算综合风险"""
        # 基于多个风险因素

        risk = 0

        # ROI风险
        if metrics.roi < -0.7:
            risk += 40
        elif metrics.roi < -0.5:
            risk += 30
        elif metrics.roi < -0.3:
            risk += 20

        # 平衡风险
        if metrics.balance_score < 30:
            risk += 30

        # 能量亏损风险
        if metrics.energy_net < -500:
            risk += 20

        return min(risk, 100)

    def _compute_burnout_risk(self, metrics: RelationMetrics, stats: Dict) -> float:
        """计算情感耗竭风险"""
        # 高投入+低回报=耗竭

        invest = metrics.investment_score
        response = metrics.response_score

        if invest > 80 and response < 30:
            return 90  # 极高风险
        elif invest > 70 and response < 40:
            return 70
        elif invest > 60 and response < 50:
            return 50
        else:
            return max(invest - response, 0)

    def _compute_rejection_risk(self, metrics: RelationMetrics, stats: Dict) -> float:
        """计算被拒绝风险"""
        # 基于对方的回避行为

        # 从统计提取
        reply_rate = stats.get('timing', {}).get('reply_rate_other', 0.5)
        avg_delay = stats.get('timing', {}).get('avg_reply_delay_other', 3600)

        risk = 0

        # 低回复率
        if reply_rate < 0.3:
            risk += 40
        elif reply_rate < 0.5:
            risk += 20

        # 长延迟
        if avg_delay > 7200:  # 2小时+
            risk += 30
        elif avg_delay > 3600:  # 1小时+
            risk += 15

        # 结合不平衡
        if metrics.balance_score < 40:
            risk += 20

        return min(risk, 100)

    def _predict_satisfaction(self, metrics: RelationMetrics) -> float:
        """预测满意度"""
        # 基于ROI和健康度

        if metrics.roi > 0.3:
            return 80 + min(metrics.roi * 20, 20)
        elif metrics.roi > 0:
            return 60 + metrics.roi * 40
        elif metrics.roi > -0.3:
            return 40 + metrics.roi * 40
        else:
            return max(20, 40 + metrics.roi * 20)

    def _predict_longevity(self, metrics: RelationMetrics, stats: Dict) -> float:
        """预测关系寿命"""
        # 基于稳定性、健康度、ROI

        base = 6  # 基准6个月

        # 健康度加权
        health_factor = metrics.health_score / 50

        # ROI加权
        roi_factor = 1 + metrics.roi

        longevity = base * health_factor * roi_factor

        return max(longevity, 1)

    def _compute_improvement_potential(self, metrics: RelationMetrics) -> float:
        """计算改善潜力"""
        # 低ROI但高投入=有改善空间(通过减少投入)
        # 或者高ROI但低投入=可以增加投入获得更多

        if metrics.roi < 0 and metrics.investment_score > 60:
            # 通过减少投入可以改善
            return 70

        if metrics.roi > 0 and metrics.investment_score < 50:
            # 增加投入可能获得更多
            return 80

        # 中间状态
        return 50

    def _generate_recommendation(self, metrics: RelationMetrics) -> str:
        """生成行动建议"""
        roi = metrics.roi
        balance = metrics.balance_score
        risk = metrics.risk_level

        # 根据ROI和风险生成建议
        if roi < -0.7 and risk > 70:
            return "强烈建议止损：情感亏损严重，风险极高"

        if roi < -0.5:
            return "建议减少投入：当前ROI为负，需要调整策略"

        if roi < -0.3:
            return "谨慎观察：关系处于亏损状态，建议设定止损线"

        if balance < 30:
            return "关系严重失衡：建议重新评估对方的态度"

        if roi < 0:
            return "关注平衡：适当减少主动，观察对方反应"

        if roi > 0.3:
            return "关系健康：可以继续投入，但保持警觉"

        if roi > 0:
            return "关系正向：保持当前模式，注意维护"

        return "继续观察：收集更多数据后评估"

    def _determine_urgency(self, metrics: RelationMetrics) -> str:
        """确定紧急程度"""
        if metrics.risk_level > 80:
            return "critical"
        elif metrics.risk_level > 60:
            return "high"
        elif metrics.risk_level > 40:
            return "normal"
        else:
            return "low"

    def _suggest_timing(self, metrics: RelationMetrics, stats: Dict) -> str:
        """建议最佳行动时机"""
        # 根据当前状态建议何时采取行动

        if metrics.urgency_level == "critical":
            return "立即行动：不要再等待"

        if metrics.urgency_level == "high":
            return "本周内行动：避免进一步亏损"

        if metrics.roi < 0 and metrics.investment_score > 70:
            return "在下次主动之前先观察对方的回应"

        if metrics.balance_score < 40:
            return "对方下次主动时回应，测试其态度变化"

        return "继续保持当前节奏，观察趋势变化"


# ============================================================
# 报告生成
# ============================================================

def create_quantification_report(metrics: RelationMetrics) -> str:
    """生成量化报告"""
    lines = []

    lines.append("=" * 60)
    lines.append("关系量化分析报告")
    lines.append("=" * 60)

    # 核心指标
    lines.append("\n【核心指标】")
    lines.append(f"  情感投入指数: {metrics.investment_score:.1f}/100")
    lines.append(f"  对方回应指数: {metrics.response_score:.1f}/100")
    lines.append(f"  能量净值: {metrics.energy_net:+.1f}")
    lines.append(f"  关系ROI: {metrics.roi:+.2f}")

    # 健康指标
    lines.append("\n【健康指标】")
    lines.append(f"  整体健康度: {metrics.health_score:.1f}/100")
    lines.append(f"  平衡度: {metrics.balance_score:.1f}/100")
    lines.append(f"  稳定性: {metrics.stability_score:.1f}/100")

    # 风险指标
    lines.append("\n【风险指标】")
    lines.append(f"  综合风险: {metrics.risk_level:.1f}/100")
    lines.append(f"  耗竭风险: {metrics.burnout_risk:.1f}/100")
    lines.append(f"  拒绝风险: {metrics.rejection_risk:.1f}/100")

    # 预测
    lines.append("\n【预测分析】")
    lines.append(f"  预测满意度: {metrics.satisfaction_predicted:.1f}/100")
    lines.append(f"  预测关系寿命: {metrics.longevity_predicted:.1f}月")
    lines.append(f"  改善潜力: {metrics.improvement_potential:.1f}/100")

    # 建议
    lines.append("\n【行动建议】")
    urgency_display = {
        'critical': '⚠️ 紧急',
        'high': '⚡ 高优先',
        'normal': '📊 正常',
        'low': '✅ 低'
    }
    lines.append(f"  紧急程度: {urgency_display.get(metrics.urgency_level, '未知')}")
    lines.append(f"  建议行动: {metrics.action_recommendation}")
    lines.append(f"  最佳时机: {metrics.optimal_timing}")

    # ROI解读
    lines.append("\n【ROI解读】")
    if metrics.roi > 0.5:
        lines.append("  📈 高回报关系：投入获得良好回报")
    elif metrics.roi > 0:
        lines.append("  📊 正向关系：持续获得回报")
    elif metrics.roi > -0.3:
        lines.append("  📉 轻微亏损：需要关注平衡")
    elif metrics.roi > -0.5:
        lines.append("  ⚠️ 明显亏损：建议调整策略")
    else:
        lines.append("  🚨 严重亏损：强烈建议止损")

    return "\n".join(lines)


__all__ = [
    'RelationMetrics', 'EnergyTransaction', 'RelationLedger',
    'RelationQuantifier', 'create_quantification_report'
]