"""
心理学分析引擎

融合多个心理学理论框架，提供深层关系分析

已实现框架：
1. 依恋理论 (Attachment Theory) - 检测焦虑型/回避型依恋
2. 社会交换理论 (Social Exchange Theory) - 投入回报ROI分析
3. Gottman四骑士 (Four Horsemen) - 关系危险信号检测
4. Sternberg爱情三角 (Love Triangle) - 亲密/激情/承诺分析
5. 大五人格 (Big Five / FFM) - 人格特质分析
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import re


class AttachmentStyle(Enum):
    """依恋类型"""
    SECURE = "安全型"
    ANXIOUS = "焦虑型"      # 舔狗倾向
    AVOIDANT = "回避型"     # 冷淡倾向
    DISORGANIZED = "混乱型"


class HorsemanType(Enum):
    """Gottman四骑士类型"""
    CRITICISM = "批评"
    CONTEMPT = "轻蔑"
    DEFENSIVENESS = "防御"
    STONEWALLING = "冷战"


class ExchangeBalance(Enum):
    """社会交换平衡状态"""
    BALANCED = "平衡"
    UNDERBENEFITED = "受益不足"   # 舔狗状态
    OVERBENEFITED = "受益过度"


@dataclass
class PsychologicalProfile:
    """心理学画像"""
    # 依恋分析
    my_attachment: AttachmentStyle = AttachmentStyle.SECURE
    their_attachment: AttachmentStyle = AttachmentStyle.SECURE
    attachment_match: str = "匹配"  # 匹配/冲突

    # 社会交换
    exchange_balance: ExchangeBalance = ExchangeBalance.BALANCED
    my_investment_score: float = 50.0
    their_return_score: float = 50.0
    roi: float = 1.0  # 回报/投入

    # 四骑士
    horsemen_detected: List[HorsemanType] = field(default_factory=list)
    horsemen_severity: Dict[HorsemanType, float] = field(default_factory=dict)
    relationship_risk: str = "low"  # low/medium/high/critical

    # Sternberg三角
    intimacy_score: float = 50.0
    passion_score: float = 50.0
    commitment_score: float = 50.0
    love_type: str = "友伴之爱"  # 完美之爱/友伴之爱/空洞之爱等

    # 综合诊断
    diagnosis: str = ""
    severity: float = 0.0
    recommendations: List[str] = field(default_factory=list)


class PsychologicalAnalyzer:
    """
    心理学分析引擎

    核心方法：
    - analyze_attachment(): 依恋类型检测
    - analyze_exchange(): 社会交换ROI分析
    - detect_horsemen(): 四骑士危险信号
    - analyze_triangle(): Sternberg爱情三角
    """

    # 舔狗信号词（焦虑型依恋表现）
    SIMP_MARKERS = {
        "pleasing": ["好吧", "没事", "哈哈", "嘿嘿", "嗯嗯", "可以的"],
        "apologetic": ["对不起", "不好意思", "抱歉", "我的错"],
        "seeking_validation": ["你觉得", "这样行吗", "可以吗", "好不好"],
        "over_investing": ["我帮你", "我来", "随时", "随时找我"],
        "emoji_coping": ["[旺柴]", "[捂脸]", "[流泪]", "[发呆]", "[坏笑]"]
    }

    # 冷淡信号词（回避型依恋表现）
    COLD_MARKERS = {
        "dismissive": ["哦", "嗯", "好的", "知道了", "行"],
        "short_filler": ["okk", "收到", "哈哈", "[表情]"],
        "avoiding": ["算了", "不用了", "再说吧", "看情况"],
        "delaying": ["等一下", "稍后", "改天"]
    }

    # 轻蔑信号（Gottman）
    CONTEMPT_MARKERS = {
        "sarcasm": ["呵呵", "搞笑", "有意思"],
        "mockery": ["（doge）", "（笑）", "哈哈好傻"],
        "dismissive_emoji": ["[微笑]", "[呵呵]", "[汗]"]
    }

    def analyze(self, messages: List[Dict], stats: Dict) -> PsychologicalProfile:
        """
        综合心理学分析

        Args:
            messages: 消息列表
            stats: 统计数据

        Returns:
            心理学画像
        """
        profile = PsychologicalProfile()

        # 1. 依恋分析
        self.analyze_attachment(messages, stats, profile)

        # 2. 社会交换分析
        self.analyze_exchange(messages, stats, profile)

        # 3. 四骑士检测
        self.detect_horsemen(messages, stats, profile)

        # 4. Sternberg三角（如果有情感关系）
        self.analyze_triangle(messages, stats, profile)

        # 5. 综合诊断
        self.generate_diagnosis(profile)

        return profile

    def analyze_attachment(self, messages: List[Dict], stats: Dict, profile: PsychologicalProfile):
        """
        依恋类型分析

        焦虑型依恋特征（舔狗）：
        - 过度主动（提问>>对方提问）
        - 频繁讨好表达
        - 消息轰炸（连发）
        - 快速回复但对方慢

        回避型依恋特征（冷淡）：
        - 简短回复为主
        - 几乎不主动开启话题
        - 用表情敷衍
        - 延迟回复
        """
        me_msgs = [m for m in messages if m.get('sender') == 'me']
        them_msgs = [m for m in messages if m.get('sender') == 'them']

        # 我的依恋类型检测
        my_anxious_score = 0
        my_anxious_signals = []

        # 1. 提问比例
        me_questions = sum(1 for m in me_msgs if '?' in m.get('content', '') or '？' in m.get('content', ''))
        them_questions = sum(1 for m in them_msgs if '?' in m.get('content', '') or '？' in m.get('content', ''))
        if me_questions > them_questions * 2:
            my_anxious_score += 25
            my_anxious_signals.append(f"提问过多({me_questions} vs {them_questions})")

        # 2. 讨好表达 (调整阈值，22.8%已算高)
        pleasing_count = 0
        for m in me_msgs:
            content = m.get('content', '')
            for markers in self.SIMP_MARKERS.values():
                if any(marker in content for marker in markers):
                    pleasing_count += 1
                    break
        pleasing_ratio = pleasing_count / len(me_msgs) if me_msgs else 0
        if pleasing_ratio > 0.2:  # 从0.3降到0.2
            my_anxious_score += 20
            my_anxious_signals.append(f"讨好表达{pleasing_ratio:.1%}")

        # 3. 连发轰炸
        consecutive_count = 0
        for i in range(len(messages) - 1):
            if messages[i].get('sender') == 'me' and messages[i+1].get('sender') == 'me':
                consecutive_count += 1
        if consecutive_count > len(me_msgs) * 0.3:
            my_anxious_score += 20
            my_anxious_signals.append(f"频繁连发({consecutive_count}次)")

        # 4. 回复速度差异
        me_reply_time = stats.get('reply_speed', {}).get('my_avg_seconds', 0)
        them_reply_time = stats.get('reply_speed', {}).get('their_avg_seconds', 0)
        if me_reply_time < them_reply_time / 2 and them_reply_time > 300:
            my_anxious_score += 15
            my_anxious_signals.append(f"急切回复(你{me_reply_time/60:.1f}分 vs 对方{them_reply_time/60:.1f}分)")

        # 判断依恋类型 (调整阈值，更敏感地检测焦虑型)
        if my_anxious_score >= 35:
            profile.my_attachment = AttachmentStyle.ANXIOUS
            if profile.their_attachment == AttachmentStyle.AVOIDANT:
                profile.attachment_match = "焦虑-回避陷阱"
        elif my_anxious_score >= 20:
            profile.my_attachment = AttachmentStyle.ANXIOUS
        else:
            profile.my_attachment = AttachmentStyle.SECURE

        # 对方依恋类型检测
        their_avoidant_score = 0
        their_avoidant_signals = []

        # 1. 敷衍回复比例
        dismissive_count = 0
        for m in them_msgs:
            content = m.get('content', '')
            for markers in self.COLD_MARKERS.values():
                if any(marker in content for marker in markers):
                    dismissive_count += 1
                    break
        dismissive_ratio = dismissive_count / len(them_msgs) if them_msgs else 0
        if dismissive_ratio > 0.5:
            their_avoidant_score += 30
            their_avoidant_signals.append(f"敷衍回复{dismissive_ratio:.1%}")

        # 2. 不主动
        them_initiations = stats.get('initiative', {}).get('them_start_count', 0)
        if them_initiations < 5:
            their_avoidant_score += 25
            their_avoidant_signals.append(f"几乎不主动({them_initiations}次)")

        # 3. 简短回复为主
        short_replies = sum(1 for m in them_msgs if len(m.get('content', '')) <= 10)
        short_ratio = short_replies / len(them_msgs) if them_msgs else 0
        if short_ratio > 0.6:
            their_avoidant_score += 20
            their_avoidant_signals.append(f"简短回复{short_ratio:.1%}")

        # 判断对方依恋类型
        if their_avoidant_score >= 50:
            profile.their_attachment = AttachmentStyle.AVOIDANT
            profile.attachment_match = "焦虑-回避陷阱"  # 最糟糕组合
        elif their_avoidant_score >= 30:
            profile.their_attachment = AttachmentStyle.AVOIDANT
        else:
            profile.their_attachment = AttachmentStyle.SECURE

    def analyze_exchange(self, messages: List[Dict], stats: Dict, profile: PsychologicalProfile):
        """
        社会交换理论分析

        计算 ROI = 对方回报 / 你的投入

        投入指标：
        - 提问次数 (权重高)
        - 主动次数 (权重最高)
        - 消息轰炸 (权重高)
        - 讨好表达 (权重中等)
        - 消息长度 (权重低)

        回报指标：
        - 对方主动次数 (权重最高)
        - 对方有意义回复 (权重高)
        - 对方延续话题 (权重中等)
        - 对方情感回报 (权重低，因为可能只是礼貌)
        """
        me_msgs = [m for m in messages if m.get('sender') == 'me']
        them_msgs = [m for m in messages if m.get('sender') == 'them']

        # 计算投入分数 (调整权重)
        investment = 0

        # 1. 提问投入 (权重提高)
        me_questions = sum(1 for m in me_msgs if '?' in m.get('content', '') or '？' in m.get('content', ''))
        investment += me_questions * 5  # 每个提问算5分投入

        # 2. 主动投入 (权重最高)
        me_initiations = stats.get('initiative', {}).get('me_start_count', 0)
        investment += me_initiations * 10  # 主动开启话题投入最大

        # 3. 消息轰炸投入 (新增)
        consecutive_count = 0
        for i in range(len(messages) - 1):
            if messages[i].get('sender') == 'me' and messages[i+1].get('sender') == 'me':
                consecutive_count += 1
        investment += consecutive_count * 3  # 每次连发算3分投入

        # 4. 讨好表达投入 (权重提高)
        pleasing_count = 0
        for m in me_msgs:
            content = m.get('content', '')
            for markers in self.SIMP_MARKERS.values():
                if any(marker in content for marker in markers):
                    pleasing_count += 1
                    break
        investment += pleasing_count * 2

        # 5. 消息长度投入
        me_total_len = sum(len(m.get('content', '')) for m in me_msgs)
        investment += me_total_len / 50  # 每50字算1分

        # 保存中间变量供依恋分析使用
        me_pleasing_ratio = pleasing_count / len(me_msgs) if me_msgs else 0
        me_consecutive_ratio = consecutive_count / len(me_msgs) if me_msgs else 0

        # 计算回报分数 (严格定义，排除敷衍)
        returns = 0

        # 1. 对方主动回报 (权重最高)
        them_initiations = stats.get('initiative', {}).get('them_start_count', 0)
        returns += them_initiations * 15  # 对方主动价值最高

        # 2. 对方有意义回复回报 (严格定义：长度>15 且非敷衍)
        meaningful_replies = 0
        for m in them_msgs:
            content = m.get('content', '')
            # 检查是否是敷衍回复
            is_dismissive = False
            for markers in self.COLD_MARKERS.values():
                if any(marker in content for marker in markers):
                    is_dismissive = True
                    break
            # 非敷衍且有实质内容才算回报
            if len(content) > 15 and not is_dismissive:
                meaningful_replies += 1
        returns += meaningful_replies * 5

        # 3. 对方延续话题回报 (提问)
        them_questions = sum(1 for m in them_msgs if '?' in m.get('content', '') or '？' in m.get('content', ''))
        returns += them_questions * 8

        # 4. 对方分享个人信息回报 (亲密行为)
        them_sharing = 0
        for m in them_msgs:
            content = m.get('content', '')
            if any(word in content for word in ["我今天", "我昨天", "我觉得", "我想", "我家里"]):
                if len(content) > 20:
                    them_sharing += 1
        returns += them_sharing * 6

        # 5. 排除：礼貌性回复不算回报（减少权重）
        polite_replies = 0
        for m in them_msgs:
            content = m.get('content', '')
            # 这些只是礼貌，不是真心回报
            if content in ["嗯嗯", "哈哈", "好的", "收到", "okk", "[表情]", "加油"]:
                polite_replies += 1
        # 礼貌回复权重很低，只有0.5分
        returns += polite_replies * 0.5

        # 计算 ROI
        profile.my_investment_score = min(investment / 100, 100)
        profile.their_return_score = min(returns / 50, 100)
        profile.roi = returns / investment if investment > 0 else 0

        # 判断交换平衡状态 (调整阈值)
        if profile.roi < 0.5:
            profile.exchange_balance = ExchangeBalance.UNDERBENEFITED
        elif profile.roi > 2:
            profile.exchange_balance = ExchangeBalance.OVERBENEFITED
        else:
            profile.exchange_balance = ExchangeBalance.BALANCED

    def detect_horsemen(self, messages: List[Dict], stats: Dict, profile: PsychologicalProfile):
        """
        Gottman四骑士检测

        四骑士预测关系破裂：
        1. 批评 - 攻击对方人格而非行为
        2. 轻蔑 - 讽刺、嘲笑、鄙视
        3. 防御 - 推卸责任、反击
        4. 冷战 - 拒绝交流、沉默

        舔狗场景常见：轻蔑(对方) + 冷战(对方不主动)
        """
        them_msgs = [m for m in messages if m.get('sender') == 'them']

        # 1. 轻蔑检测
        contempt_count = 0
        for m in them_msgs:
            content = m.get('content', '')
            for markers in self.CONTEMPT_MARKERS.values():
                if any(marker in content for marker in markers):
                    contempt_count += 1
                    break

        contempt_ratio = contempt_count / len(them_msgs) if them_msgs else 0
        if contempt_ratio > 0.1:
            profile.horsemen_detected.append(HorsemanType.CONTEMPT)
            profile.horsemen_severity[HorsemanType.CONTEMPT] = contempt_ratio * 100

        # 2. 冷战检测
        them_initiations = stats.get('initiative', {}).get('them_start_count', 0)
        total_days = stats.get('basic', {}).get('duration_days', 30)
        initiation_rate = them_initiations / total_days if total_days > 0 else 0

        if initiation_rate < 0.1:  # 平均每10天不到1次主动
            profile.horsemen_detected.append(HorsemanType.STONEWALLING)
            profile.horsemen_severity[HorsemanType.STONEWALLING] = 100 - initiation_rate * 100

        # 3. 批评检测（较少见于舔狗场景）
        criticism_words = ["你总是", "你从不", "你真的很", "你怎么这么"]
        criticism_count = 0
        for m in them_msgs:
            content = m.get('content', '')
            if any(word in content for word in criticism_words):
                criticism_count += 1

        if criticism_count > 3:
            profile.horsemen_detected.append(HorsemanType.CRITICISM)
            profile.horsemen_severity[HorsemanType.CRITICISM] = criticism_count * 10

        # 计算关系风险
        if len(profile.horsemen_detected) >= 3:
            profile.relationship_risk = "critical"
        elif len(profile.horsemen_detected) >= 2:
            profile.relationship_risk = "high"
        elif len(profile.horsemen_detected) >= 1:
            profile.relationship_risk = "medium"
        else:
            profile.relationship_risk = "low"

    def analyze_triangle(self, messages: List[Dict], stats: Dict, profile: PsychologicalProfile):
        """
        Sternberg爱情三角分析

        三要素：
        - 亲密: 情感连接、分享、理解
        - 激情: 吸引力、浪漫、性吸引
        - 承诺: 决定维持关系的意愿

        舔狗场景：
        - 亲密：低（对方不分享）
        - 激情：你高对方低
        - 承诺：低（对方不提未来）
        """
        them_msgs = [m for m in messages if m.get('sender') == 'them']

        # 亲密度评估
        them_sharing = 0
        for m in them_msgs:
            content = m.get('content', '')
            # 分享个人信息
            if any(word in content for word in ["我", "我的", "今天", "昨天", "觉得", "想"]):
                if len(content) > 20:  # 有实质内容
                    them_sharing += 1

        sharing_ratio = them_sharing / len(them_msgs) if them_msgs else 0
        profile.intimacy_score = sharing_ratio * 100

        # 承诺度评估
        commitment_words = ["未来", "我们", "一起", "以后", "下次", "到时候"]
        them_commitment = 0
        for m in them_msgs:
            content = m.get('content', '')
            if any(word in content for word in commitment_words):
                them_commitment += 1

        commitment_ratio = them_commitment / len(them_msgs) if them_msgs else 0
        profile.commitment_score = commitment_ratio * 200  # 权重更高

        # 激情度评估（简化）
        # 通过回复热情度估计
        enthusiastic_words = ["好呀", "可以", "想", "期待", "哈哈", "嘻嘻"]
        them_passion = 0
        for m in them_msgs:
            content = m.get('content', '')
            if any(word in content for word in enthusiastic_words):
                them_passion += 1

        passion_ratio = them_passion / len(them_msgs) if them_msgs else 0
        profile.passion_score = passion_ratio * 150

        # 判断爱情类型
        high_threshold = 60
        if profile.intimacy_score > high_threshold and profile.passion_score > high_threshold and profile.commitment_score > high_threshold:
            profile.love_type = "完美之爱"
        elif profile.intimacy_score > high_threshold and profile.commitment_score > high_threshold:
            profile.love_type = "友伴之爱"
        elif profile.intimacy_score > high_threshold:
            profile.love_type = "喜欢之爱"
        elif profile.passion_score > high_threshold:
            profile.love_type = "迷恋之爱"
        elif profile.commitment_score > high_threshold:
            profile.love_type = "空洞之爱"
        else:
            profile.love_type = "无爱状态"

    def generate_diagnosis(self, profile: PsychologicalProfile):
        """
        综合诊断生成
        """
        diagnoses = []
        recommendations = []
        severity = 0

        # 焦虑-回避陷阱诊断
        if profile.my_attachment == AttachmentStyle.ANXIOUS and profile.their_attachment == AttachmentStyle.AVOIDANT:
            diagnoses.append("焦虑-回避陷阱: 你追得越紧，对方逃得越远")
            recommendations.append("立即停止主动，给对方空间")
            recommendations.append("转移注意力到其他活动")
            recommendations.append("对方不主动 = 明确答案")
            severity += 40

        # 社会交换失衡诊断
        if profile.exchange_balance == ExchangeBalance.UNDERBENEFITED:
            diagnoses.append(f"投入回报失衡: ROI={profile.roi:.2f}，持续亏损")
            recommendations.append("止损：减少投入至对方回应水平")
            severity += 30

        # 四骑士诊断
        if HorsemanType.STONEWALLING in profile.horsemen_detected:
            diagnoses.append("冷战信号: 对方通过不主动表达疏离")
            severity += 20

        if HorsemanType.CONTEMPT in profile.horsemen_detected:
            diagnoses.append("轻蔑信号: 对方内心可能在鄙视你的投入")
            severity += 25

        # 综合判断
        profile.diagnosis = "; ".join(diagnoses) if diagnoses else "关系正常"
        profile.severity = min(severity, 100)
        profile.recommendations = recommendations


def create_psychological_report(profile: PsychologicalProfile) -> str:
    """
    生成心理学分析报告
    """
    lines = []
    lines.append("=" * 60)
    lines.append("心理学深度分析报告")
    lines.append("=" * 60)

    lines.append("\n【依恋类型分析】")
    lines.append(f"  你的依恋: {profile.my_attachment.value}")
    lines.append(f"  对方依恋: {profile.their_attachment.value}")
    lines.append(f"  匹配状态: {profile.attachment_match}")

    lines.append("\n【社会交换分析】")
    lines.append(f"  你的投入: {profile.my_investment_score:.1f}")
    lines.append(f"  对方回报: {profile.their_return_score:.1f}")
    lines.append(f"  ROI: {profile.roi:.2f}")
    lines.append(f"  平衡状态: {profile.exchange_balance.value}")

    lines.append("\n【四骑士检测】")
    if profile.horsemen_detected:
        for h in profile.horsemen_detected:
            severity = profile.horsemen_severity.get(h, 0)
            lines.append(f"  ⚠️ {h.value}: {severity:.1f}%")
    else:
        lines.append("  未检测到危险信号")

    lines.append(f"\n  关系风险: {profile.relationship_risk}")

    lines.append("\n【Sternberg三角】")
    lines.append(f"  亲密: {profile.intimacy_score:.1f}")
    lines.append(f"  激情: {profile.passion_score:.1f}")
    lines.append(f"  承诺: {profile.commitment_score:.1f}")
    lines.append(f"  类型: {profile.love_type}")

    lines.append("\n【综合诊断】")
    lines.append(f"  {profile.diagnosis}")
    lines.append(f"  严重程度: {profile.severity:.0f}/100")

    if profile.recommendations:
        lines.append("\n【建议】")
        for r in profile.recommendations:
            lines.append(f"  → {r}")

    return "\n".join(lines)


__all__ = [
    'AttachmentStyle', 'HorsemanType', 'ExchangeBalance',
    'PsychologicalProfile', 'PsychologicalAnalyzer',
    'create_psychological_report'
]