"""
大五人格模型 (Big Five / FFM)

五大人格特质：
1. 开放性 (Openness) - 好奇心、创造力、新体验
2. 尽责性 (Conscientiousness) - 自律、组织、可靠
3. 外向性 (Extraversion) - 社交、活力、积极
4. 宜人性 (Agreeableness) - 合作、信任、同理心
5. 神经质 (Neuroticism) - 情绪稳定性、焦虑、敏感

应用场景：
- 分析用户的人格特质（从聊天行为推断）
- 分析对方的人格特质（从回复模式推断）
- 人格匹配度评估
- 基于人格的决策建议
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import re


class BigFiveTrait(Enum):
    """大五人格特质"""
    OPENNESS = "开放性"
    CONSCIENTIOUSNESS = "尽责性"
    EXTRAVERSION = "外向性"
    AGREEABLENESS = "宜人性"
    NEUROTICISM = "神经质"


@dataclass
class PersonalityProfile:
    """人格画像"""
    # 五个维度的分数 (0-100)
    openness: float = 50.0
    conscientiousness: float = 50.0
    extraversion: float = 50.0
    agreeableness: float = 50.0
    neuroticism: float = 50.0

    # 描述性标签
    dominant_traits: List[str] = field(default_factory=list)
    personality_type: str = ""

    # 行为特征
    communication_style: str = ""
    relationship_tendency: str = ""

    def get_scores(self) -> Dict[BigFiveTrait, float]:
        """获取所有维度分数"""
        return {
            BigFiveTrait.OPENNESS: self.openness,
            BigFiveTrait.CONSCIENTIOUSNESS: self.conscientiousness,
            BigFiveTrait.EXTRAVERSION: self.extraversion,
            BigFiveTrait.AGREEABLENESS: self.agreeableness,
            BigFiveTrait.NEUROTICISM: self.neuroticism
        }

    def get_high_traits(self, threshold: float = 60) -> List[BigFiveTrait]:
        """获取高分特质"""
        scores = self.get_scores()
        return [trait for trait, score in scores.items() if score >= threshold]

    def get_low_traits(self, threshold: float = 40) -> List[BigFiveTrait]:
        """获取低分特质"""
        scores = self.get_scores()
        return [trait for trait, score in scores.items() if score <= threshold]


@dataclass
class PersonalityMatch:
    """人格匹配分析"""
    compatibility_score: float = 50.0
    complementary_traits: List[Tuple[BigFiveTrait, str]] = field(default_factory=list)
    conflict_traits: List[Tuple[BigFiveTrait, str]] = field(default_factory=list)
    advice: List[str] = field(default_factory=list)


class BigFiveAnalyzer:
    """
    大五人格分析器

    从聊天行为推断人格特质
    """

    # ============ 语言特征词库 ============

    # 开放性相关词汇
    OPENNESS_MARKERS = {
        "high": [
            "觉得", "想法", "可能", "也许", "探索", "尝试", "新", "有趣",
            "思考", "哲学", "艺术", "音乐", "读书", "学习", "创意",
            "不同", "独特", "奇怪", "为什么", "怎么样", "什么意思"
        ],
        "low": [
            "传统", "保守", "规矩", "应该", "正常", "标准", "按部就班",
            "踏实", "稳定", "安全", "习惯", "老样子"
        ]
    }

    # 尽责性相关词汇
    CONSCIENTIOUSNESS_MARKERS = {
        "high": [
            "计划", "安排", "目标", "进度", "完成", "deadline", "作业",
            "学习", "工作", "努力", "坚持", "自律", "效率", "时间"
        ],
        "low": [
            "拖延", "懒", "随缘", "再说", "不想", "躺平", "摆烂",
            "随便", "都行", "无所谓", "摸鱼"
        ]
    }

    # 外向性相关词汇
    EXTRAVERSION_MARKERS = {
        "high": [
            "哈哈", "哈哈哈", "好玩", "有趣", "朋友", "一起", "聚会",
            "出门", "玩", "热闹", "开心", "兴奋", "社交", "群"
        ],
        "low": [
            "累", "社恐", "宅", "一个人", "安静", "独处", "内耗",
            "社死", "尴尬", "不想说话", "社畜"
        ]
    }

    # 宜人性相关词汇
    AGREEABLENESS_MARKERS = {
        "high": [
            "好的", "可以", "没问题", "谢谢", "辛苦", "理解", "体谅",
            "帮忙", "关心", "照顾", "抱歉", "不好意思", "对不起", "感恩"
        ],
        "low": [
            "无语", "烦", "讨厌", "恶心", "傻", "智障", "有病",
            "滚", "靠", "服了", "离谱", "绝了"
        ]
    }

    # 神经质相关词汇
    NEUROTICISM_MARKERS = {
        "high": [
            "焦虑", "担心", "害怕", "紧张", "压力", "崩溃", "抑郁",
            "失眠", "emo", "难受", "痛苦", "想哭", "受不了", "[流泪]"
        ],
        "low": [
            "没事", "放松", "淡定", "冷静", "无所谓", "随便", "不在乎",
            "正常", "还好", "挺好的"
        ]
    }

    # ============ 行为模式 ============

    def analyze(self, messages: List[Dict], sender: str = "me") -> PersonalityProfile:
        """
        分析人格特质

        Args:
            messages: 消息列表
            sender: 分析对象 ("me" 或 "them")

        Returns:
            人格画像
        """
        # 筛选目标消息
        target_msgs = [m for m in messages if m.get('sender') == sender]

        if not target_msgs:
            return PersonalityProfile()

        # 提取文本内容
        all_text = " ".join(m.get('content', '') for m in target_msgs)

        # 计算各维度分数
        openness = self._calc_openness(target_msgs, all_text)
        conscientiousness = self._calc_conscientiousness(target_msgs, all_text, messages)
        extraversion = self._calc_extraversion(target_msgs, all_text)
        agreeableness = self._calc_agreeableness(target_msgs, all_text)
        neuroticism = self._calc_neuroticism(target_msgs, all_text)

        # 创建画像
        profile = PersonalityProfile(
            openness=openness,
            conscientiousness=conscientiousness,
            extraversion=extraversion,
            agreeableness=agreeableness,
            neuroticism=neuroticism
        )

        # 生成描述
        self._generate_descriptions(profile, target_msgs)

        return profile

    def _calc_openness(self, msgs: List[Dict], all_text: str) -> float:
        """
        计算开放性

        高开放性特征：
        - 使用抽象/思考性词汇
        - 讨论多元话题
        - 表达好奇心
        - 分享新观点/体验
        """
        score = 50.0  # 基准分

        # 高开放性词汇
        high_count = sum(1 for word in self.OPENNESS_MARKERS["high"] if word in all_text)
        score += min(high_count * 2, 20)

        # 低开放性词汇
        low_count = sum(1 for word in self.OPENNESS_MARKERS["low"] if word in all_text)
        score -= min(low_count * 2, 15)

        # 话题多样性（通过消息长度变化估计）
        lengths = [len(m.get('content', '')) for m in msgs if m.get('content')]
        if lengths:
            length_variety = max(lengths) - min(lengths) if len(lengths) > 1 else 0
            if length_variety > 100:  # 话题深度不一，可能有深度思考
                score += 10

        # 提问频率（好奇心）
        questions = sum(1 for m in msgs if '?' in m.get('content', '') or '？' in m.get('content', ''))
        question_ratio = questions / len(msgs) if msgs else 0
        if question_ratio > 0.2:
            score += 10  # 高提问率 = 高好奇心

        return min(max(score, 0), 100)

    def _calc_conscientiousness(self, msgs: List[Dict], all_text: str, all_msgs: List[Dict]) -> float:
        """
        计算尽责性

        高尽责性特征：
        - 讨论计划、目标
        - 稳定的回复时间
        - 完整的表达
        - 少拖延表达
        """
        score = 50.0

        # 高尽责性词汇
        high_count = sum(1 for word in self.CONSCIENTIOUSNESS_MARKERS["high"] if word in all_text)
        score += min(high_count * 2, 20)

        # 低尽责性词汇
        low_count = sum(1 for word in self.CONSCIENTIOUSNESS_MARKERS["low"] if word in all_text)
        score -= min(low_count * 2, 15)

        # 回复时间稳定性（如果有时间数据）
        timestamps = [m.get('timestamp', 0) for m in msgs if m.get('timestamp', 0) > 0]
        if len(timestamps) > 2:
            # 计算回复时间标准差
            import numpy as np
            if len(timestamps) > 3:
                reply_times = np.diff(timestamps)
                std = np.std(reply_times)
                if std < 3600:  # 1小时内标准差，回复时间稳定
                    score += 10
                elif std > 86400:  # 回复时间很不稳定
                    score -= 10

        # 消息完整性（不频繁连发短句）
        short_msgs = sum(1 for m in msgs if len(m.get('content', '')) < 5)
        short_ratio = short_msgs / len(msgs) if msgs else 0
        if short_ratio > 0.3:
            score -= 10  # 大量短句 = 可能不够深思熟虑

        return min(max(score, 0), 100)

    def _calc_extraversion(self, msgs: List[Dict], all_text: str) -> float:
        """
        计算外向性

        高外向性特征：
        - 活跃表达（哈哈、表情）
        - 主动分享
        - 社交话题
        - 积极情绪词
        """
        score = 50.0

        # 高外向性词汇
        high_count = sum(1 for word in self.EXTRAVERSION_MARKERS["high"] if word in all_text)
        score += min(high_count * 2, 25)

        # 低外向性词汇
        low_count = sum(1 for word in self.EXTRAVERSION_MARKERS["low"] if word in all_text)
        score -= min(low_count * 2, 20)

        # 表情使用频率
        emoji_count = all_text.count('[') + all_text.count(']')
        emoji_ratio = emoji_count / len(all_text) if all_text else 0
        if emoji_ratio > 0.05:
            score += 10

        # 消息频率（主动程度）
        if len(msgs) > 50:  # 消息数量多
            score += 10

        return min(max(score, 0), 100)

    def _calc_agreeableness(self, msgs: List[Dict], all_text: str) -> float:
        """
        计算宜人性

        高宜人性特征：
        - 礼貌表达
        - 同理心词汇
        - 合作性语言
        - 少冲突表达
        """
        score = 50.0

        # 高宜人性词汇
        high_count = sum(1 for word in self.AGREEABLENESS_MARKERS["high"] if word in all_text)
        score += min(high_count * 2, 25)

        # 低宜人性词汇
        low_count = sum(1 for word in self.AGREEABLENESS_MARKERS["low"] if word in all_text)
        score -= min(low_count * 3, 30)

        # 感谢/道歉频率
        thanks = sum(1 for m in msgs
                     for word in ["谢谢", "感谢", "辛苦", "麻烦"]
                     if word in m.get('content', ''))
        if thanks > 5:
            score += 10

        # 冲突词汇
        conflicts = sum(1 for m in msgs
                       for word in ["你总是", "你从不", "你怎么", "服了", "无语"]
                       if word in m.get('content', ''))
        score -= min(conflicts * 5, 20)

        return min(max(score, 0), 100)

    def _calc_neuroticism(self, msgs: List[Dict], all_text: str) -> float:
        """
        计算神经质

        高神经质特征：
        - 负面情绪词汇
        - 焦虑表达
        - 情绪波动
        - 自我怀疑
        """
        score = 50.0

        # 高神经质词汇
        high_count = sum(1 for word in self.NEUROTICISM_MARKERS["high"] if word in all_text)
        score += min(high_count * 3, 30)

        # 低神经质词汇
        low_count = sum(1 for word in self.NEUROTICISM_MARKERS["low"] if word in all_text)
        score -= min(low_count * 2, 20)

        # 情绪波动指标（通过消息长度变化）
        lengths = [len(m.get('content', '')) for m in msgs if m.get('content')]
        if len(lengths) > 5:
            import numpy as np
            cv = np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0
            if cv > 1.0:  # 高变异性 = 情绪波动大
                score += 10

        # 自我贬低表达
        self_deprecating = sum(1 for m in msgs
                              for word in ["我太", "我不行", "我废", "我烂", "我菜"]
                              if word in m.get('content', ''))
        score += min(self_deprecating * 3, 15)

        return min(max(score, 0), 100)

    def _generate_descriptions(self, profile: PersonalityProfile, msgs: List[Dict]):
        """生成人格描述"""

        # 确定主导特质
        high_traits = profile.get_high_traits(60)
        low_traits = profile.get_low_traits(40)

        trait_names = {
            BigFiveTrait.OPENNESS: "高开放性",
            BigFiveTrait.CONSCIENTIOUSNESS: "高尽责性",
            BigFiveTrait.EXTRAVERSION: "高外向性",
            BigFiveTrait.AGREEABLENESS: "高宜人性",
            BigFiveTrait.NEUROTICISM: "高神经质"
        }

        low_trait_names = {
            BigFiveTrait.OPENNESS: "低开放性",
            BigFiveTrait.CONSCIENTIOUSNESS: "低尽责性",
            BigFiveTrait.EXTRAVERSION: "低外向性",
            BigFiveTrait.AGREEABLENESS: "低宜人性",
            BigFiveTrait.NEUROTICISM: "情绪稳定"
        }

        profile.dominant_traits = [trait_names[t] for t in high_traits]
        profile.dominant_traits.extend([low_trait_names[t] for t in low_traits])

        # 确定人格类型
        scores = profile.get_scores()
        dominant = max(scores, key=scores.get)

        type_descriptions = {
            BigFiveTrait.OPENNESS: "探索者型",
            BigFiveTrait.CONSCIENTIOUSNESS: "执行者型",
            BigFiveTrait.EXTRAVERSION: "社交者型",
            BigFiveTrait.AGREEABLENESS: "协作者型",
            BigFiveTrait.NEUROTICISM: "敏感者型"
        }
        profile.personality_type = type_descriptions.get(dominant, "平衡型")

        # 沟通风格
        if profile.extraversion > 60:
            profile.communication_style = "热情活跃，善于表达"
        elif profile.extraversion < 40:
            profile.communication_style = "内敛深沉，善于倾听"
        else:
            profile.communication_style = "适中平衡，因人而异"

        if profile.agreeableness > 60:
            profile.communication_style += "，温和友善"
        elif profile.agreeableness < 40:
            profile.communication_style += "，直言不讳"

        # 关系倾向
        if profile.neuroticism > 60:
            profile.relationship_tendency = "情感敏感，需要安全感"
        elif profile.neuroticism < 40:
            profile.relationship_tendency = "情绪稳定，适应力强"

        if profile.agreeableness > 60 and profile.neuroticism > 60:
            profile.relationship_tendency += "，易陷入讨好模式"
        elif profile.agreeableness < 40 and profile.neuroticism < 40:
            profile.relationship_tendency += "，独立自主"

    def analyze_match(self,
                      my_profile: PersonalityProfile,
                      their_profile: PersonalityProfile) -> PersonalityMatch:
        """
        分析人格匹配度

        Args:
            my_profile: 我的人格画像
            their_profile: 对方的人格画像

        Returns:
            匹配分析结果
        """
        match = PersonalityMatch()

        my_scores = my_profile.get_scores()
        their_scores = their_profile.get_scores()

        # 计算匹配度
        # 相似性匹配：外向性和宜人性相似更好
        # 互补性匹配：神经质差异大更好（一方稳定可以安抚另一方）

        # 1. 相似性维度
        similarity_traits = [
            BigFiveTrait.EXTRAVERSION,
            BigFiveTrait.AGREEABLENESS,
            BigFiveTrait.OPENNESS
        ]

        similarity_score = 0
        for trait in similarity_traits:
            diff = abs(my_scores[trait] - their_scores[trait])
            similarity_score += 100 - diff  # 越相似越好

        similarity_score /= len(similarity_traits)

        # 2. 互补性维度
        # 神经质：差异大有好处（稳定方可以安抚焦虑方）
        neuro_diff = abs(my_scores[BigFiveTrait.NEUROTICISM] - their_scores[BigFiveTrait.NEUROTICISM])
        if neuro_diff > 30:  # 差异大
            match.complementary_traits.append(
                (BigFiveTrait.NEUROTICISM,
                 f"情绪特质互补：一方稳定({min(my_scores[BigFiveTrait.NEUROTICISM], their_scores[BigFiveTrait.NEUROTICISM]):.0f})可安抚另一方({max(my_scores[BigFiveTrait.NEUROTICISM], their_scores[BigFiveTrait.NEUROTICISM]):.0f})")
            )

        # 尽责性：差异可以互补
        consc_diff = abs(my_scores[BigFiveTrait.CONSCIENTIOUSNESS] - their_scores[BigFiveTrait.CONSCIENTIOUSNESS])
        if consc_diff > 25:
            match.complementary_traits.append(
                (BigFiveTrait.CONSCIENTIOUSNESS,
                 f"行动风格互补：一方({max(my_scores[BigFiveTrait.CONSCIENTIOUSNESS], their_scores[BigFiveTrait.CONSCIENTIOUSNESS]):.0f})可带动另一方({min(my_scores[BigFiveTrait.CONSCIENTIOUSNESS], their_scores[BigFiveTrait.CONSCIENTIOUSNESS]):.0f})")
            )

        # 3. 冲突维度
        # 外向性差异过大
        extra_diff = abs(my_scores[BigFiveTrait.EXTRAVERSION] - their_scores[BigFiveTrait.EXTRAVERSION])
        if extra_diff > 40:
            match.conflict_traits.append(
                (BigFiveTrait.EXTRAVERSION,
                 f"社交需求差异大({my_scores[BigFiveTrait.EXTRAVERSION]:.0f} vs {their_scores[BigFiveTrait.EXTRAVERSION]:.0f})：可能产生生活节奏冲突")
            )

        # 宜人性差异
        agree_diff = abs(my_scores[BigFiveTrait.AGREEABLENESS] - their_scores[BigFiveTrait.AGREEABLENESS])
        if agree_diff > 35:
            match.conflict_traits.append(
                (BigFiveTrait.AGREEABLENESS,
                 f"处事风格差异({my_scores[BigFiveTrait.AGREEABLENESS]:.0f} vs {their_scores[BigFiveTrait.AGREEABLENESS]:.0f})：可能产生沟通方式冲突")
            )

        # 4. 计算综合匹配度
        match.compatibility_score = similarity_score * 0.6 + (100 - len(match.conflict_traits) * 15) * 0.4

        # 5. 生成建议
        if match.compatibility_score > 70:
            match.advice.append("人格特质匹配度高，相处自然")
        elif match.compatibility_score > 50:
            match.advice.append("人格特质有差异，需要互相理解")
        else:
            match.advice.append("人格差异较大，需要更多沟通磨合")

        for trait, desc in match.conflict_traits:
            match.advice.append(f"注意：{desc}")

        for trait, desc in match.complementary_traits:
            match.advice.append(f"优势：{desc}")

        return match


def create_big_five_report(profile: PersonalityProfile, is_self: bool = True) -> str:
    """生成大五人格报告"""
    subject = "你" if is_self else "对方"

    lines = []
    lines.append("=" * 50)
    lines.append(f"大五人格分析 - {subject}")
    lines.append("=" * 50)

    lines.append("\n[五维分数]")
    lines.append(f"  开放性 (O): {profile.openness:.1f} {'█' * int(profile.openness/10)}")
    lines.append(f"  尽责性 (C): {profile.conscientiousness:.1f} {'█' * int(profile.conscientiousness/10)}")
    lines.append(f"  外向性 (E): {profile.extraversion:.1f} {'█' * int(profile.extraversion/10)}")
    lines.append(f"  宜人性 (A): {profile.agreeableness:.1f} {'█' * int(profile.agreeableness/10)}")
    lines.append(f"  神经质 (N): {profile.neuroticism:.1f} {'█' * int(profile.neuroticism/10)}")

    if profile.dominant_traits:
        lines.append("\n[主导特质]")
        for trait in profile.dominant_traits:
            lines.append(f"  - {trait}")

    lines.append(f"\n[人格类型] {profile.personality_type}")
    lines.append(f"[沟通风格] {profile.communication_style}")
    lines.append(f"[关系倾向] {profile.relationship_tendency}")

    return "\n".join(lines)


def create_match_report(match: PersonalityMatch) -> str:
    """生成匹配分析报告"""
    lines = []
    lines.append("=" * 50)
    lines.append("人格匹配分析")
    lines.append("=" * 50)

    lines.append(f"\n[匹配度] {match.compatibility_score:.1f}/100")

    if match.complementary_traits:
        lines.append("\n[互补优势]")
        for trait, desc in match.complementary_traits:
            lines.append(f"  + {desc}")

    if match.conflict_traits:
        lines.append("\n[潜在冲突]")
        for trait, desc in match.conflict_traits:
            lines.append(f"  ! {desc}")

    if match.advice:
        lines.append("\n[建议]")
        for advice in match.advice:
            lines.append(f"  → {advice}")

    return "\n".join(lines)


__all__ = [
    'BigFiveTrait', 'PersonalityProfile', 'PersonalityMatch',
    'BigFiveAnalyzer', 'create_big_five_report', 'create_match_report'
]