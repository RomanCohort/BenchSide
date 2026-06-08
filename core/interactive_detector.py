"""
交互式角色判断系统

通过多维度信号自动推断关系类型，并提供交互式确认机制
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import re

from .relation_types import RelationType, RelationProfile, RELATION_PROFILES


@dataclass
class RelationSignal:
    """关系信号"""
    signal_type: str      # signal类型
    evidence: str         # 证据（具体消息片段）
    confidence: float     # 置信度 (0-1)
    target_relation: RelationType  # 指向的关系类型


@dataclass
class RelationInference:
    """关系推断结果"""
    predicted_type: RelationType
    confidence: float
    signals: List[RelationSignal] = field(default_factory=list)
    alternative_types: List[Tuple[RelationType, float]] = field(default_factory=list)
    needs_confirmation: bool = True
    reasoning: str = ""
    suggested_questions: List[str] = field(default_factory=list)


class InteractiveRelationDetector:
    """
    交互式关系类型检测器

    通过分析：
    1. 联系人备注/名称
    2. 对话内容特征
    3. 消息行为模式
    4. 时间分布规律
    来推断关系类型
    """

    # 关系类型信号词库
    RELATION_SIGNALS = {
        # 情感关系信号
        "romantic": {
            "keywords": [
                "爱你", "想你", "宝贝", "亲爱的", "老公", "老婆",
                "晚安", "早安", "亲亲", "抱抱", "么么哒",
                "我们", "未来", "结婚", "一起", "永远"
            ],
            "emojis": ["❤", "💕", "💖", "💗", "😘", "💋", "🥰", "😍"],
            "time_pattern": "late_night",  # 深夜聊天
            "relation_types": [RelationType.BOYFRIEND, RelationType.GIRLFRIEND]
        },

        "ambiguous": {
            "keywords": [
                "哈哈", "嗯嗯", "好呀", "可以哦", "行吧",
                "那...", "其实...", "就是说...", "不知道"
            ],
            "behavior": "mixed_initiative",
            "relation_types": [RelationType.CRUSH, RelationType.AMBIGUOUS, RelationType.PURSUING]
        },

        "pursuing": {
            "keywords": [
                "在吗", "干嘛呢", "最近怎么样", "有空吗",
                "周末", "一起", "出来", "约"
            ],
            "behavior": "high_initiative",
            "relation_types": [RelationType.PURSUING, RelationType.CRUSH]
        },

        # 职业关系信号
        "professional": {
            "keywords": [
                "收到", "好的", "明白", "没问题", "我会", "确认",
                "汇报", "进度", "完成", "任务", "工作", "项目",
                "尽快", "今天", "明天", "紧急", "急"
            ],
            "emojis": ["👍", "👌", "💪", "🙏"],
            "time_pattern": "work_hours",  # 工作时间
            "relation_types": [RelationType.BOSS, RelationType.COLLEAGUE, RelationType.CLIENT]
        },

        "boss_signals": {
            "keywords": [
                "尽快", "马上", "今天", "明天", "本周",
                "汇报", "结果", "进度", "情况", "问题",
                "怎么", "为什么", "怎么回事"
            ],
            "behavior": "them_dominant",
            "relation_types": [RelationType.BOSS]
        },

        "client_signals": {
            "keywords": [
                "需求", "要求", "希望", "满意", "不满意",
                "价格", "费用", "报价", "合同", "付款",
                "投诉", "问题", "服务"
            ],
            "relation_types": [RelationType.CLIENT]
        },

        # 师生关系信号
        "academic": {
            "keywords": [
                "论文", "实验", "研究", "项目", "课题",
                "导师", "老师", "教授", "师兄", "师姐",
                "毕业", "答辩", "发表", "投稿"
            ],
            "relation_types": [RelationType.ADVISOR, RelationType.TEACHER, RelationType.SENIOR]
        },

        "advisor_signals": {
            "keywords": [
                "论文", "实验", "数据", "进度", "组会",
                "下周", "讨论", "修改", "完善", "继续"
            ],
            "behavior": "me_initiative",
            "relation_types": [RelationType.ADVISOR]
        },

        # 亲情关系信号
        "family": {
            "keywords": [
                "妈", "爸", "爹", "娘", "父", "母",
                "儿", "女", "娃", "宝", "孩子",
                "回家", "吃饭", "身体", "健康", "注意",
                "想你", "爱你", "担心", "牵挂"
            ],
            "relation_types": [RelationType.PARENT, RelationType.SIBLING]
        },

        "parent_signals": {
            "keywords": [
                "吃了吗", "身体", "休息", "早点睡", "别太累",
                "注意", "小心", "多穿", "别感冒",
                "什么时候回来", "想你了", "爸妈"
            ],
            "behavior": "them_caring",
            "relation_types": [RelationType.PARENT]
        },

        # 友情关系信号
        "friendship": {
            "keywords": [
                "哈哈哈", "笑死", "太逗了", "绝了",
                "姐妹", "兄弟", "哥们", "闺蜜",
                "出来", "聚聚", "喝酒", "吃饭",
                "帮你", "陪你", "一起", "走"
            ],
            "emojis": ["😂", "🤣", "😏", "😜", "🤪", "🙌"],
            "relation_types": [RelationType.BEST_FRIEND, RelationType.FRIEND]
        }
    }

    # 时间模式定义
    TIME_PATTERNS = {
        "late_night": (22, 6),    # 22:00 - 06:00 深夜
        "work_hours": (9, 18),    # 09:00 - 18:00 工作时间
        "evening": (18, 22),      # 18:00 - 22:00 晚上
        "morning": (6, 9)         # 06:00 - 09:00 早晨
    }

    # 交互式确认问题模板
    CONFIRMATION_QUESTIONS = {
        RelationType.BOSS: [
            "这个人是不是你的领导/上级？",
            "你们的关系是工作相关的吗？"
        ],
        RelationType.ADVISOR: [
            "这个人是不是你的导师/指导老师？",
            "你们主要讨论学术/研究相关内容吗？"
        ],
        RelationType.BOYFRIEND: [
            "这个人是不是你的男朋友？",
            "你们是恋爱关系吗？"
        ],
        RelationType.GIRLFRIEND: [
            "这个人是不是你的女朋友？",
            "你们是恋爱关系吗？"
        ],
        RelationType.PARENT: [
            "这个人是不是你的父母？",
            "这是家人吗？"
        ],
        RelationType.CRUSH: [
            "这个人是不是你暗恋/追求的对象？",
            "你们还没有确立恋爱关系吗？"
        ],
        RelationType.BEST_FRIEND: [
            "这个人是不是你最好的朋友？",
            "你们关系很亲密但不是恋爱关系吗？"
        ]
    }

    def __init__(self):
        self.inference_history: List[RelationInference] = []

    def detect(self,
               contact_name: str,
               messages: List[Dict],
               stats: Optional[Dict] = None,
               user_gender: Optional[str] = None) -> RelationInference:
        """
        综合检测关系类型

        Args:
            contact_name: 联系人名称/备注
            messages: 消息历史
            stats: 统计数据
            user_gender: 用户性别（用于判断男朋友/女朋友）

        Returns:
            关系推断结果
        """
        signals: List[RelationSignal] = []

        # 1. 从名称推断
        name_signals = self._infer_from_name(contact_name)
        signals.extend(name_signals)

        # 2. 从消息内容推断
        content_signals = self._infer_from_content(messages)
        signals.extend(content_signals)

        # 3. 从行为模式推断
        if stats:
            behavior_signals = self._infer_from_behavior(stats)
            signals.extend(behavior_signals)

        # 4. 从时间分布推断
        time_signals = self._infer_from_time(messages)
        signals.extend(time_signals)

        # 5. 综合推断
        inference = self._synthesize_inference(signals, user_gender)

        # 记录历史
        self.inference_history.append(inference)

        return inference

    def _infer_from_name(self, name: str) -> List[RelationSignal]:
        """从联系人名称推断"""
        signals = []
        name_lower = name.lower()

        # 检查各关系类型的名称模式
        name_patterns = {
            RelationType.PARENT: ["妈", "爸", "爹", "娘", "父", "母", "老爸", "老妈", "爸爸妈妈"],
            RelationType.BOSS: ["总", "经理", "主管", "领导", "老板", "主任", "总监", "部长"],
            RelationType.ADVISOR: ["导师", "老师", "教授", "导员", "辅导员", "指导老师"],
            RelationType.SENIOR: ["师兄", "师姐", "学长", "学姐"],
            RelationType.BEST_FRIEND: ["闺蜜", "兄弟", "基友", "死党", "铁哥们", "好姐妹"],
            RelationType.BOYFRIEND: ["男朋友", "男友", "老公", "对象", "爱人", "亲爱的"],
            RelationType.GIRLFRIEND: ["女朋友", "女友", "老婆", "对象", "宝贝", "亲爱的"],
            RelationType.CLIENT: ["客户", "甲方", "老板", "顾客", "买家"],
            RelationType.SIBLING: ["姐", "妹", "哥", "弟", "兄", "姐姐", "妹妹", "哥哥", "弟弟"]
        }

        for rel_type, patterns in name_patterns.items():
            for pattern in patterns:
                if pattern in name_lower:
                    signals.append(RelationSignal(
                        signal_type="name_pattern",
                        evidence=f"名称包含'{pattern}'",
                        confidence=0.8,
                        target_relation=rel_type
                    ))
                    break

        return signals

    def _infer_from_content(self, messages: List[Dict]) -> List[RelationSignal]:
        """从消息内容推断"""
        signals = []

        if not messages:
            return signals

        # 提取文本消息
        text_messages = [m.get("content", "") for m in messages
                         if m.get("type") == "text" and m.get("sender") == "them"]

        # 合并文本
        all_text = " ".join(text_messages).lower()

        # 检查信号词库
        for signal_name, signal_config in self.RELATION_SIGNALS.items():
            keywords = signal_config.get("keywords", [])

            # 统计关键词出现次数
            matches = []
            for kw in keywords:
                if kw in all_text:
                    count = all_text.count(kw)
                    matches.append((kw, count))

            if matches:
                # 计算置信度
                total_matches = sum(c for _, c in matches)
                confidence = min(total_matches / 10, 0.7)

                # 获取指向的关系类型
                target_relations = signal_config.get("relation_types", [])

                if target_relations:
                    signals.append(RelationSignal(
                        signal_type="content_keywords",
                        evidence=f"关键词匹配: {matches[:3]}",
                        confidence=confidence,
                        target_relation=target_relations[0]
                    ))

        # 检查表情符号
        emojis = signal_config.get("emojis", [])
        if emojis:
            emoji_count = sum(all_text.count(e) for e in emojis)
            if emoji_count > 5:
                signals.append(RelationSignal(
                    signal_type="emoji_pattern",
                    evidence=f"表情符号频繁使用: {emoji_count}次",
                    confidence=0.5,
                    target_relation=RelationType.BEST_FRIEND  # 友情
                ))

        return signals

    def _infer_from_behavior(self, stats: Dict) -> List[RelationSignal]:
        """从行为模式推断"""
        signals = []

        # 主动性分析
        my_start_ratio = stats.get("initiative", {}).get("my_start_ratio", 0.5)

        if my_start_ratio > 0.7:
            # 高主动性
            signals.append(RelationSignal(
                signal_type="behavior_pattern",
                evidence=f"主动发起{my_start_ratio*100:.0f}%的对话",
                confidence=0.6,
                target_relation=RelationType.PURSUING
            ))
        elif my_start_ratio < 0.3:
            # 低主动性（对方主导）
            signals.append(RelationSignal(
                signal_type="behavior_pattern",
                evidence=f"对方主动发起{(1-my_start_ratio)*100:.0f}%的对话",
                confidence=0.5,
                target_relation=RelationType.BOSS
            ))

        # 回复速度分析
        their_speed = stats.get("reply_speed", {}).get("their_avg_seconds", 0)
        my_speed = stats.get("reply_speed", {}).get("my_avg_seconds", 0)

        if their_speed < 60 and my_speed > 300:
            # 对方秒回，我慢回
            signals.append(RelationSignal(
                signal_type="reply_pattern",
                evidence="对方回复很快",
                confidence=0.4,
                target_relation=RelationType.PARENT  # 父母关心
            ))
        elif their_speed > my_speed * 3:
            # 对方慢回
            signals.append(RelationSignal(
                signal_type="reply_pattern",
                evidence="对方回复较慢",
                confidence=0.3,
                target_relation=RelationType.BOSS  # 领导忙
            ))

        # 冷淡词分析
        cold_ratio = stats.get("cold_response", {}).get("their_cold_count", 0) / \
                     max(stats.get("basic", {}).get("their_messages", 1), 1)

        if cold_ratio > 0.3:
            signals.append(RelationSignal(
                signal_type="cold_pattern",
                evidence="对方回复较为冷淡",
                confidence=0.3,
                target_relation=RelationType.CRUSH  # 可能是追求对象
            ))

        return signals

    def _infer_from_time(self, messages: List[Dict]) -> List[RelationSignal]:
        """从时间分布推断"""
        signals = []

        if not messages:
            return signals

        # 统计各时间段的消息分布
        from collections import Counter
        hour_counts = Counter()

        for m in messages:
            if "timestamp" in m:
                from datetime import datetime
                hour = datetime.fromtimestamp(m["timestamp"]).hour
                hour_counts[hour] += 1

        if not hour_counts:
            return signals

        # 找出最活跃时间段
        total = sum(hour_counts.values())
        late_night_count = sum(hour_counts[h] for h in range(22, 24)) + \
                          sum(hour_counts[h] for h in range(0, 6))
        work_hours_count = sum(hour_counts[h] for h in range(9, 18))

        late_night_ratio = late_night_count / total if total > 0 else 0
        work_hours_ratio = work_hours_count / total if total > 0 else 0

        if late_night_ratio > 0.3:
            # 深夜聊天多
            signals.append(RelationSignal(
                signal_type="time_pattern",
                evidence=f"深夜聊天占比{late_night_ratio*100:.0f}%",
                confidence=0.5,
                target_relation=RelationType.BOYFRIEND  # 情感关系
            ))

        if work_hours_ratio > 0.6:
            # 工作时间聊天多
            signals.append(RelationSignal(
                signal_type="time_pattern",
                evidence=f"工作时间聊天占比{work_hours_ratio*100:.0f}%",
                confidence=0.6,
                target_relation=RelationType.BOSS  # 职业关系
            ))

        return signals

    def _synthesize_inference(self,
                               signals: List[RelationSignal],
                               user_gender: Optional[str]) -> RelationInference:
        """综合推断"""
        if not signals:
            # 无信号，默认为其他
            return RelationInference(
                predicted_type=RelationType.OTHER,
                confidence=0.3,
                signals=[],
                needs_confirmation=True,
                reasoning="未能检测到明确的关系信号，建议手动确认",
                suggested_questions=["请告诉我这个联系人对你来说是什么关系？"]
            )

        # 统计各关系类型的信号支持
        relation_scores: Dict[RelationType, float] = {}

        for signal in signals:
            rel = signal.target_relation
            if rel not in relation_scores:
                relation_scores[rel] = 0.0
            relation_scores[rel] += signal.confidence

        # 排序
        sorted_relations = sorted(
            relation_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 获取最可能的关系类型
        predicted_type = sorted_relations[0][0]
        confidence = min(sorted_relations[0][1] / 3, 1.0)  # 归一化

        # 处理男朋友/女朋友的性别区分
        if user_gender:
            if predicted_type == RelationType.BOYFRIEND and user_gender == "female":
                pass  # 正确，女性对男性
            elif predicted_type == RelationType.GIRLFRIEND and user_gender == "male":
                pass  # 正确，男性对女性
            elif predicted_type == RelationType.BOYFRIEND:
                # 如果检测到男朋友但用户是男性，可能是女朋友
                if user_gender == "male":
                    predicted_type = RelationType.GIRLFRIEND

        # 备选类型
        alternative_types = sorted_relations[1:4] if len(sorted_relations) > 1 else []

        # 是否需要确认
        needs_confirmation = confidence < 0.7

        # 生成推理说明
        reasoning = self._generate_reasoning(signals, predicted_type, confidence)

        # 生成确认问题
        suggested_questions = self.CONFIRMATION_QUESTIONS.get(predicted_type, [
            f"请确认这个人是否是【{RELATION_PROFILES[predicted_type].display_name}】？"
        ])

        return RelationInference(
            predicted_type=predicted_type,
            confidence=confidence,
            signals=signals,
            alternative_types=alternative_types,
            needs_confirmation=needs_confirmation,
            reasoning=reasoning,
            suggested_questions=suggested_questions
        )

    def _generate_reasoning(self,
                             signals: List[RelationSignal],
                             predicted_type: RelationType,
                             confidence: float) -> str:
        """生成推理说明"""
        lines = []

        lines.append(f"推断关系类型: {RELATION_PROFILES[predicted_type].display_name}")
        lines.append(f"置信度: {confidence*100:.0f}%")
        lines.append("")
        lines.append("检测到的信号:")
        for signal in signals:
            if signal.target_relation == predicted_type:
                lines.append(f"  - [{signal.signal_type}] {signal.evidence} (置信度: {signal.confidence*100:.0f}%)")

        return "\n".join(lines)

    def confirm_relation(self,
                         inference: RelationInference,
                         confirmed_type: RelationType) -> RelationProfile:
        """
        用户确认关系类型

        Args:
            inference: 推断结果
            confirmed_type: 用户确认的类型

        Returns:
            关系画像
        """
        # 获取关系画像
        profile = RELATION_PROFILES.get(confirmed_type, RELATION_PROFILES[RelationType.OTHER])

        # 更新置信度（如果确认与预测一致）
        if confirmed_type == inference.predicted_type:
            profile = RelationProfile(
                relation_type=profile.relation_type,
                display_name=profile.display_name,
                dimension_weights=profile.dimension_weights,
                score_weights=profile.score_weights,
                psychological_frameworks=profile.psychological_frameworks,
                suggestion_style=profile.suggestion_style,
                signal_words=profile.signal_words,
                disabled_actions=profile.disabled_actions
            )

        return profile

    def interactive_detect(self,
                           contact_name: str,
                           messages: List[Dict],
                           stats: Optional[Dict] = None) -> Tuple[RelationInference, str]:
        """
        交互式检测（返回用户需要回答的问题）

        Returns:
            (推断结果, 用户需要回答的问题)
        """
        inference = self.detect(contact_name, messages, stats)

        if inference.needs_confirmation:
            question = inference.suggested_questions[0]
        else:
            question = f"检测到关系类型为【{RELATION_PROFILES[inference.predicted_type].display_name}】，是否正确？"

        return inference, question


class RelationEvolutionTracker:
    """
    关系演变追踪器

    追踪关系在不同时间点的变化
    """

    def __init__(self):
        self.evolution_history: List[Dict] = []

    def track(self,
              timestamp: float,
              relation_type: RelationType,
              scores: Dict,
              signals: List[RelationSignal]):
        """记录关系状态"""
        self.evolution_history.append({
            "timestamp": timestamp,
            "relation_type": relation_type,
            "scores": scores,
            "signals": signals
        })

    def detect_evolution(self) -> Optional[str]:
        """检测关系演变"""
        if len(self.evolution_history) < 2:
            return None

        # 比较最近两次记录
        recent = self.evolution_history[-1]
        previous = self.evolution_history[-2]

        # 检查关系类型变化
        if recent["relation_type"] != previous["relation_type"]:
            return f"关系类型从{previous['relation_type'].value}变为{recent['relation_type'].value}"

        # 检查指标变化
        if "scores" in recent and "scores" in previous:
            loved_change = recent["scores"].get("loved_index", 0) - previous["scores"].get("loved_index", 0)
            if loved_change > 5:
                return "关系有所升温"
            elif loved_change < -5:
                return "关系有所降温"

        return None

    def predict_next_stage(self) -> Optional[RelationType]:
        """预测下一阶段"""
        if len(self.evolution_history) < 3:
            return None

        # 分析趋势
        types = [h["relation_type"] for h in self.evolution_history[-3:]]
        scores = [h["scores"].get("loved_index", 50) for h in self.evolution_history[-3:]]

        # 如果指标持续上升，可能进入下一阶段
        if all(scores[i] < scores[i+1] for i in range(len(scores)-1)):
            # 预测升级
            current = types[-1]
            evolution_map = {
                RelationType.CRUSH: RelationType.AMBIGUOUS,
                RelationType.AMBIGUOUS: RelationType.BOYFRIEND,
                RelationType.PURSUING: RelationType.AMBIGUOUS
            }
            return evolution_map.get(current)

        return None


__all__ = [
    'InteractiveRelationDetector', 'RelationInference', 'RelationSignal',
    'RelationEvolutionTracker'
]