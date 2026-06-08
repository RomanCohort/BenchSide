"""
算法对齐模块 - 与原系统 she-love-me 的算法保持一致

确保新系统计算的指标与原系统完全相同
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime


# ============================================================
# 原系统常量（从 stats_analyzer.py 复制）
# ============================================================

# 冷淡回复词库
COLD_WORDS = {
    "嗯", "哦", "好", "行", "哦哦", "嗯嗯", "好的", "ok", "OK", "好吧",
    "随便", "可以", "知道了", "知道", "哈", "哈哈", "em", "em...", "emmm"
}

# 早安/晚安关键词
GOODNIGHT_WORDS = {"晚安", "good night", "goodnight", "晚安啦", "晚安嗷", "晚安哦"}
GOODMORNING_WORDS = {"早安", "早上好", "早啊", "早哦", "good morning", "早", "早起了"}

# 对话间隔阈值（秒）: 超过此时间视为新对话开始
NEW_CONVERSATION_GAP = 3 * 3600  # 3小时

# 语言学词库
HEDGING_WORDS = ["也许", "可能", "感觉", "好像", "大概", "应该", "似乎", "觉得", "不确定", "说不定"]
CONDITIONAL_MARKERS = ["如果", "要是", "假如", "万一", "要不然", "若是", "倘若"]
POSITIVE_EMOTION_WORDS = [
    "开心", "高兴", "快乐", "幸福", "爱", "喜欢", "想你", "好想", "哈哈", "嘻嘻",
    "棒", "好棒", "厉害", "可爱", "美", "帅", "甜", "暖", "温柔", "期待",
    "谢谢", "感谢", "宝贝", "亲爱", "老公", "老婆", "宝宝"
]
NEGATIVE_EMOTION_WORDS = [
    "烦", "累", "难过", "伤心", "痛苦", "委屈", "生气", "愤怒", "失望", "绝望",
    "算了", "无所谓", "随便", "不想", "放弃", "好累", "心累", "难受", "伤", "哭"
]


@dataclass
class OriginalScores:
    """原系统的三个核心指数"""
    simp_index: int      # 主动指数 (0-100)
    loved_index: int     # 被爱指数 (0-100)
    cold_index: int      # 冷淡指数 (0-100)


class OriginalAlgorithm:
    """
    原系统算法实现

    与 stats_analyzer.py 中的 compute_scores 完全一致
    """

    @staticmethod
    def compute_simp_index(basic: Dict, initiative: Dict, reply: Dict,
                           bombing: Dict, unanswered: Dict, goodnight: Dict) -> int:
        """
        计算主动指数（与原系统完全一致）

        权重分配：
        - 消息占比: 20分
        - 主动发起: 25分
        - 回复速度: 20分
        - 连续轰炸: 15分
        - 晚安主动: 10分
        - 已读不回: 10分
        """
        total = basic["total_messages"]
        my_total = basic["my_messages"]

        simp_score = 0.0

        # 1. 消息占比（你发超过60%加分）
        my_ratio = my_total / total if total > 0 else 0.5
        simp_score += 20 * min(my_ratio / 0.7, 1.0)

        # 2. 主动发起占比
        total_starts = initiative["my_starts"] + initiative["their_starts"]
        my_start_ratio = initiative["my_starts"] / total_starts if total_starts > 0 else 0.5
        simp_score += 25 * min(my_start_ratio / 0.75, 1.0)

        # 3. 回复速度差（你比对方快多少）
        my_speed = reply["my_avg_seconds"]
        their_speed = reply["their_avg_seconds"]
        if their_speed > 0 and my_speed > 0:
            speed_ratio = their_speed / my_speed
            simp_score += 20 * min(speed_ratio / 10, 1.0)
        elif my_speed > 0 and their_speed == 0:
            simp_score += 20

        # 4. 连续轰炸
        bomb_ratio = bombing["my_bomb_count"] / max(total_starts, 1)
        simp_score += 15 * min(bomb_ratio / 0.3, 1.0)

        # 5. 晚安主动率
        total_goodnight = goodnight["my_goodnight"] + goodnight["their_goodnight"]
        if total_goodnight > 0:
            my_gn_ratio = goodnight["my_goodnight"] / total_goodnight
            simp_score += 10 * min(my_gn_ratio / 0.8, 1.0)

        # 6. 已读不回忍受
        if unanswered["my_unanswered"] > 5:
            simp_score += 10

        return min(int(simp_score), 100)

    @staticmethod
    def compute_loved_index(basic: Dict, initiative: Dict, reply: Dict,
                            goodnight: Dict, cold: Dict,
                            message_length: Dict) -> int:
        """
        计算被爱指数（与原系统完全一致）

        权重分配：
        - 对方消息占比: 20分
        - 对方主动发起: 25分
        - 对方回复速度: 20分
        - 对方消息长度: 15分
        - 对方说晚安: 10分
        - 对方不敷衍: 10分
        """
        total = basic["total_messages"]
        their_total = basic["their_messages"]

        loved_score = 0.0

        # 1. 对方消息占比
        their_ratio = their_total / total if total > 0 else 0.5
        loved_score += 20 * min(their_ratio / 0.5, 1.0)

        # 2. 对方主动发起
        total_starts = initiative["my_starts"] + initiative["their_starts"]
        loved_score += 25 * min(initiative["their_starts"] / max(total_starts * 0.4, 1), 1.0)

        # 3. 对方回复速度快
        my_speed = reply["my_avg_seconds"]
        their_speed = reply["their_avg_seconds"]
        if their_speed > 0 and my_speed > 0:
            their_responsiveness = my_speed / their_speed
            loved_score += 20 * min(their_responsiveness / 3, 1.0)

        # 4. 对方消息长（用心）
        my_len = message_length["my_avg_chars"]
        their_len = message_length["their_avg_chars"]
        if my_len > 0:
            len_ratio = their_len / my_len
            loved_score += 15 * min(len_ratio / 1.0, 1.0)

        # 5. 对方说晚安主动
        total_goodnight = goodnight["my_goodnight"] + goodnight["their_goodnight"]
        if total_goodnight > 0:
            their_gn_ratio = goodnight["their_goodnight"] / total_goodnight
            loved_score += 10 * min(their_gn_ratio / 0.6, 1.0)

        # 6. 对方不敷衍（冷淡少）
        their_cold_ratio = cold["their_cold_count"] / max(their_total, 1)
        loved_score += 10 * (1 - min(their_cold_ratio / 0.3, 1.0))

        return min(int(loved_score), 100)

    @staticmethod
    def compute_cold_index(basic: Dict, reply: Dict, cold: Dict) -> int:
        """
        计算冷淡指数（与原系统完全一致）

        权重分配：
        - 冷淡词占比: 40分
        - 回复速度慢: 30分
        - 消息比例失衡: 30分
        """
        my_total = basic["my_messages"]
        their_total = basic["their_messages"]

        cold_score = 0.0

        # 1. 冷淡词占比
        cold_ratio = cold["their_cold_count"] / max(their_total, 1)
        cold_score += 40 * min(cold_ratio / 0.3, 1.0)

        # 2. 对方回复速度慢
        my_speed = reply["my_avg_seconds"]
        their_speed = reply["their_avg_seconds"]
        if their_speed > 0 and my_speed > 0 and their_speed > my_speed * 5:
            cold_score += 30

        # 3. 消息比例失衡
        if their_total > 0 and my_total / their_total > 2:
            cold_score += 30

        return min(int(cold_score), 100)

    @staticmethod
    def compute_all_scores(stats: Dict) -> OriginalScores:
        """计算所有指数"""
        return OriginalScores(
            simp_index=OriginalAlgorithm.compute_simp_index(
                stats["basic"], stats["initiative"], stats["reply_speed"],
                stats["bombing"], stats["unanswered"], stats["goodnight"]
            ),
            loved_index=OriginalAlgorithm.compute_loved_index(
                stats["basic"], stats["initiative"], stats["reply_speed"],
                stats["goodnight"], stats["cold_response"], stats["message_length"]
            ),
            cold_index=OriginalAlgorithm.compute_cold_index(
                stats["basic"], stats["reply_speed"], stats["cold_response"]
            )
        )


class AlignedStatsCalculator:
    """
    对齐的统计计算器

    完全复刻原系统的统计逻辑
    """

    def __init__(self):
        pass

    def calculate_basic_stats(self, messages: List[Dict]) -> Dict:
        """计算基础统计"""
        valid = [m for m in messages if m["sender"] in ("me", "them")]
        my_msgs = [m for m in valid if m["sender"] == "me"]
        their_msgs = [m for m in valid if m["sender"] == "them"]
        total = len(valid)

        timestamps = [m["timestamp"] for m in valid]
        date_start = datetime.fromtimestamp(min(timestamps)).strftime("%Y-%m-%d") if timestamps else ""
        date_end = datetime.fromtimestamp(max(timestamps)).strftime("%Y-%m-%d") if timestamps else ""
        total_days = max((max(timestamps) - min(timestamps)) // 86400, 1) if timestamps else 1

        return {
            "total_messages": total,
            "my_messages": len(my_msgs),
            "their_messages": len(their_msgs),
            "my_ratio": round(len(my_msgs) / total, 3) if total > 0 else 0,
            "their_ratio": round(len(their_msgs) / total, 3) if total > 0 else 0,
            "date_range": [date_start, date_end],
            "total_days": total_days,
            "avg_daily": round(total / total_days, 1)
        }

    def calculate_initiative(self, messages: List[Dict]) -> Dict:
        """计算主动性（谁先发起对话）"""
        valid = [m for m in messages if m["sender"] in ("me", "them")]

        # 切分对话
        conversations = self._detect_conversations(valid)

        my_starts = sum(1 for c in conversations if c[0]["sender"] == "me")
        their_starts = sum(1 for c in conversations if c[0]["sender"] == "them")

        return {
            "my_starts": my_starts,
            "their_starts": their_starts,
            "my_start_ratio": round(my_starts / max(my_starts + their_starts, 1), 3)
        }

    def calculate_reply_speed(self, messages: List[Dict]) -> Dict:
        """计算回复速度"""
        valid = [m for m in messages if m["sender"] in ("me", "them")]

        my_reply_times = []
        their_reply_times = []
        last_sender = None
        last_time = None

        for msg in valid:
            if last_sender is not None and msg["sender"] != last_sender:
                gap = msg["timestamp"] - last_time
                if 10 <= gap <= 86400:  # 忽略太短和太长
                    if msg["sender"] == "me":
                        my_reply_times.append(gap)
                    else:
                        their_reply_times.append(gap)
            last_sender = msg["sender"]
            last_time = msg["timestamp"]

        avg_my = sum(my_reply_times) / len(my_reply_times) if my_reply_times else 0
        avg_their = sum(their_reply_times) / len(their_reply_times) if their_reply_times else 0

        return {
            "my_avg_seconds": round(avg_my),
            "their_avg_seconds": round(avg_their),
            "speed_ratio": round(avg_their / max(avg_my, 1), 1)
        }

    def calculate_bombing(self, messages: List[Dict]) -> Dict:
        """检测连续轰炸"""
        valid = [m for m in messages if m["sender"] in ("me", "them")]

        my_bombs = 0
        their_bombs = 0
        my_consecutive = 0
        their_consecutive = 0

        last_sender = None
        for msg in valid:
            sender = msg["sender"]
            if sender == last_sender:
                if sender == "me":
                    my_consecutive += 1
                else:
                    their_consecutive += 1
            else:
                if last_sender == "me" and my_consecutive >= 3:
                    my_bombs += 1
                if last_sender == "them" and their_consecutive >= 3:
                    their_bombs += 1
                my_consecutive = 1 if sender == "me" else 0
                their_consecutive = 1 if sender == "them" else 0
            last_sender = sender

        return {
            "my_bomb_count": my_bombs,
            "their_bomb_count": their_bombs
        }

    def calculate_cold_response(self, messages: List[Dict]) -> Dict:
        """检测冷淡回复"""
        text_msgs = [m for m in messages if m.get("type") == "text" and m["sender"] in ("me", "them")]

        my_cold = 0
        their_cold = 0

        for msg in text_msgs:
            content = msg["content"].strip()
            is_cold = content in COLD_WORDS or len(content) <= 2
            if is_cold:
                if msg["sender"] == "me":
                    my_cold += 1
                else:
                    their_cold += 1

        return {
            "my_cold_count": my_cold,
            "their_cold_count": their_cold
        }

    def calculate_unanswered(self, messages: List[Dict]) -> Dict:
        """检测未回复"""
        valid = [m for m in messages if m["sender"] in ("me", "them")]

        my_unanswered = 0
        their_unanswered = 0
        last_sender = None
        last_time = None

        for msg in valid:
            if last_sender is not None and msg["sender"] != last_sender:
                gap = msg["timestamp"] - last_time
                if gap > 7200:  # 超过2小时
                    if last_sender == "me":
                        my_unanswered += 1
                    else:
                        their_unanswered += 1
            last_sender = msg["sender"]
            last_time = msg["timestamp"]

        return {
            "my_unanswered": my_unanswered,
            "their_unanswered": their_unanswered
        }

    def calculate_goodnight(self, messages: List[Dict]) -> Dict:
        """检测晚安"""
        text_msgs = [m for m in messages if m.get("type") == "text" and m["sender"] in ("me", "them")]

        my_goodnight = 0
        their_goodnight = 0

        for msg in text_msgs:
            content = msg["content"].lower().strip()
            if any(w in content for w in GOODNIGHT_WORDS):
                if msg["sender"] == "me":
                    my_goodnight += 1
                else:
                    their_goodnight += 1

        return {
            "my_goodnight": my_goodnight,
            "their_goodnight": their_goodnight
        }

    def calculate_message_length(self, messages: List[Dict]) -> Dict:
        """计算消息长度"""
        text_msgs = [m for m in messages if m.get("type") == "text" and m["sender"] in ("me", "them")]

        my_text = [m for m in text_msgs if m["sender"] == "me"]
        their_text = [m for m in text_msgs if m["sender"] == "them"]

        my_avg = sum(len(m["content"]) for m in my_text) / len(my_text) if my_text else 0
        their_avg = sum(len(m["content"]) for m in their_text) / len(their_text) if their_text else 0

        return {
            "my_avg_chars": round(my_avg, 1),
            "their_avg_chars": round(their_avg, 1)
        }

    def _detect_conversations(self, messages: List[Dict]) -> List[List[Dict]]:
        """切分对话"""
        if not messages:
            return []

        conversations = []
        current = [messages[0]]

        for msg in messages[1:]:
            gap = msg["timestamp"] - current[-1]["timestamp"]
            if gap > NEW_CONVERSATION_GAP:
                conversations.append(current)
                current = [msg]
            else:
                current.append(msg)

        if current:
            conversations.append(current)

        return conversations

    def compute_full_stats(self, messages: List[Dict]) -> Dict:
        """计算完整统计（与原系统输出格式一致）"""
        stats = {
            "basic": self.calculate_basic_stats(messages),
            "initiative": self.calculate_initiative(messages),
            "reply_speed": self.calculate_reply_speed(messages),
            "message_length": self.calculate_message_length(messages),
            "bombing": self.calculate_bombing(messages),
            "cold_response": self.calculate_cold_response(messages),
            "unanswered": self.calculate_unanswered(messages),
            "goodnight": self.calculate_goodnight(messages)
        }

        # 计算三个核心指数
        scores = OriginalAlgorithm.compute_all_scores(stats)
        stats["scores"] = {
            "simp_index": scores.simp_index,
            "loved_index": scores.loved_index,
            "cold_index": scores.cold_index
        }

        return stats


def verify_alignment(original_stats: Dict, new_stats: Dict) -> Dict:
    """
    验证新旧系统计算结果是否一致

    Args:
        original_stats: 原系统 stats.json
        new_stats: 新系统计算结果

    Returns:
        对比结果
    """
    results = {
        "aligned": True,
        "differences": []
    }

    # 对比三个核心指数
    for key in ["simp_index", "loved_index", "cold_index"]:
        orig = original_stats.get("scores", {}).get(key, 0)
        new = new_stats.get("scores", {}).get(key, 0)

        if orig != new:
            results["aligned"] = False
            results["differences"].append({
                "metric": key,
                "original": orig,
                "new": new,
                "diff": new - orig
            })

    return results


__all__ = [
    'OriginalAlgorithm', 'OriginalScores', 'AlignedStatsCalculator',
    'verify_alignment', 'COLD_WORDS', 'GOODNIGHT_WORDS', 'GOODMORNING_WORDS'
]