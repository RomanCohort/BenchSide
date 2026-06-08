"""
关系类型定义 - 多场景支持

支持多种关系类型的差异化分析
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional


class RelationType(Enum):
    """关系类型枚举"""
    # 情感关系
    BOYFRIEND = "男朋友"          # 女对男
    GIRLFRIEND = "女朋友"        # 男对女
    CRUSH = "暗恋对象"
    PURSUING = "追求中"
    AMBIGUOUS = "暧昧阶段"
    EX = "前任"

    # 职业关系
    BOSS = "上级领导"
    COLLEAGUE = "同事伙伴"
    SUBORDINATE = "下属员工"
    CLIENT = "客户甲方"
    PARTNER = "合作伙伴"

    # 师生关系
    ADVISOR = "导师"
    TEACHER = "老师"
    STUDENT = "学生"
    SENIOR = "师兄师姐"
    JUNIOR = "学弟学妹"

    # 亲情关系
    PARENT = "父母"
    CHILD = "子女"
    SIBLING = "兄弟姐妹"

    # 友情关系
    BEST_FRIEND = "闺蜜/兄弟"
    FRIEND = "普通朋友"
    ONLINE_FRIEND = "网友"

    # 其他关系
    LANDLORD = "房东"
    DOCTOR = "医生"
    SERVICE = "服务人员"
    OTHER = "其他"


@dataclass
class RelationProfile:
    """
    关系画像配置

    不同关系类型有不同的：
    - 分析重点
    - 评分权重
    - 建议策略
    - 心理学框架
    """
    relation_type: RelationType
    display_name: str

    # 分析维度权重
    dimension_weights: Dict[str, float] = field(default_factory=dict)

    # 评分指标权重
    score_weights: Dict[str, float] = field(default_factory=dict)

    # 适用的心理学框架
    psychological_frameworks: List[str] = field(default_factory=list)

    # 建议策略
    suggestion_style: str = "supportive"  # supportive, professional, casual

    # 关键词/信号词
    signal_words: Dict[str, List[str]] = field(default_factory=dict)

    # 禁用建议（某些场景不适合）
    disabled_actions: List[str] = field(default_factory=list)


# ============================================================
# 关系画像配置库
# ============================================================

RELATION_PROFILES: Dict[RelationType, RelationProfile] = {

    # ============ 情感关系 ============

    RelationType.BOYFRIEND: RelationProfile(
        relation_type=RelationType.BOYFRIEND,
        display_name="男朋友",
        dimension_weights={
            "initiative": 0.25,      # 主动性
            "responsiveness": 0.20,  # 响应度
            "intimacy": 0.20,        # 亲密感
            "commitment": 0.15,      # 承诺感
            "support": 0.20          # 支持度
        },
        score_weights={
            "simp_index": 0.3,
            "loved_index": 0.4,
            "cold_index": -0.3
        },
        psychological_frameworks=[
            "attachment_theory",      # 依恋理论
            "sternberg_triangle",    # Sternberg三角
            "gottman_four_horsemen"  # Gottman四骑士
        ],
        suggestion_style="supportive",
        signal_words={
            "positive": ["爱你", "想你", "宝贝", "亲爱的", "晚安", "早安"],
            "negative": ["算了", "随便", "无所谓", "不想说了"],
            "commitment": ["未来", "结婚", "一起", "我们"],
            "cold": ["嗯", "哦", "好的", "知道了"]
        },
        disabled_actions=[]
    ),

    RelationType.GIRLFRIEND: RelationProfile(
        relation_type=RelationType.GIRLFRIEND,
        display_name="女朋友",
        dimension_weights={
            "initiative": 0.20,
            "responsiveness": 0.25,  # 女生更看重响应
            "intimacy": 0.25,
            "commitment": 0.15,
            "support": 0.15
        },
        score_weights={
            "simp_index": 0.25,
            "loved_index": 0.45,
            "cold_index": -0.3
        },
        psychological_frameworks=[
            "attachment_theory",
            "sternberg_triangle",
            "gottman_four_horsemen",
            "love_languages"         # 爱的语言（女生更在意）
        ],
        suggestion_style="supportive",
        signal_words={
            "positive": ["爱你", "想你", "宝贝", "亲爱的", "嘻嘻", "哈哈"],
            "negative": ["哼", "不理你了", "你猜", "随便"],
            "commitment": ["未来", "结婚", "宝宝", "我们"],
            "cold": ["哦", "嗯", "好的", "行吧"]
        }
    ),

    RelationType.CRUSH: RelationProfile(
        relation_type=RelationType.CRUSH,
        display_name="暗恋对象",
        dimension_weights={
            "initiative": 0.30,      # 主动最重要
            "responsiveness": 0.25,
            "interest_signals": 0.25, # 兴趣信号
            "opportunity": 0.20       # 机会把握
        },
        score_weights={
            "simp_index": 0.35,
            "loved_index": 0.35,
            "cold_index": -0.3
        },
        psychological_frameworks=[
            "attachment_theory",        # 焦虑型依恋检测
            "social_exchange_theory",  # 投入回报分析
            "gottman_four_horsemen",   # 危险信号检测
            "uncertainty_reduction"    # 不确定性减少理论
        ],
        suggestion_style="cautious",   # 谨慎风格
        signal_words={
            "positive": ["哈哈", "嗯嗯", "好呀", "可以"],
            "negative": ["...", "？", "哦"],
            "interest": ["你呢", "在干嘛", "最近怎么样"],
            # 舔狗信号词
            "simp_markers": ["[旺柴]", "[捂脸]", "[流泪]", "哈哈", "没事", "好吧"],
            # 冷淡信号词
            "cold_markers": ["嗯", "哦", "好的", "[表情]", "okk", "收到"]
        },
        disabled_actions=["suggest_meeting", "chase_harder"]  # 不约见面，不追更紧
    ),

    # ============ 职业关系 ============

    RelationType.BOSS: RelationProfile(
        relation_type=RelationType.BOSS,
        display_name="领导",
        dimension_weights={
            "responsiveness": 0.30,   # 响应最重要
            "professionalism": 0.25,  # 专业度
            "reliability": 0.25,      # 可靠性
            "communication": 0.20     # 沟通效率
        },
        score_weights={
            "simp_index": -0.2,       # 不应该太主动
            "loved_index": 0.3,
            "cold_index": -0.5        # 冷淡很危险
        },
        psychological_frameworks=[
            "power_dynamics",         # 权力动态
            "organizational_behavior" # 组织行为学
        ],
        suggestion_style="professional",
        signal_words={
            "urgent": ["尽快", "马上", "紧急", "今天"],
            "task": ["完成", "汇报", "确认", "跟进"],
            "approval": ["好的", "收到", "明白", "没问题"],
            "warning": ["？", "...", "你自己看着办"]
        },
        disabled_actions=["send_meme", "share_life", "take_space"]
    ),

    RelationType.COLLEAGUE: RelationProfile(
        relation_type=RelationType.COLLEAGUE,
        display_name="同事",
        dimension_weights={
            "cooperation": 0.30,      # 合作
            "communication": 0.25,
            "support": 0.25,
            "boundary": 0.20          # 边界感
        },
        score_weights={
            "simp_index": 0.2,
            "loved_index": 0.3,
            "cold_index": -0.3
        },
        psychological_frameworks=[
            "social_exchange_theory",
            "boundary_theory"
        ],
        suggestion_style="professional",
        signal_words={
            "work": ["帮忙", "请教", "一起", "合作"],
            "casual": ["哈哈", "谢谢", "辛苦了"],
            "boundary": ["下班", "周末", "私事"]
        },
        disabled_actions=["suggest_meeting"]  # 不约见面
    ),

    RelationType.CLIENT: RelationProfile(
        relation_type=RelationType.CLIENT,
        display_name="客户",
        dimension_weights={
            "responsiveness": 0.35,   # 响应最重要
            "professionalism": 0.30,
            "satisfaction": 0.20,
            "trust": 0.15
        },
        score_weights={
            "simp_index": -0.1,
            "loved_index": 0.4,
            "cold_index": -0.5
        },
        psychological_frameworks=[
            "service_quality_model",
            "trust_building"
        ],
        suggestion_style="professional",
        signal_words={
            "urgent": ["急", "今天", "明天", "尽快"],
            "satisfied": ["好的", "谢谢", "满意", "不错"],
            "complaint": ["问题", "不对", "怎么回事", "不满意"]
        },
        disabled_actions=["send_meme", "share_life", "take_space", "be_supportive"]
    ),

    # ============ 师生关系 ============

    RelationType.ADVISOR: RelationProfile(
        relation_type=RelationType.ADVISOR,
        display_name="导师",
        dimension_weights={
            "respect": 0.25,          # 尊重
            "responsiveness": 0.25,
            "progress_report": 0.25,  # 进度汇报
            "initiative": 0.25        # 主动汇报
        },
        score_weights={
            "simp_index": 0.3,        # 需要主动汇报
            "loved_index": 0.3,
            "cold_index": -0.4
        },
        psychological_frameworks=[
            "mentorship_model",
            "academic_relationship"
        ],
        suggestion_style="respectful",
        signal_words={
            "task": ["论文", "实验", "项目", "进度"],
            "meeting": ["见面", "讨论", "汇报", "组会"],
            "approval": ["好的", "收到", "明白", "我会"],
            "warning": ["？", "进度", "怎么回事"]
        },
        disabled_actions=["send_meme", "suggest_meeting"]
    ),

    RelationType.TEACHER: RelationProfile(
        relation_type=RelationType.TEACHER,
        display_name="老师",
        dimension_weights={
            "respect": 0.30,
            "responsiveness": 0.25,
            "engagement": 0.25,
            "gratitude": 0.20
        },
        score_weights={
            "simp_index": 0.2,
            "loved_index": 0.35,
            "cold_index": -0.3
        },
        psychological_frameworks=[
            "teacher_student_relationship"
        ],
        suggestion_style="respectful",
        signal_words={
            "academic": ["作业", "考试", "成绩", "课程"],
            "help": ["请教", "帮忙", "问题"],
            "thanks": ["谢谢老师", "辛苦了"]
        },
        disabled_actions=["send_meme", "share_life"]
    ),

    # ============ 亲情关系 ============

    RelationType.PARENT: RelationProfile(
        relation_type=RelationType.PARENT,
        display_name="父母",
        dimension_weights={
            "care": 0.30,             # 关心
            "communication": 0.25,
            "filial_piety": 0.25,     # 孝顺
            "sharing": 0.20           # 分享生活
        },
        score_weights={
            "simp_index": 0.25,
            "loved_index": 0.4,
            "cold_index": -0.35
        },
        psychological_frameworks=[
            "family_systems_theory",
            "attachment_theory"
        ],
        suggestion_style="warm",
        signal_words={
            "care": ["身体", "吃饭", "休息", "注意"],
            "worry": ["担心", "别太累", "早点睡"],
            "love": ["想你", "爱你", "回家"]
        },
        disabled_actions=["take_space"]  # 不对父母冷淡
    ),

    # ============ 友情关系 ============

    RelationType.BEST_FRIEND: RelationProfile(
        relation_type=RelationType.BEST_FRIEND,
        display_name="闺蜜/兄弟",
        dimension_weights={
            "trust": 0.25,
            "sharing": 0.25,
            "support": 0.25,
            "fun": 0.25
        },
        score_weights={
            "simp_index": 0.2,
            "loved_index": 0.4,
            "cold_index": -0.2
        },
        psychological_frameworks=[
            "friendship_quality_model"
        ],
        suggestion_style="casual",
        signal_words={
            "close": ["姐妹", "兄弟", "宝贝", "亲爱的"],
            "support": ["帮你", "陪你", "一起", "来"],
            "fun": ["哈哈哈", "笑死", "太逗了"]
        }
    ),

    RelationType.FRIEND: RelationProfile(
        relation_type=RelationType.FRIEND,
        display_name="朋友",
        dimension_weights={
            "friendliness": 0.30,
            "reliability": 0.25,
            "boundary": 0.25,
            "reciprocity": 0.20
        },
        score_weights={
            "simp_index": 0.25,
            "loved_index": 0.35,
            "cold_index": -0.3
        },
        psychological_frameworks=[
            "social_exchange_theory"
        ],
        suggestion_style="casual",
        signal_words={
            "friendly": ["哈哈", "好的", "谢谢", "下次"],
            "distant": ["哦", "嗯", "好的"]
        }
    ),

    # ============ 默认配置 ============
    RelationType.OTHER: RelationProfile(
        relation_type=RelationType.OTHER,
        display_name="其他",
        dimension_weights={
            "communication": 0.30,
            "responsiveness": 0.30,
            "friendliness": 0.40
        },
        score_weights={
            "simp_index": 0.25,
            "loved_index": 0.35,
            "cold_index": -0.3
        },
        psychological_frameworks=[],
        suggestion_style="neutral",
        signal_words={},
        disabled_actions=[]
    )
}


def get_relation_profile(relation_type: RelationType) -> RelationProfile:
    """获取关系画像"""
    return RELATION_PROFILES.get(relation_type, RELATION_PROFILES[RelationType.OTHER])


def detect_relation_type(contact_name: str,
                         messages: List[Dict],
                         user_hint: Optional[str] = None) -> RelationType:
    """
    自动检测关系类型

    Args:
        contact_name: 联系人名称/备注
        messages: 消息历史
        user_hint: 用户提示

    Returns:
        推测的关系类型
    """
    # 1. 用户明确指定
    if user_hint:
        for rt in RelationType:
            if user_hint in rt.value:
                return rt

    # 2. 从备注名推断
    name_lower = contact_name.lower()

    # 亲情
    if any(kw in name_lower for kw in ["妈", "爸", "父", "母", "爹", "娘"]):
        return RelationType.PARENT
    if any(kw in name_lower for kw in ["姐", "妹", "哥", "弟", "兄"]):
        return RelationType.SIBLING

    # 师生
    if any(kw in name_lower for kw in ["导师", "老师", "教授", "导员"]):
        return RelationType.ADVISOR
    if any(kw in name_lower for kw in ["师兄", "师姐", "学长", "学姐"]):
        return RelationType.SENIOR

    # 职业
    if any(kw in name_lower for kw in ["总", "经理", "主管", "领导", "老板"]):
        return RelationType.BOSS
    if any(kw in name_lower for kw in ["同事", "同事"]):
        return RelationType.COLLEAGUE
    if any(kw in name_lower for kw in ["客户", "甲方"]):
        return RelationType.CLIENT

    # 友情
    if any(kw in name_lower for kw in ["闺蜜", "兄弟", "基友", "死党"]):
        return RelationType.BEST_FRIEND

    # 情感
    if any(kw in name_lower for kw in ["男朋友", "男友", "老公", "对象"]):
        return RelationType.BOYFRIEND
    if any(kw in name_lower for kw in ["女朋友", "女友", "老婆", "对象"]):
        return RelationType.GIRLFRIEND

    # 3. 默认返回其他
    return RelationType.OTHER


__all__ = [
    'RelationType', 'RelationProfile', 'RELATION_PROFILES',
    'get_relation_profile', 'detect_relation_type'
]