"""
关系类型分类系统

人际关系不只是恋爱，还包括：
- 亲情关系：父母、子女、兄弟姐妹、亲戚
- 友情关系：普通朋友、闺蜜/兄弟、同事朋友
- 职场关系：上下级、同事、客户、合作伙伴
- 师生关系：老师、学生、导师、学徒
- 恋爱关系：暧昧、热恋、稳定、分手边缘
- 陌生人：初次接触、泛泛之交

每种关系类型有不同的：
- 互动模式特征
- 心理学框架适用性
- 分析维度权重
- 建议策略
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import re


class RelationCategory(Enum):
    """关系大类"""
    FAMILY = "亲情"
    FRIENDSHIP = "友情"
    WORK = "职场"
    MENTORSHIP = "师生"
    ROMANTIC = "恋爱"
    ACQUAINTANCE = "泛交"


class FamilyRelation(Enum):
    """亲情关系"""
    PARENT = "父母"
    CHILD = "子女"
    SIBLING = "兄弟姐妹"
    GRANDPARENT = "祖父母"
    GRANDCHILD = "孙辈"
    RELATIVE = "亲戚"


class FriendshipRelation(Enum):
    """友情关系"""
    CLOSE_FRIEND = "闺蜜/兄弟"
    GOOD_FRIEND = "好友"
    CASUAL_FRIEND = "普通朋友"
    WORK_FRIEND = "同事朋友"
    ONLINE_FRIEND = "网友"


class WorkRelation(Enum):
    """职场关系"""
    SUPERIOR = "上级"
    SUBORDINATE = "下属"
    COLLEAGUE = "同事"
    CLIENT = "客户"
    PARTNER = "合作伙伴"
    COMPETITOR = "竞争对手"


class MentorshipRelation(Enum):
    """师生关系"""
    TEACHER = "老师"
    STUDENT = "学生"
    MENTOR = "导师"
    MENTEE = "学徒"


class RomanticRelation(Enum):
    """恋爱关系"""
    CRUSH = "暗恋"
    PURSUING = "追求中"
    AMBIGUOUS = "暧昧期"
    DATING = "热恋期"
    STABLE = "稳定期"
    ENGAGED = "婚约"
    MARRIED = "已婚"
    SEPARATED = "分居"
    DIVORCED = "离异"
    EX = "前任"


class AcquaintanceRelation(Enum):
    """泛交关系"""
    NEW_CONTACT = "新认识"
    CASUAL = "泛泛之交"
    SOCIAL = "社交圈"


@dataclass
class RelationTypeResult:
    """关系类型识别结果"""
    category: RelationCategory = RelationCategory.ACQUAINTANCE
    specific_type: str = ""
    confidence: float = 0.5
    indicators: List[str] = field(default_factory=list)
    analysis_framework: str = ""  # 适用的分析框架


class RelationTypeDetector:
    """
    关系类型检测器

    根据聊天内容、称呼、互动模式识别关系类型
    """

    # ============ 关系指示词 ============

    # 亲情称呼
    FAMILY_TERMS = {
        'parent': ['爸', '妈妈', '老妈', '老爸', '爹', '娘', 'father', 'mother', 'mom', 'dad'],
        'child': ['儿子', '女儿', '娃', '孩子', '宝贝', 'son', 'daughter'],
        'sibling': ['哥', '姐', '弟', '妹', '哥哥', '姐姐', '弟弟', '妹妹', 'brother', 'sister'],
        'grandparent': ['爷爷', '奶奶', '外公', '外婆', '姥姥', '姥爷'],
        'relative': ['舅', '叔', '伯', '姨', '姑', '表', '堂', '侄', '外甥']
    }

    # 恋爱称呼/词汇
    ROMANTIC_TERMS = {
        'direct': ['老公', '老婆', '男朋友', '女朋友', '男友', '女友', '对象', '另一半',
                   '亲爱的', '宝贝', 'honey', 'baby', 'babe'],
        'indirect': ['喜欢你', '爱你', '想你', '在一起', '分手', '表白', '追求',
                     '暧昧', '暗恋', '约会', '约会', '谈恋爱'],
        'future': ['结婚', '婚礼', '未来', '以后', '一辈子', '孩子']
    }

    # 职场称呼/词汇
    WORK_TERMS = {
        'superior': ['老板', '领导', '经理', '总监', '总', '主任', 'chief', 'boss', 'manager'],
        'colleague': ['同事', '工作', '项目', '开会', '汇报', 'deadline', '加班', '摸鱼'],
        'client': ['客户', '甲方', '乙方', '合作', '合同', '订单', '需求']
    }

    # 师生称呼
    MENTOR_TERMS = {
        'teacher': ['老师', '教授', '导师', '师父', 'teacher', 'prof', 'tutor'],
        'student': ['学生', '同学', '师弟', '师妹', '师兄', '师姐', 'student']
    }

    # 友情称呼
    FRIEND_TERMS = {
        'close': ['闺蜜', '兄弟', '死党', '铁哥们', 'bestie', 'best friend'],
        'casual': ['朋友', '友', '哥们', '姐妹', 'friend', 'buddy', 'pal']
    }

    def detect(self,
               contact_name: str,
               messages: List[Dict],
               stats: Dict) -> RelationTypeResult:
        """
        检测关系类型

        Args:
            contact_name: 联系人名称
            messages: 消息列表
            stats: 统计数据

        Returns:
            关系类型结果
        """
        result = RelationTypeResult()

        # 合并所有文本
        all_text = ' '.join(m.get('content', '') for m in messages)
        all_text_lower = all_text.lower()

        # 按优先级检测
        scores = {}

        # 1. 检测恋爱关系（优先级最高，因为行为模式明显）
        romantic_score, romantic_type = self._detect_romantic(all_text_lower, stats)
        scores['romantic'] = (romantic_score, romantic_type)

        # 2. 检测职场关系
        work_score, work_type = self._detect_work(all_text_lower)
        scores['work'] = (work_score, work_type)

        # 3. 检测师生关系
        mentor_score, mentor_type = self._detect_mentorship(contact_name, all_text_lower)
        scores['mentor'] = (mentor_score, mentor_type)

        # 4. 检测友情关系
        friend_score, friend_type = self._detect_friendship(all_text_lower, stats)
        scores['friend'] = (friend_score, friend_type)

        # 5. 检测亲情关系（降低权重，避免误判）
        family_score, family_type = self._detect_family(contact_name, all_text_lower)
        scores['family'] = (family_score * 0.8, family_type)  # 降低亲情权重

        # 选择得分最高的关系类型
        # 但如果有明显的恋爱行为模式，优先选择恋爱
        if romantic_score > 0.3:  # 有一定恋爱特征
            result.category = RelationCategory.ROMANTIC
            result.specific_type = romantic_type
            result.confidence = romantic_score
            result.analysis_framework = "依恋理论+社会交换"
            result.indicators = self._get_romantic_indicators(stats)
            return result

        # 其他情况按得分排序
        best_type = max(scores, key=lambda x: scores[x][0])
        best_score, best_specific = scores[best_type]

        if best_score > 0.5:
            if best_type == 'family':
                result.category = RelationCategory.FAMILY
                result.analysis_framework = "家庭系统理论"
            elif best_type == 'work':
                result.category = RelationCategory.WORK
                result.analysis_framework = "组织行为学"
            elif best_type == 'mentor':
                result.category = RelationCategory.MENTORSHIP
                result.analysis_framework = "教育心理学"
            elif best_type == 'friend':
                result.category = RelationCategory.FRIENDSHIP
                result.analysis_framework = "社会心理学"

            result.specific_type = best_specific
            result.confidence = best_score
            return result

        # 默认：泛交
        result.category = RelationCategory.ACQUAINTANCE
        result.specific_type = "泛泛之交"
        result.confidence = 0.5
        result.analysis_framework = "基础社交分析"

        return result

    def _get_romantic_indicators(self, stats: Dict) -> List[str]:
        """获取恋爱关系指示器"""
        indicators = []

        init_ratio = stats.get('initiative', {}).get('my_start_ratio', 0.5)
        if init_ratio > 0.7:
            indicators.append("高主动发起比例 ({:.0%})".format(init_ratio))

        avg_daily = stats.get('basic', {}).get('avg_daily', 0)
        if avg_daily > 10:
            indicators.append("高频互动 (日均{:.0f}条)".format(avg_daily))

        return indicators

    def _detect_family(self, contact_name: str, text: str) -> Tuple[float, str]:
        """检测亲情关系"""
        # 检查联系人名称
        name_lower = contact_name.lower()

        for rel_type, terms in self.FAMILY_TERMS.items():
            for term in terms:
                if term in name_lower or term in text:
                    if rel_type == 'parent':
                        return 0.9, "父母"
                    elif rel_type == 'child':
                        return 0.9, "子女"
                    elif rel_type == 'sibling':
                        return 0.85, "兄弟姐妹"
                    elif rel_type == 'grandparent':
                        return 0.85, "祖父母"
                    elif rel_type == 'relative':
                        return 0.7, "亲戚"

        return 0.0, ""

    def _detect_romantic(self, text: str, stats: Dict) -> Tuple[float, str]:
        """检测恋爱关系"""
        score = 0.0

        # 直接称呼
        for term in self.ROMANTIC_TERMS['direct']:
            if term in text:
                score += 0.4
                break

        # 情感表达
        for term in self.ROMANTIC_TERMS['indirect']:
            if term in text:
                score += 0.2
                break

        # 未来规划
        for term in self.ROMANTIC_TERMS['future']:
            if term in text:
                score += 0.2
                break

        # ========== 行为模式检测（新增）==========
        # 高主动发起率（追求特征）
        init_ratio = stats.get('initiative', {}).get('my_start_ratio', 0.5)
        if init_ratio > 0.8:
            score += 0.35  # 强追求信号
        elif init_ratio > 0.7:
            score += 0.25
        elif init_ratio > 0.6:
            score += 0.15

        # 高频互动（恋爱特征）
        avg_daily = stats.get('basic', {}).get('avg_daily', 0)
        if avg_daily > 20:
            score += 0.2
        elif avg_daily > 10:
            score += 0.1

        # 消息轰炸（追求特征）
        # 从聊天历史检测连发
        # （这里简化处理，实际应该从统计数据获取）

        # 确定具体类型
        # 根据主动比例和互动模式判断
        if init_ratio > 0.85:  # 极高主动 = 单相思/追求
            return min(score, 1.0), "单相思"
        elif score > 0.8 and init_ratio < 0.6:  # 高特征但主动平衡 = 热恋
            return min(score, 1.0), "热恋期"
        elif score > 0.6:  # 有特征但主动不平衡 = 追求中
            return min(score, 1.0), "追求中"
        elif score > 0.4:
            return min(score, 1.0), "暧昧期"

        return score, "潜在恋爱关系"

    def _detect_work(self, text: str) -> Tuple[float, str]:
        """检测职场关系"""
        score = 0.0
        work_type = ""

        # 上级词汇
        for term in self.WORK_TERMS['superior']:
            if term in text:
                score += 0.5
                work_type = "上级"
                break

        # 同事词汇
        colleague_count = sum(1 for term in self.WORK_TERMS['colleague'] if term in text)
        if colleague_count > 2:
            score += 0.4
            work_type = "同事"

        # 客户词汇
        for term in self.WORK_TERMS['client']:
            if term in text:
                score += 0.4
                work_type = "客户"
                break

        return min(score, 1.0), work_type

    def _detect_mentorship(self, contact_name: str, text: str) -> Tuple[float, str]:
        """检测师生关系"""
        name_lower = contact_name.lower()

        # 老师称呼
        for term in self.MENTOR_TERMS['teacher']:
            if term in name_lower or term in text:
                return 0.8, "老师"

        # 学生称呼
        for term in self.MENTOR_TERMS['student']:
            if term in name_lower or term in text:
                return 0.8, "学生"

        return 0.0, ""

    def _detect_friendship(self, text: str, stats: Dict) -> Tuple[float, str]:
        """检测友情关系"""
        score = 0.0

        # 亲密朋友称呼
        for term in self.FRIEND_TERMS['close']:
            if term in text:
                return 0.8, "闺蜜/兄弟"

        # 普通朋友
        for term in self.FRIEND_TERMS['casual']:
            if term in text:
                score += 0.3
                break

        # 互动频率判断亲密度
        avg_daily = stats.get('basic', {}).get('avg_daily', 0)
        if avg_daily > 10:
            score += 0.3
            return min(score, 1.0), "好友"
        elif avg_daily > 5:
            score += 0.2
            return min(score, 1.0), "普通朋友"

        return min(score, 1.0), "普通朋友"


# ============================================================
# 关系类型适配的分析配置
# ============================================================

@dataclass
class AnalysisConfig:
    """分析配置"""
    # 适用的心理学框架
    frameworks: List[str] = field(default_factory=list)

    # 专家权重调整
    expert_weights: Dict[str, float] = field(default_factory=dict)

    # 关注的维度
    focus_dimensions: List[str] = field(default_factory=list)

    # 忽略的维度
    ignore_dimensions: List[str] = field(default_factory=list)

    # 特殊建议模板
    advice_templates: Dict[str, str] = field(default_factory=dict)


# 各关系类型的分析配置
RELATION_ANALYSIS_CONFIGS = {
    RelationCategory.FAMILY: AnalysisConfig(
        frameworks=["家庭系统理论", "依恋理论"],
        expert_weights={
            'attachment': 0.4,   # 依恋更重要
            'behavior': 0.2,
            'role': 0.2,
            'social': 0.1,
            'decision': 0.1
        },
        focus_dimensions=['依恋模式', '沟通模式', '边界感'],
        ignore_dimensions=['ROI', '追求策略'],
        advice_templates={
            'boundary': "家庭关系中需要建立健康的边界",
            'communication': "尝试非暴力沟通方式"
        }
    ),

    RelationCategory.ROMANTIC: AnalysisConfig(
        frameworks=["依恋理论", "社会交换理论", "Sternberg爱情三角", "Gottman四骑士"],
        expert_weights={
            'attachment': 0.3,
            'behavior': 0.2,
            'role': 0.25,   # 关系角色很重要
            'social': 0.1,
            'decision': 0.15
        },
        focus_dimensions=['依恋模式', '投入回报比', '关系角色', '危险信号'],
        advice_templates={
            'unrequited': "单相思场景：建议止损或撤退",
            'anxious_trap': "焦虑-回避陷阱：需要打破循环"
        }
    ),

    RelationCategory.FRIENDSHIP: AnalysisConfig(
        frameworks=["社会心理学", "社会交换理论"],
        expert_weights={
            'attachment': 0.2,
            'behavior': 0.3,   # 行为模式更重要
            'role': 0.2,
            'social': 0.2,     # 社交风格
            'decision': 0.1
        },
        focus_dimensions=['互动平衡', '信任度', '支持度'],
        ignore_dimensions=['承诺', '激情'],
        advice_templates={
            'unbalanced': "友谊需要双向投入",
            'drifting': "友谊淡化是正常的，接受或主动维护"
        }
    ),

    RelationCategory.WORK: AnalysisConfig(
        frameworks=["组织行为学", "权力动力学"],
        expert_weights={
            'attachment': 0.1,
            'behavior': 0.3,
            'role': 0.3,    # 职位角色最重要
            'social': 0.2,
            'decision': 0.1
        },
        focus_dimensions=['权力关系', '专业度', '沟通效率'],
        ignore_dimensions=['依恋', '激情', '承诺'],
        advice_templates={
            'superior': "与上级沟通：保持专业，主动汇报",
            'subordinate': "管理下属：清晰指令，及时反馈",
            'colleague': "同事关系：协作优先，避免冲突"
        }
    ),

    RelationCategory.MENTORSHIP: AnalysisConfig(
        frameworks=["教育心理学", "发展心理学"],
        expert_weights={
            'attachment': 0.2,
            'behavior': 0.2,
            'role': 0.3,
            'social': 0.1,
            'decision': 0.2
        },
        focus_dimensions=['指导关系', '学习效果', '尊重边界'],
        advice_templates={
            'teacher': "保持耐心，因材施教",
            'student': "主动请教，尊重老师时间"
        }
    ),

    RelationCategory.ACQUAINTANCE: AnalysisConfig(
        frameworks=["基础社交分析"],
        expert_weights={
            'attachment': 0.1,
            'behavior': 0.3,
            'role': 0.2,
            'social': 0.3,
            'decision': 0.1
        },
        focus_dimensions=['社交礼仪', '第一印象'],
        advice_templates={
            'general': "保持礼貌，适度互动"
        }
    )
}


def get_analysis_config(category: RelationCategory) -> AnalysisConfig:
    """获取关系类型的分析配置"""
    return RELATION_ANALYSIS_CONFIGS.get(category, AnalysisConfig())


def create_relation_type_report(result: RelationTypeResult) -> str:
    """生成关系类型报告"""
    lines = []
    lines.append("=" * 50)
    lines.append("关系类型识别")
    lines.append("=" * 50)

    lines.append(f"\n[关系类别] {result.category.value}")
    lines.append(f"[具体类型] {result.specific_type}")
    lines.append(f"[置信度] {result.confidence:.1%}")
    lines.append(f"[分析框架] {result.analysis_framework}")

    if result.indicators:
        lines.append("\n[识别依据]")
        for indicator in result.indicators:
            lines.append(f"  - {indicator}")

    # 获取分析配置
    config = get_analysis_config(result.category)

    lines.append("\n[分析配置]")
    lines.append(f"  适用框架: {', '.join(config.frameworks)}")
    lines.append(f"  关注维度: {', '.join(config.focus_dimensions)}")

    return "\n".join(lines)


__all__ = [
    'RelationCategory',
    'FamilyRelation', 'FriendshipRelation', 'WorkRelation',
    'MentorshipRelation', 'RomanticRelation', 'AcquaintanceRelation',
    'RelationTypeResult', 'RelationTypeDetector',
    'AnalysisConfig', 'RELATION_ANALYSIS_CONFIGS', 'get_analysis_config',
    'create_relation_type_report'
]