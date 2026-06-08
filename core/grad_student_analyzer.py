"""
研究生/博士生社交关系分析系统

目标人群特点：
- 社交圈：导师、同门、同学、对象、家人
- 压力大：学业+情感+就业
- 孤独感强：实验室/图书馆两点一线
- 心理风险高：抑郁、焦虑、耗竭常见

核心功能：
1. 导师关系分析（权力不对等）
2. 同门关系分析（竞争与合作）
3. 情感关系分析（异地恋常见）
4. 整体支持系统评估
5. 学业压力与社交平衡
6. 心理健康预警

关系类型：
- advisor: 导师（权力不对等，关键关系）
- lab_mate: 同门（合作/竞争）
- classmate: 同学（社交支持）
- romantic: 恋人（情感支持）
- family: 家人（后盾）
- friend: 朋友（社交）
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


# ============================================================
# 研究生特有的关系类型和指标
# ============================================================

class GradRelationType(Enum):
    """研究生关系类型"""
    ADVISOR = "advisor"           # 导师
    LAB_MATE = "lab_mate"         # 同门
    CLASSMATE = "classmate"       # 同学
    ROMANTIC = "romantic"         # 恋人
    FAMILY = "family"             # 家人
    FRIEND = "friend"             # 朋友
    COLLABORATOR = "collaborator" # 合作者


@dataclass
class AdvisorRelationMetrics:
    """导师关系特有指标"""
    # 权力动态
    power_gap: float = 0              # 权力差距 0-100
    my_submissiveness: float = 0      # 我的顺从度
    advisor_responsiveness: float = 0 # 导师响应度

    # 指导质量
    guidance_quality: float = 50      # 指导质量
    meeting_frequency: float = 0      # 见面频率
    feedback_timeliness: float = 0    # 反馈及时性

    # 压力指标
    pressure_level: float = 0         # 压力水平
    expectation_gap: float = 0        # 期望差距

    # 风险
    toxic_risk: float = 0             # 有毒关系风险
    neglect_risk: float = 0           # 被忽视风险


@dataclass
class LabMateMetrics:
    """同门关系指标"""
    # 合作程度
    collaboration_score: float = 50
    help_seeking: float = 0           # 寻求帮助频率
    help_giving: float = 0            # 提供帮助频率

    # 竞争
    competition_level: float = 0      # 竞争程度
    resource_conflict: float = 0      # 资源冲突

    # 社交
    social_closeness: float = 50      # 社交亲密度
    gossip_risk: float = 0            # 八卦风险


@dataclass
class GradStudentProfile:
    """研究生画像"""
    # 基本信息
    year: int = 1                     # 年级
    major: str = ""                   # 专业

    # 社交圈构成
    has_advisor: bool = False
    num_lab_mates: int = 0
    has_romantic: bool = False
    lives_alone: bool = True          # 是否独居

    # 压力指标
    academic_pressure: float = 50     # 学业压力
    employment_pressure: float = 50   # 就业压力
    publication_pressure: float = 50  # 发表压力

    # 支持系统
    advisor_support: float = 50       # 导师支持
    peer_support: float = 50          # 同辈支持
    family_support: float = 50        # 家庭支持
    romantic_support: float = 50      # 情感支持

    # 风险
    isolation_risk: float = 0
    burnout_risk: float = 0
    depression_risk: float = 0


# ============================================================
# 研究生关系量化器
# ============================================================

class GradRelationQuantifier:
    """研究生关系量化器"""

    # 研究生特有的能量权重
    GRAD_ENERGY_RATES = {
        # 导师相关
        'advisor_message': 5,          # 给导师发消息消耗更多能量
        'advisor_meeting': 30,         # 组会/单独见面
        'advisor_feedback_wait': 20,   # 等待导师反馈
        'advisor_pressure': 15,        # 导师施压

        # 同门相关
        'lab_collaboration': 10,       # 合作
        'lab_competition': -5,         # 竞争消耗
        'lab_help': 8,                 # 帮助同门
        'lab_gossip': -10,             # 八卦负面影响

        # 学业压力
        'deadline': 20,                # 截止日期压力
        'rejection': 30,               # 论文被拒
        'revision': 15,                # 修改重投
    }

    def quantify_advisor_relation(
        self,
        messages: List[Dict],
        stats: Dict
    ) -> Tuple[Dict, AdvisorRelationMetrics]:
        """
        量化导师关系

        导师关系是研究生最关键的关系之一
        """
        metrics = AdvisorRelationMetrics()
        energy = {'invest': 0, 'return': 0}

        # 1. 基础互动分析
        my_msgs = stats.get('basic', {}).get('my_messages', 0)
        their_msgs = stats.get('basic', {}).get('their_messages', 0)

        # 导师回复率
        reply_rate = their_msgs / max(my_msgs, 1)
        metrics.advisor_responsiveness = min(reply_rate * 50, 100)

        # 2. 权力动态
        # 学生主动发起比例高 = 顺从度高（需要导师认可）
        my_init_ratio = stats.get('initiative', {}).get('my_start_ratio', 0.5)
        metrics.my_submissiveness = my_init_ratio * 100

        # 权力差距
        metrics.power_gap = 80  # 导师-学生天然权力差距

        # 3. 回复延迟分析
        # 导师回复慢 = 忙碌或忽视
        their_reply_time = stats.get('reply_speed', {}).get('their_avg_seconds', 3600)
        if their_reply_time > 86400:  # 超过1天
            metrics.feedback_timeliness = 20
            metrics.neglect_risk = 40
        elif their_reply_time > 43200:  # 超过12小时
            metrics.feedback_timeliness = 50
        else:
            metrics.feedback_timeliness = 80

        # 4. 消息内容分析
        # 压力词汇
        pressure_words = ['deadline', '进度', '论文', '毕业', '发表', '实验', '数据']
        # （需要实际消息内容分析）

        # 5. 能量计算
        energy['invest'] = my_msgs * self.GRAD_ENERGY_RATES['advisor_message']
        energy['return'] = their_msgs * 3  # 导师回复价值高

        # 等待消耗
        wait_energy = their_reply_time / 3600 * self.GRAD_ENERGY_RATES['advisor_feedback_wait'] * 0.1
        energy['invest'] += wait_energy

        # 6. 指导质量评估
        metrics.guidance_quality = (
            metrics.advisor_responsiveness * 0.4 +
            metrics.feedback_timeliness * 0.3 +
            min(their_msgs / 10, 30)  # 互动频率
        )

        # 7. 风险评估
        if metrics.guidance_quality < 30:
            metrics.toxic_risk = 50
            metrics.neglect_risk = 60

        return energy, metrics

    def quantify_lab_mate_relation(
        self,
        messages: List[Dict],
        stats: Dict
    ) -> Tuple[Dict, LabMateMetrics]:
        """
        量化同门关系

        同门关系复杂：既有合作也有竞争
        """
        metrics = LabMateMetrics()
        energy = {'invest': 0, 'return': 0}

        # 1. 互动平衡
        my_msgs = stats.get('basic', {}).get('my_messages', 0)
        their_msgs = stats.get('basic', {}).get('their_messages', 0)

        balance = min(my_msgs, their_msgs) / max(my_msgs, their_msgs, 1)
        metrics.collaboration_score = balance * 100

        # 2. 帮助行为
        # （需要消息内容分析）
        # 寻求帮助 vs 提供帮助

        # 3. 社交亲密度
        # 基于消息频率、表情使用等
        emojis = stats.get('message_types', {}).get('emoji', {})
        total_emojis = emojis.get('me', 0) + emojis.get('them', 0)
        metrics.social_closeness = min(total_emojis, 100)

        # 4. 能量计算
        energy['invest'] = my_msgs * 2
        energy['return'] = their_msgs * 2

        # 合作加成
        if metrics.collaboration_score > 50:
            energy['return'] += self.GRAD_ENERGY_RATES['lab_collaboration']

        return energy, metrics

    def assess_grad_student(
        self,
        relations: Dict[str, Tuple[Dict, Dict]],
        profile: GradStudentProfile
    ) -> Dict:
        """
        评估研究生整体状态
        """
        assessment = {
            'overall_energy': 0,
            'support_system': {},
            'risks': {},
            'recommendations': []
        }

        total_energy = 0

        # 1. 导师关系（权重最高）
        if 'advisor' in relations:
            energy, metrics = relations['advisor']
            advisor_energy = energy.get('return', 0) - energy.get('invest', 0)
            total_energy += advisor_energy * 1.5  # 导师关系权重1.5

            assessment['support_system']['advisor'] = {
                'energy': advisor_energy,
                'guidance_quality': metrics.guidance_quality,
                'neglect_risk': metrics.neglect_risk
            }

            # 风险
            if metrics.neglect_risk > 50:
                assessment['risks']['advisor_neglect'] = metrics.neglect_risk
                assessment['recommendations'].append(
                    "导师响应度低，建议：主动预约办公时间沟通，或寻找其他指导来源"
                )

        # 2. 同门关系
        lab_energies = []
        for rel_id, (energy, metrics) in relations.items():
            if 'lab_mate' in rel_id:
                lab_energy = energy.get('return', 0) - energy.get('invest', 0)
                lab_energies.append(lab_energy)
                total_energy += lab_energy

        if lab_energies:
            assessment['support_system']['lab_mates'] = {
                'count': len(lab_energies),
                'avg_energy': np.mean(lab_energies)
            }

        # 3. 情感关系
        if 'romantic' in relations:
            energy, _ = relations['romantic']
            romantic_energy = energy.get('return', 0) - energy.get('invest', 0)
            total_energy += romantic_energy * 1.2  # 情感支持重要

            assessment['support_system']['romantic'] = {
                'energy': romantic_energy
            }

            if romantic_energy < -100:
                assessment['risks']['romantic_drain'] = abs(romantic_energy)
                assessment['recommendations'].append(
                    "情感关系消耗大，建议：评估关系健康度，设定边界"
                )

        # 4. 家人支持
        if 'family' in relations:
            energy, _ = relations['family']
            family_energy = energy.get('return', 0) - energy.get('invest', 0)
            total_energy += family_energy

        # 5. 整体评估
        assessment['overall_energy'] = total_energy

        # 6. 孤立风险
        num_supports = len([s for s in assessment['support_system'].values()
                           if s.get('energy', 0) > 0])
        if num_supports < 2:
            assessment['risks']['isolation'] = 70
            assessment['recommendations'].append(
                "支持系统薄弱，建议：主动与同门建立联系，参加学术/社交活动"
            )

        # 7. 耗竭风险
        if total_energy < -300:
            assessment['risks']['burnout'] = min(abs(total_energy) / 10, 100)
            assessment['recommendations'].append(
                "⚠️ 整体能量严重亏损，建议：减少低回报关系投入，增加自我关怀"
            )

        # 8. 研究生特有建议
        if profile.academic_pressure > 70 and num_supports < 3:
            assessment['recommendations'].append(
                "学业压力大但支持不足，建议：寻找心理咨询资源"
            )

        return assessment


# ============================================================
# 研究生心理健康预警
# ============================================================

class GradMentalHealthAssessor:
    """研究生心理健康评估器"""

    # 研究生特有的风险因素
    GRAD_RISK_FACTORS = {
        'advisor_conflict': {
            'description': '导师关系冲突',
            'weight': 2.0,
            'signs': ['导师长期不回复', '导师施压过大', '缺乏指导']
        },
        'academic_isolation': {
            'description': '学术孤立',
            'weight': 1.8,
            'signs': ['无同门支持', '独自做研究', '缺乏讨论']
        },
        'publication_pressure': {
            'description': '发表压力',
            'weight': 1.5,
            'signs': ['毕业要求高', '投稿被拒', '修改周期长']
        },
        'employment_anxiety': {
            'description': '就业焦虑',
            'weight': 1.4,
            'signs': ['毕业年级', '无明确方向', '竞争激烈']
        },
        'romantic_strain': {
            'description': '情感压力',
            'weight': 1.3,
            'signs': ['异地恋', '关系恶化', '单身孤独']
        },
        'impostor_syndrome': {
            'description': '冒名顶替综合症',
            'weight': 1.2,
            'signs': ['自我怀疑', '比较焦虑', '觉得不配']
        }
    }

    def assess(
        self,
        relations: Dict,
        profile: GradStudentProfile
    ) -> Dict:
        """
        评估研究生心理健康风险
        """
        risks = {
            'overall': 0,
            'depression': 0,
            'anxiety': 0,
            'burnout': 0,
            'factors': [],
            'level': 'normal',
            'actions': []
        }

        # 1. 抑郁风险评估
        depression_score = 0

        # 孤立因素
        has_advisor_support = False
        has_peer_support = False
        has_romantic_support = False

        for rel_id, (energy, _) in relations.items():
            net = energy.get('return', 0) - energy.get('invest', 0)
            if 'advisor' in rel_id and net > 0:
                has_advisor_support = True
            if 'lab_mate' in rel_id and net > 0:
                has_peer_support = True
            if 'romantic' in rel_id and net > 0:
                has_romantic_support = True

        supports = [has_advisor_support, has_peer_support, has_romantic_support]
        if sum(supports) == 0:
            depression_score += 40
            risks['factors'].append("完全缺乏支持系统")
        elif sum(supports) == 1:
            depression_score += 20
            risks['factors'].append("支持系统薄弱")

        # 独居
        if profile.lives_alone:
            depression_score += 15
            risks['factors'].append("独居增加孤立风险")

        risks['depression'] = min(depression_score, 100)

        # 2. 焦虑风险评估
        anxiety_score = 0

        # 学业压力
        if profile.academic_pressure > 70:
            anxiety_score += 25
            risks['factors'].append("学业压力高")

        # 发表压力
        if profile.publication_pressure > 70:
            anxiety_score += 20
            risks['factors'].append("发表压力大")

        # 导师关系不确定
        if 'advisor' in relations:
            energy, metrics = relations['advisor']
            if hasattr(metrics, 'neglect_risk') and metrics.neglect_risk > 40:
                anxiety_score += 25
                risks['factors'].append("导师关系不稳定")

        risks['anxiety'] = min(anxiety_score, 100)

        # 3. 耗竭风险评估
        burnout_score = 0

        total_invest = sum(e.get('invest', 0) for e, _ in relations.values())
        total_return = sum(e.get('return', 0) for e, _ in relations.values())

        if total_invest > total_return * 2:
            burnout_score += 40
            risks['factors'].append("投入远超回报")

        if profile.academic_pressure > 80:
            burnout_score += 30

        risks['burnout'] = min(burnout_score, 100)

        # 4. 综合风险
        risks['overall'] = (
            risks['depression'] * 0.35 +
            risks['anxiety'] * 0.35 +
            risks['burnout'] * 0.30
        )

        # 5. 预警等级
        if risks['overall'] > 70:
            risks['level'] = 'critical'
            risks['actions'].append("🚨 建议尽快联系学校心理咨询中心")
        elif risks['overall'] > 50:
            risks['level'] = 'warning'
            risks['actions'].append("⚠️ 建议寻求心理支持")
        elif risks['overall'] > 30:
            risks['level'] = 'watch'
            risks['actions'].append("👀 关注自身状态")
        else:
            risks['level'] = 'normal'

        # 6. 研究生特有建议
        if risks['depression'] > 40:
            risks['actions'].append("保持规律作息，每天至少一次社交互动")

        if risks['anxiety'] > 40:
            risks['actions'].append("设定合理目标，避免过度比较")

        if not has_advisor_support:
            risks['actions'].append("主动与导师沟通，或寻找其他学术支持")

        if not has_peer_support and profile.num_lab_mates > 0:
            risks['actions'].append("尝试与同门建立联系，组织学术讨论或聚餐")

        return risks


# ============================================================
# 报告生成
# ============================================================

def create_grad_student_report(
    assessment: Dict,
    risks: Dict,
    profile: GradStudentProfile
) -> str:
    """生成研究生关系分析报告"""
    lines = []

    lines.append("=" * 60)
    lines.append("研究生社交关系与心理健康评估报告")
    lines.append("=" * 60)

    # 基本信息
    lines.append(f"\n【基本信息】")
    lines.append(f"  年级: {'研一' if profile.year == 1 else '研二' if profile.year == 2 else '研三' if profile.year == 3 else '博士'}")
    lines.append(f"  独居: {'是' if profile.lives_alone else '否'}")

    # 支持系统
    lines.append(f"\n【支持系统】")
    support = assessment.get('support_system', {})

    if 'advisor' in support:
        adv = support['advisor']
        lines.append(f"  导师: 指导质量 {adv['guidance_quality']:.0f}/100")
        if adv.get('neglect_risk', 0) > 30:
            lines.append(f"    ⚠️ 被忽视风险: {adv['neglect_risk']:.0f}")

    if 'lab_mates' in support:
        lab = support['lab_mates']
        lines.append(f"  同门: {lab['count']}人, 平均能量 {lab['avg_energy']:+.0f}")

    if 'romantic' in support:
        rom = support['romantic']
        lines.append(f"  恋人: 能量 {rom['energy']:+.0f}")

    # 整体能量
    lines.append(f"\n【整体能量净值】{assessment['overall_energy']:+.0f}")

    # 风险评估
    level_display = {
        'normal': '✅ 正常',
        'watch': '👀 关注',
        'warning': '⚠️ 警告',
        'critical': '🚨 紧急'
    }
    lines.append(f"\n【心理健康预警】{level_display.get(risks['level'], '未知')}")
    lines.append(f"  综合风险: {risks['overall']:.0f}/100")
    lines.append(f"  抑郁风险: {risks['depression']:.0f}/100")
    lines.append(f"  焦虑风险: {risks['anxiety']:.0f}/100")
    lines.append(f"  耗竭风险: {risks['burnout']:.0f}/100")

    # 风险因素
    if risks['factors']:
        lines.append(f"\n【识别到的风险因素】")
        for factor in risks['factors']:
            lines.append(f"  • {factor}")

    # 建议
    if risks['actions']:
        lines.append(f"\n【行动建议】")
        for i, action in enumerate(risks['actions'], 1):
            lines.append(f"  {i}. {action}")

    return "\n".join(lines)


__all__ = [
    'GradRelationType', 'AdvisorRelationMetrics', 'LabMateMetrics',
    'GradStudentProfile', 'GradRelationQuantifier', 'GradMentalHealthAssessor',
    'create_grad_student_report'
]