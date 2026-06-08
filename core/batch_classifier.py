"""
批量人格分类系统

对所有联系人进行人格分类，生成完整的人格画像库

功能：
1. 扫描所有联系人数据
2. 对每个人进行 MoE 分类
3. 生成人格画像报告
4. 支持人格匹配分析
5. 导出分类结果
"""
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from .moe_classifier import (
    MoEClassifier, PersonalityClassification,
    AttachmentType, BehaviorPattern, RelationshipRole,
    create_classification_report
)
from .big_five import (
    BigFiveAnalyzer, PersonalityProfile,
    create_big_five_report, create_match_report
)


@dataclass
class PersonProfile:
    """完整人格画像"""
    name: str
    contact_id: str
    target: str = "me"  # 分析对象: "me" 用户自己, "them" 对方

    # MoE 分类结果
    moe_classification: PersonalityClassification = None

    # 大五人格
    big_five: PersonalityProfile = None

    # 综合评估
    personality_type: str = ""
    relationship_style: str = ""
    interaction_pattern: str = ""

    # 风险提示
    risk_warnings: List[str] = field(default_factory=list)

    # 建议
    recommendations: List[str] = field(default_factory=list)

    # 数据来源
    message_count: int = 0
    analysis_date: str = ""


@dataclass
class BatchClassificationResult:
    """批量分类结果"""
    total_contacts: int = 0
    classifications: Dict[str, PersonProfile] = field(default_factory=dict)
    summary_stats: Dict = field(default_factory=dict)
    analysis_timestamp: str = ""


class BatchClassifier:
    """
    批量人格分类器

    对所有联系人进行人格分析
    """

    def __init__(self, data_dir: Path = None):
        """
        初始化

        Args:
            data_dir: 数据目录路径（默认 ~/.she-love-me/data/contacts）
        """
        if data_dir is None:
            data_dir = Path.home() / 'she-love-me' / 'data' / 'contacts'

        self.data_dir = data_dir
        self.moe_classifier = MoEClassifier()
        self.big_five_analyzer = BigFiveAnalyzer()

    def classify_all(self, min_messages: int = 10, include_partner: bool = True) -> BatchClassificationResult:
        """
        对所有联系人进行分类

        Args:
            min_messages: 最少消息数量要求
            include_partner: 是否同时分析对方

        Returns:
            批量分类结果
        """
        result = BatchClassificationResult()
        result.analysis_timestamp = datetime.now().isoformat()

        # 扫描所有联系人
        contact_dirs = list(self.data_dir.iterdir())

        for contact_dir in contact_dirs:
            if not contact_dir.is_dir():
                continue

            # 加载数据
            msg_path = contact_dir / 'messages.json'
            stats_path = contact_dir / 'stats.json'

            if not msg_path.exists():
                continue

            with open(msg_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                messages = data.get('messages', [])

            # 消息数量检查
            if len(messages) < min_messages:
                continue

            # 加载统计
            stats = {}
            if stats_path.exists():
                with open(stats_path, 'r', encoding='utf-8') as f:
                    stats = json.load(f)

            # 获取联系人名称
            contact_name = contact_dir.name
            contact_id = data.get('contact_id', contact_name)

            # 执行分类 - 用户自己
            profile_me = self._classify_person(
                contact_name,
                contact_id,
                messages,
                stats,
                target='me'
            )
            result.classifications[f"{contact_id}_me"] = profile_me
            result.total_contacts += 1

            # 执行分类 - 对方
            if include_partner:
                profile_them = self._classify_person(
                    contact_name,
                    contact_id,
                    messages,
                    stats,
                    target='them'
                )
                result.classifications[f"{contact_id}_them"] = profile_them
                result.total_contacts += 1

        # 生成汇总统计
        result.summary_stats = self._generate_summary_stats(result.classifications)

        return result

    def _classify_person(self,
                         name: str,
                         contact_id: str,
                         messages: List[Dict],
                         stats: Dict,
                         target: str = 'me') -> PersonProfile:
        """
        对单个联系人进行分类

        Args:
            name: 联系人名称
            contact_id: 联系人ID
            messages: 消息列表
            stats: 统计数据
            target: 分析目标 ("me" 或 "them")

        Returns:
            人格画像
        """
        profile = PersonProfile(
            name=f"{name} ({'用户' if target == 'me' else '对方'})",
            contact_id=contact_id,
            target=target,
            message_count=len(messages),
            analysis_date=datetime.now().isoformat()
        )

        # 1. MoE 分类
        profile.moe_classification = self.moe_classifier.classify(
            messages, stats, target=target
        )

        # 2. 大五人格分析
        profile.big_five = self.big_five_analyzer.analyze(
            messages, sender=target
        )

        # 3. 生成综合评估
        self._generate_comprehensive_evaluation(profile)

        # 4. 生成风险提示和建议
        self._generate_recommendations(profile)

        return profile

    def _generate_comprehensive_evaluation(self, profile: PersonProfile):
        """生成综合评估"""
        moe = profile.moe_classification
        bf = profile.big_five

        # 人格类型
        types = []
        if moe.attachment == AttachmentType.ANXIOUS:
            types.append("焦虑依恋")
        elif moe.attachment == AttachmentType.AVOIDANT:
            types.append("回避依恋")

        if moe.behavior == BehaviorPattern.ACTIVE:
            types.append("主动型")
        elif moe.behavior == BehaviorPattern.PASSIVE:
            types.append("被动型")

        if moe.role == RelationshipRole.PURSUER:
            types.append("追求者角色")
        elif moe.role == RelationshipRole.DISTANCER:
            types.append("疏离者角色")
        elif moe.role == RelationshipRole.CARETAKER:
            types.append("照顾者角色")

        profile.personality_type = " | ".join(types) if types else "平衡型"

        # 关系风格
        if moe.attachment == AttachmentType.ANXIOUS and moe.role == RelationshipRole.PURSUER:
            profile.relationship_style = "舔狗型：过度关注、过度投入"
        elif moe.attachment == AttachmentType.AVOIDANT and moe.role == RelationshipRole.DISTANCER:
            profile.relationship_style = "回避型：保持距离、不愿深入"
        elif moe.role == RelationshipRole.CARETAKER:
            profile.relationship_style = "照顾型：过度付出、讨好模式"
        elif moe.behavior == BehaviorPattern.BALANCED and moe.attachment == AttachmentType.SECURE:
            profile.relationship_style = "健康型：平衡互动、安全依恋"
        else:
            profile.relationship_style = "混合型：多种特征组合"

        # 互动模式
        patterns = []
        if moe.key_traits:
            patterns.extend(moe.key_traits)

        # 大五人格补充
        if bf.neuroticism > 60:
            patterns.append("情绪敏感")
        if bf.agreeableness > 65:
            patterns.append("高宜人性")
        if bf.extraversion > 70:
            patterns.append("外向活跃")

        profile.interaction_pattern = "；".join(patterns) if patterns else "无明显特征"

    def _generate_recommendations(self, profile: PersonProfile):
        """生成风险提示和建议"""
        moe = profile.moe_classification
        bf = profile.big_five

        # 风险提示
        risks = []

        if moe.attachment == AttachmentType.ANXIOUS:
            risks.append("⚠️ 焦虑依恋：过度关注对方回应，容易情绪波动")

        if moe.role == RelationshipRole.PURSUER:
            risks.append("⚠️ 追求者角色：投入远大于回报，可能陷入舔狗模式")

        if moe.role == RelationshipRole.CARETAKER:
            risks.append("⚠️ 照顾者角色：过度付出可能导致自我消耗")

        if bf.neuroticism > 70:
            risks.append("⚠️ 高神经质：情绪不稳定，需要建立安全感")

        if bf.agreeableness > 70 and bf.neuroticism > 60:
            risks.append("⚠️ 讨好型人格组合：高宜人性+高神经质 = 易陷入讨好循环")

        profile.risk_warnings = risks

        # 建议
        recommendations = []

        if moe.attachment == AttachmentType.ANXIOUS:
            recommendations.append("建议：建立自我边界，减少对对方回应的过度关注")

        if moe.role == RelationshipRole.PURSUER:
            recommendations.append("建议：匹配对方能量，停止过度追求")
            recommendations.append("建议：设定止损线，观察对方是否会主动")

        if moe.role == RelationshipRole.CARETAKER:
            recommendations.append("建议：学会接受而非一味付出")
            recommendations.append("建议：表达自己的需求，而非隐忍")

        if bf.neuroticism > 60:
            recommendations.append("建议：建立稳定的自我支持系统")
            recommendations.append("建议：练习情绪调节技巧")

        profile.recommendations = recommendations

    def _generate_summary_stats(self, classifications: Dict[str, PersonProfile]) -> Dict:
        """生成汇总统计"""
        stats = {
            'attachment_distribution': {},
            'behavior_distribution': {},
            'role_distribution': {},
            'high_neuroticism_count': 0,
            'pursuer_count': 0,
            'healthy_count': 0,
            'risk_contacts': []
        }

        for contact_id, profile in classifications.items():
            moe = profile.moe_classification

            # 依恋分布
            attachment = moe.attachment.name
            stats['attachment_distribution'][attachment] = \
                stats['attachment_distribution'].get(attachment, 0) + 1

            # 行为分布
            behavior = moe.behavior.name
            stats['behavior_distribution'][behavior] = \
                stats['behavior_distribution'].get(behavior, 0) + 1

            # 角色分布
            role = moe.role.name
            stats['role_distribution'][role] = \
                stats['role_distribution'].get(role, 0) + 1

            # 高神经质计数
            if profile.big_five and profile.big_five.neuroticism > 60:
                stats['high_neuroticism_count'] += 1

            # 追求者计数
            if moe.role == RelationshipRole.PURSUER:
                stats['pursuer_count'] += 1
                stats['risk_contacts'].append(contact_id)

            # 健康型计数
            if moe.attachment == AttachmentType.SECURE and \
               moe.behavior == BehaviorPattern.BALANCED:
                stats['healthy_count'] += 1

        return stats

    def export_results(self,
                       result: BatchClassificationResult,
                       output_dir: Path = None) -> Dict[str, Path]:
        """
        导出分类结果

        Args:
            result: 分类结果
            output_dir: 输出目录

        Returns:
            导出文件路径字典
        """
        if output_dir is None:
            output_dir = Path.home() / 'she-love-me' / 'analysis'

        output_dir.mkdir(parents=True, exist_ok=True)

        output_paths = {}

        # 1. JSON 汇总文件
        summary_path = output_dir / 'all_classifications.json'
        summary_data = {
            'analysis_timestamp': result.analysis_timestamp,
            'total_contacts': result.total_contacts,
            'summary_stats': result.summary_stats,
            'classifications': {
                contact_id: {
                    'name': profile.name,
                    'message_count': profile.message_count,
                    'personality_type': profile.personality_type,
                    'relationship_style': profile.relationship_style,
                    'interaction_pattern': profile.interaction_pattern,
                    'moe': {
                        'attachment': profile.moe_classification.attachment.name,
                        'behavior': profile.moe_classification.behavior.name,
                        'role': profile.moe_classification.role.name,
                        'key_traits': profile.moe_classification.key_traits
                    },
                    'big_five': {
                        'openness': round(profile.big_five.openness, 1) if profile.big_five else 50,
                        'conscientiousness': round(profile.big_five.conscientiousness, 1) if profile.big_five else 50,
                        'extraversion': round(profile.big_five.extraversion, 1) if profile.big_five else 50,
                        'agreeableness': round(profile.big_five.agreeableness, 1) if profile.big_five else 50,
                        'neuroticism': round(profile.big_five.neuroticism, 1) if profile.big_five else 50
                    },
                    'risk_warnings': profile.risk_warnings,
                    'recommendations': profile.recommendations
                }
                for contact_id, profile in result.classifications.items()
            }
        }

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        output_paths['summary'] = summary_path

        # 2. 报告文件
        report_path = output_dir / 'classification_report.txt'
        report_lines = []

        report_lines.append("=" * 70)
        report_lines.append("批量人格分类报告")
        report_lines.append("=" * 70)
        report_lines.append(f"\n分析时间: {result.analysis_timestamp}")
        report_lines.append(f"联系人总数: {result.total_contacts}")

        report_lines.append("\n[汇总统计]")
        report_lines.append(f"  依恋分布: {result.summary_stats['attachment_distribution']}")
        report_lines.append(f"  行为分布: {result.summary_stats['behavior_distribution']}")
        report_lines.append(f"  角色分布: {result.summary_stats['role_distribution']}")
        report_lines.append(f"  高神经质人数: {result.summary_stats['high_neuroticism_count']}")
        report_lines.append(f"  追求者角色人数: {result.summary_stats['pursuer_count']}")
        report_lines.append(f"  健康型人数: {result.summary_stats['healthy_count']}")

        if result.summary_stats['risk_contacts']:
            report_lines.append(f"\n[高风险联系人]")
            for contact_id in result.summary_stats['risk_contacts']:
                profile = result.classifications[contact_id]
                report_lines.append(f"  - {profile.name}: {profile.relationship_style}")

        report_lines.append("\n" + "=" * 70)
        report_lines.append("各联系人详细分析")
        report_lines.append("=" * 70)

        for contact_id, profile in result.classifications.items():
            report_lines.append(f"\n\n{'─' * 60}")
            report_lines.append(f"联系人: {profile.name}")
            report_lines.append(f"消息数: {profile.message_count}")
            report_lines.append(f"{'─' * 60}")

            report_lines.append(f"\n[人格类型] {profile.personality_type}")
            report_lines.append(f"[关系风格] {profile.relationship_style}")
            report_lines.append(f"[互动模式] {profile.interaction_pattern}")

            if profile.risk_warnings:
                report_lines.append("\n[风险提示]")
                for risk in profile.risk_warnings:
                    report_lines.append(f"  {risk}")

            if profile.recommendations:
                report_lines.append("\n[建议]")
                for rec in profile.recommendations:
                    report_lines.append(f"  {rec}")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        output_paths['report'] = report_path

        return output_paths


def classify_all_contacts(min_messages: int = 10) -> BatchClassificationResult:
    """
    快捷函数：对所有联系人进行分类

    Args:
        min_messages: 最少消息数量

    Returns:
        分类结果
    """
    classifier = BatchClassifier()
    return classifier.classify_all(min_messages)


def export_all_classifications(min_messages: int = 10) -> Dict[str, Path]:
    """
    快捷函数：分类并导出结果

    Args:
        min_messages: 最少消息数量

    Returns:
        导出文件路径
    """
    classifier = BatchClassifier()
    result = classifier.classify_all(min_messages)
    return classifier.export_results(result)


__all__ = [
    'PersonProfile', 'BatchClassificationResult', 'BatchClassifier',
    'classify_all_contacts', 'export_all_classifications'
]