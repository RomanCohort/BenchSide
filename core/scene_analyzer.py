"""
场景差异化分析器

针对不同关系类型提供定制化分析
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from .relation_types import RelationType, RelationProfile, RELATION_PROFILES, get_relation_profile


@dataclass
class SceneAnalysisResult:
    """场景分析结果"""
    relation_type: RelationType
    profile: RelationProfile

    # 基础指标（通用）
    basic_scores: Dict[str, float] = field(default_factory=dict)

    # 场景特定指标
    scene_scores: Dict[str, float] = field(default_factory=dict)

    # 综合评分
    overall_score: float = 0.0
    overall_grade: str = "C"

    # 场景特定建议
    suggestions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # 风险评估
    risk_level: str = "low"  # low, medium, high, critical

    # 改进方向
    improvement_areas: List[str] = field(default_factory=list)


class SceneAnalyzer:
    """
    场景差异化分析器

    核心思想：
    - 不同关系类型有不同的分析重点
    - 不同场景有不同的风险阈值
    - 建议策略需要匹配场景特点
    """

    # 场景特定的评分公式
    SCENE_SCORE_FORMULAS = {
        RelationType.BOSS: {
            "responsiveness_score": lambda s: min(s['reply_speed'] / 60, 100),
            "professionalism_score": lambda s: 100 - s['cold_index'],
            "reliability_score": lambda s: s['loved_index'] * 1.5,
            "overall": lambda scores: (
                scores['responsiveness_score'] * 0.3 +
                scores['professionalism_score'] * 0.25 +
                scores['reliability_score'] * 0.25 +
                scores['communication_score'] * 0.2
            )
        },
        RelationType.ADVISOR: {
            "progress_report_score": lambda s: s['initiative'] * 1.2,
            "respect_score": lambda s: 100 - s['cold_index'],
            "academic_engagement_score": lambda s: s['loved_index'],
            "overall": lambda scores: (
                scores['progress_report_score'] * 0.25 +
                scores['respect_score'] * 0.25 +
                scores['academic_engagement_score'] * 0.25 +
                scores['responsiveness_score'] * 0.25
            )
        },
        RelationType.BOYFRIEND: {
            "intimacy_score": lambda s: s['loved_index'],
            "commitment_score": lambda s: s['commitment_signals'] if 'commitment_signals' in s else 50,
            "attachment_score": lambda s: 100 - abs(s['simp_index'] - s['loved_index']),
            "overall": lambda scores: (
                scores['intimacy_score'] * 0.25 +
                scores['commitment_score'] * 0.2 +
                scores['attachment_score'] * 0.25 +
                scores['support_score'] * 0.3
            )
        },
        RelationType.PARENT: {
            "care_score": lambda s: s['loved_index'],
            "communication_score": lambda s: s['message_frequency'],
            "filial_score": lambda s: s['initiative'],
            "overall": lambda scores: (
                scores['care_score'] * 0.3 +
                scores['communication_score'] * 0.25 +
                scores['filial_score'] * 0.25 +
                scores['sharing_score'] * 0.2
            )
        }
    }

    # 场景特定的建议模板
    SUGGESTION_TEMPLATES = {
        RelationType.BOSS: {
            "low_responsiveness": [
                "领导消息要及时回复，建议设置提醒",
                "重要事项先回复'收到'，再详细汇报",
                "避免在非工作时间发非紧急消息"
            ],
            "high_cold_risk": [
                "⚠️ 领导对你态度冷淡，需要反思工作表现",
                "主动汇报进度，展示工作成果",
                "询问是否有什么问题需要沟通"
            ],
            "positive": [
                "保持当前的沟通节奏",
                "定期汇报工作进展",
                "注意边界，不要过度私聊"
            ]
        },
        RelationType.ADVISOR: {
            "low_progress_report": [
                "导师期望你主动汇报进度",
                "每周发送一次进度简报",
                "有问题及时请教，不要憋着"
            ],
            "high_cold_risk": [
                "⚠️ 导师可能对你的进度不满意",
                "尽快安排一次当面沟通",
                "准备好进度汇报和问题清单"
            ],
            "positive": [
                "保持主动汇报的习惯",
                "论文/项目进度要定期更新",
                "节日可以适当问候，但不要太频繁"
            ]
        },
        RelationType.BOYFRIEND: {
            "attachment_anxiety": [
                "你的主动程度和对方回应不匹配",
                "建议：先观察几天，看他是否会主动",
                "不要频繁追问'你怎么了'"
            ],
            "commitment_unclear": [
                "关系中承诺感不明确",
                "可以自然地聊聊未来规划",
                "观察他是否提到'我们'而非'我'"
            ],
            "cold_signals": [
                "⚠️ 他可能正在疏远或压力大",
                "不要追问，给他一些空间",
                "用轻松话题测试，如分享搞笑视频"
            ]
        },
        RelationType.PARENT: {
            "low_contact": [
                "父母期望更频繁的联系",
                "每周至少主动联系一次",
                "分享一些生活小事，让他们放心"
            ],
            "positive": [
                "保持与父母的沟通习惯",
                "节日记得问候和感谢",
                "多分享生活中的好事"
            ]
        },
        RelationType.CLIENT: {
            "low_responsiveness": [
                "⚠️ 客户消息必须第一时间回复",
                "设置专门提醒，避免漏掉客户消息",
                "回复要专业、完整"
            ],
            "complaint_signals": [
                "⚠️ 客户可能不满意",
                "立即跟进，了解具体问题",
                "主动提出解决方案"
            ]
        }
    }

    # 风险阈值（不同场景不同）
    RISK_THRESHOLDS = {
        RelationType.BOSS: {
            "critical": {"cold_index": 40, "reply_time": 7200},  # 2小时不回
            "high": {"cold_index": 30, "reply_time": 3600},
            "medium": {"cold_index": 20, "reply_time": 1800}
        },
        RelationType.ADVISOR: {
            "critical": {"cold_index": 50, "silence_days": 7},
            "high": {"cold_index": 35, "silence_days": 5},
            "medium": {"cold_index": 20, "silence_days": 3}
        },
        RelationType.BOYFRIEND: {
            "critical": {"cold_index": 60, "silence_days": 3},
            "high": {"cold_index": 45, "silence_days": 2},
            "medium": {"cold_index": 30, "silence_days": 1}
        }
    }

    def analyze(self,
                stats: Dict,
                relation_type: RelationType,
                custom_metrics: Optional[Dict] = None) -> SceneAnalysisResult:
        """
        执行场景差异化分析

        Args:
            stats: 统计数据（stats.json内容）
            relation_type: 关系类型
            custom_metrics: 自定义指标

        Returns:
            场景分析结果
        """
        profile = get_relation_profile(relation_type)

        # 1. 计算基础指标
        basic_scores = {
            "simp_index": stats.get('scores', {}).get('simp_index', 50),
            "loved_index": stats.get('scores', {}).get('loved_index', 50),
            "cold_index": stats.get('scores', {}).get('cold_index', 0),
            "reply_speed": stats.get('reply_speed', {}).get('my_avg_seconds', 0),
            "initiative": stats.get('initiative', {}).get('my_start_ratio', 0.5) * 100,
            "message_frequency": stats.get('basic', {}).get('avg_daily', 5)
        }

        # 2. 计算场景特定指标
        scene_scores = self._calculate_scene_scores(
            basic_scores, relation_type, custom_metrics
        )

        # 3. 计算综合评分
        overall_score = self._calculate_overall_score(
            basic_scores, scene_scores, relation_type
        )

        # 4. 评级
        overall_grade = self._get_grade(overall_score)

        # 5. 风险评估
        risk_level = self._assess_risk(basic_scores, relation_type)

        # 6. 生成建议
        suggestions, warnings = self._generate_suggestions(
            basic_scores, scene_scores, relation_type, risk_level
        )

        # 7. 改进方向
        improvement_areas = self._identify_improvements(
            basic_scores, scene_scores, relation_type
        )

        return SceneAnalysisResult(
            relation_type=relation_type,
            profile=profile,
            basic_scores=basic_scores,
            scene_scores=scene_scores,
            overall_score=overall_score,
            overall_grade=overall_grade,
            suggestions=suggestions,
            warnings=warnings,
            risk_level=risk_level,
            improvement_areas=improvement_areas
        )

    def _calculate_scene_scores(self,
                                 basic_scores: Dict,
                                 relation_type: RelationType,
                                 custom_metrics: Optional[Dict]) -> Dict:
        """计算场景特定指标"""
        formulas = self.SCENE_SCORE_FORMULAS.get(relation_type, {})

        scene_scores = {}
        for metric_name, formula in formulas.items():
            if metric_name != "overall":
                try:
                    scene_scores[metric_name] = min(max(formula(basic_scores), 0), 100)
                except:
                    scene_scores[metric_name] = 50  # 默认值

        # 添加自定义指标
        if custom_metrics:
            scene_scores.update(custom_metrics)

        return scene_scores

    def _calculate_overall_score(self,
                                  basic_scores: Dict,
                                  scene_scores: Dict,
                                  relation_type: RelationType) -> float:
        """计算综合评分"""
        formulas = self.SCENE_SCORE_FORMULAS.get(relation_type, {})

        if "overall" in formulas:
            try:
                all_scores = {**basic_scores, **scene_scores}
                return min(max(formulas["overall"](all_scores), 0), 100)
            except:
                pass

        # 默认公式
        profile = get_relation_profile(relation_type)
        weights = profile.score_weights

        score = 0.0
        for metric, weight in weights.items():
            base_score = basic_scores.get(metric, 50)
            score += base_score * weight

        return min(max(score, 0), 100)

    def _get_grade(self, score: float) -> str:
        """评级"""
        if score >= 80:
            return "A"
        elif score >= 60:
            return "B"
        elif score >= 40:
            return "C"
        elif score >= 20:
            return "D"
        else:
            return "F"

    def _assess_risk(self,
                     basic_scores: Dict,
                     relation_type: RelationType) -> str:
        """风险评估"""
        thresholds = self.RISK_THRESHOLDS.get(relation_type, {})

        if not thresholds:
            # 默认阈值
            cold = basic_scores.get('cold_index', 0)
            if cold > 50:
                return "critical"
            elif cold > 35:
                return "high"
            elif cold > 20:
                return "medium"
            return "low"

        # 检查各风险级别
        for level, conditions in sorted(
            thresholds.items(),
            key=lambda x: ["low", "medium", "high", "critical"].index(x[0]),
            reverse=True
        ):
            for metric, threshold in conditions.items():
                value = basic_scores.get(metric, 0)
                if value > threshold:
                    return level

        return "low"

    def _generate_suggestions(self,
                              basic_scores: Dict,
                              scene_scores: Dict,
                              relation_type: RelationType,
                              risk_level: str) -> tuple:
        """生成建议"""
        templates = self.SUGGESTION_TEMPLATES.get(relation_type, {})
        suggestions = []
        warnings = []

        # 根据风险级别
        if risk_level in ["high", "critical"]:
            if "high_cold_risk" in templates:
                warnings.extend(templates["high_cold_risk"])

        # 根据指标
        cold = basic_scores.get('cold_index', 0)
        reply_speed = basic_scores.get('reply_speed', 0)

        if reply_speed > 1800 and relation_type in [RelationType.BOSS, RelationType.CLIENT]:
            if "low_responsiveness" in templates:
                suggestions.extend(templates["low_responsiveness"])

        if cold > 30:
            if "cold_signals" in templates:
                warnings.extend(templates["cold_signals"])

        # 默认正向建议
        if risk_level == "low" and "positive" in templates:
            suggestions.extend(templates["positive"])

        return suggestions, warnings

    def _identify_improvements(self,
                               basic_scores: Dict,
                               scene_scores: Dict,
                               relation_type: RelationType) -> List[str]:
        """识别改进方向"""
        improvements = []
        profile = get_relation_profile(relation_type)

        # 找出薄弱维度
        for metric, weight in profile.dimension_weights.items():
            if weight > 0.2:  # 重要维度
                score = scene_scores.get(metric, basic_scores.get(metric, 50))
                if score < 60:
                    improvements.append(f"提升{metric}: 当前{score:.0f}分，建议目标70分")

        return improvements


def create_scene_report(result: SceneAnalysisResult) -> str:
    """
    创建场景报告

    Args:
        result: 分析结果

    Returns:
        格式化的报告文本
    """
    lines = []

    lines.append("=" * 50)
    lines.append(f"「关系洞察」分析报告 - {result.profile.display_name}")
    lines.append("=" * 50)

    lines.append("\n📊 基础指标:")
    for metric, value in result.basic_scores.items():
        lines.append(f"  {metric}: {value:.1f}")

    if result.scene_scores:
        lines.append("\n🎯 场景指标:")
        for metric, value in result.scene_scores.items():
            lines.append(f"  {metric}: {value:.1f}")

    lines.append(f"\n⭐ 综合评分: {result.overall_score:.1f} ({result.overall_grade})")
    lines.append(f"⚠️ 风险等级: {result.risk_level}")

    if result.warnings:
        lines.append("\n🚨 警告:")
        for w in result.warnings:
            lines.append(f"  {w}")

    if result.suggestions:
        lines.append("\n💡 建议:")
        for s in result.suggestions:
            lines.append(f"  {s}")

    if result.improvement_areas:
        lines.append("\n📈 改进方向:")
        for i in result.improvement_areas:
            lines.append(f"  {i}")

    return "\n".join(lines)


__all__ = ['SceneAnalyzer', 'SceneAnalysisResult', 'create_scene_report']