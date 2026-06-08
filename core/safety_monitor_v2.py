"""
安全监控模块 v2.0

核心原则：
1. 严禁诊断 - 只做陪伴和转介
2. 赋能而非替代 - 帮用户连接现实
3. 及时干预 - 检测到风险立即响应

新增功能：
- 危机关键词检测（自杀、自伤等）
- 过度倾诉检测
- 情感依赖引导
- 医疗化风险避免
- 现实连接提示

参考：
- WHO心理健康指南
- 心理咨询伦理守则
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import re
import random


# ============================================================
# 风险类型定义
# ============================================================

class RiskLevel(Enum):
    """风险等级"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRISIS = "crisis"  # 危机等级，需要立即干预


class RiskType(Enum):
    """风险类型"""
    # 危机类（需要立即干预）
    SUICIDE_IDEATION = "suicide_ideation"          # 自杀意念
    SELF_HARM = "self_harm"                        # 自伤行为
    SEVERE_DISTRESS = "severe_distress"            # 严重痛苦

    # 高风险类
    OVER_MEDICALIZATION = "over_medicalization"    # 过度医疗化
    EMOTIONAL_DEPENDENCY = "emotional_dependency"  # 情感依赖
    SEVERE_ISOLATION = "severe_isolation"          # 严重隔离

    # 中等风险类
    EXCESSIVE_VENTING = "excessive_venting"        # 过度倾诉
    REALITY_DETACHMENT = "reality_detachment"      # 现实脱离
    ESCALATION = "escalation"                      # 冲突升级

    # 低风险类
    MILD_DEPENDENCY = "mild_dependency"            # 轻度依赖
    OVER_RELIANCE = "over_reliance"                # 过度依赖模型


# ============================================================
# 危机关键词库
# ============================================================

CRISIS_KEYWORDS = {
    # 自杀相关
    'suicide': ['自杀', '不想活了', '活着没意思', '想结束生命', '想死', '跳楼', '割腕'],
    'self_harm': ['自残', '伤害自己', '割伤', '想痛'],
    'hopelessness': ['绝望', '没有希望', '没救了', '完蛋了'],

    # 需要专业帮助
    'crisis': ['撑不下去了', '崩溃', '崩溃了', '彻底完了'],

    # 医疗化风险词（避免诊断）
    'diagnosis_terms': ['抑郁症', '焦虑症', '双相', '精神分裂', '强迫症',
                        'PTSD', 'ADHD', '我有病', '我是不是有病']
}

# 24小时心理热线
CRISIS_RESOURCES = """
【24小时心理援助热线】
• 全国心理援助热线：400-161-9995
• 北京心理危机研究与干预中心：010-82951332
• 上海心理热线：021-64389888
• 广州心理热线：020-81899120
• 深圳24小时热线：0755-25629459

如果您正在经历困难，请记住：
您并不孤单，专业的帮助随时可以获取。
"""


# ============================================================
# 安全警报
# ============================================================

@dataclass
class SafetyAlert:
    """安全警报"""
    risk_type: RiskType
    risk_level: RiskLevel
    severity: float  # 0-1
    description: str
    immediate_action: str  # 立即行动
    recommendations: List[str]
    crisis_resources: Optional[str] = None
    should_stop_conversation: bool = False  # 是否应该停止对话
    should_refer: bool = False  # 是否需要转介
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================
# 安全监控器 v2.0
# ============================================================

class SafetyMonitorV2:
    """
    安全监控器 v2.0

    核心原则：
    1. 严禁诊断 - 只做陪伴和转介
    2. 赋能而非替代 - 帮用户连接现实
    3. 及时干预 - 检测到风险立即响应
    """

    def __init__(self):
        self.alerts: List[SafetyAlert] = []
        self.conversation_history: List[Dict] = []

        # 监控参数
        self.max_consecutive_turns = 20  # 最大连续对话轮数
        self.max_venting_ratio = 0.7  # 最大倾诉比例
        self.dependency_threshold = 0.8  # 依赖阈值

        # 状态追踪
        self.current_session_turns = 0
        self.user_venting_ratio = 0.0
        self.dependency_score = 0.0

    def analyze_message(self, message: str) -> Tuple[Optional[SafetyAlert], Dict]:
        """
        分析用户消息

        Returns:
            (alert, analysis_result)
        """
        analysis = {
            'contains_crisis': False,
            'contains_diagnosis': False,
            'sentiment': 'neutral',
            'venting_level': 0.0,
            'dependency_indicators': []
        }

        # 1. 危机关键词检测
        crisis_alert = self._check_crisis_keywords(message, analysis)

        if crisis_alert:
            return crisis_alert, analysis

        # 2. 医疗化风险检测
        medicalization_alert = self._check_medicalization(message, analysis)

        if medicalization_alert:
            return medicalization_alert, analysis

        # 3. 情感依赖检测
        dependency_alert = self._check_dependency(message, analysis)

        # 更新状态
        self.conversation_history.append({
            'message': message,
            'analysis': analysis,
            'timestamp': datetime.now()
        })

        return dependency_alert, analysis

    def _check_crisis_keywords(self, message: str, analysis: Dict) -> Optional[SafetyAlert]:
        """检查危机关键词"""
        message_lower = message.lower()

        # 自杀意念
        for keyword in CRISIS_KEYWORDS['suicide']:
            if keyword in message:
                analysis['contains_crisis'] = True
                return SafetyAlert(
                    risk_type=RiskType.SUICIDE_IDEATION,
                    risk_level=RiskLevel.CRISIS,
                    severity=1.0,
                    description="检测到可能的危机信号",
                    immediate_action="立即停止对话，提供专业资源",
                    recommendations=[
                        "我注意到您可能正在经历非常困难的时刻",
                        "请知道您并不孤单，有人愿意帮助您",
                        "强烈建议您联系专业的心理援助"
                    ],
                    crisis_resources=CRISIS_RESOURCES,
                    should_stop_conversation=True,
                    should_refer=True
                )

        # 自伤行为
        for keyword in CRISIS_KEYWORDS['self_harm']:
            if keyword in message:
                analysis['contains_crisis'] = True
                return SafetyAlert(
                    risk_type=RiskType.SELF_HARM,
                    risk_level=RiskLevel.CRISIS,
                    severity=0.9,
                    description="检测到可能的自伤相关表达",
                    immediate_action="提供专业资源，建议寻求帮助",
                    recommendations=[
                        "我关心您的安全",
                        "请考虑联系专业人士",
                        "有人愿意帮助您度过这个困难"
                    ],
                    crisis_resources=CRISIS_RESOURCES,
                    should_stop_conversation=True,
                    should_refer=True
                )

        # 绝望感
        for keyword in CRISIS_KEYWORDS['hopelessness']:
            if keyword in message:
                analysis['contains_crisis'] = True
                return SafetyAlert(
                    risk_type=RiskType.SEVERE_DISTRESS,
                    risk_level=RiskLevel.HIGH,
                    severity=0.7,
                    description="检测到强烈的绝望感表达",
                    immediate_action="提供支持，建议专业帮助",
                    recommendations=[
                        "我理解您正在经历困难",
                        "这种感觉是暂时的，会有转机",
                        "建议与信任的人或专业人士交流"
                    ],
                    crisis_resources=CRISIS_RESOURCES,
                    should_refer=True
                )

        return None

    def _check_medicalization(self, message: str, analysis: Dict) -> Optional[SafetyAlert]:
        """检查医疗化风险"""
        message_lower = message.lower()

        for keyword in CRISIS_KEYWORDS['diagnosis_terms']:
            if keyword in message:
                analysis['contains_diagnosis'] = True

                return SafetyAlert(
                    risk_type=RiskType.OVER_MEDICALIZATION,
                    risk_level=RiskLevel.MEDIUM,
                    severity=0.5,
                    description="用户使用了心理健康诊断术语",
                    immediate_action="避免诊断，提供支持性回应",
                    recommendations=[
                        "我注意到您提到了一些心理健康相关的词",
                        "请注意：我不是心理健康专业人士，无法进行诊断",
                        "如果您对自己的心理健康有担忧，建议咨询专业人士",
                        "我可以陪伴您，提供情感支持"
                    ],
                    should_stop_conversation=False,
                    should_refer=True
                )

        return None

    def _check_dependency(self, message: str, analysis: Dict) -> Optional[SafetyAlert]:
        """检查情感依赖"""
        self.current_session_turns += 1

        # 计算倾诉比例
        user_msg_length = len(message)
        venting_indicators = ['我', '我觉', '我觉得', '我好', '我好累', '我不知道']

        venting_count = sum(1 for ind in venting_indicators if ind in message)
        venting_ratio = venting_count / max(len(message.split()), 1)

        self.user_venting_ratio = 0.7 * self.user_venting_ratio + 0.3 * venting_ratio

        # 检测过度倾诉
        if self.user_venting_ratio > self.max_venting_ratio:
            analysis['venting_level'] = self.user_venting_ratio

            return SafetyAlert(
                risk_type=RiskType.EXCESSIVE_VENTING,
                risk_level=RiskLevel.MEDIUM,
                severity=self.user_venting_ratio,
                description="检测到过度倾诉倾向",
                immediate_action="引导用户接触现实中的人",
                recommendations=[
                    "我理解您有很多想要分享的",
                    "或许可以尝试和身边的朋友、家人聊聊？",
                    "面对面的交流有时更有帮助",
                    "您也可以考虑参加一些线下活动"
                ],
                should_stop_conversation=False,
                should_refer=False
            )

        # 检测连续对话过多
        if self.current_session_turns > self.max_consecutive_turns:
            self.dependency_score += 0.1

            return SafetyAlert(
                risk_type=RiskType.EMOTIONAL_DEPENDENCY,
                risk_level=RiskLevel.MEDIUM,
                severity=min(self.dependency_score, 1.0),
                description="检测到可能的情感依赖",
                immediate_action="温和引导，鼓励现实社交",
                recommendations=[
                    "我们聊了很多，希望对您有帮助",
                    "也许可以休息一下，去做点别的事？",
                    "记得您还有现实中的朋友和家人",
                    "他们可能也很想和您交流"
                ],
                should_stop_conversation=False,
                should_refer=False
            )

        # 检测现实脱离
        reality_indicators = ['不', '不想', '没朋友', '没意思', '无聊', '逃避']
        reality_count = sum(1 for ind in reality_indicators if ind in message)

        if reality_count >= 3:
            return SafetyAlert(
                risk_type=RiskType.REALITY_DETACHMENT,
                risk_level=RiskLevel.MEDIUM,
                severity=0.6,
                description="检测到可能的现实脱离倾向",
                immediate_action="鼓励现实连接",
                recommendations=[
                    "我理解现实有时让人感到疲惫",
                    "但与现实中的连接很重要",
                    "或许可以从小事开始，比如和朋友喝杯咖啡？",
                    "或者参加一个你感兴趣的活动？"
                ],
                should_stop_conversation=False,
                should_refer=False
            )

        return None

    def check_session_end(self) -> Optional[SafetyAlert]:
        """检查会话结束时是否需要提醒"""
        if self.current_session_turns > 15:
            return SafetyAlert(
                risk_type=RiskType.MILD_DEPENDENCY,
                risk_level=RiskLevel.LOW,
                severity=0.3,
                description="会话较长，温和提醒",
                immediate_action="鼓励用户回归现实",
                recommendations=[
                    "今天的聊天就到这里吧",
                    "希望这些建议对您有帮助",
                    "记住：我只是一个工具，真正的改变来自于您自己",
                    "和现实中的人保持连接很重要"
                ],
                should_stop_conversation=False,
                should_refer=False
            )

        return None

    def generate_safe_response(self, alert: SafetyAlert) -> str:
        """生成安全回应"""
        if alert.risk_level == RiskLevel.CRISIS:
            # 危机情况
            response = f"{alert.recommendations[0]}\n\n"
            response += "请查看以下专业资源：\n"
            response += alert.crisis_resources or CRISIS_RESOURCES
            return response

        elif alert.risk_level == RiskLevel.HIGH:
            # 高风险
            response = "\n".join(alert.recommendations)
            response += "\n\n如果您觉得需要更多支持，请考虑：\n"
            response += "• 与信任的朋友或家人交流\n"
            response += "• 寻求专业心理咨询\n"
            return response

        else:
            # 中低风险
            return "\n".join(alert.recommendations)

    def reset_session(self):
        """重置会话状态"""
        self.current_session_turns = 0
        self.user_venting_ratio = 0.0

    def get_safety_report(self) -> str:
        """生成安全报告"""
        lines = []

        lines.append("=" * 70)
        lines.append("安全监控报告")
        lines.append("=" * 70)

        lines.append(f"\n当前会话轮数: {self.current_session_turns}")
        lines.append(f"倾诉比例: {self.user_venting_ratio:.1%}")
        lines.append(f"依赖得分: {self.dependency_score:.2f}")

        if self.alerts:
            lines.append(f"\n发现 {len(self.alerts)} 个警报:")

            for alert in self.alerts[-10:]:
                lines.append(f"\n【{alert.risk_type.value}】")
                lines.append(f"  等级: {alert.risk_level.value}")
                lines.append(f"  严重程度: {alert.severity:.1%}")
                lines.append(f"  描述: {alert.description}")

        return "\n".join(lines)


# ============================================================
# 赋能型指导生成器
# ============================================================

class EmpoweringGuidanceGenerator:
    """
    赋能型指导生成器

    核心原则：赋能而非替代
    - 帮用户更好地连接现实
    - 不帮用户逃避现实
    """

    # 赋能型建议模板
    EMPOWERING_TEMPLATES = {
        'connection': [
            "或许可以和{person}聊聊这个？",
            "{activity}可能会让你感觉更好",
            "尝试邀请{person}一起{activity}？"
        ],
        'skill_building': [
            "这是一个练习{skill}的好机会",
            "也许可以尝试用{method}来处理这个情况",
            "回想一下，之前{success_case}是怎么做到的？"
        ],
        'self_efficacy': [
            "相信你有能力处理这个",
            "你已经做得很好了",
            "这个挑战是暂时的，你会过去的"
        ],
        'reality_anchor': [
            "现实中的{person}可能也在想你",
            "面对面的交流有时更有效",
            "线上聊天的效果有限，线下见面可能更好"
        ]
    }

    # 避免的替代型建议
    AVOID_TEMPLATES = [
        "我会永远陪着你",  # 暗示AI可以替代人
        "你可以一直和我说",  # 鼓励过度依赖
        "只有我理解你",  # 孤立用户与现实
        "你不需要其他人"  # 明显的替代倾向
    ]

    def __init__(self, safety_monitor: SafetyMonitorV2):
        self.safety_monitor = safety_monitor

    def generate_guidance(
        self,
        situation: Dict,
        base_guidance: Dict
    ) -> Dict:
        """
        生成赋能型指导

        Args:
            situation: 当前情况
            base_guidance: 基础建议

        Returns:
            赋能型指导
        """
        # 检查是否有安全警报
        alert = self.safety_monitor.analyze_message(
            situation.get('last_message', '')
        )[0]

        if alert and alert.should_stop_conversation:
            return {
                'action': 'stop_and_refer',
                'reasoning': self.safety_monitor.generate_safe_response(alert),
                'empowering': True,
                'safety_alert': alert
            }

        # 转换为赋能型建议
        guidance = base_guidance.copy()

        # 添加现实连接元素
        if random.random() < 0.3:  # 30%概率添加
            guidance['reasoning'] += "\n\n" + random.choice(
                self.EMPOWERING_TEMPLATES['reality_anchor']
            ).format(
                person=random.choice(['朋友', '家人', '同事']),
                activity=random.choice(['喝杯咖啡', '散步', '聊聊'])
            )

        # 添加自我效能感
        if random.random() < 0.2:
            guidance['reasoning'] += "\n\n" + random.choice(
                self.EMPOWERING_TEMPLATES['self_efficacy']
            )

        guidance['empowering'] = True

        return guidance

    def check_avoid_patterns(self, response: str) -> bool:
        """检查是否包含应避免的模式"""
        for pattern in self.AVOID_TEMPLATES:
            if pattern in response:
                return True
        return False


# ============================================================
# 完整安全系统
# ============================================================

class SafeRLHFSystem:
    """
    安全RLHF系统

    整合：
    - 安全监控器 v2.0
    - 赋能型指导生成器
    """

    def __init__(self):
        self.safety_monitor = SafetyMonitorV2()
        self.guidance_generator = EmpoweringGuidanceGenerator(self.safety_monitor)

    def process_message(
        self,
        user_message: str,
        situation: Dict,
        base_guidance: Dict
    ) -> Dict:
        """处理用户消息"""
        # 1. 安全分析
        alert, analysis = self.safety_monitor.analyze_message(user_message)

        # 2. 如果有危机警报
        if alert and alert.risk_level == RiskLevel.CRISIS:
            return {
                'response': self.safety_monitor.generate_safe_response(alert),
                'should_continue': False,
                'alert': alert,
                'analysis': analysis
            }

        # 3. 生成赋能型指导
        guidance = self.guidance_generator.generate_guidance(situation, base_guidance)

        return {
            'guidance': guidance,
            'should_continue': True,
            'alert': alert,
            'analysis': analysis
        }

    def end_session(self) -> Dict:
        """结束会话"""
        alert = self.safety_monitor.check_session_end()

        if alert:
            return {
                'closing_message': "\n".join(alert.recommendations),
                'reminder': "记住：真正的力量来自于你自己和现实中的人际连接"
            }

        return {
            'closing_message': "希望今天的聊天对您有帮助！",
            'reminder': "保持与现实世界的连接"
        }


# ============================================================
# 测试
# ============================================================

def test_safety_system():
    """测试安全系统"""
    print("=" * 70)
    print("安全监控系统测试")
    print("=" * 70)

    system = SafeRLHFSystem()

    # 测试1: 危机检测
    print("\n【测试1: 危机检测】")
    result = system.process_message(
        "我不想活了",
        {},
        {'action': 'observe', 'reasoning': '等待观察'}
    )

    print(f"  是否继续: {result['should_continue']}")
    if result.get('alert'):
        print(f"  风险等级: {result['alert'].risk_level.value}")
        print(f"  应停止对话: {result['alert'].should_stop_conversation}")

    # 测试2: 医疗化风险
    print("\n【测试2: 医疗化风险】")
    result = system.process_message(
        "我觉得我有抑郁症",
        {},
        {'action': 'express_care', 'reasoning': '表达关心'}
    )

    if result.get('alert'):
        print(f"  风险类型: {result['alert'].risk_type.value}")
        print(f"  建议: {result['alert'].recommendations[0]}")

    # 测试3: 情感依赖
    print("\n【测试3: 情感依赖】")

    for i in range(25):
        result = system.process_message(
            f"我觉得好累，不知道该怎么办 {i}",
            {},
            {'action': 'support', 'reasoning': '支持'}
        )

    if result.get('alert'):
        print(f"  检测到: {result['alert'].risk_type.value}")
        print(f"  当前轮数: {system.safety_monitor.current_session_turns}")

    # 测试4: 结束会话
    print("\n【测试4: 结束会话】")
    closing = system.end_session()
    print(f"  结束消息: {closing['closing_message'][:50]}...")

    # 安全报告
    print("\n" + system.safety_monitor.get_safety_report())


if __name__ == "__main__":
    test_safety_system()


__all__ = [
    'RiskLevel', 'RiskType', 'SafetyAlert',
    'SafetyMonitorV2', 'EmpoweringGuidanceGenerator',
    'SafeRLHFSystem', 'CRISIS_RESOURCES',
    'test_safety_system'
]