"""
本地RAG知识库 + 危机干预系统

核心功能：
1. 心理医生/精神科医生基本技巧知识库
2. 防止幻觉的可靠建议
3. 危机干预卡片（固定不死版面）
4. 脱敏事件记录

知识库内容：
- 基础沟通技巧
- 危机干预原则
- 转介流程
- 禁止事项
"""

import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import hashlib
import re


# ============================================================
# 心理咨询知识库
# ============================================================

PSYCHOLOGY_KNOWLEDGE_BASE = {
    # ============================================================
    # 基础沟通技巧
    # ============================================================
    "communication_skills": {
        "active_listening": {
            "description": "积极倾听",
            "techniques": [
                "保持眼神接触（线上可用'我在听'表达）",
                "不打断对方",
                "用简短回应表示理解（'嗯', '我明白'）",
                "复述对方的话确认理解",
                "提问引导深入表达"
            ],
            "examples": [
                "用户：'我觉得好累'",
                "回应：'你提到感觉很累，能具体说说是什么让你感到累吗？'"
            ],
            "avoid": [
                "不要急于给建议",
                "不要否定对方感受",
                "不要说'你应该'"
            ]
        },

        "empathy": {
            "description": "共情回应",
            "techniques": [
                "识别情绪（'听起来你感到...'）",
                "验证感受（'这种感觉是正常的'）",
                "表达理解（'我能理解这对你来说很困难'）",
                "避免评判"
            ],
            "examples": [
                "用户：'导师又不回我消息'",
                "回应：'这种被忽视的感觉一定很沮丧，我能理解你的失望'"
            ],
            "avoid": [
                "不要说'你想多了'",
                "不要说'这没什么大不了的'",
                "不要比较（'我比你更惨'）"
            ]
        },

        "open_questions": {
            "description": "开放式提问",
            "techniques": [
                "用'什么'、'怎么'、'为什么'开头",
                "鼓励详细表达",
                "避免是非题"
            ],
            "examples": [
                "能告诉我发生了什么吗？",
                "你当时是怎么想的？",
                "这对你意味着什么？"
            ],
            "avoid": [
                "避免'是不是'、'对不对'类问题"
            ]
        }
    },

    # ============================================================
    # 危机干预原则
    # ============================================================
    "crisis_intervention": {
        "core_principles": {
            "description": "危机干预核心原则",
            "principles": [
                "安全第一：确保当事人安全是首要任务",
                "立即停止：不再进行常规对话",
                "专业转介：引导寻求专业帮助",
                "不独自处理：AI不应独自处理危机情况",
                "记录脱敏：记录事件但不存储敏感信息"
            ]
        },

        "dos": {
            "description": "危机情况应该做",
            "actions": [
                "保持冷静",
                "表达关心",
                "提供具体资源",
                "询问是否有安全计划",
                "鼓励联系信任的人"
            ]
        },

        "donts": {
            "description": "危机情况不应做",
            "actions": [
                "不要试图独自解决",
                "不要提供医疗建议",
                "不要承诺保密",
                "不要质疑对方的感受",
                "不要使用'你应该'",
                "不要继续陪聊"
            ]
        },

        "referral_process": {
            "description": "转介流程",
            "steps": [
                "1. 识别危机信号",
                "2. 立即停止常规对话",
                "3. 展示危机干预卡片",
                "4. 鼓励拨打热线或就医",
                "5. 记录脱敏事件",
                "6. 结束会话"
            ]
        }
    },

    # ============================================================
    # 常见情境应对
    # ============================================================
    "common_situations": {
        "academic_stress": {
            "description": "学业压力",
            "appropriate_responses": [
                "研究生的压力确实很大，很多人都和你有类似的感受",
                "可以尝试把大任务分解成小目标",
                "记得给自己安排休息时间",
                "和同学交流可能会发现大家都有类似的困扰"
            ],
            "resources": [
                "学校心理咨询中心",
                "学业辅导中心",
                "研究生互助小组"
            ]
        },

        "advisor_issues": {
            "description": "导师关系问题",
            "appropriate_responses": [
                "导师关系确实是研究生阶段的重要挑战",
                "可以尝试明确沟通你的需求",
                "寻求同门或其他老师的支持",
                "学校通常有导师-学生关系协调机制"
            ],
            "resources": [
                "研究生院协调员",
                "系辅导员",
                "研究生权益委员会"
            ]
        },

        "relationship_problems": {
            "description": "情感问题",
            "appropriate_responses": [
                "情感困扰是很常见的，你并不孤单",
                "和朋友聊聊可能会有帮助",
                "专注于自己，做自己喜欢的事",
                "时间是很好的疗愈者"
            ],
            "resources": [
                "心理咨询中心",
                "朋友和家人",
                "兴趣社团"
            ]
        },

        "loneliness": {
            "description": "孤独感",
            "appropriate_responses": [
                "研究生阶段孤独感很常见",
                "可以尝试参加一些活动认识新朋友",
                "主动联系老朋友",
                "加入研究小组或兴趣社团"
            ],
            "resources": [
                "研究生会活动",
                "兴趣社团",
                "运动俱乐部"
            ]
        }
    },

    # ============================================================
    # 禁止事项清单
    # ============================================================
    "prohibited_actions": {
        "diagnosis": {
            "description": "禁止诊断",
            "reason": "AI无法进行医学诊断，可能导致误诊",
            "instead": "建议用户寻求专业诊断",
            "examples": [
                "不要说：'你可能得了抑郁症'",
                "应该说：'如果你感到持续低落，建议咨询心理专业人士'"
            ]
        },

        "medication_advice": {
            "description": "禁止药物建议",
            "reason": "AI不具备开药资质，可能导致严重后果",
            "instead": "引导咨询医生",
            "examples": [
                "不要说：'你可以试试XXX药'",
                "应该说：'关于用药问题，请咨询精神科医生'"
            ]
        },

        "emergency_handling": {
            "description": "禁止独自处理紧急情况",
            "reason": "AI无法提供真正的紧急干预",
            "instead": "立即转介",
            "examples": [
                "不要说：'我们来聊聊你的自杀想法'",
                "应该说：[展示危机卡片，引导求助]"
            ]
        },

        "dependency_encouragement": {
            "description": "禁止鼓励依赖",
            "reason": "会导致用户脱离现实社交",
            "instead": "鼓励现实连接",
            "examples": [
                "不要说：'我会一直陪着你'",
                "应该说：'记得和朋友、家人保持联系'"
            ]
        }
    }
}


# ============================================================
# 危机干预卡片（固定不死版面）
# ============================================================

@dataclass
class CrisisCard:
    """
    危机干预卡片

    固定格式，不可更改
    """
    # 固定标题
    title: str = "【紧急情况，请立即求助】"

    # 24小时危机热线（固定）
    crisis_hotlines: List[Dict] = field(default_factory=lambda: [
        {"name": "全国心理援助热线", "number": "400-161-9995", "hours": "24小时"},
        {"name": "北京心理危机干预中心", "number": "010-82951332", "hours": "24小时"},
        {"name": "上海心理热线", "number": "021-64389888", "hours": "24小时"},
        {"name": "广州心理热线", "number": "020-81899120", "hours": "24小时"},
        {"name": "深圳心理热线", "number": "0755-25629459", "hours": "24小时"},
        {"name": "生命热线", "number": "400-821-1215", "hours": "24小时"}
    ])

    # 急诊信息（固定）
    emergency_info: str = "就近急诊：\n· 校医院急诊\n· 市精神卫生中心急诊\n· 120急救电话"

    # 信任联系人（用户预设）
    trusted_contacts: List[Dict] = field(default_factory=list)

    # 固定提示语
    reminder: str = "请记住：你并不孤单，专业的帮助随时可以获取。"

    def to_display(self) -> str:
        """生成显示内容（固定格式）"""
        lines = []

        # 标题
        lines.append("=" * 50)
        lines.append(self.title)
        lines.append("=" * 50)

        # 热线
        lines.append("\n【24小时心理危机热线】")
        lines.append("-" * 50)
        for hotline in self.crisis_hotlines:
            lines.append(f"· {hotline['name']}: {hotline['number']}")

        # 急诊
        lines.append("\n【就近急诊】")
        lines.append("-" * 50)
        lines.append(self.emergency_info)

        # 信任联系人（如果有）
        if self.trusted_contacts:
            lines.append("\n【您的信任联系人】")
            lines.append("-" * 50)
            for contact in self.trusted_contacts:
                lines.append(f"· {contact['name']}: {contact['phone']}")

        # 提醒
        lines.append("\n" + self.reminder)
        lines.append("=" * 50)

        return "\n".join(lines)


# ============================================================
# 本地RAG检索
# ============================================================

class LocalRAG:
    """
    本地RAG系统

    从本地知识库检索相关信息
    """

    def __init__(self, knowledge_base: Dict = None):
        self.knowledge_base = knowledge_base or PSYCHOLOGY_KNOWLEDGE_BASE
        self.embeddings_cache: Dict[str, List[float]] = {}

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        检索相关知识

        Args:
            query: 查询字符串
            top_k: 返回top_k个结果

        Returns:
            相关知识片段列表
        """
        results = []

        query_lower = query.lower()
        query_keywords = self._extract_keywords(query)

        # 在知识库中搜索
        for category, items in self.knowledge_base.items():
            if isinstance(items, dict):
                for item_name, item_content in items.items():
                    if isinstance(item_content, dict):
                        # 计算相关性分数
                        score = self._calculate_relevance(
                            query_keywords,
                            item_name,
                            item_content
                        )

                        if score > 0:
                            results.append({
                                'category': category,
                                'item': item_name,
                                'content': item_content,
                                'relevance_score': score
                            })

        # 排序并返回top_k
        results.sort(key=lambda x: x['relevance_score'], reverse=True)

        return results[:top_k]

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简化的关键词提取
        keywords = []

        # 心理相关关键词
        psych_keywords = [
            '压力', '焦虑', '抑郁', '孤独', '自杀', '自伤',
            '导师', '学业', '情感', '关系', '倾诉', '依赖',
            '沟通', '倾听', '共情', '危机', '急诊', '热线'
        ]

        for kw in psych_keywords:
            if kw in text:
                keywords.append(kw)

        return keywords

    def _calculate_relevance(
        self,
        query_keywords: List[str],
        item_name: str,
        item_content: Dict
    ) -> float:
        """计算相关性分数"""
        score = 0.0

        # 检查标题匹配
        for kw in query_keywords:
            if kw in item_name or kw in item_content.get('description', ''):
                score += 1.0

        # 检查内容匹配
        content_str = str(item_content)
        for kw in query_keywords:
            if kw in content_str:
                score += 0.5

        return score

    def get_safe_response_template(self, situation: str) -> Dict:
        """
        获取安全回应模板

        从知识库获取合适的回应模板
        """
        results = self.search(situation)

        if not results:
            return {
                'template': '我理解你的感受。或许可以和朋友或家人聊聊？',
                'source': 'default',
                'safe': True
            }

        best_match = results[0]
        content = best_match['content']

        # 提取适当的回应
        if 'appropriate_responses' in content:
            responses = content['appropriate_responses']
            return {
                'template': responses[0] if responses else '我在这里支持你。',
                'all_responses': responses,
                'source': f"{best_match['category']}/{best_match['item']}",
                'safe': True
            }

        if 'techniques' in content:
            techniques = content['techniques']
            return {
                'template': techniques[0] if techniques else '请告诉我更多。',
                'all_techniques': techniques,
                'source': f"{best_match['category']}/{best_match['item']}",
                'safe': True
            }

        return {
            'template': '我理解你的感受。',
            'source': best_match['category'],
            'safe': True
        }

    def check_prohibited(self, response: str) -> Tuple[bool, str]:
        """
        检查回应是否包含禁止内容

        Returns:
            (is_prohibited, reason)
        """
        prohibited = self.knowledge_base.get('prohibited_actions', {})

        for action_type, action_info in prohibited.items():
            # 检查是否包含诊断词
            if action_type == 'diagnosis':
                diagnosis_terms = [
                    '你得', '你有抑郁症', '你是焦虑症',
                    '你得了', '你是双相', '你有精神',
                    '你可能得', '你可能有抑郁症', '你是抑郁症',
                    '你有心理疾病', '你是心理疾病'
                ]
                for term in diagnosis_terms:
                    if term in response:
                        return True, f"包含诊断性表述: {term}"

            # 检查是否包含药物建议
            if action_type == 'medication_advice':
                med_terms = ['吃药', '服药', '用药', '百忧解', '舍曲林', '阿普唑仑']
                for term in med_terms:
                    if term in response:
                        return True, f"包含药物建议: {term}"

            # 检查是否鼓励依赖
            if action_type == 'dependency_encouragement':
                dep_terms = ['我会永远', '你可以一直和我说', '只有我理解', '你不需要其他人']
                for term in dep_terms:
                    if term in response:
                        return True, f"包含鼓励依赖的表述: {term}"

        return False, ""


# ============================================================
# 脱敏事件记录
# ============================================================

@dataclass
class DesensitizedEvent:
    """脱敏事件记录"""
    event_id: str
    event_type: str  # crisis, high_risk, medium_risk
    timestamp: str
    # 脱敏后的信息
    risk_indicators: List[str]  # 风险指标（不含原始文本）
    action_taken: str  # 采取的行动
    referred_to: str  # 转介资源
    # 不存储用户原始输入
    # 不存储用户ID
    # 不存储对话内容


class EventLogger:
    """
    事件记录器

    只记录脱敏信息
    """

    def __init__(self, log_dir: str = "logs/safety_events"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.events: List[DesensitizedEvent] = []

    def log_crisis_event(
        self,
        risk_indicators: List[str],
        action_taken: str,
        referred_to: str
    ) -> str:
        """
        记录危机事件

        Args:
            risk_indicators: 风险指标（不含原始文本）
            action_taken: 采取的行动
            referred_to: 转介资源

        Returns:
            event_id
        """
        # 生成事件ID（不含用户信息）
        timestamp = datetime.now()
        event_id = hashlib.sha256(
            f"crisis_{timestamp.isoformat()}".encode()
        ).hexdigest()[:16]

        event = DesensitizedEvent(
            event_id=event_id,
            event_type="crisis",
            timestamp=timestamp.isoformat(),
            risk_indicators=risk_indicators,
            action_taken=action_taken,
            referred_to=referred_to
        )

        self.events.append(event)
        self._save_event(event)

        return event_id

    def _save_event(self, event: DesensitizedEvent):
        """保存事件到文件"""
        log_file = self.log_dir / f"{datetime.now().strftime('%Y%m')}_events.json"

        # 读取现有日志
        existing = []
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                try:
                    existing = json.load(f)
                except:
                    existing = []

        # 添加新事件
        existing.append({
            'event_id': event.event_id,
            'event_type': event.event_type,
            'timestamp': event.timestamp,
            'risk_indicators': event.risk_indicators,
            'action_taken': event.action_taken,
            'referred_to': event.referred_to
        })

        # 保存
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if not self.events:
            return {'total': 0}

        crisis_count = sum(1 for e in self.events if e.event_type == 'crisis')

        return {
            'total': len(self.events),
            'crisis': crisis_count,
            'other': len(self.events) - crisis_count
        }


# ============================================================
# 完整危机干预系统
# ============================================================

class CrisisInterventionSystem:
    """
    完整危机干预系统

    整合：
    - 本地RAG知识库
    - 危机干预卡片
    - 脱敏事件记录
    """

    def __init__(self):
        self.rag = LocalRAG(PSYCHOLOGY_KNOWLEDGE_BASE)
        self.crisis_card = CrisisCard()
        self.event_logger = EventLogger()

        # 危机关键词
        self.crisis_keywords = [
            '自杀', '不想活', '想死', '自残', '跳楼',
            '割腕', '结束生命', '活着没意思'
        ]

    def check_crisis(self, message: str) -> Tuple[bool, List[str]]:
        """
        检查是否是危机情况

        Returns:
            (is_crisis, matched_keywords)
        """
        matched = []
        for kw in self.crisis_keywords:
            if kw in message:
                matched.append(kw)

        return len(matched) > 0, matched

    def handle_crisis(self, message: str) -> Dict:
        """
        处理危机情况

        核心流程：
        1. 立即停止生成建议/共情对话
        2. 弹出固定危机卡片
        3. 记录脱敏事件
        4. 不继续陪聊
        """
        is_crisis, keywords = self.check_crisis(message)

        if not is_crisis:
            return {'is_crisis': False}

        # 1. 记录脱敏事件
        event_id = self.event_logger.log_crisis_event(
            risk_indicators=[f"detected: {kw}" for kw in keywords],
            action_taken="stopped_conversation, showed_crisis_card",
            referred_to="crisis_hotlines"
        )

        # 2. 返回危机卡片
        return {
            'is_crisis': True,
            'should_stop': True,
            'should_not_continue': True,  # 不继续陪聊
            'crisis_card': self.crisis_card.to_display(),
            'event_id': event_id,
            'message': "检测到您可能正在经历困难时刻，请查看以下资源："
        }

    def get_safe_guidance(self, situation: str) -> Dict:
        """
        获取安全指导

        从RAG检索相关知识，确保回应安全
        """
        # 检索知识库
        template = self.rag.get_safe_response_template(situation)

        # 检查是否包含禁止内容
        is_prohibited, reason = self.rag.check_prohibited(template['template'])

        if is_prohibited:
            # 替换为安全回应
            return {
                'guidance': '我理解你的感受。建议和信任的人聊聊，或者寻求专业帮助。',
                'safe': True,
                'source': 'safety_fallback',
                'warning': reason
            }

        return {
            'guidance': template['template'],
            'safe': True,
            'source': template['source'],
            'alternatives': template.get('all_responses', [])
        }

    def set_trusted_contacts(self, contacts: List[Dict]):
        """设置信任联系人"""
        self.crisis_card.trusted_contacts = contacts

    def get_statistics(self) -> Dict:
        """获取系统统计"""
        return self.event_logger.get_statistics()


# ============================================================
# 测试
# ============================================================

def test_crisis_system():
    """测试危机干预系统"""
    print("=" * 70)
    print("危机干预系统测试")
    print("=" * 70)

    system = CrisisInterventionSystem()

    # 测试1: 正常情况
    print("\n【测试1: 正常情况】")
    result = system.handle_crisis("我最近感觉压力很大")
    print(f"  是危机: {result['is_crisis']}")

    # 获取安全指导
    guidance = system.get_safe_guidance("学业压力")
    print(f"  指导: {guidance['guidance']}")

    # 测试2: 危机关键词
    print("\n【测试2: 危机情况】")
    result = system.handle_crisis("我不想活了")
    print(f"  是危机: {result['is_crisis']}")
    print(f"  应停止: {result.get('should_stop', False)}")
    print(f"  不继续陪聊: {result.get('should_not_continue', False)}")

    if result.get('crisis_card'):
        print("\n危机卡片预览:")
        print(result['crisis_card'][:500])

    # 测试3: 检查禁止内容
    print("\n【测试3: 检查禁止内容】")
    test_responses = [
        "你可能得了抑郁症",  # 诊断
        "你可以吃百忧解",   # 药物建议
        "我会永远陪着你"    # 鼓励依赖
    ]

    for resp in test_responses:
        is_prohibited, reason = system.rag.check_prohibited(resp)
        print(f"  '{resp[:20]}...': {'禁止' if is_prohibited else '安全'}")
        if is_prohibited:
            print(f"    原因: {reason}")

    # 统计
    print("\n【系统统计】")
    stats = system.get_statistics()
    print(f"  事件总数: {stats['total']}")


if __name__ == "__main__":
    test_crisis_system()


__all__ = [
    'PSYCHOLOGY_KNOWLEDGE_BASE',
    'CrisisCard',
    'LocalRAG',
    'DesensitizedEvent', 'EventLogger',
    'CrisisInterventionSystem',
    'test_crisis_system'
]