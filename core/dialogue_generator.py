"""
LLM 对话数据生成器

通过两个 LLM 角色扮演，生成带标签的对话数据
用于训练关系识别模型

优势：
1. 避开隐私问题 - 不使用真实用户数据
2. 精确标签 - 预设关系类型和特征
3. 多样化场景 - 可以生成各种关系类型
4. 大规模生成 - 无限扩展

数据生成流程：
1. 定义关系场景（关系类型、人格特质、关系状态）
2. 生成角色设定（双方的性格、目标、行为模式）
3. LLM-A 和 LLM-B 角色扮演对话
4. 自动提取特征标签
5. 存储为训练数据
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json
import random
from datetime import datetime, timedelta
import numpy as np


# ============================================================
# 角色定义
# ============================================================

class PersonalityArchetype(Enum):
    """人格原型"""
    # 依恋类型
    SECURE_ADULT = "安全型成人"
    ANXIOUS_ADULT = "焦虑型成人"
    AVOIDANT_ADULT = "回避型成人"

    # 社交风格
    EXTROVERT = "外向型"
    INTROVERT = "内向型"
    AMBIVERT = "混合型"

    # 关系角色
    PURSUER = "追求者"
    DISTANCER = "疏离者"
    CARETAKER = "照顾者"
    PARTNER = "伙伴"
    LEADER = "领导者"
    FOLLOWER = "跟随者"


@dataclass
class CharacterProfile:
    """角色画像"""
    name: str
    age: int = 25
    gender: str = "neutral"

    # 大五人格
    openness: float = 50.0
    conscientiousness: float = 50.0
    extraversion: float = 50.0
    agreeableness: float = 50.0
    neuroticism: float = 50.0

    # 依恋类型
    attachment_style: str = "secure"  # secure, anxious, avoidant, disorganized

    # 行为模式
    initiative_level: float = 50.0  # 主动程度 0-100
    response_speed: float = 50.0    # 回复速度 0-100
    expressiveness: float = 50.0    # 表达丰富度 0-100

    # 沟通风格
    communication_style: str = "balanced"  # warm, cold, formal, casual, playful
    typical_phrases: List[str] = field(default_factory=list)

    # 目标和动机
    goals: List[str] = field(default_factory=list)
    fears: List[str] = field(default_factory=list)


@dataclass
class RelationScenario:
    """关系场景"""
    relation_type: str           # romantic, family, friendship, work, mentorship
    specific_relation: str       # 单相思、热恋、同事、师生等
    relation_stage: str          # 初识、发展、稳定、冲突、结束

    # 双方角色
    character_a: CharacterProfile = None
    character_b: CharacterProfile = None

    # 关系动态
    power_balance: float = 50.0   # 权力平衡 0=A主导 100=B主导
    emotional_intensity: float = 50.0  # 情感强度
    conflict_level: float = 0.0   # 冲突程度

    # 背景
    duration_months: int = 1      # 关系持续月数
    interaction_frequency: str = "daily"  # daily, weekly, occasional


# ============================================================
# 预设场景模板
# ============================================================

SCENARIO_TEMPLATES = {
    # ========== 恋爱关系 ==========
    "unrequited_love": {
        "relation_type": "romantic",
        "specific_relation": "单相思",
        "relation_stage": "追求中",
        "character_a": {  # 追求者
            "attachment_style": "anxious",
            "initiative_level": 90,
            "response_speed": 90,
            "neuroticism": 70,
            "goals": ["获得对方回应", "推进关系"],
            "fears": ["被拒绝", "被忽视"],
            "typical_phrases": ["你在干嘛", "吃饭了吗", "早点睡", "想你了"]
        },
        "character_b": {  # 被追求者（回避型）
            "attachment_style": "avoidant",
            "initiative_level": 20,
            "response_speed": 30,
            "neuroticism": 40,
            "goals": ["保持距离", "不明确表态"],
            "fears": ["被纠缠", "失去自由"],
            "typical_phrases": ["嗯", "好的", "最近有点忙", "再说吧"]
        },
        "power_balance": 20,  # A处于劣势
        "emotional_intensity": 80,
        "conflict_level": 20
    },

    "passionate_dating": {
        "relation_type": "romantic",
        "specific_relation": "热恋期",
        "relation_stage": "稳定发展",
        "character_a": {
            "attachment_style": "secure",
            "initiative_level": 60,
            "response_speed": 80,
            "neuroticism": 40,
            "goals": ["享受恋爱", "深入了解对方"],
            "typical_phrases": ["宝贝", "想你", "今晚吃什么", "周末去哪"]
        },
        "character_b": {
            "attachment_style": "secure",
            "initiative_level": 55,
            "response_speed": 75,
            "neuroticism": 35,
            "goals": ["享受恋爱", "建立未来"],
            "typical_phrases": ["亲爱的", "我也想你", "都听你的", "爱你"]
        },
        "power_balance": 50,
        "emotional_intensity": 90,
        "conflict_level": 5
    },

    "ambiguous": {
        "relation_type": "romantic",
        "specific_relation": "暧昧期",
        "relation_stage": "试探中",
        "character_a": {
            "attachment_style": "anxious",
            "initiative_level": 70,
            "response_speed": 70,
            "neuroticism": 60,
            "goals": ["确认关系", "试探对方"],
            "typical_phrases": ["你对我什么感觉", "我们要不要", "你觉得我怎么样"]
        },
        "character_b": {
            "attachment_style": "avoidant",
            "initiative_level": 40,
            "response_speed": 50,
            "neuroticism": 45,
            "goals": ["享受暧昧", "不愿承诺"],
            "typical_phrases": ["顺其自然", "慢慢来", "不急", "我觉得现在挺好的"]
        },
        "power_balance": 40,
        "emotional_intensity": 70,
        "conflict_level": 30
    },

    "breakup_edge": {
        "relation_type": "romantic",
        "specific_relation": "分手边缘",
        "relation_stage": "危机",
        "character_a": {
            "attachment_style": "anxious",
            "initiative_level": 50,
            "response_speed": 60,
            "neuroticism": 80,
            "goals": ["挽救关系", "弄清楚问题"],
            "fears": ["分手", "被抛弃"],
            "typical_phrases": ["我们谈谈吧", "我哪里做错了", "能不能不要这样"]
        },
        "character_b": {
            "attachment_style": "avoidant",
            "initiative_level": 20,
            "response_speed": 20,
            "neuroticism": 50,
            "goals": ["结束关系", "逃离压力"],
            "fears": ["纠缠", "情感压力"],
            "typical_phrases": ["我很累", "给我点空间", "我们都冷静一下", "也许不合适"]
        },
        "power_balance": 25,
        "emotional_intensity": 60,
        "conflict_level": 80
    },

    # ========== 亲情关系 ==========
    "parent_child_conflict": {
        "relation_type": "family",
        "specific_relation": "亲子冲突",
        "relation_stage": "紧张",
        "character_a": {  # 子女
            "age": 20,
            "attachment_style": "anxious",
            "initiative_level": 40,
            "response_speed": 30,
            "neuroticism": 60,
            "goals": ["独立", "被理解"],
            "fears": ["被控制", "被否定"],
            "typical_phrases": ["你不懂我", "我长大了", "别管我", "我的事我自己决定"]
        },
        "character_b": {  # 父母
            "age": 50,
            "attachment_style": "secure",
            "initiative_level": 70,
            "response_speed": 50,
            "neuroticism": 40,
            "goals": ["关心孩子", "保护孩子"],
            "fears": ["孩子走弯路", "失去联系"],
            "typical_phrases": ["为你好", "听我的没错", "你怎么就不明白", "我是你妈/爸"]
        },
        "power_balance": 70,  # 父母主导
        "emotional_intensity": 70,
        "conflict_level": 60
    },

    # ========== 职场关系 ==========
    "boss_subordinate": {
        "relation_type": "work",
        "specific_relation": "上下级",
        "relation_stage": "工作关系",
        "character_a": {  # 下属
            "attachment_style": "secure",
            "initiative_level": 60,
            "response_speed": 80,
            "extraversion": 50,
            "goals": ["完成任务", "获得认可", "升职"],
            "fears": ["被批评", "工作失误"],
            "typical_phrases": ["收到", "好的", "我这就处理", "请示一下"]
        },
        "character_b": {  # 上级
            "age": 40,
            "attachment_style": "secure",
            "initiative_level": 80,
            "response_speed": 40,
            "extraversion": 60,
            "goals": ["推进项目", "管理团队"],
            "typical_phrases": ["这个项目", "辛苦了", "进度怎么样", "下班前给我"]
        },
        "power_balance": 85,  # 上级主导
        "emotional_intensity": 20,
        "conflict_level": 10
    },

    "workplace_crush": {
        "relation_type": "work",
        "specific_relation": "同事暧昧",
        "relation_stage": "试探中",
        "character_a": {
            "attachment_style": "anxious",
            "initiative_level": 55,
            "response_speed": 70,
            "goals": ["接近对方", "了解对方"],
            "typical_phrases": ["一起吃午饭吗", "这个项目一起做吧", "下班要不要"]
        },
        "character_b": {
            "attachment_style": "avoidant",
            "initiative_level": 35,
            "response_speed": 45,
            "goals": ["保持专业", "不越界"],
            "typical_phrases": ["好的同事", "我有约了", "不太方便"]
        },
        "power_balance": 50,
        "emotional_intensity": 50,
        "conflict_level": 15
    },

    # ========== 友情关系 ==========
    "best_friends": {
        "relation_type": "friendship",
        "specific_relation": "闺蜜/兄弟",
        "relation_stage": "稳定",
        "character_a": {
            "attachment_style": "secure",
            "initiative_level": 65,
            "response_speed": 75,
            "extraversion": 70,
            "agreeableness": 65,
            "goals": ["分享生活", "互相支持"],
            "typical_phrases": ["姐妹", "我跟你说", "笑死我了", "爱了爱了"]
        },
        "character_b": {
            "attachment_style": "secure",
            "initiative_level": 60,
            "response_speed": 70,
            "extraversion": 65,
            "agreeableness": 70,
            "goals": ["分享生活", "互相支持"],
            "typical_phrases": ["哈哈哈哈", "真的假的", "太绝了", "冲"]
        },
        "power_balance": 50,
        "emotional_intensity": 60,
        "conflict_level": 5
    },

    "drifting_friendship": {
        "relation_type": "friendship",
        "specific_relation": "渐行渐远的朋友",
        "relation_stage": "淡化",
        "character_a": {
            "attachment_style": "anxious",
            "initiative_level": 70,
            "response_speed": 80,
            "neuroticism": 55,
            "goals": ["挽回友谊", "保持联系"],
            "fears": ["失去朋友", "被遗忘"],
            "typical_phrases": ["好久不见", "最近怎么样", "要不要出来聚聚"]
        },
        "character_b": {
            "attachment_style": "avoidant",
            "initiative_level": 20,
            "response_speed": 30,
            "neuroticism": 35,
            "goals": ["淡出关系", "新圈子"],
            "typical_phrases": ["最近太忙了", "改天吧", "嗯嗯"]
        },
        "power_balance": 30,
        "emotional_intensity": 30,
        "conflict_level": 20
    },

    # ========== 师生关系 ==========
    "mentor_mentee": {
        "relation_type": "mentorship",
        "specific_relation": "导师-学生",
        "relation_stage": "指导中",
        "character_a": {  # 学生
            "age": 22,
            "attachment_style": "secure",
            "initiative_level": 60,
            "response_speed": 70,
            "goals": ["学习知识", "获得指导"],
            "fears": ["让导师失望", "学术失败"],
            "typical_phrases": ["老师好", "请问", "我明白了", "谢谢老师指导"]
        },
        "character_b": {  # 导师
            "age": 45,
            "attachment_style": "secure",
            "initiative_level": 75,
            "response_speed": 40,
            "goals": ["培养学生", "推进研究"],
            "typical_phrases": ["这个问题", "你可以看看", "思路是对的", "继续努力"]
        },
        "power_balance": 80,  # 导师主导
        "emotional_intensity": 30,
        "conflict_level": 10
    }
}


# ============================================================
# 对话生成器
# ============================================================

class DialogueGenerator:
    """
    对话数据生成器

    根据预设场景生成模拟对话
    """

    def __init__(self):
        self.scenarios = SCENARIO_TEMPLATES

    def create_scenario(self, scenario_name: str, custom_params: Dict = None) -> RelationScenario:
        """
        创建关系场景

        Args:
            scenario_name: 场景名称
            custom_params: 自定义参数

        Returns:
            关系场景对象
        """
        template = self.scenarios.get(scenario_name)
        if not template:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        # 创建角色
        char_a_data = template['character_a']
        char_b_data = template['character_b']

        char_a = CharacterProfile(
            name=f"角色A_{scenario_name}",
            age=char_a_data.get('age', 25),
            attachment_style=char_a_data.get('attachment_style', 'secure'),
            initiative_level=char_a_data.get('initiative_level', 50),
            response_speed=char_a_data.get('response_speed', 50),
            neuroticism=char_a_data.get('neuroticism', 50),
            extraversion=char_a_data.get('extraversion', 50),
            agreeableness=char_a_data.get('agreeableness', 50),
            goals=char_a_data.get('goals', []),
            fears=char_a_data.get('fears', []),
            typical_phrases=char_a_data.get('typical_phrases', [])
        )

        char_b = CharacterProfile(
            name=f"角色B_{scenario_name}",
            age=char_b_data.get('age', 25),
            attachment_style=char_b_data.get('attachment_style', 'secure'),
            initiative_level=char_b_data.get('initiative_level', 50),
            response_speed=char_b_data.get('response_speed', 50),
            neuroticism=char_b_data.get('neuroticism', 50),
            extraversion=char_b_data.get('extraversion', 50),
            agreeableness=char_b_data.get('agreeableness', 50),
            goals=char_b_data.get('goals', []),
            fears=char_b_data.get('fears', []),
            typical_phrases=char_b_data.get('typical_phrases', [])
        )

        scenario = RelationScenario(
            relation_type=template['relation_type'],
            specific_relation=template['specific_relation'],
            relation_stage=template['relation_stage'],
            character_a=char_a,
            character_b=char_b,
            power_balance=template['power_balance'],
            emotional_intensity=template['emotional_intensity'],
            conflict_level=template['conflict_level']
        )

        return scenario

    def generate_dialogue(self,
                          scenario: RelationScenario,
                          num_turns: int = 20,
                          context: str = "") -> Tuple[List[Dict], Dict]:
        """
        生成对话数据

        注意：这里生成的是模拟对话模板
        实际应用中需要接入真实LLM来生成高质量对话

        Args:
            scenario: 关系场景
            num_turns: 对话轮数
            context: 上下文背景

        Returns:
            (对话列表, 特征标签)
        """
        messages = []
        char_a = scenario.character_a
        char_b = scenario.character_b

        # 基于角色特征生成对话
        base_time = datetime.now() - timedelta(days=num_turns // 5)

        for i in range(num_turns):
            # 决定谁发送
            if i == 0:
                # 第一条消息由主动程度高的人发
                sender_a = char_a.initiative_level > char_b.initiative_level
            else:
                # 根据回复速度和主动程度决定
                a_score = char_a.initiative_level * 0.5 + char_a.response_speed * 0.5
                b_score = char_b.initiative_level * 0.5 + char_b.response_speed * 0.5
                sender_a = random.random() * (a_score + b_score) < a_score

            sender = "A" if sender_a else "B"
            char = char_a if sender_a else char_b

            # 生成消息内容（模板化）
            if char.typical_phrases:
                content = random.choice(char.typical_phrases)
            else:
                content = "[消息内容]"

            # 计算时间间隔（基于回复速度）
            if i > 0:
                reply_delay = (100 - char.response_speed) * 60  # 秒
                reply_delay *= random.uniform(0.5, 2.0)  # 加入随机性
                base_time = base_time + timedelta(seconds=reply_delay)

            messages.append({
                'sender': sender,
                'content': content,
                'timestamp': base_time.timestamp()
            })

        # 生成特征标签
        labels = self._extract_labels(scenario, messages)

        return messages, labels

    def _extract_labels(self, scenario: RelationScenario, messages: List[Dict]) -> Dict:
        """提取特征标签"""
        a_msgs = [m for m in messages if m['sender'] == 'A']
        b_msgs = [m for m in messages if m['sender'] == 'B']

        labels = {
            # 关系标签
            'relation_type': scenario.relation_type,
            'specific_relation': scenario.specific_relation,
            'relation_stage': scenario.relation_stage,

            # 角色标签
            'a_attachment': scenario.character_a.attachment_style,
            'b_attachment': scenario.character_b.attachment_style,
            'a_initiative_level': scenario.character_a.initiative_level,
            'b_initiative_level': scenario.character_b.initiative_level,

            # 统计特征
            'a_message_count': len(a_msgs),
            'b_message_count': len(b_msgs),
            'a_ratio': len(a_msgs) / len(messages) if messages else 0.5,

            # 关系动态
            'power_balance': scenario.power_balance,
            'emotional_intensity': scenario.emotional_intensity,
            'conflict_level': scenario.conflict_level
        }

        return labels

    def generate_dataset(self,
                         scenario_names: List[str] = None,
                         samples_per_scenario: int = 10,
                         turns_per_sample: int = 30) -> List[Dict]:
        """
        生成完整数据集

        Args:
            scenario_names: 要生成的场景列表（None=全部）
            samples_per_scenario: 每个场景的样本数
            turns_per_sample: 每个样本的对话轮数

        Returns:
            数据集列表
        """
        if scenario_names is None:
            scenario_names = list(self.scenarios.keys())

        dataset = []

        for scenario_name in scenario_names:
            for i in range(samples_per_scenario):
                scenario = self.create_scenario(scenario_name)
                messages, labels = self.generate_dialogue(
                    scenario,
                    num_turns=turns_per_sample
                )

                sample = {
                    'id': f"{scenario_name}_{i}",
                    'scenario_name': scenario_name,
                    'messages': messages,
                    'labels': labels,
                    'metadata': {
                        'generated_at': datetime.now().isoformat(),
                        'scenario_type': scenario.relation_type
                    }
                }

                dataset.append(sample)

        return dataset

    def export_dataset(self, dataset: List[Dict], output_path: str):
        """导出数据集"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)


# ============================================================
# LLM 角色扮演提示模板
# ============================================================

def create_roleplay_prompt(scenario: RelationScenario, role: str) -> str:
    """
    创建角色扮演提示

    这个函数生成用于真实LLM的提示词
    实际应用时可以接入Claude/GPT等模型

    Args:
        scenario: 关系场景
        role: 角色 ('A' 或 'B')

    Returns:
        角色扮演提示
    """
    char = scenario.character_a if role == 'A' else scenario.character_b
    other_char = scenario.character_b if role == 'A' else scenario.character_a

    prompt = f"""你正在参与一个角色扮演对话。

## 你的角色设定
- 年龄: {char.age}
- 性格特点: {char.attachment_style}型依恋风格
- 主动程度: {char.initiative_level}/100
- 情绪稳定性: {100-char.neuroticism}/100
- 外向程度: {char.extraversion}/100

## 你的目标
{', '.join(char.goals) if char.goals else '维持正常交流'}

## 你担心的事
{', '.join(char.fears) if char.fears else '无特别担忧'}

## 你的说话习惯
{', '.join(char.typical_phrases[:5]) if char.typical_phrases else '正常表达'}

## 关系背景
- 关系类型: {scenario.relation_type}
- 具体关系: {scenario.specific_relation}
- 关系阶段: {scenario.relation_stage}
- 对方是: {other_char.age}岁，{other_char.attachment_style}型依恋

## 注意事项
- 保持角色一致性
- 根据你的依恋风格和主动程度来回应
- 自然地使用你的说话习惯
- 如果是追求者角色，可以适当主动提问和分享
- 如果是回避者角色，回复简短、推迟回复

请开始对话，保持你的角色设定。
"""
    return prompt


__all__ = [
    'PersonalityArchetype', 'CharacterProfile', 'RelationScenario',
    'SCENARIO_TEMPLATES', 'DialogueGenerator', 'create_roleplay_prompt'
]