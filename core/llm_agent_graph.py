"""
LLM Agent 驱动的社交图谱生成器

核心思想：
- 使用多个 LLM Agent 模拟不同人格的人
- Agent 之间进行真实的社交互动（生成文本消息）
- 构建动态、带文本属性的社交关系图谱
- 用于大规模反馈学习训练

应用：
1. 生成合成训练数据（解决隐私问题）
2. 模拟各种社交场景（正常、冲突、恢复）
3. 测试韧性算法在不同情境下的表现
4. 收集反馈信号进行模型改进

架构：
- PersonaAgent: 模拟特定人格的个体
- SocialNetwork: Agent 组成的动态网络
- InteractionSimulator: 模拟社交互动
- FeedbackCollector: 收集反馈信号
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import re


# ============================================================
# 人格定义
# ============================================================

class AttachmentStyle(Enum):
    """依恋风格"""
    SECURE = "secure"
    ANXIOUS = "anxious"
    AVOIDANT = "avoidant"
    FEARFUL = "fearful"


class PersonalityType(Enum):
    """人格类型（简化版Big Five）"""
    HIGH_EXTRAVERSION = "外向型"
    LOW_EXTRAVERSION = "内向型"
    HIGH_NEUROTICISM = "情绪不稳定型"
    LOW_NEUROTICISM = "情绪稳定型"
    HIGH_AGREEABLENESS = "亲和型"
    LOW_AGREEABLENESS = "独立型"


@dataclass
class AgentPersona:
    """Agent 人格设定"""
    agent_id: str
    name: str
    age: int = 25
    occupation: str = "研究生"

    # 人格特质
    attachment_style: AttachmentStyle = AttachmentStyle.SECURE
    extraversion: float = 0.5         # 外向性 0-1
    neuroticism: float = 0.3          # 神经质 0-1
    agreeableness: float = 0.6        # 亲和性 0-1
    conscientiousness: float = 0.5    # 尽责性 0-1

    # 当前状态
    current_mood: float = 0.5         # 当前情绪 0-1
    energy_level: float = 0.5         # 能量水平 0-1
    stress_level: float = 0.3         # 压力水平 0-1

    # 社交偏好
    preferred_reply_delay: float = 300  # 秒
    preferred_msg_length: float = 30     # 字符
    emoji_usage: float = 0.3              # 表情使用频率

    # 记忆
    interaction_history: List[Dict] = field(default_factory=list)
    relationship_memories: Dict[str, List] = field(default_factory=dict)


# ============================================================
# Agent 行为生成
# ============================================================

class LLMSimulatedAgent:
    """
    LLM 模拟的 Agent

    模拟特定人格的社交行为
    """

    def __init__(self, persona: AgentPersona):
        self.persona = persona
        self.message_templates = self._build_message_templates()

    def _build_message_templates(self) -> Dict:
        """
        构建消息模板库

        根据人格特质定制
        """
        templates = {
            'greeting': [],
            'question': [],
            'response': [],
            'emotional': [],
            'conflict': [],
            'recovery': []
        }

        # 根据依恋风格定制
        if self.persona.attachment_style == AttachmentStyle.ANXIOUS:
            templates['question'] = [
                "你在忙吗？",
                "最近怎么没回复？",
                "是不是不想理我了？",
                "我想你了..."
            ]
            templates['emotional'] = [
                "有点担心...",
                "感觉不太安心",
                "希望你能理解我"
            ]
            templates['recovery'] = [
                "没事了，只是有点敏感",
                "谢谢你的耐心"
            ]

        elif self.persona.attachment_style == AttachmentStyle.AVOIDANT:
            templates['response'] = [
                "嗯",
                "好的",
                "再说吧",
                "最近有点忙"
            ]
            templates['emotional'] = [
                "没什么",
                "不需要担心"
            ]

        elif self.persona.attachment_style == AttachmentStyle.SECURE:
            templates['greeting'] = [
                "最近怎么样？",
                "今天有什么有趣的事吗？"
            ]
            templates['question'] = [
                "需要帮忙吗？",
                "想聊聊天吗？"
            ]
            templates['emotional'] = [
                "很高兴和你分享",
                "感觉挺好的"
            ]

        else:  # FEARFUL
            templates['emotional'] = [
                "有点矛盾...",
                "不知道该怎么说"
            ]

        # 根据外向性定制
        if self.persona.extraversion > 0.6:
            templates['greeting'].extend([
                "哈喽！",
                "最近去玩了什么？",
                "周末约吗？"
            ])
        elif self.persona.extraversion < 0.4:
            templates['greeting'].extend([
                "在吗",
                "有空聊聊吗"
            ])

        return templates

    def generate_message(
        self,
        context: Dict,
        partner_persona: AgentPersona,
        relationship_state: Dict
    ) -> Dict:
        """
        生成消息

        Args:
            context: 上下文（之前的消息、话题等）
            partner_persona: 对方的人格
            relationship_state: 关系状态

        Returns:
            消息字典
        """
        # 确定消息类型
        msg_type = self._determine_message_type(context, relationship_state)

        # 选择模板
        templates = self.message_templates.get(msg_type, ["嗯"])

        # 添加人格影响
        selected_template = self._select_with_personality(templates)

        # 添加情绪修饰
        message_content = self._apply_emotional_modifiers(
            selected_template,
            relationship_state
        )

        # 计算回复延迟
        reply_delay = self._calculate_reply_delay(partner_persona, relationship_state)

        # 计算消息长度
        msg_length = self._calculate_message_length(msg_type)

        # 添加表情（如果人格倾向于使用）
        if self.persona.emoji_usage > 0.3 and np.random.rand() < self.persona.emoji_usage:
            message_content = self._add_emoji(message_content)

        return {
            'sender': self.persona.agent_id,
            'content': message_content,
            'timestamp': datetime.now().timestamp() + reply_delay,
            'delay_seconds': reply_delay,
            'length': msg_length,
            'type': msg_type,
            'emotion': self.persona.current_mood,
            'attachment_signal': self._get_attachment_signal()
        }

    def _determine_message_type(
        self,
        context: Dict,
        relationship_state: Dict
    ) -> str:
        """确定消息类型"""
        # 基于上下文和关系状态

        last_msg = context.get('last_message', {})
        last_type = last_msg.get('type', 'greeting')

        # 关系紧张度
        tension = relationship_state.get('tension', 0)

        if tension > 0.7:
            return 'conflict'
        elif tension > 0.4:
            return 'emotional'

        # 基于上一条消息类型
        if last_type == 'question':
            return 'response'
        elif last_type == 'greeting':
            return np.random.choice(['greeting', 'question'])
        elif last_type == 'conflict':
            return np.random.choice(['conflict', 'recovery'])

        return np.random.choice(['greeting', 'question', 'emotional'])

    def _select_with_personality(self, templates: List[str]) -> str:
        """根据人格选择模板"""
        if not templates:
            return "嗯"

        # 高神经质 = 更可能选择情绪化模板
        if self.persona.neuroticism > 0.6:
            # 找更情绪化的
            emotional_keywords = ['担心', '不安', '难过', '矛盾']
            for t in templates:
                if any(k in t for k in emotional_keywords):
                    return t

        # 低亲和性 = 更可能选择简短模板
        if self.persona.agreeableness < 0.4:
            short_templates = [t for t in templates if len(t) < 10]
            if short_templates:
                return np.random.choice(short_templates)

        return np.random.choice(templates)

    def _apply_emotional_modifiers(
        self,
        template: str,
        relationship_state: Dict
    ) -> str:
        """应用情绪修饰"""
        # 当前情绪低 = 添加消极语气
        if self.persona.current_mood < 0.3:
            modifiers = ['...', '（叹气）', '唉']
            if np.random.rand() < 0.3:
                template = template + np.random.choice(modifiers)

        # 高压力 = 更简短
        if self.persona.stress_level > 0.7:
            template = template.split('，')[0]  # 只取第一部分

        return template

    def _calculate_reply_delay(
        self,
        partner: AgentPersona,
        relationship_state: Dict
    ) -> float:
        """计算回复延迟"""
        base_delay = self.persona.preferred_reply_delay

        # 依恋风格影响
        if self.persona.attachment_style == AttachmentStyle.ANXIOUS:
            # 焦虑型 = 快速回复
            base_delay *= 0.5
        elif self.persona.attachment_style == AttachmentStyle.AVOIDANT:
            # 回避型 = 慢回复
            base_delay *= 2

        # 关系紧张 = 慢回复
        tension = relationship_state.get('tension', 0)
        base_delay *= (1 + tension)

        # 压力大 = 慢回复
        base_delay *= (1 + self.persona.stress_level * 0.5)

        # 添加随机波动
        base_delay *= (0.8 + np.random.rand() * 0.4)

        return base_delay

    def _calculate_message_length(self, msg_type: str) -> int:
        """计算消息长度"""
        base_lengths = {
            'greeting': 15,
            'question': 20,
            'response': 10,
            'emotional': 30,
            'conflict': 40,
            'recovery': 25
        }

        length = base_lengths.get(msg_type, 20)

        # 外向性影响
        length *= (1 + self.persona.extraversion - 0.5)

        return int(length)

    def _add_emoji(self, message: str) -> str:
        """添加表情"""
        emojis = ['😊', '❤', '👍', '🙏', '😂', '🤔', '😊', '✨']

        # 根据情绪选择
        if self.persona.current_mood > 0.6:
            positive_emojis = ['😊', '❤', '👍', '✨']
            return message + np.random.choice(positive_emojis)
        elif self.persona.current_mood < 0.4:
            neutral_emojis = ['🤔', '🙏']
            return message + np.random.choice(neutral_emojis)
        else:
            return message + np.random.choice(emojis)

    def _get_attachment_signal(self) -> Dict:
        """获取依恋信号"""
        return {
            'style': self.persona.attachment_style.value,
            'anxiety_score': self._calculate_anxiety_score(),
            'avoidance_score': self._calculate_avoidance_score()
        }

    def _calculate_anxiety_score(self) -> float:
        """计算焦虑得分"""
        base = 0

        if self.persona.attachment_style == AttachmentStyle.ANXIOUS:
            base = 0.7
        elif self.persona.attachment_style == AttachmentStyle.FEARFUL:
            base = 0.5

        # 神经质增加焦虑
        base += self.persona.neuroticism * 0.2

        return min(base, 1)

    def _calculate_avoidance_score(self) -> float:
        """计算回避得分"""
        base = 0

        if self.persona.attachment_style == AttachmentStyle.AVOIDANT:
            base = 0.7
        elif self.persona.attachment_style == AttachmentStyle.FEARFUL:
            base = 0.5

        # 低外向增加回避
        base += (1 - self.persona.extraversion) * 0.1

        return min(base, 1)

    def update_state(self, interaction_result: Dict):
        """更新 Agent 状态"""
        # 根据互动结果调整情绪
        partner_response = interaction_result.get('partner_response', {})

        # 收到快速回复 = 提升情绪
        if partner_response.get('delay_seconds', 300) < 60:
            self.persona.current_mood = min(self.persona.current_mood + 0.1, 1)

        # 收到慢回复 = 降低情绪（焦虑型更敏感）
        if partner_response.get('delay_seconds', 300) > 3600:
            if self.persona.attachment_style == AttachmentStyle.ANXIOUS:
                self.persona.current_mood = max(self.persona.current_mood - 0.2, 0)
            else:
                self.persona.current_mood = max(self.persona.current_mood - 0.1, 0)

        # 情绪化回复
        if partner_response.get('emotion', 0.5) < 0.3:
            self.persona.stress_level = min(self.persona.stress_level + 0.1, 1)

        # 记录互动
        self.persona.interaction_history.append(interaction_result)


# ============================================================
# 动态社交图谱生成器
# ============================================================

@dataclass
class RelationshipEdge:
    """关系边"""
    source_id: str
    target_id: str
    relationship_type: str  # romantic, friend, family, work

    # 动态状态
    current_tension: float = 0
    current_closeness: float = 0.5
    current_balance: float = 0.5

    # 能量流动
    energy_flow: float = 0

    # 文本历史
    message_history: List[Dict] = field(default_factory=list)

    # 时序特征
    tension_history: List[Tuple[datetime, float]] = field(default_factory=list)
    closeness_history: List[Tuple[datetime, float]] = field(default_factory=list)


@dataclass
class DynamicSocialGraph:
    """动态社交图谱"""
    agents: Dict[str, LLMSimulatedAgent] = field(default_factory=dict)
    edges: Dict[Tuple[str, str], RelationshipEdge] = field(default_factory=dict)

    # 图级特征
    timestamp: datetime = field(default_factory=datetime.now)

    # 场景标签
    scenario: str = "normal_interaction"
    perturbation_type: str = ""


class SocialGraphGenerator:
    """
    社交图谱生成器

    使用 LLM Agent 生成动态图谱
    """

    # 预定义场景
    SCENARIOS = {
        'normal_interaction': {
            'description': '正常日常互动',
            'duration_hours': 24,
            'tension_level': 0.1
        },
        'conflict_escalation': {
            'description': '冲突升级',
            'duration_hours': 48,
            'tension_level': 0.7
        },
        'recovery_process': {
            'description': '冲突后恢复',
            'duration_hours': 72,
            'tension_level': 0.3,
            'previous_tension': 0.7
        },
        'gradual_drift': {
            'description': '逐渐疏远',
            'duration_hours': 168,  # 一周
            'tension_level': 0.5,
            'trend': 'declining'
        },
        'reconnection': {
            'description': '重新连接',
            'duration_hours': 24,
            'tension_level': 0.2,
            'previous_tension': 0.6,
            'trend': 'improving'
        },
        'single_pursuit': {
            'description': '单方面追求',
            'duration_hours': 72,
            'tension_level': 0.4,
            'balance': 0.8  # 一方过度主动
        },
        'mutual_support': {
            'description': '双向支持',
            'duration_hours': 48,
            'tension_level': 0.1,
            'balance': 0.5,
            'support_level': 0.8
        }
    }

    def __init__(self):
        self.current_graph = None
        self.scenario_history = []

    def create_agents(
        self,
        num_agents: int = 5,
        custom_personas: List[AgentPersona] = None
    ) -> Dict[str, LLMSimulatedAgent]:
        """
        创建 Agent 组

        Args:
            num_agents: Agent 数量
            custom_personas: 自定义人格列表
        """
        agents = {}

        if custom_personas:
            for persona in custom_personas:
                agents[persona.agent_id] = LLMSimulatedAgent(persona)
        else:
            # 生成随机人格
            for i in range(num_agents):
                persona = self._generate_random_persona(i)
                agents[persona.agent_id] = LLMSimulatedAgent(persona)

        return agents

    def _generate_random_persona(self, index: int) -> AgentPersona:
        """生成随机人格"""
        # 依恋风格分布（基于人口统计）
        attachment_probs = [0.5, 0.25, 0.15, 0.1]  # secure, anxious, avoidant, fearful
        attachment = np.random.choice(
            list(AttachmentStyle),
            p=attachment_probs
        )

        # 人格特质
        extraversion = np.random.beta(5, 5)  # 中心偏移的分布
        neuroticism = np.random.beta(3, 7)   # 大多数人较低
        agreeableness = np.random.beta(6, 4) # 大多数人较高

        # 依恋风格影响人格
        if attachment == AttachmentStyle.ANXIOUS:
            neuroticism = max(neuroticism, 0.5)
        elif attachment == AttachmentStyle.AVOIDANT:
            extraversion = min(extraversion, 0.4)

        names = ['小明', '小红', '小李', '小王', '小张', '小陈', '小刘', '小赵']
        occupations = ['研究生', '博士生', '上班族', '自由职业']

        return AgentPersona(
            agent_id=f"agent_{index}",
            name=names[index % len(names)],
            age=np.random.randint(22, 35),
            occupation=np.random.choice(occupations),
            attachment_style=attachment,
            extraversion=extraversion,
            neuroticism=neuroticism,
            agreeableness=agreeableness,
            current_mood=np.random.uniform(0.4, 0.7),
            energy_level=np.random.uniform(0.3, 0.8),
            stress_level=np.random.uniform(0.1, 0.4)
        )

    def setup_relationships(
        self,
        agents: Dict[str, LLMSimulatedAgent],
        relationship_config: List[Dict] = None
    ) -> Dict[Tuple[str, str], RelationshipEdge]:
        """
        设置关系网络
        """
        edges = {}

        if relationship_config:
            for config in relationship_config:
                source = config['source']
                target = config['target']
                rel_type = config['type']

                edge = RelationshipEdge(
                    source_id=source,
                    target_id=target,
                    relationship_type=rel_type,
                    current_closeness=config.get('initial_closeness', 0.5),
                    current_balance=config.get('initial_balance', 0.5)
                )
                edges[(source, target)] = edge
        else:
            # 随机生成关系
            agent_ids = list(agents.keys())
            rel_types = ['friend', 'romantic', 'family', 'work']

            # 每个agent至少有一个关系
            for i, agent_id in enumerate(agent_ids):
                # 随机选择另一个agent
                other_idx = np.random.randint(len(agent_ids))
                while other_idx == i:
                    other_idx = np.random.randint(len(agent_ids))

                other_id = agent_ids[other_idx]

                # 创建关系
                edge_key = tuple(sorted([agent_id, other_id]))
                if edge_key not in edges:
                    rel_type = np.random.choice(rel_types)
                    edge = RelationshipEdge(
                        source_id=edge_key[0],
                        target_id=edge_key[1],
                        relationship_type=rel_type,
                        current_closeness=np.random.uniform(0.3, 0.8)
                    )
                    edges[edge_key] = edge

        return edges

    def simulate_scenario(
        self,
        agents: Dict[str, LLMSimulatedAgent],
        edges: Dict[Tuple[str, str], RelationshipEdge],
        scenario_name: str,
        num_interactions: int = 50
    ) -> DynamicSocialGraph:
        """
        模拟特定场景

        Args:
            agents: Agent 组
            edges: 关系网络
            scenario_name: 场景名称
            num_interactions: 互动次数
        """
        scenario_config = self.SCENARIOS.get(scenario_name, self.SCENARIOS['normal_interaction'])

        graph = DynamicSocialGraph(
            agents=agents,
            edges=edges,
            scenario=scenario_name
        )

        # 设置初始紧张度
        for edge in edges.values():
            edge.current_tension = scenario_config.get('tension_level', 0.1)

            if scenario_config.get('previous_tension'):
                # 恢复场景：从之前的高紧张度开始
                edge.current_tension = scenario_config['previous_tension']

        # 模拟互动
        edge_keys = list(edges.keys())
        for i in range(num_interactions):
            # 选择一条关系边
            edge_key = edge_keys[np.random.randint(len(edge_keys))]
            edge = edges[edge_key]

            # 选择主动方
            source_agent = agents[edge.source_id]
            target_agent = agents[edge.target_id]

            # 根据场景调整主动比例
            if scenario_name == 'single_pursuit':
                # 单方面追求：一方更主动
                if np.random.rand() < 0.8:
                    active_agent = source_agent
                    passive_agent = target_agent
                else:
                    active_agent = target_agent
                    passive_agent = source_agent
            else:
                # 正常选择
                if np.random.rand() < 0.5:
                    active_agent = source_agent
                    passive_agent = target_agent
                else:
                    active_agent = target_agent
                    passive_agent = source_agent

            # 构建上下文
            context = {
                'last_message': edge.message_history[-1] if edge.message_history else {},
                'interaction_count': len(edge.message_history),
                'scenario': scenario_name
            }

            # 关系状态
            relationship_state = {
                'tension': edge.current_tension,
                'closeness': edge.current_closeness,
                'balance': edge.current_balance
            }

            # 生成消息
            message = active_agent.generate_message(
                context,
                passive_agent.persona,
                relationship_state
            )

            # 添加到历史
            edge.message_history.append(message)

            # 更新关系状态
            self._update_edge_state(edge, message, scenario_config)

            # 生成回复（如果场景需要）
            if scenario_name != 'gradual_drift' or np.random.rand() < 0.3:
                # 逐渐疏远场景：回复概率低
                reply_context = {'last_message': message}
                reply = passive_agent.generate_message(
                    reply_context,
                    active_agent.persona,
                    relationship_state
                )
                edge.message_history.append(reply)

                # 更新状态
                self._update_edge_state(edge, reply, scenario_config)

        # 记录
        self.scenario_history.append({
            'scenario': scenario_name,
            'timestamp': graph.timestamp,
            'num_interactions': num_interactions,
            'final_tensions': {
                str(k): e.current_tension for k, e in edges.items()
            }
        })

        return graph

    def _update_edge_state(
        self,
        edge: RelationshipEdge,
        message: Dict,
        scenario_config: Dict
    ) -> None:
        """更新边状态"""
        # 紧张度变化
        msg_type = message.get('type', 'greeting')

        tension_changes = {
            'conflict': 0.1,
            'emotional': 0.02,
            'recovery': -0.05,
            'greeting': -0.01,
            'question': 0,
            'response': -0.01
        }

        edge.current_tension += tension_changes.get(msg_type, 0)

        # 场景趋势影响
        if scenario_config.get('trend') == 'declining':
            edge.current_tension += 0.02
        elif scenario_config.get('trend') == 'improving':
            edge.current_tension -= 0.03

        # 边界检查
        edge.current_tension = max(0, min(edge.current_tension, 1))

        # 记录历史
        edge.tension_history.append((datetime.now(), edge.current_tension))
        edge.closeness_history.append((datetime.now(), 1 - edge.current_tension * 0.5))

        # 计算能量流
        # 焦虑型发送者 + 高紧张 = 高投入
        sender_signal = message.get('attachment_signal', {})
        if sender_signal.get('anxiety_score', 0) > 0.5 and edge.current_tension > 0.3:
            edge.energy_flow -= 10  # 消耗能量
        elif sender_signal.get('avoidance_score', 0) > 0.5:
            edge.energy_flow += 5   # 保存能量

    def generate_training_dataset(
        self,
        num_graphs: int = 100,
        scenarios_per_graph: int = 3
    ) -> List[Dict]:
        """
        生成训练数据集

        Args:
            num_graphs: 图数量
            scenarios_per_graph: 每个图的场景数

        Returns:
            数据集
        """
        dataset = []

        for g in range(num_graphs):
            # 创建agents
            agents = self.create_agents(num_agents=np.random.randint(3, 7))

            # 创建关系
            edges = self.setup_relationships(agents)

            # 模拟多个场景
            scenario_sequence = np.random.choice(
                list(self.SCENARIOS.keys()),
                size=scenarios_per_graph,
                replace=False
            )

            for scenario in scenario_sequence:
                graph = self.simulate_scenario(
                    agents, edges, scenario,
                    num_interactions=np.random.randint(30, 100)
                )

                # 提取数据
                graph_data = self._extract_graph_data(graph)

                # 添加标签
                graph_data['scenario_label'] = scenario
                graph_data['resilience_label'] = self._compute_resilience_label(graph)

                dataset.append(graph_data)

        return dataset

    def _extract_graph_data(self, graph: DynamicSocialGraph) -> Dict:
        """提取图数据"""
        data = {
            'agents': {},
            'edges': {},
            'messages': [],
            'features': {}
        }

        # Agent 信息
        for agent_id, agent in graph.agents.items():
            data['agents'][agent_id] = {
                'name': agent.persona.name,
                'attachment_style': agent.persona.attachment_style.value,
                'extraversion': agent.persona.extraversion,
                'neuroticism': agent.persona.neuroticism,
                'current_mood': agent.persona.current_mood,
                'stress_level': agent.persona.stress_level
            }

        # Edge 信息
        for edge_key, edge in graph.edges.items():
            data['edges'][str(edge_key)] = {
                'type': edge.relationship_type,
                'tension': edge.current_tension,
                'closeness': edge.current_closeness,
                'energy_flow': edge.energy_flow,
                'message_count': len(edge.message_history)
            }

        # 消息文本
        for edge_key, edge in graph.edges.items():
            for msg in edge.message_history[-20:]:  # 最近20条
                data['messages'].append({
                    'edge': str(edge_key),
                    'sender': msg['sender'],
                    'content': msg['content'],
                    'type': msg['type'],
                    'emotion': msg['emotion'],
                    'delay': msg['delay_seconds']
                })

        return data

    def _compute_resilience_label(self, graph: DynamicSocialGraph) -> float:
        """计算韧性标签"""
        # 基于最终紧张度和恢复情况
        tensions = [e.current_tension for e in graph.edges.values()]

        if not tensions:
            return 0.5

        # 平均紧张度
        avg_tension = np.mean(tensions)

        # 紧张度变化（如果有历史）
        tension_changes = []
        for e in graph.edges.values():
            if len(e.tension_history) > 1:
                change = e.tension_history[-1][1] - e.tension_history[0][1]
                tension_changes.append(change)

        if tension_changes:
            avg_change = np.mean(tension_changes)
            # 负变化（紧张度下降）= 高韧性
            resilience = 1 - avg_tension - avg_change
        else:
            resilience = 1 - avg_tension

        return max(0, min(resilience, 1))


# ============================================================
# 反馈学习接口
# ============================================================

class FeedbackLearningInterface:
    """
    反馈学习接口

    收集反馈信号，用于模型改进
    """

    def __init__(self):
        self.feedback_buffer = []
        self.reward_history = []

    def collect_feedback(
        self,
        graph_data: Dict,
        model_prediction: float,
        ground_truth: float
    ) -> Dict:
        """
        收集反馈

        Args:
            graph_data: 图数据
            model_prediction: 模型预测
            ground_truth: 真实标签
        """
        feedback = {
            'graph_id': graph_data.get('scenario_label', ''),
            'prediction': model_prediction,
            'ground_truth': ground_truth,
            'error': abs(model_prediction - ground_truth),
            'reward': 1 - abs(model_prediction - ground_truth),  # RL reward
            'timestamp': datetime.now().timestamp()
        }

        self.feedback_buffer.append(feedback)
        self.reward_history.append(feedback['reward'])

        return feedback

    def compute_gradient_signal(self) -> Dict:
        """计算梯度信号（用于更新模型）"""
        if len(self.feedback_buffer) < 10:
            return {'status': 'insufficient_data'}

        # 误差分布
        errors = [f['error'] for f in self.feedback_buffer]

        # 高误差案例 = 需要重点学习
        high_error_cases = [
            f for f in self.feedback_buffer
            if f['error'] > 0.2
        ]

        # 分析错误模式
        error_patterns = {}
        for case in high_error_cases:
            scenario = case['graph_id']
            if scenario not in error_patterns:
                error_patterns[scenario] = []
            error_patterns[scenario].append(case['error'])

        # 每个场景的平均误差
        scenario_errors = {
            s: np.mean(errors) for s, errors in error_patterns.items()
        }

        return {
            'total_samples': len(self.feedback_buffer),
            'average_error': np.mean(errors),
            'average_reward': np.mean(self.reward_history),
            'scenario_errors': scenario_errors,
            'suggested_focus': max(scenario_errors.items(), key=lambda x: x[1])[0] if scenario_errors else None
        }


# ============================================================
# 报告生成
# ============================================================

def create_synthetic_data_report(
    graph: DynamicSocialGraph,
    dataset_stats: Dict = None
) -> str:
    """生成合成数据报告"""
    lines = []

    lines.append("=" * 60)
    lines.append("LLM Agent 生成的社交图谱报告")
    lines.append("=" * 60)

    lines.append("\n【重要说明】")
    lines.append("  本数据由 LLM Agent 模拟生成，用于训练和测试")
    lines.append("  不涉及真实用户数据，完全隐私安全")

    # Agent 信息
    lines.append("\n【Agent 组成】")
    for agent_id, agent in graph.agents.items():
        persona = agent.persona
        lines.append(f"  {persona.name} ({persona.agent_id}):")
        lines.append(f"    依恋风格: {persona.attachment_style.value}")
        lines.append(f"    外向性: {persona.extraversion:.2f}")
        lines.append(f"    神经质: {persona.neuroticism:.2f}")

    # 关系状态
    lines.append("\n【关系状态】")
    for edge_key, edge in graph.edges.items():
        lines.append(f"  {edge_key}: {edge.relationship_type}")
        lines.append(f"    紧张度: {edge.current_tension:.2f}")
        lines.append(f"    亲密度: {edge.current_closeness:.2f}")
        lines.append(f"    消息数: {len(edge.message_history)}")

    # 场景信息
    lines.append("\n【模拟场景】")
    lines.append(f"  场景类型: {graph.scenario}")

    # 消息样本
    lines.append("\n【消息样本】")
    for edge_key, edge in graph.edges.items():
        if edge.message_history:
            lines.append(f"  {edge_key}:")
            for msg in edge.message_history[-3:]:
                sender_name = graph.agents[msg['sender']].persona.name
                lines.append(f"    {sender_name}: \"{msg['content']}\"")

    if dataset_stats:
        lines.append("\n【数据集统计】")
        lines.append(f"  总图数: {dataset_stats.get('num_graphs', 0)}")
        lines.append(f"  总消息数: {dataset_stats.get('total_messages', 0)}")

    return "\n".join(lines)


__all__ = [
    'AttachmentStyle', 'PersonalityType', 'AgentPersona',
    'LLMSimulatedAgent',
    'RelationshipEdge', 'DynamicSocialGraph',
    'SocialGraphGenerator',
    'FeedbackLearningInterface',
    'create_synthetic_data_report'
]