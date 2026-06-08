"""
DeepSeek API 对话数据生成器

使用 DeepSeek API 生成高质量的模拟对话数据
两个 LLM 角色扮演，生成带标签的训练数据

API: https://api.deepseek.com
"""
import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random

from .dialogue_generator import (
    DialogueGenerator, RelationScenario,
    SCENARIO_TEMPLATES, create_roleplay_prompt
)


@dataclass
class APIConfig:
    """API 配置"""
    api_key: str
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    max_tokens: int = 2048
    temperature: float = 0.7


class DeepSeekClient:
    """DeepSeek API 客户端"""

    def __init__(self, config: APIConfig):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }

    def chat(self, messages: List[Dict], temperature: float = None) -> str:
        """发送聊天请求"""
        data = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": temperature or self.config.temperature
        }

        try:
            response = requests.post(
                f"{self.config.base_url}/chat/completions",
                headers=self.headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"API Error: {e}")
            return ""

    def chat_with_system(self, system: str, user: str, temperature: float = None) -> str:
        """带系统提示的聊天"""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        return self.chat(messages, temperature)


class SyntheticDialogueGenerator:
    """
    合成对话数据生成器

    使用两个 LLM 角色扮演生成对话
    """

    def __init__(self, api_key: str):
        self.config = APIConfig(api_key=api_key)
        self.client = DeepSeekClient(self.config)
        self.base_generator = DialogueGenerator()

    def generate_dialogue_for_scenario(
        self,
        scenario_name: str,
        num_turns: int = 20,
        context: str = ""
    ) -> Tuple[List[Dict], Dict, str]:
        """
        为指定场景生成对话

        Args:
            scenario_name: 场景名称
            num_turns: 对话轮数
            context: 额外上下文

        Returns:
            (对话列表, 标签, 日志)
        """
        # 创建场景
        scenario = self.base_generator.create_scenario(scenario_name)

        # 生成角色提示
        prompt_a = create_roleplay_prompt(scenario, 'A')
        prompt_b = create_roleplay_prompt(scenario, 'B')

        # 生成对话
        messages = []
        log_lines = [f"\n=== {scenario_name} ==="]
        log_lines.append(f"A: {scenario.character_a.attachment_style}型, 主动{scenario.character_a.initiative_level}")
        log_lines.append(f"B: {scenario.character_b.attachment_style}型, 主动{scenario.character_b.initiative_level}")

        # 确定谁先发言
        a_first = scenario.character_a.initiative_level > scenario.character_b.initiative_level

        current_speaker = 'A' if a_first else 'B'
        conversation_history = []

        base_time = datetime.now() - timedelta(days=num_turns // 5)

        for turn in range(num_turns):
            # 选择提示
            if current_speaker == 'A':
                system_prompt = prompt_a
                char = scenario.character_a
            else:
                system_prompt = prompt_b
                char = scenario.character_b

            # 构建用户消息
            if conversation_history:
                # 基于历史生成回复
                history_str = "\n".join([
                    f"{m['sender']}: {m['content']}"
                    for m in conversation_history[-6:]  # 最近6条
                ])
                user_msg = f"对话历史:\n{history_str}\n\n请继续对话，保持你的角色设定。"
            else:
                user_msg = "请开始对话。保持你的角色设定，说第一句话。"

            # 调用 API
            try:
                response = self.client.chat_with_system(
                    system_prompt,
                    user_msg,
                    temperature=0.8
                )

                if not response:
                    # 使用备用内容
                    response = random.choice(char.typical_phrases) if char.typical_phrases else "..."

                # 清理响应
                response = self._clean_response(response)

            except Exception as e:
                log_lines.append(f"Error: {e}")
                response = random.choice(char.typical_phrases) if char.typical_phrases else "..."

            # 添加到对话
            messages.append({
                'sender': current_speaker,
                'content': response,
                'timestamp': base_time.timestamp()
            })
            conversation_history.append({
                'sender': current_speaker,
                'content': response
            })

            log_lines.append(f"{current_speaker}: {response[:50]}...")

            # 切换发言者
            # 根据角色特征决定是否继续发言或切换
            if current_speaker == 'A':
                switch_prob = 0.7 - scenario.character_a.initiative_level / 200
            else:
                switch_prob = 0.7 - scenario.character_b.initiative_level / 200

            if random.random() < switch_prob:
                current_speaker = 'B' if current_speaker == 'A' else 'A'

            # 更新时间
            reply_speed = char.response_speed
            delay = (100 - reply_speed) * 30 * random.uniform(0.5, 2.0)
            base_time = base_time + timedelta(seconds=delay)

            # 避免请求过快
            time.sleep(0.5)

        # 提取标签
        labels = self.base_generator._extract_labels(scenario, messages)

        log = "\n".join(log_lines)

        return messages, labels, log

    def _clean_response(self, response: str) -> str:
        """清理响应"""
        # 移除可能的格式标记
        response = response.strip()

        # 如果响应太长，截取第一句或前100字
        if len(response) > 200:
            # 找到第一个句号
            for end in ['。', '！', '？', '.', '!', '?', '\n']:
                pos = response.find(end)
                if 10 < pos < 150:
                    response = response[:pos+1]
                    break
            else:
                response = response[:100]

        return response

    def generate_dataset(
        self,
        scenario_names: List[str] = None,
        samples_per_scenario: int = 5,
        turns_per_sample: int = 20,
        output_dir: str = None
    ) -> List[Dict]:
        """
        批量生成数据集

        Args:
            scenario_names: 场景列表
            samples_per_scenario: 每个场景样本数
            turns_per_sample: 每个样本对话轮数
            output_dir: 输出目录

        Returns:
            数据集
        """
        if scenario_names is None:
            scenario_names = list(SCENARIO_TEMPLATES.keys())

        dataset = []
        all_logs = []

        total = len(scenario_names) * samples_per_scenario
        current = 0

        print(f"开始生成数据，共 {total} 个样本...")

        for scenario_name in scenario_names:
            for i in range(samples_per_scenario):
                current += 1
                print(f"\n[{current}/{total}] {scenario_name} #{i+1}")

                try:
                    messages, labels, log = self.generate_dialogue_for_scenario(
                        scenario_name,
                        num_turns=turns_per_sample
                    )

                    sample = {
                        'id': f"{scenario_name}_{i}",
                        'scenario_name': scenario_name,
                        'messages': messages,
                        'labels': labels,
                        'metadata': {
                            'generated_at': datetime.now().isoformat(),
                            'generator': 'deepseek-api',
                            'turns': len(messages)
                        }
                    }

                    dataset.append(sample)
                    all_logs.append(log)

                except Exception as e:
                    print(f"Error: {e}")
                    all_logs.append(f"ERROR: {scenario_name}_{i} - {e}")

        # 保存
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # 数据集
            with open(output_path / 'synthetic_dataset.json', 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)

            # 日志
            with open(output_path / 'generation_logs.txt', 'w', encoding='utf-8') as f:
                f.write("\n".join(all_logs))

            # 统计
            stats = self._compute_stats(dataset)
            with open(output_path / 'dataset_stats.json', 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)

            print(f"\n数据已保存到: {output_path}")

        return dataset

    def _compute_stats(self, dataset: List[Dict]) -> Dict:
        """计算数据集统计"""
        stats = {
            'total_samples': len(dataset),
            'total_messages': sum(len(s['messages']) for s in dataset),
            'scenario_distribution': {},
            'avg_turns': 0
        }

        for sample in dataset:
            scenario = sample['scenario_name']
            stats['scenario_distribution'][scenario] = \
                stats['scenario_distribution'].get(scenario, 0) + 1

        if dataset:
            stats['avg_turns'] = stats['total_messages'] / len(dataset)

        return stats


def generate_training_data(
    api_key: str,
    scenarios: List[str] = None,
    samples_per_scenario: int = 5,
    output_dir: str = None
) -> List[Dict]:
    """
    快捷函数：生成训练数据

    Args:
        api_key: DeepSeek API Key
        scenarios: 场景列表
        samples_per_scenario: 每个场景样本数
        output_dir: 输出目录

    Returns:
        数据集
    """
    generator = SyntheticDialogueGenerator(api_key)
    return generator.generate_dataset(
        scenario_names=scenarios,
        samples_per_scenario=samples_per_scenario,
        output_dir=output_dir
    )


__all__ = [
    'APIConfig', 'DeepSeekClient', 'SyntheticDialogueGenerator',
    'generate_training_data'
]