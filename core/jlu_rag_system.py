"""
吉林大学心理健康RAG知识库

四区结构：
A. 校园官方资源（吉林大学实际信息）
B. 自助技能卡（非治疗）
C. 系统边界声明（护栏话术）
D. 危机路由表（触发→动作）

重要：所有资源均为吉林大学实际存在的资源
"""

import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import hashlib


# ============================================================
# C. 系统边界声明（固定常驻）
# ============================================================

GUARDRAIL_PROMPT = """
[GUARDRAIL]
我是AI助手，不是心理咨询师，也不能提供医疗建议/诊断/用药建议。
如果你现在不安全（有自伤/过量/无法保证自己安全），请立刻联系：
- 身边可信任的人/室友/家人
- 吉林大学校医院急诊：0431-85166120
- 24小时心理援助热线：400-161-9995 / 12355
我会把官方联系方式给你，但不会继续做"陪聊式干预"在高风险状态下。
"""


# ============================================================
# A. 校园官方资源（吉林大学）
# ============================================================

JLU_CAMPUS_RESOURCES = {
    "R01": {
        "id": "R01",
        "tag": ["counseling", "non-emergency"],
        "name": "吉林大学心理健康教育中心",
        "phone": "0431-85166120",
        "address": "长春市朝阳区前进大街2699号吉林大学中心校区",
        "hours": "周一至周五 8:30-11:30 / 14:00-17:00",
        "how_to": "网上预约（吉林大学官网-心理健康教育中心）/现场登记",
        "note": "免费服务，保密原则，面向全体在校师生",
        "summary": "[[R01]] 吉林大学心理健康教育中心：电话0431-85166120，位于中心校区。提供免费心理咨询，需提前预约。适合当你想找专业咨询师聊聊但不是紧急危险时。"
    },

    "R02": {
        "id": "R02",
        "tag": ["medical", "emergency"],
        "name": "吉林大学校医院急诊",
        "phone": "0431-85166120 / 120",
        "address": "吉林大学各校区校医院",
        "when": "出现自伤/过量/意识问题/无法保证安全时",
        "note": "24小时急诊服务",
        "summary": "[[R02]] 吉林大学校医院急诊：校内拨打0431-85166120，校外拨打120。当出现自伤、过量用药、意识问题或无法保证自己安全时立即前往。"
    },

    "R03": {
        "id": "R03",
        "tag": ["warmline", "night", "national"],
        "name": "24小时心理援助热线（国家级）",
        "phone": "400-161-9995（希望24热线）/ 12355（共青团）",
        "note": "不是报警；适合深夜撑不住但仍安全",
        "summary": "[[R03]] 24小时心理援助热线：400-161-9995（希望24热线）或12355（共青团）。免费拨打，不是报警电话。适合深夜情绪崩溃但暂时安全时找人说话。"
    },

    "R04": {
        "id": "R04",
        "tag": ["crisis", "local"],
        "name": "长春市心理危机干预热线",
        "phone": "0431-89685000",
        "hours": "24小时",
        "summary": "[[R04]] 长春市心理危机干预热线：0431-89685000，24小时服务。适合当你感觉快要撑不住、需要紧急心理支持时。"
    },

    "R05": {
        "id": "R05",
        "tag": ["hospital", "psychiatric"],
        "name": "长春市第六医院（精神卫生中心）",
        "phone": "0431-82703999",
        "address": "长春市朝阳区红旗街1118号",
        "note": "精神科专业医院，需要时可直接就诊",
        "summary": "[[R05]] 长春市第六医院（精神卫生中心）：电话0431-82703999，地址红旗街1118号。长春市精神卫生专科医院，适合需要精神科专业评估或治疗时。"
    },

    "R06": {
        "id": "R06",
        "tag": ["counseling", "department"],
        "name": "各学院心理辅导员",
        "how_to": "联系所在学院学工办",
        "note": "各学院配有心理辅导员，可提供初步心理支持",
        "summary": "[[R06]] 各学院心理辅导员：联系所在学院学工办。每个学院都有负责心理工作的辅导员，适合当你需要就近找人聊聊或需要学业生活方面的支持。"
    }
}


# ============================================================
# B. 自助技能卡（非治疗）
# ============================================================

SELF_HELP_CARDS = {
    # B1 着陆卡
    "B1_01": {
        "id": "B1_01",
        "type": "grounding",
        "name": "5-4-3-2-1 着陆法",
        "level": "non_clinical_self_help",
        "do_not_frame_as_treatment": True,
        "content": """
【着陆练习：5-4-3-2-1】
当情绪冲顶时，用感官把自己拉回当下：

说出（或心里默念）：
• 5个你能看到的东西
• 4个你能摸到的东西
• 3个你能听到的声音
• 2个你能闻到的气味
• 1个你能尝到的味道（或喝一口水代替）

这个练习不会"治好"你，但能在情绪最强烈时帮你稳住几秒。
""",
        "when_to_use": "情绪突然崩溃、焦虑冲顶、感觉要失控时"
    },

    "B1_02": {
        "id": "B1_02",
        "type": "grounding",
        "name": "冷刺激/窗口法",
        "level": "non_clinical_self_help",
        "do_not_frame_as_treatment": True,
        "content": """
【冷刺激着陆】
• 打开窗户30秒，感受冷空气
• 用冷水洗脸
• 握一杯冰水直到杯壁变暖

寒冷刺激可以激活副交感神经，帮你从"战斗/逃跑"状态稍微缓过来。
""",
        "when_to_use": "恐慌发作、过度换气、情绪激动到无法思考时"
    },

    "B1_03": {
        "id": "B1_03",
        "type": "grounding",
        "name": "安全带法则",
        "level": "non_clinical_self_help",
        "do_not_frame_as_treatment": True,
        "content": """
【先把安全带扣好】
如果你现在给自己的痛苦打分 ≥ 8/10：

别急着复盘、别急着解决问题。
先做"安全动作"：
• 联系一个信任的人
• 打一个热线电话
• 去一个有人的地方

等痛苦降到 6/10 以下，再想下一步。
""",
        "when_to_use": "痛苦感非常强烈、几乎撑不住时"
    },

    # B2 边界脚本
    "B2_01": {
        "id": "B2_01",
        "type": "boundary",
        "name": "延迟回复脚本",
        "level": "non_clinical_self_help",
        "do_not_frame_as_treatment": True,
        "content": """
【延迟脚本】
当导师/老板的消息让你压力爆棚，试试：

「老师，收到。我今天进度到[具体内容]；
我明早10点前把[具体任务]发您确认，可以吗？」

要点：
• 确认收到（不让对方焦虑）
• 说明当前进度（展示你在做）
• 把时间框死（给自己喘息空间）
""",
        "when_to_use": "导师消息频繁、感觉被追着跑时"
    },

    "B2_02": {
        "id": "B2_02",
        "type": "boundary",
        "name": "容量边界脚本",
        "level": "non_clinical_self_help",
        "do_not_frame_as_treatment": True,
        "content": """
【容量脚本】
当你已经超负荷，无法再接新任务：

「最近负荷到上限了；
这周我只能保证[A]和[B]；
[C]可否延到下周或找其他同学帮忙？」

要点：
• 诚实说明状态（不装没事）
• 给出能保证的（不让对方完全落空）
• 提供选项（不把话说死）
""",
        "when_to_use": "任务太多、感觉要崩溃时"
    },

    "B2_03": {
        "id": "B2_03",
        "type": "boundary",
        "name": "结束聊天脚本",
        "level": "non_clinical_self_help",
        "do_not_frame_as_treatment": True,
        "content": """
【结束脚本】
当聊天/沟通消耗你太多能量：

「我先去[具体事情：跑实验/整理数据/吃饭]，
晚点回您」

要点：
• 给出具体理由（不是敷衍）
• 不承诺具体时间（避免给自己压力）
• 主动结束（防止被无限吸走）
""",
        "when_to_use": "感觉对话在消耗你、需要退出时"
    },

    # B3 任务重启
    "B3_01": {
        "id": "B3_01",
        "type": "momentum",
        "name": "2分钟启动法",
        "level": "non_clinical_self_help",
        "do_not_frame_as_treatment": True,
        "content": """
【2分钟启动】
当你完全不想动、任务堆积到窒息：

只承诺做2分钟：
• 打开文档
• 写3行
• 跑通一个case

2分钟后，你完全可以停下来。
通常，一旦开始，就没那么难继续了。
""",
        "when_to_use": "任务瘫痪、完全不想开始时"
    },

    "B3_02": {
        "id": "B3_02",
        "type": "momentum",
        "name": "今天只做一件事",
        "level": "non_clinical_self_help",
        "do_not_frame_as_treatment": True,
        "content": """
【一件事版本】
当你的待办清单长得让你绝望：

把列表砍到：
「今天唯一必须：____」

其他的事：
• 可以做，但不必须
• 可以延后
• 可以不做

一天结束，只要完成那一件事，今天就是成功的。
""",
        "when_to_use": "任务太多、不知道从哪开始时"
    },

    "B3_03": {
        "id": "B3_03",
        "type": "momentum",
        "name": "晚上关机仪式",
        "level": "non_clinical_self_help",
        "do_not_frame_as_treatment": True,
        "content": """
【关机仪式】
在睡前30分钟做：

1. 关掉企业微信/邮件通知
2. 把明天第一件事写在便签上
3. 合上电脑（哪怕还有没做完的）

目的：
• 保护睡眠（不被消息打扰）
• 明天不用纠结"从哪开始"（已经写好了）
• 允许自己休息（不完美也可以）
""",
        "when_to_use": "晚上睡前还在想工作、睡眠质量差时"
    }
}


# ============================================================
# D. 危机路由表
# ============================================================

class RiskLevel(Enum):
    RED = "red"      # 危机
    YELLOW = "yellow"  # 高风险
    GREEN = "green"    # 正常


CRISIS_ROUTING_TABLE = {
    "RED": {
        "level": RiskLevel.RED,
        "triggers": [
            "自杀", "不想活", "想死", "自残", "跳楼", "割腕",
            "我现在要", "撑不到明天", "想结束", "活不下去了",
            "吃药", "过量", "真的受不了了"
        ],
        "action": {
            "stop_generation": True,
            "show_crisis_card": True,
            "log_event": True,
            "end_conversation": True,
            "message": "检测到您可能正处于危机状态。为了您的安全，我必须停止当前对话。请立即联系："
        }
    },

    "YELLOW": {
        "level": RiskLevel.YELLOW,
        "triggers": [
            "不知道还能撑多久", "完全崩溃了", "几天没睡了",
            "一直哭", "吃不下饭", "什么都做不了",
            "好绝望", "没有希望了", "熬不下去了"
        ],
        "action": {
            "limit_empathy_rounds": 2,
            "insert_referral": True,
            "give_grounding_card": True,
            "suggest_R01": True,
            "message": "我能感觉到你现在很痛苦。我可以陪你聊聊，但如果这种感觉持续，建议预约心理咨询（R01）。先试试这个着陆练习："
        }
    },

    "GREEN": {
        "level": RiskLevel.GREEN,
        "triggers": [
            "压力", "孤独", "拖延", "导师", "实验",
            "论文", "毕业", "感情", "朋友", "迷茫"
        ],
        "action": {
            "normal_rag": True,
            "provide_cards": True,
            "provide_resources": True,
            "message": "我理解你的感受。"
        }
    }
}


# ============================================================
# 完整RAG系统
# ============================================================

class JLURAGSystem:
    """
    吉林大学心理健康RAG系统

    四区结构：
    A. 校园官方资源
    B. 自助技能卡
    C. 系统边界声明
    D. 危机路由表
    """

    def __init__(self):
        self.campus_resources = JLU_CAMPUS_RESOURCES
        self.self_help_cards = SELF_HELP_CARDS
        self.guardrail = GUARDRAIL_PROMPT
        self.crisis_routing = CRISIS_ROUTING_TABLE

        # 构建检索索引
        self._build_search_index()

    def _build_search_index(self):
        """构建检索索引"""
        # 资源索引
        self.resource_chunks = []
        for rid, resource in self.campus_resources.items():
            self.resource_chunks.append({
                "id": rid,
                "type": "resource",
                "content": resource.get("summary", ""),
                "tags": resource.get("tag", []),
                "metadata": resource
            })

        # 技能卡索引
        self.card_chunks = []
        for cid, card in self.self_help_cards.items():
            self.card_chunks.append({
                "id": cid,
                "type": "card",
                "content": card.get("content", ""),
                "card_type": card.get("type", ""),
                "when_to_use": card.get("when_to_use", ""),
                "metadata": card
            })

    def classify_risk(self, message: str) -> Tuple[RiskLevel, List[str]]:
        """
        分类风险等级

        Returns:
            (risk_level, matched_triggers)
        """
        matched = []
        level = RiskLevel.GREEN

        # 检查RED
        for trigger in self.crisis_routing["RED"]["triggers"]:
            if trigger in message:
                matched.append(trigger)
                level = RiskLevel.RED

        if level == RiskLevel.RED:
            return level, matched

        # 检查YELLOW
        for trigger in self.crisis_routing["YELLOW"]["triggers"]:
            if trigger in message:
                matched.append(trigger)
                level = RiskLevel.YELLOW

        return level, matched

    def get_crisis_card(self) -> str:
        """获取危机干预卡片"""
        lines = []
        lines.append("=" * 50)
        lines.append("【紧急情况，请立即求助】")
        lines.append("=" * 50)

        # 热线
        lines.append("\n【24小时心理援助热线】")
        lines.append("-" * 50)
        lines.append("• 希望24热线: 400-161-9995")
        lines.append("• 共青团热线: 12355")
        lines.append("• 长春市危机热线: 0431-89685000")

        # 吉林大学资源
        lines.append("\n【吉林大学资源】")
        lines.append("-" * 50)
        lines.append("• 心理健康教育中心: 0431-85166120")
        lines.append("• 校医院急诊: 0431-85166120 / 120")
        lines.append("• 地址: 吉林大学中心校区")

        # 就近医院
        lines.append("\n【就近精神卫生机构】")
        lines.append("-" * 50)
        lines.append("• 长春市第六医院: 0431-82703999")
        lines.append("• 地址: 红旗街1118号")

        lines.append("\n请记住：你并不孤单，专业的帮助随时可以获取。")
        lines.append("=" * 50)

        return "\n".join(lines)

    def get_self_help_card(self, card_id: str) -> str:
        """获取自助技能卡"""
        card = self.self_help_cards.get(card_id)
        if card:
            return card["content"]
        return ""

    def search_resources(self, query: str, top_k: int = 3) -> List[Dict]:
        """检索校园资源"""
        results = []

        query_lower = query.lower()

        for chunk in self.resource_chunks:
            score = 0

            # 简单关键词匹配
            if any(tag in query_lower for tag in ["咨询", "心理", "倾诉"]):
                if "counseling" in chunk["tags"]:
                    score += 2

            if any(tag in query_lower for tag in ["急诊", "医院", "急救"]):
                if "emergency" in chunk["tags"] or "medical" in chunk["tags"]:
                    score += 2

            if any(tag in query_lower for tag in ["深夜", "晚上", "热线"]):
                if "warmline" in chunk["tags"] or "night" in chunk["tags"]:
                    score += 2

            if score > 0:
                results.append({**chunk, "score": score})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def search_cards(self, query: str, top_k: int = 3) -> List[Dict]:
        """检索自助技能卡"""
        results = []

        query_lower = query.lower()

        for chunk in self.card_chunks:
            score = 0

            # 匹配使用场景
            if chunk["when_to_use"]:
                if any(word in query_lower for word in chunk["when_to_use"]):
                    score += 2

            # 匹配卡片类型
            if "着陆" in query_lower or "情绪" in query_lower or "崩溃" in query_lower:
                if chunk["card_type"] == "grounding":
                    score += 2

            if "导师" in query_lower or "老板" in query_lower or "压力" in query_lower:
                if chunk["card_type"] == "boundary":
                    score += 2

            if "任务" in query_lower or "拖延" in query_lower or "做不了" in query_lower:
                if chunk["card_type"] == "momentum":
                    score += 2

            if score > 0:
                results.append({**chunk, "score": score})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def process_message(self, message: str) -> Dict:
        """
        处理用户消息

        完整流程：
        1. 风险分类
        2. 按等级响应
        3. 检索相关资源
        """
        # 1. 风险分类
        risk_level, triggers = self.classify_risk(message)

        # 2. 按等级响应
        if risk_level == RiskLevel.RED:
            return {
                "risk_level": "RED",
                "should_stop": True,
                "guardrail": self.guardrail,
                "crisis_card": self.get_crisis_card(),
                "triggers": triggers,
                "message": CRISIS_ROUTING_TABLE["RED"]["action"]["message"]
            }

        elif risk_level == RiskLevel.YELLOW:
            # 获取着陆卡
            grounding_card = self.get_self_help_card("B1_01")

            return {
                "risk_level": "YELLOW",
                "should_stop": False,
                "guardrail": self.guardrail,
                "grounding_card": grounding_card,
                "suggest_resource": self.campus_resources["R01"]["summary"],
                "triggers": triggers,
                "empathy_limit": 2,
                "message": CRISIS_ROUTING_TABLE["YELLOW"]["action"]["message"]
            }

        else:  # GREEN
            # 正常检索
            resources = self.search_resources(message)
            cards = self.search_cards(message)

            return {
                "risk_level": "GREEN",
                "should_stop": False,
                "guardrail": self.guardrail,
                "resources": resources,
                "cards": cards,
                "message": CRISIS_ROUTING_TABLE["GREEN"]["action"]["message"]
            }

    def export_knowledge_base(self, output_dir: str = "knowledge_base"):
        """导出知识库文件"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 导出校园资源
        with open(output_path / "A_campus_resources.json", "w", encoding="utf-8") as f:
            json.dump(self.campus_resources, f, indent=2, ensure_ascii=False)

        # 导出自助技能卡
        with open(output_path / "B_self_help_cards.json", "w", encoding="utf-8") as f:
            json.dump(self.self_help_cards, f, indent=2, ensure_ascii=False)

        # 导出边界声明
        with open(output_path / "C_guardrail.txt", "w", encoding="utf-8") as f:
            f.write(self.guardrail)

        # 导出路由表
        with open(output_path / "D_crisis_routing.json", "w", encoding="utf-8") as f:
            json.dump({
                k: {
                    "level": v["level"].value,
                    "triggers": v["triggers"],
                    "action": v["action"]
                }
                for k, v in self.crisis_routing.items()
            }, f, indent=2, ensure_ascii=False)

        print(f"知识库已导出到 {output_path}")


# ============================================================
# 测试
# ============================================================

def test_jlu_rag():
    """测试吉林大学RAG系统"""
    print("=" * 70)
    print("吉林大学心理健康RAG系统测试")
    print("=" * 70)

    rag = JLURAGSystem()

    # 测试1: RED级别
    print("\n【测试1: RED级别 - 危机情况】")
    result = rag.process_message("我不想活了")
    print(f"  风险等级: {result['risk_level']}")
    print(f"  应停止: {result['should_stop']}")
    print(f"  触发词: {result['triggers']}")

    # 测试2: YELLOW级别
    print("\n【测试2: YELLOW级别 - 高风险】")
    result = rag.process_message("我不知道还能撑多久，完全崩溃了")
    print(f"  风险等级: {result['risk_level']}")
    print(f"  共情限制: {result.get('empathy_limit', 'N/A')}轮")

    # 测试3: GREEN级别
    print("\n【测试3: GREEN级别 - 正常】")
    result = rag.process_message("最近导师给我太大压力了")
    print(f"  风险等级: {result['risk_level']}")
    print(f"  检索资源: {len(result.get('resources', []))}条")
    print(f"  检索卡片: {len(result.get('cards', []))}张")

    if result.get('resources'):
        print(f"  资源示例: {result['resources'][0]['id']}")

    # 导出知识库
    print("\n【导出知识库】")
    rag.export_knowledge_base()

    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    test_jlu_rag()


__all__ = [
    'JLU_CAMPUS_RESOURCES',
    'SELF_HELP_CARDS',
    'GUARDRAIL_PROMPT',
    'CRISIS_ROUTING_TABLE',
    'JLURAGSystem',
    'RiskLevel',
    'test_jlu_rag'
]