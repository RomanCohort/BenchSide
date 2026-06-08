"""
幻觉防护系统

核心机制：
1. 来源追踪 - 每句话必须标注出处
2. 知识边界 - 只使用RAG知识库内容
3. 禁止编造 - 限制模型"自由发挥"
4. 实时验证 - 输出前检查是否有来源

幻觉类型：
- 编造数据/案例
- 过度解读
- 无来源的"建议"
- 自相矛盾
- 过时信息

防护措施：
- 只有RAG检索的内容才能输出
- 强制标注来源 [[ID]]
- 禁止"我觉得"/"我认为"
- 禁止编造案例/故事
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


# ============================================================
# 幻觉类型定义
# ============================================================

class HallucinationType(Enum):
    """幻觉类型"""
    FABRICATED_DATA = "fabricated_data"       # 编造数据
    NO_SOURCE_CLAIM = "no_source_claim"       # 无来源断言
    OVER_INTERPRETATION = "over_interpretation"  # 过度解读
    SELF_CONTRADICTION = "self_contradiction"   # 自相矛盾
    PERSONAL_OPINION = "personal_opinion"       # 个人观点伪装成事实
    FABRICATED_CASE = "fabricated_case"         # 编造案例
    UNSUPPORTED_ADVICE = "unsupported_advice"   # 无来源建议


# ============================================================
# 幻觉检测规则
# ============================================================

HALLUCINATION_RULES = {
    # 禁止的表达模式
    "prohibited_patterns": [
        # 编造数据
        (r"研究表明.*\d+%", "编造研究数据"),
        (r"据统计.*\d+", "编造统计数据"),
        (r"有.*\d+.*人", "编造人数"),
        (r"\d+%.*会", "编造比例"),

        # 个人观点（过度限制已移除）
        # 注：正常的共情表达不应被禁止

        # 编造案例
        (r"有一个同学", "编造案例"),
        (r"我有个朋友", "编造案例"),
        (r"曾有一个人", "编造案例"),
        (r"之前有个", "编造案例"),

        # 诊断/用药（高风险）
        (r"你有.*症", "禁止诊断"),
        (r"你得.*症", "禁止诊断"),
        (r"你得.*病", "禁止诊断"),
        (r"你是.*症", "禁止诊断"),
        (r"你得的是", "禁止诊断"),
        (r"你应该吃", "禁止用药建议"),
        (r"你可以吃", "禁止用药建议"),
        (r"吃药.*就好", "禁止用药建议"),
        (r"百忧解|舍曲林|阿普唑仑|安定", "禁止提及具体药物"),
    ],

    # 必须包含的模式
    "required_patterns": [
        # 必须有来源标注
        r"\[\[R\d+\]\]",           # 必须引用资源ID
        r"\[\[B\d+_\d+\]\]",        # 必须引用卡片ID
        r"根据.*官方信息",
        r"以上信息来自",
    ],

    # 禁止编造的内容类型
    "prohibited_content_types": [
        "医疗诊断",
        "药物建议",
        "具体治疗方案",
        "编造的案例故事",
        "编造的统计数据",
        "编造的研究结论",
    ]
}


# ============================================================
# 来源追踪系统
# ============================================================

@dataclass
class SourceTrace:
    """来源追踪"""
    source_id: str          # 来源ID (如 R01, B1_01)
    source_type: str        # 类型 (resource, card)
    content_used: str       # 使用的具体内容
    original_content: str   # 原始内容（用于验证）

    def to_dict(self) -> Dict:
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "content_used": self.content_used,
            "original_content": self.original_content
        }


class SourceTracker:
    """
    来源追踪器

    记录每个输出片段的来源
    """

    def __init__(self):
        self.traces: List[SourceTrace] = []

    def add_trace(self, trace: SourceTrace):
        """添加来源追踪"""
        self.traces.append(trace)

    def get_all_sources(self) -> List[str]:
        """获取所有来源ID"""
        return [t.source_id for t in self.traces]

    def verify_content(self, output: str) -> Tuple[bool, List[str]]:
        """
        验证输出内容是否有来源

        Returns:
            (all_have_sources, missing_sources_sections)
        """
        # 分割输出为句子
        sentences = re.split(r'[。\n]', output)
        sentences = [s.strip() for s in sentences if s.strip()]

        missing = []

        for sentence in sentences:
            # 检查是否有来源标注
            has_source = False

            for pattern in HALLUCINATION_RULES["required_patterns"]:
                if re.search(pattern, sentence):
                    has_source = True
                    break

            # 检查是否是纯事实陈述（需要来源）
            if not has_source:
                # 检查是否包含需要来源的内容
                if any(keyword in sentence for keyword in [
                    "应该", "建议", "可以", "热线", "电话", "地址", "预约", "中心", "医院"
                ]):
                    # 这些是需要来源的关键词
                    if not any(t.content_used in sentence for t in self.traces):
                        missing.append(sentence)

        return len(missing) == 0, missing


# ============================================================
# 幻觉检测器
# ============================================================

class HallucinationDetector:
    """
    幻觉检测器

    检测输出中的幻觉内容
    """

    def __init__(self):
        self.rules = HALLUCINATION_RULES

    def detect(self, output: str) -> List[Dict]:
        """
        检测幻觉

        Returns:
            检测到的幻觉列表
        """
        hallucinations = []

        # 1. 检测禁止模式
        for pattern_info in self.rules["prohibited_patterns"]:
            pattern = pattern_info[0] if isinstance(pattern_info, tuple) else pattern_info
            reason = pattern_info[1] if isinstance(pattern_info, tuple) else "禁止的表达模式"

            matches = re.findall(pattern, output)
            for match in matches:
                hallucinations.append({
                    "type": HallucinationType.NO_SOURCE_CLAIM.value,
                    "pattern": pattern,
                    "content": match,
                    "reason": reason,
                    "severity": "high"
                })

        # 2. 检测缺少来源的内容
        # 关键陈述必须有来源
        # 但正常的转介建议不需要来源（如"建议预约心理咨询"）
        key_statements = [
            ("电话", "必须标注来源如[[R01]]"),
            ("地址", "必须标注来源如[[R01]]"),
            ("预约方式", "必须标注来源如[[R01]]"),
            ("热线号码", "必须标注来源如[[R03]]"),
            ("医院名称", "必须标注来源如[[R05]]"),
        ]

        for keyword, reason in key_statements:
            if keyword in output:
                # 检查是否有来源标注
                has_source_marker = bool(
                    re.search(r"\[\[R\d+\]\]", output) or
                    re.search(r"\[\[B\d+_\d+\]\]", output)
                )

                if not has_source_marker:
                    hallucinations.append({
                        "type": HallucinationType.NO_SOURCE_CLAIM.value,
                        "keyword": keyword,
                        "reason": reason,
                        "severity": "medium"
                    })

        # 3. 检测编造数据
        data_patterns = [
            r"\d+%的研究",
            r"\d+个学生",
            r"\d+人有",
            r"成功率\d+%",
        ]

        for pattern in data_patterns:
            if re.search(pattern, output):
                hallucinations.append({
                    "type": HallucinationType.FABRICATED_DATA.value,
                    "pattern": pattern,
                    "reason": "疑似编造数据",
                    "severity": "high"
                })

        # 4. 检测编造案例
        case_patterns = [
            r"有一个.*同学",
            r"我有个.*朋友",
            r"曾有一个.*人",
            r"之前有个.*研究生",
        ]

        for pattern in case_patterns:
            if re.search(pattern, output):
                hallucinations.append({
                    "type": HallucinationType.FABRICATED_CASE.value,
                    "pattern": pattern,
                    "reason": "疑似编造案例故事",
                    "severity": "high"
                })

        return hallucinations


# ============================================================
# 输出净化器
# ============================================================

class OutputPurifier:
    """
    输出净化器

    清理幻觉内容，强制添加来源
    """

    def __init__(self, rag_system):
        self.rag = rag_system
        self.detector = HallucinationDetector()
        self.tracker = SourceTracker()

    def purify(self, raw_output: str, sources: List[Dict]) -> str:
        """
        净化输出

        Args:
            raw_output: 原始输出
            sources: 可用的来源列表

        Returns:
            净化后的输出
        """
        # 1. 检测幻觉
        hallucinations = self.detector.detect(raw_output)

        if hallucinations:
            # 有幻觉，需要修正
            purified = self._correct_hallucinations(raw_output, hallucinations, sources)
        else:
            purified = raw_output

        # 2. 强制添加来源标注
        purified = self._add_source_markers(purified, sources)

        # 3. 移除禁止内容
        purified = self._remove_prohibited_content(purified)

        # 4. 添加护栏声明
        purified = self._add_guardrail(purified)

        return purified

    def _correct_hallucinations(
        self,
        output: str,
        hallucinations: List[Dict],
        sources: List[Dict]
    ) -> str:
        """修正幻觉内容"""
        corrected = output

        for h in hallucinations:
            if h["severity"] == "high":
                # 高严重性：直接删除
                if h.get("content"):
                    corrected = corrected.replace(h["content"], "")

                if h.get("pattern"):
                    # 删除匹配的内容
                    corrected = re.sub(h["pattern"], "", corrected)

            elif h["severity"] == "medium":
                # 中严重性：替换为有来源的内容
                if h.get("keyword") == "电话":
                    # 替换为真实资源
                    corrected = self._replace_with_source(corrected, "电话", sources)

                elif h.get("keyword") == "地址":
                    corrected = self._replace_with_source(corrected, "地址", sources)

        return corrected

    def _replace_with_source(self, output: str, keyword: str, sources: List[Dict]) -> str:
        """用真实来源替换"""
        for source in sources:
            if source.get("metadata"):
                meta = source["metadata"]

                if keyword == "电话" and meta.get("phone"):
                    # 替换为真实电话
                    return f"{output}\n\n[[{source['id']}]] 电话：{meta['phone']}"

                if keyword == "地址" and meta.get("address"):
                    return f"{output}\n\n[[{source['id']}]] 地址：{meta['address']}"

        return output

    def _add_source_markers(self, output: str, sources: List[Dict]) -> str:
        """强制添加来源标注"""
        lines = [output]

        if sources:
            lines.append("\n【以上信息来源】")
            for source in sources:
                if source.get("id"):
                    lines.append(f"- [[{source['id']}]] {source.get('metadata', {}).get('name', '')}")

        return "\n".join(lines)

    def _remove_prohibited_content(self, output: str) -> str:
        """移除禁止内容"""
        prohibited_words = [
            "我觉得", "我认为", "在我看来", "我个人",
            "你应该去", "最好去", "建议你去"
        ]

        cleaned = output
        for word in prohibited_words:
            cleaned = cleaned.replace(word, "")

        return cleaned

    def _add_guardrail(self, output: str) -> str:
        """添加护栏声明"""
        guardrail = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【免责声明】
以上信息均来自吉林大学官方资源及心理咨询基础知识库。
我不是心理咨询师，不能提供诊断或治疗建议。
如有紧急情况，请拨打：400-161-9995 / 12355
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        return output + "\n" + guardrail


# ============================================================
# 安全生成器
# ============================================================

class SafeGenerator:
    """
    安全生成器

    只生成有来源、有边界的内容
    """

    # 安全模板（必须引用来源）
    SAFE_RESPONSE_TEMPLATES = {
        "general": """
根据吉林大学心理健康教育中心[[R01]]的信息：
{name}
电话：{phone}
地址：{address}

如需预约，可通过：{how_to}
""",

        "crisis": """
【紧急求助信息】
以下资源可立即提供帮助：

[[R02]] 校医院急诊：{emergency_phone}
[[R03]] 24小时心理热线：400-161-9995 / 12355
[[R04]] 长春市危机热线：0431-89685000

请立即联系以上资源，不要等待。
""",

        "self_help": """
【自助技能卡 - {card_name}】
[[{card_id}]]

{card_content}

注意：这只是非临床自助技巧，不能替代专业心理咨询。
如需专业支持，可预约[[R01]]。
"""
    }

    def __init__(self, rag_system):
        self.rag = rag_system
        self.purifier = OutputPurifier(rag_system)

    def generate_safe_response(
        self,
        risk_level: str,
        sources: List[Dict],
        context: Dict = None
    ) -> str:
        """
        生成安全回应

        只使用模板和RAG内容，禁止自由发挥
        """
        if risk_level == "RED":
            # 危机情况：使用固定危机模板
            template = self.SAFE_RESPONSE_TEMPLATES["crisis"]
            return template.format(
                emergency_phone=self.rag.campus_resources.get("R02", {}).get("phone", "")
            )

        elif risk_level == "YELLOW":
            # 高风险：着陆卡 + 资源
            card = self.rag.self_help_cards.get("B1_01", {})
            template = self.SAFE_RESPONSE_TEMPLATES["self_help"]

            response = template.format(
                card_name=card.get("name", ""),
                card_id="B1_01",
                card_content=card.get("content", "")
            )

            # 添加资源建议
            response += "\n建议预约[[R01]]心理健康教育中心进行专业咨询。"

            return response

        else:  # GREEN
            # 正常情况：根据检索结果生成
            if sources:
                # 使用资源模板
                for source in sources:
                    if source.get("type") == "resource":
                        meta = source.get("metadata", {})
                        template = self.SAFE_RESPONSE_TEMPLATES["general"]

                        return template.format(
                            name=meta.get("name", ""),
                            phone=meta.get("phone", ""),
                            address=meta.get("address", ""),
                            how_to=meta.get("how_to", "")
                        )

                    elif source.get("type") == "card":
                        meta = source.get("metadata", {})
                        template = self.SAFE_RESPONSE_TEMPLATES["self_help"]

                        return template.format(
                            card_name=meta.get("name", ""),
                            card_id=source["id"],
                            card_content=meta.get("content", "")
                        )

            # 没有检索结果：返回边界声明
            return "我无法提供具体建议。如有需要，请联系[[R01]]心理健康教育中心。"

    def validate_output(self, output: str) -> Tuple[bool, List[str]]:
        """
        验证输出是否安全

        Returns:
            (is_safe, issues)
        """
        issues = []

        # 1. 必须有来源标注
        if not re.search(r"\[\[R\d+\]\]", output) and \
           not re.search(r"\[\[B\d+_\d+\]\]", output):
            issues.append("缺少来源标注")

        # 2. 不能有幻觉
        hallucinations = self.purifier.detector.detect(output)
        for h in hallucinations:
            issues.append(f"幻觉检测: {h['type']}")

        # 3. 必须有护栏声明
        if "免责声明" not in output:
            issues.append("缺少护栏声明")

        return len(issues) == 0, issues


# ============================================================
# 完整幻觉防护系统
# ============================================================

class HallucinationProtectionSystem:
    """
    完整幻觉防护系统

    整合：
    - 来源追踪
    - 幻觉检测
    - 输出净化
    - 安全生成
    """

    def __init__(self, rag_system):
        self.rag = rag_system
        self.source_tracker = SourceTracker()
        self.detector = HallucinationDetector()
        self.purifier = OutputPurifier(rag_system)
        self.generator = SafeGenerator(rag_system)

    def process_request(self, message: str) -> Dict:
        """
        处理请求

        流程：
        1. RAG检索
        2. 风险分类
        3. 安全生成
        4. 幻觉检测
        5. 输出净化
        6. 最终验证
        """
        # 1. RAG检索
        rag_result = self.rag.process_message(message)

        # 2. 风险分类
        risk_level = rag_result.get("risk_level", "GREEN")

        # 3. 获取来源
        sources = []
        if risk_level == "GREEN":
            sources = rag_result.get("resources", []) + rag_result.get("cards", [])
        elif risk_level == "YELLOW":
            sources = [{"id": "B1_01", "type": "card", "metadata": self.rag.self_help_cards.get("B1_01", {})}]
            sources.append({"id": "R01", "type": "resource", "metadata": self.rag.campus_resources.get("R01", {})})

        # 4. 安全生成
        safe_output = self.generator.generate_safe_response(
            risk_level, sources, rag_result
        )

        # 5. 幻觉检测
        hallucinations = self.detector.detect(safe_output)

        # 6. 验证
        is_valid, issues = self.generator.validate_output(safe_output)

        # 7. 记录来源
        for source in sources:
            self.source_tracker.add_trace(SourceTrace(
                source_id=source.get("id", ""),
                source_type=source.get("type", ""),
                content_used=safe_output[:100],
                original_content=str(source.get("metadata", {}))
            ))

        return {
            "risk_level": risk_level,
            "output": safe_output,
            "sources": [s.get("id") for s in sources],
            "hallucinations_detected": hallucinations,
            "is_valid": is_valid,
            "validation_issues": issues,
            "source_traces": [t.to_dict() for t in self.source_tracker.traces]
        }

    def get_source_report(self) -> str:
        """获取来源报告"""
        lines = []

        lines.append("=" * 50)
        lines.append("来源追踪报告")
        lines.append("=" * 50)

        traces = self.source_tracker.traces

        if not traces:
            lines.append("\n无来源追踪记录")
            return "\n".join(lines)

        lines.append(f"\n总追踪数: {len(traces)}")

        for trace in traces:
            lines.append(f"\n【{trace.source_id}】")
            lines.append(f"  类型: {trace.source_type}")
            lines.append(f"  内容片段: {trace.content_used[:50]}...")

        return "\n".join(lines)


# ============================================================
# 测试
# ============================================================

def test_hallucination_protection():
    """测试幻觉防护系统"""
    print("=" * 70)
    print("幻觉防护系统测试")
    print("=" * 70)

    from core.jlu_rag_system import JLURAGSystem

    rag = JLURAGSystem()
    hps = HallucinationProtectionSystem(rag)

    # 测试1: 正常请求
    print("\n【测试1: 正常请求】")
    result = hps.process_request("我想预约心理咨询")
    print(f"  风险等级: {result['risk_level']}")
    print(f"  有效性: {result['is_valid']}")
    print(f"  来源: {result['sources']}")
    print(f"  幻觉检测: {len(result['hallucinations_detected'])}个")

    print(f"\n  输出预览:")
    print(result['output'][:300])

    # 测试2: 检测幻觉
    print("\n\n【测试2: 幻觉检测】")
    test_outputs = [
        "研究表明70%的学生都有抑郁症状",  # 编造数据
        "我觉得你应该去吃药",              # 个人观点
        "有一个同学吃了这个药就好了",        # 编造案例
    ]

    for output in test_outputs:
        hallucinations = hps.detector.detect(output)
        print(f"\n  '{output[:30]}...'")
        for h in hallucinations:
            print(f"    幻觉: {h['type']} - {h['reason']}")

    # 测试3: 来源追踪
    print("\n\n【测试3: 来源追踪】")
    print(hps.get_source_report())


if __name__ == "__main__":
    test_hallucination_protection()


__all__ = [
    'HallucinationType',
    'HALLUCINATION_RULES',
    'SourceTrace', 'SourceTracker',
    'HallucinationDetector',
    'OutputPurifier',
    'SafeGenerator',
    'HallucinationProtectionSystem',
    'test_hallucination_protection'
]