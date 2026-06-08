"""
吉林大学心理健康支持系统 - 快速启动

用法:
    python quick_start.py
"""
import sys
sys.path.insert(0, '.')

def main():
    print("=" * 70)
    print("吉林大学心理健康支持系统")
    print("=" * 70)

    print("\n【系统初始化】")

    from core.jlu_rag_system import JLURAGSystem
    from core.hallucination_protection import HallucinationProtectionSystem

    rag = JLURAGSystem()
    hps = HallucinationProtectionSystem(rag)

    print("  ✓ RAG系统加载完成")
    print("  ✓ 幻觉防护系统加载完成")
    print("  ✓ 吉林大学资源已加载 (R01-R06)")
    print("  ✓ 自助技能卡已加载 (B1_01-B3_03)")

    print("\n" + "=" * 70)
    print("交互模式")
    print("=" * 70)
    print("\n输入 'exit' 或 'quit' 退出")
    print("输入 'help' 查看帮助")
    print("输入 'resources' 查看吉林大学资源")
    print()

    while True:
        try:
            user_input = input("\n[用户] ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['exit', 'quit']:
                print("\n[系统] 感谢使用！如有需要，请拨打400-161-9995")
                break

            if user_input.lower() == 'help':
                print("\n[系统] 可用命令:")
                print("  help      - 查看帮助")
                print("  resources - 查看吉林大学心理健康资源")
                print("  cards     - 查看自助技能卡")
                print("  exit      - 退出系统")
                continue

            if user_input.lower() == 'resources':
                print("\n[系统] 吉林大学心理健康资源:")
                for rid, resource in rag.campus_resources.items():
                    print(f"  [{rid}] {resource['name']}")
                    print(f"       电话: {resource.get('phone', 'N/A')}")
                    print(f"       地址: {resource.get('address', 'N/A')}")
                continue

            if user_input.lower() == 'cards':
                print("\n[系统] 自助技能卡:")
                for cid, card in rag.self_help_cards.items():
                    print(f"  [{cid}] {card['name']} ({card['type']})")
                continue

            # 处理用户消息
            result = hps.process_request(user_input)

            print(f"\n[系统] 风险等级: {result['risk_level']}")
            print(f"[系统] 来源: {result['sources']}")
            print()
            print(result['output'])

        except KeyboardInterrupt:
            print("\n\n[系统] 感谢使用！")
            break
        except Exception as e:
            print(f"\n[系统] 错误: {e}")
            continue


if __name__ == "__main__":
    main()