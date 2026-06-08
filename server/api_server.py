"""
微信小程序后端API服务器

功能：
- 对话接口 /api/chat
- 韧性预测 /api/predict
- 资源列表 /api/resources

运行方式：
    python server/api_server.py

端口：5000
"""

import sys
sys.path.insert(0, '.')

from flask import Flask, request, jsonify
from flask_cors import CORS
import json

from core.jlu_rag_system import JLURAGSystem
from core.hallucination_protection import HallucinationProtectionSystem

app = Flask(__name__)
CORS(app)

# 初始化系统
print("初始化系统...")
rag_system = JLURAGSystem()
hps = HallucinationProtectionSystem(rag_system)
print("系统初始化完成")


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    对话接口

    Request:
        {
            "message": "用户消息",
            "user_id": "用户ID (可选)"
        }

    Response:
        {
            "risk_level": "GREEN/YELLOW/RED",
            "output": "回复内容",
            "sources": ["R01", ...],
            "crisis_card": {...} (RED时),
            "grounding_card": "..." (YELLOW时)
        }
    """
    data = request.get_json()

    if not data or 'message' not in data:
        return jsonify({'error': '缺少message参数'}), 400

    message = data.get('message', '')
    user_id = data.get('user_id', 'anonymous')

    # 处理消息
    result = hps.process_request(message)

    return jsonify(result)


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    韧性预测接口

    Request:
        {
            "nodes": [...],
            "edges": [...]
        }

    Response:
        {
            "resilience_score": 0.65,
            "risk_level": "medium",
            "factors": [...]
        }
    """
    data = request.get_json()

    if not data:
        return jsonify({'error': '缺少数据'}), 400

    # TODO: 实现预测逻辑
    # 使用 Federated GNN 进行预测

    # 模拟响应
    result = {
        'resilience_score': 0.65,
        'risk_level': 'medium',
        'factors': [
            {'name': '导师关系', 'score': 0.5, 'impact': -0.1},
            {'name': '同门支持', 'score': 0.8, 'impact': 0.2}
        ],
        'recommendation': '建议加强与导师的沟通，参考[[B2_01]]边界脚本'
    }

    return jsonify(result)


@app.route('/api/resources', methods=['GET'])
def resources():
    """
    获取资源列表
    """
    resources_list = []

    for rid, resource in rag_system.campus_resources.items():
        resources_list.append({
            'id': rid,
            'name': resource.get('name', ''),
            'phone': resource.get('phone', ''),
            'address': resource.get('address', ''),
            'hours': resource.get('hours', ''),
            'how_to': resource.get('how_to', ''),
            'note': resource.get('note', ''),
            'tags': resource.get('tag', [])
        })

    return jsonify({
        'resources': resources_list,
        'count': len(resources_list)
    })


@app.route('/api/cards', methods=['GET'])
def cards():
    """
    获取技能卡列表
    """
    cards_list = []

    for cid, card in rag_system.self_help_cards.items():
        cards_list.append({
            'id': cid,
            'name': card.get('name', ''),
            'type': card.get('type', ''),
            'content': card.get('content', ''),
            'when_to_use': card.get('when_to_use', '')
        })

    return jsonify({
        'cards': cards_list,
        'count': len(cards_list)
    })


@app.route('/api/health', methods=['GET'])
def health():
    """
    健康检查
    """
    return jsonify({
        'status': 'ok',
        'system': 'JLU Mental Health Support System',
        'version': '1.0.0'
    })


@app.route('/', methods=['GET'])
def index():
    """
    根路径
    """
    return jsonify({
        'name': 'JLU Mental Health Support System API',
        'endpoints': [
            '/api/chat',
            '/api/predict',
            '/api/resources',
            '/api/cards',
            '/api/health'
        ],
        'hotline': '400-161-9995'
    })


if __name__ == '__main__':
    print("=" * 50)
    print("JLU心理健康支持系统 - API服务器")
    print("=" * 50)
    print("端口: 5000")
    print("热线: 400-161-9995")
    print("=" * 50)

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )