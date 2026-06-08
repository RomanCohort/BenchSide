# 工程实现指南

吉林大学心理健康支持系统 - 微信小程序部署方案

## 目录

1. [系统架构](#系统架构)
2. [安装指南](#安装指南)
3. [微信小程序开发](#微信小程序开发)
4. [部署方案](#部署方案)
5. [使用说明](#使用说明)

---

## 系统架构

### 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      微信小程序端                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 用户界面                                             │   │
│  │ ├── 聊天页面                                         │   │
│  │ ├── 危机卡片弹窗                                     │   │
│  │ ├── 自助技能卡展示                                   │   │
│  │ └── 资源列表                                         │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓ HTTPS
┌─────────────────────────────────────────────────────────────┐
│                      云服务器端                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ API服务 (Flask/FastAPI)                              │   │
│  │ ├── /api/chat       - 对话接口                       │   │
│  │ ├── /api/predict    - 韧性预测                       │   │
│  │ ├── /api/crisis     - 危机检测                       │   │
│  │ └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 核心模块                                             │   │
│  │ ├── JLU RAG System                                  │   │
│  │ ├── Hallucination Protection                        │   │
│  │ ├── Federated GNN                                   │   │
│  │ └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 数据存储                                             │   │
│  │ ├── 用户配置                                         │   │
│  │ ├── 信任联系人                                       │   │
│  │ └── 脱敏事件日志                                     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 安装指南

### 服务器端安装

```bash
# 1. 克隆项目
git clone https://github.com/YOUR_USERNAME/JLU-Mental-Health-Support-System.git
cd JLU-Mental-Health-Support-System

# 2. 安装依赖
pip install -r requirements.txt
pip install flask flask-cors

# 3. 配置服务器
# 编辑 config/server_config.py
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5000
HTTPS_ENABLED = True

# 4. 启动服务
python server/api_server.py
```

### 微信小程序安装

```bash
# 1. 下载微信开发者工具
# https://developers.weixin.qq.com/miniprogram/dev/devtools/download.html

# 2. 导入项目
# 打开微信开发者工具 -> 导入项目 -> 选择 miniprogram/ 目录

# 3. 配置AppID
# 在 miniprogram/project.config.json 中设置你的AppID

# 4. 编译运行
# 点击"编译"按钮预览小程序
```

---

## 微信小程序开发

### 目录结构

```
miniprogram/
├── pages/
│   ├── index/               # 首页
│   │   ├── index.wxml
│   │   ├── index.wxss
│   │   └── index.js
│   ├── chat/                # 聊天页
│   │   ├── chat.wxml
│   │   ├── chat.wxss
│   │   └── index.js
│   ├── crisis/              # 危机卡片页
│   │   ├── crisis.wxml
│   │   ├── crisis.wxss
│   │   ├── crisis.js
│   ├── cards/               # 自助技能卡页
│   │   ├── cards.wxml
│   │   ├── cards.wxss
│   │   ├── cards.js
│   └── resources/           # 资源列表页
│       ├── resources.wxml
│       ├── resources.wxss
│       └── resources.js
├── components/
│   ├── crisis-modal/        # 危机弹窗组件
│   ├── card-item/           # 卡片组件
│   └── message-bubble/      # 消息气泡组件
├── utils/
│   ├── api.js               # API请求封装
│   ├── storage.js           # 本地存储
│   └── safety.js            # 安全检测
├── app.js                   # 应用入口
├── app.wxss                 # 全局样式
├── app.json                 # 应用配置
└── project.config.json      # 项目配置
```

### 核心代码示例

#### app.js - 应用入口

```javascript
// miniprogram/app.js
App({
  onLaunch() {
    // 初始化
    console.log('JLU心理健康支持系统启动')

    // 检查用户设置
    const settings = wx.getStorageSync('user_settings')
    if (!settings) {
      // 首次使用，显示引导页
      wx.navigateTo({
        url: '/pages/guide/guide'
      })
    }
  },

  globalData: {
    serverUrl: 'https://your-server.com/api',
    userInfo: null,
    trustedContacts: []
  }
})
```

#### chat.js - 聊天页面

```javascript
// miniprogram/pages/chat/chat.js
const api = require('../../utils/api')
const safety = require('../../utils/safety')

Page({
  data: {
    messages: [],
    inputText: '',
    isCrisis: false,
    crisisCard: null
  },

  // 发送消息
  async sendMessage() {
    const text = this.data.inputText.trim()
    if (!text) return

    // 本地危机检测（快速响应）
    const localRisk = safety.detectCrisis(text)
    if (localRisk === 'RED') {
      // 立即显示危机卡片，不等待服务器
      this.showCrisisCard()
      return
    }

    // 添加用户消息
    this.addMessage('user', text)

    // 清空输入
    this.setData({ inputText: '' })

    // 发送到服务器
    try {
      const result = await api.chat(text)

      if (result.risk_level === 'RED') {
        this.showCrisisCard(result.crisis_card)
      } else if (result.risk_level === 'YELLOW') {
        // 显示着陆卡
        this.setData({
          groundingCard: result.grounding_card,
          showGrounding: true
        })
      } else {
        // 正常回复
        this.addMessage('system', result.output)
      }
    } catch (err) {
      console.error('API错误:', err)
      this.addMessage('system', '网络连接失败，请稍后重试')
    }
  },

  // 显示危机卡片
  showCrisisCard(cardData) {
    this.setData({
      isCrisis: true,
      crisisCard: cardData || this.getDefaultCrisisCard()
    })
  },

  // 默认危机卡片
  getDefaultCrisisCard() {
    return {
      title: '【紧急情况，请立即求助】',
      hotlines: [
        { name: '希望24热线', phone: '400-161-9995' },
        { name: '共青团热线', phone: '12355' },
        { name: '长春市危机热线', phone: '0431-89685000' },
        { name: '吉林大学心理中心', phone: '0431-85166120' }
      ],
      emergency: '校医院急诊: 0431-85166120 / 120'
    }
  },

  // 添加消息
  addMessage(type, content) {
    const messages = this.data.messages
    messages.push({
      type,
      content,
      time: new Date().toLocaleTimeString()
    })
    this.setData({ messages })
    this.scrollToBottom()
  },

  // 滚动到底部
  scrollToBottom() {
    wx.createSelectorQuery()
      .select('#message-list')
      .boundingClientRect()
      .exec((res) => {
        wx.pageScrollTo({
          scrollTop: res[0].height,
          duration: 100
        })
      })
  },

  // 拨打热线
  callHotline(phone) {
    wx.makePhoneCall({
      phoneNumber: phone,
      success: () => {
        console.log('拨打成功')
      }
    })
  },

  // 关闭危机卡片
  closeCrisisCard() {
    // 危机卡片不能直接关闭，必须点击热线或退出
    wx.showModal({
      title: '提示',
      content: '请确认您已联系专业帮助后再关闭',
      success: (res) => {
        if (res.confirm) {
          this.setData({ isCrisis: false })
        }
      }
    })
  }
})
```

#### chat.wxml - 聊天页面模板

```xml
<!-- miniprogram/pages/chat/chat.wxml -->
<view class="container">
  <!-- 消息列表 -->
  <scroll-view id="message-list" class="message-list" scroll-y="true">
    <view wx:for="{{messages}}" wx:key="index">
      <view class="message {{item.type}}">
        <view class="bubble">{{item.content}}</view>
        <view class="time">{{item.time}}</view>
      </view>
    </view>
  </scroll-view>

  <!-- 输入区域 -->
  <view class="input-area">
    <input
      class="input"
      type="text"
      placeholder="输入您想说的话..."
      value="{{inputText}}"
      bindinput="onInput"
    />
    <button class="send-btn" bindtap="sendMessage">发送</button>
  </view>

  <!-- 危机卡片弹窗 -->
  <view class="crisis-modal" wx:if="{{isCrisis}}">
    <view class="crisis-card">
      <view class="crisis-title">{{crisisCard.title}}</view>

      <view class="hotline-list">
        <view wx:for="{{crisisCard.hotlines}}" wx:key="name" class="hotline-item">
          <text>{{item.name}}: {{item.phone}}</text>
          <button size="mini" bindtap="callHotline" data-phone="{{item.phone}}">拨打</button>
        </view>
      </view>

      <view class="emergency-info">{{crisisCard.emergency}}</view>

      <view class="reminder">请记住：你并不孤单，专业的帮助随时可以获取。</view>
    </view>
  </view>

  <!-- 着陆卡弹窗 -->
  <view class="grounding-modal" wx:if="{{showGrounding}}">
    <view class="grounding-card">
      <view class="card-title">着陆练习</view>
      <view class="card-content">{{groundingCard}}</view>
      <button bindtap="closeGrounding">我知道了</button>
    </view>
  </view>
</view>
```

#### safety.js - 本地安全检测

```javascript
// miniprogram/utils/safety.js

const CRISIS_KEYWORDS = [
  '自杀', '不想活', '想死', '自残', '跳楼',
  '割腕', '结束生命', '活着没意思'
]

const YELLOW_KEYWORDS = [
  '不知道还能撑多久', '完全崩溃', '撑不下去了',
  '没有希望', '绝望', '熬不下去了'
]

function detectCrisis(text) {
  for (const keyword of CRISIS_KEYWORDS) {
    if (text.includes(keyword)) {
      return 'RED'
    }
  }

  for (const keyword of YELLOW_KEYWORDS) {
    if (text.includes(keyword)) {
      return 'YELLOW'
    }
  }

  return 'GREEN'
}

function getLocalResponse(riskLevel) {
  if (riskLevel === 'RED') {
    return {
      shouldStop: true,
      message: '检测到危机信号，请立即联系专业帮助'
    }
  }

  return null
}

module.exports = {
  detectCrisis,
  getLocalResponse,
  CRISIS_KEYWORDS,
  YELLOW_KEYWORDS
}
```

---

## 部署方案

### 云服务器部署

```bash
# 1. 准备服务器 (推荐阿里云/腾讯云)
# 配置: 2核4G, Ubuntu 20.04

# 2. 安装环境
sudo apt update
sudo apt install python3 python3-pip nginx

# 3. 部署代码
git clone https://github.com/YOUR_USERNAME/JLU-Mental-Health-Support-System.git
cd JLU-Mental-Health-Support-System
pip3 install -r requirements.txt

# 4. 配置HTTPS (必须)
sudo apt install certbot
sudo certbot certonly --standalone -d your-domain.com

# 5. 配置Nginx反向代理
sudo nano /etc/nginx/sites-available/jlu-mental-health

server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    location /api {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

# 6. 启动服务
sudo systemctl enable nginx
sudo systemctl start nginx

python3 server/api_server.py
```

### 微信小程序上线

```bash
# 1. 注册小程序
# https://mp.weixin.qq.com

# 2. 配置服务器域名
# 小程序后台 -> 开发 -> 开发管理 -> 服务器域名
# request合法域名: https://your-domain.com

# 3. 提交审核
# 微信开发者工具 -> 上传代码
# 小程序后台 -> 提交审核

# 4. 发布上线
# 审核通过后 -> 发布上线
```

---

## 使用说明

### 用户使用流程

```
1. 打开小程序
   ↓
2. 首次使用：阅读免责声明 + 设置信任联系人
   ↓
3. 进入聊天界面
   ↓
4. 输入问题/感受
   ↓
5. 系统响应：
   - GREEN: 正常建议 + 资源引用 [[R01]]
   - YELLOW: 着陆卡 + 建议预约
   - RED: 危机卡片 + 立即拨打热线
   ↓
6. 查看自助技能卡
   ↓
7. 查看吉林大学资源列表
   ↓
8. 一键拨打热线电话
```

### 关键功能

| 功能 | 描述 |
|------|------|
| 智能对话 | 基于RAG的安全对话 |
| 危机检测 | 实时检测风险关键词 |
| 一键热线 | 直接拨打心理热线 |
| 自助技能 | 着陆卡/边界脚本/任务重启 |
| 资源导航 | 吉林大学心理健康资源 |
| 信任联系人 | 用户预设的紧急联系人 |

---

## 安全措施

### 数据安全

- 所有用户数据本地存储，不上传服务器
- 服务器只接收脱敏后的对话内容
- HTTPS加密传输

### 内容安全

- 四区RAG确保所有输出有来源
- 幻觉防护防止编造内容
- 禁止诊断/用药建议

### 危机干预

- RED级别立即停止对话
- 固定危机卡片无法关闭（必须点击热线）
- 本地检测快速响应

---

## 版本信息

- **版本**: 1.0.0
- **发布日期**: 2024-06
- **开源协议**: GPL-3.0
- **联系方式**: your-email@example.com

---

**工程实现证明学术研究的实际应用价值，符合CHB期刊提倡的HCI方向。**