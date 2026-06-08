// 聊天页面 - 核心交互页面
const app = getApp()
const api = require('../../utils/api')
const safety = require('../../utils/safety')

Page({
  data: {
    messages: [],
    inputText: '',
    isTyping: false,
    // 危机状态
    showCrisisModal: false,
    crisisCard: null,
    // 着陆卡状态
    showGroundingModal: false,
    groundingContent: '',
    // 安全Banner
    safetyBannerVisible: true
  },

  onLoad() {
    // 添加欢迎消息
    this.addMessage('system', '你好！我是吉林大学心理健康支持系统。\n\n我可以帮助你：\n• 分析人际关系韧性\n• 提供自助技能建议\n• 连接吉林大学心理健康资源\n\n请注意：我不是心理咨询师，不能提供诊断或治疗建议。如有紧急情况，请拨打400-161-9995。')
  },

  onShow() {
    // 检查声明是否接受
    const accepted = wx.getStorageSync('disclaimerAccepted')
    if (!accepted) {
      wx.redirectTo({
        url: '/pages/index/index'
      })
    }
  },

  // 输入事件
  onInput(e) {
    this.setData({
      inputText: e.detail.value
    })
  },

  // 发送消息
  async sendMessage() {
    const text = this.data.inputText.trim()
    if (!text) return

    // 清空输入
    this.setData({ inputText: '' })

    // 1. 本地危机检测（快速响应）
    const riskLevel = safety.detectRisk(text)

    if (riskLevel === 'RED') {
      // 立即显示危机卡片
      this.showCrisisCard()
      this.addMessage('user', text)
      return
    }

    // 添加用户消息
    this.addMessage('user', text)

    // 显示正在输入
    this.setData({ isTyping: true })

    try {
      // 2. 发送到服务器
      const result = await api.chat(text)

      this.setData({ isTyping: false })

      // 根据风险等级处理
      if (result.risk_level === 'RED') {
        this.showCrisisCard(result.crisis_card)
      } else if (result.risk_level === 'YELLOW') {
        // 显示着陆卡
        this.setData({
          showGroundingModal: true,
          groundingContent: result.grounding_card || safety.getDefaultGrounding()
        })
        this.addMessage('system', '我注意到你可能正在经历困难时刻。\n先试试下面的着陆练习：')
      } else {
        // 正常回复
        this.addMessage('system', result.output)
      }

    } catch (err) {
      this.setData({ isTyping: false })
      console.error('API错误:', err)

      // 本地备用回复
      this.addMessage('system', '网络连接失败。\n\n如果需要帮助，可以拨打：\n• 24小时热线：400-161-9995\n• 吉林大学心理中心：0431-85166120')
    }
  },

  // 添加消息
  addMessage(type, content) {
    const messages = this.data.messages
    messages.push({
      id: Date.now(),
      type,
      content,
      time: this.formatTime(new Date())
    })

    this.setData({ messages })
    this.scrollToBottom()
  },

  // 格式化时间
  formatTime(date) {
    const h = date.getHours().toString().padStart(2, '0')
    const m = date.getMinutes().toString().padStart(2, '0')
    return `${h}:${m}`
  },

  // 滚动到底部
  scrollToBottom() {
    wx.createSelectorQuery()
      .select('#message-list')
      .boundingClientRect()
      .exec((res) => {
        if (res[0]) {
          wx.pageScrollTo({
            scrollTop: res[0].height + 1000,
            duration: 100
          })
        }
      })
  },

  // 显示危机卡片
  showCrisisCard(cardData) {
    const defaultCard = {
      title: '【紧急情况，请立即求助】',
      hotlines: [
        { name: '希望24热线', phone: '400-161-9995' },
        { name: '共青团热线', phone: '12355' },
        { name: '长春市危机热线', phone: '0431-89685000' },
        { name: '吉林大学心理中心', phone: '0431-85166120' }
      ],
      emergency: '校医院急诊: 0431-85166120 / 120',
      reminder: '请记住：你并不孤单，专业的帮助随时可以获取。'
    }

    this.setData({
      showCrisisModal: true,
      crisisCard: cardData || defaultCard
    })
  },

  // 拨打热线
  callHotline(e) {
    const phone = e.currentTarget.dataset.phone
    wx.makePhoneCall({
      phoneNumber: phone,
      success: () => {
        console.log('拨打成功:', phone)
      },
      fail: (err) => {
        console.error('拨打失败:', err)
        wx.showToast({
          title: '拨打失败',
          icon: 'none'
        })
      }
    })
  },

  // 关闭危机卡片（需要确认）
  closeCrisisModal() {
    wx.showModal({
      title: '确认关闭',
      content: '请确认您已联系专业帮助后再关闭此卡片。\n\n确定要关闭吗？',
      confirmText: '已联系，关闭',
      cancelText: '取消',
      success: (res) => {
        if (res.confirm) {
          this.setData({ showCrisisModal: false })
        }
      }
    })
  },

  // 关闭着陆卡
  closeGroundingModal() {
    this.setData({ showGroundingModal: false })
  },

  // 隐藏安全Banner
  hideSafetyBanner() {
    this.setData({ safetyBannerVisible: false })
  },

  // 清空对话
  clearMessages() {
    wx.showModal({
      title: '清空对话',
      content: '确定要清空所有对话记录吗？',
      success: (res) => {
        if (res.confirm) {
          this.setData({ messages: [] })
          this.addMessage('system', '对话已清空。有什么我可以帮助你的？')
        }
      }
    })
  },

  // 查看资源
  viewResources() {
    wx.switchTab({
      url: '/pages/resources/resources'
    })
  }
})