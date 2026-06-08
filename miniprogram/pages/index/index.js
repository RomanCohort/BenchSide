// 首页 - 欢迎页
Page({
  data: {
    disclaimerAccepted: false,
    features: [
      { icon: '🔒', title: '隐私保护', desc: '数据不上传，本地处理' },
      { icon: '📱', title: '吉林大学资源', desc: '真实官方资源' },
      { icon: '🛡️', title: '安全干预', desc: '危机自动检测' },
      { icon: '✓', title: '幻觉防护', desc: '所有信息有来源' }
    ],
    hotlines: [
      { name: '24小时心理热线', phone: '400-161-9995' },
      { name: '吉林大学心理中心', phone: '0431-85166120' }
    ]
  },

  onLoad() {
    // 检查是否已接受声明
    const accepted = wx.getStorageSync('disclaimerAccepted')
    if (accepted) {
      this.setData({ disclaimerAccepted: true })
    }
  },

  // 接受声明
  acceptDisclaimer() {
    wx.setStorageSync('disclaimerAccepted', true)
    this.setData({ disclaimerAccepted: true })
  },

  // 开始使用
  startUsing() {
    wx.switchTab({
      url: '/pages/chat/chat'
    })
  },

  // 拨打热线
  callHotline(e) {
    const phone = e.currentTarget.dataset.phone
    wx.makePhoneCall({
      phoneNumber: phone
    })
  },

  // 查看资源
  viewResources() {
    wx.switchTab({
      url: '/pages/resources/resources'
    })
  }
})