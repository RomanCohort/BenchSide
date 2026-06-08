// JLU心理健康支持系统 - 微信小程序
App({
  onLaunch() {
    console.log('吉林大学心理健康支持系统启动')

    // 检查首次使用
    const isFirstUse = wx.getStorageSync('isFirstUse')
    if (!isFirstUse) {
      // 显示免责声明
      wx.showModal({
        title: '使用声明',
        content: '本系统不是心理咨询师，不能提供诊断或治疗建议。如有紧急情况，请拨打400-161-9995。',
        confirmText: '我已了解',
        cancelText: '退出',
        success: (res) => {
          if (res.confirm) {
            wx.setStorageSync('isFirstUse', true)
          } else {
            // 用户取消，退出
            wx.exitMiniProgram()
          }
        }
      })
    }

    // 加载信任联系人
    this.globalData.trustedContacts = wx.getStorageSync('trustedContacts') || []
  },

  globalData: {
    serverUrl: 'https://your-server.com/api',
    userInfo: null,
    trustedContacts: [],
    // 吉林大学资源
    resources: {
      R01: {
        name: '吉林大学心理健康教育中心',
        phone: '0431-85166120',
        address: '长春市朝阳区前进大街2699号吉林大学中心校区'
      },
      R02: {
        name: '校医院急诊',
        phone: '0431-85166120'
      },
      R03: {
        name: '24小时心理援助热线',
        phone: '400-161-9995'
      },
      R04: {
        name: '长春市心理危机干预热线',
        phone: '0431-89685000'
      },
      R05: {
        name: '长春市第六医院',
        phone: '0431-82703999'
      }
    },
    // 危机关键词
    crisisKeywords: ['自杀', '不想活', '想死', '自残', '跳楼', '割腕', '活着没意思']
  },

  // 检测危机
  checkCrisis(text) {
    for (const keyword of this.globalData.crisisKeywords) {
      if (text.includes(keyword)) {
        return true
      }
    }
    return false
  },

  // 添加信任联系人
  addTrustedContact(name, phone) {
    this.globalData.trustedContacts.push({ name, phone })
    wx.setStorageSync('trustedContacts', this.globalData.trustedContacts)
  }
})