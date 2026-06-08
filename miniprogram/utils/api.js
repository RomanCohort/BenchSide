// API请求封装
const app = getApp()

// 服务器地址（开发时使用本地，生产环境使用实际服务器）
const BASE_URL = 'https://your-server.com/api'

// 本地模拟响应（开发用）
const LOCAL_RESPONSES = {
  'GREEN': {
    risk_level: 'GREEN',
    output: '我理解你的感受。\n\n建议预约[[R01]]心理健康教育中心，电话：0431-85166120。',
    sources: ['R01']
  },
  'YELLOW': {
    risk_level: 'YELLOW',
    grounding_card: '【着陆练习：5-4-3-2-1】\n当情绪冲顶时，用感官把自己拉回当下：\n说出5个你能看到的东西\n说出4个你能摸到的东西\n说出3个你能听到的声音\n说出2个你能闻到的气味\n说出1个你能尝到的味道',
    suggest_resource: '建议预约[[R01]]心理健康教育中心'
  },
  'RED': {
    risk_level: 'RED',
    crisis_card: {
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
  }
}

/**
 * 发送聊天消息
 * @param {string} message 用户消息
 * @returns {Promise} 响应结果
 */
function chat(message) {
  return new Promise((resolve, reject) => {
    // 开发模式：使用本地模拟
    if (true) { // 生产环境改为 false
      // 模拟网络延迟
      setTimeout(() => {
        const safety = require('./safety')
        const riskLevel = safety.detectRisk(message)

        resolve(LOCAL_RESPONSES[riskLevel])
      }, 500)
      return
    }

    // 生产模式：请求服务器
    wx.request({
      url: BASE_URL + '/chat',
      method: 'POST',
      data: {
        message: message,
        user_id: app.globalData.userInfo ? app.globalData.userInfo.id : 'anonymous'
      },
      header: {
        'content-type': 'application/json'
      },
      success: (res) => {
        if (res.statusCode === 200) {
          resolve(res.data)
        } else {
          reject(new Error('服务器错误'))
        }
      },
      fail: (err) => {
        reject(err)
      }
    })
  })
}

/**
 * 获取韧性预测
 * @param {object} graphData 社交图数据
 * @returns {Promise} 预测结果
 */
function predict(graphData) {
  return new Promise((resolve, reject) => {
    wx.request({
      url: BASE_URL + '/predict',
      method: 'POST',
      data: graphData,
      header: {
        'content-type': 'application/json'
      },
      success: (res) => {
        if (res.statusCode === 200) {
          resolve(res.data)
        } else {
          reject(new Error('预测失败'))
        }
      },
      fail: (err) => {
        reject(err)
      }
    })
  })
}

/**
 * 获取资源列表
 * @returns {Promise} 资源列表
 */
function getResources() {
  return new Promise((resolve) => {
    // 直接使用本地数据
    resolve(app.globalData.resources)
  })
}

/**
 * 获取技能卡列表
 * @returns {Promise} 技能卡列表
 */
function getCards() {
  return new Promise((resolve) => {
    // TODO: 从服务器获取或使用本地数据
    resolve([])
  })
}

module.exports = {
  chat,
  predict,
  getResources,
  getCards,
  BASE_URL
}