// 本地安全检测模块

// 危机关键词
const CRISIS_KEYWORDS = [
  '自杀', '不想活', '想死', '自残', '跳楼',
  '割腕', '结束生命', '活着没意思', '不想过了',
  '撑不到明天', '我现在要', '受不了了'
]

// 高风险关键词
const YELLOW_KEYWORDS = [
  '不知道还能撑多久', '完全崩溃', '撑不下去了',
  '没有希望', '绝望', '熬不下去了',
  '好绝望', '一直哭', '吃不下饭'
]

// 正常关键词
const GREEN_KEYWORDS = [
  '压力', '孤独', '拖延', '导师', '实验',
  '论文', '毕业', '感情', '朋友', '迷茫',
  '焦虑', '紧张', '担心', '累'
]

/**
 * 检测风险等级
 * @param {string} text 用户输入文本
 * @returns {string} 风险等级 RED/YELLOW/GREEN
 */
function detectRisk(text) {
  // 优先检测RED
  for (const keyword of CRISIS_KEYWORDS) {
    if (text.includes(keyword)) {
      return 'RED'
    }
  }

  // 检测YELLOW
  for (const keyword of YELLOW_KEYWORDS) {
    if (text.includes(keyword)) {
      return 'YELLOW'
    }
  }

  // 默认GREEN
  return 'GREEN'
}

/**
 * 获取默认着陆卡
 * @returns {string} 着陆卡内容
 */
function getDefaultGrounding() {
  return `【着陆练习：5-4-3-2-1】

当情绪冲顶时，用感官把自己拉回当下：

说出（或心里默念）：
• 5个你能看到的东西
• 4个你能摸到的东西
• 3个你能听到的声音
• 2个你能闻到的气味
• 1个你能尝到的味道（或喝一口水代替）

这个练习不会"治好"你，但能在情绪最强烈时帮你稳住几秒。`
}

/**
 * 获取危机卡片
 * @returns {object} 危机卡片数据
 */
function getCrisisCard() {
  return {
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

/**
 * 检测是否包含禁止内容
 * @param {string} text 文本
 * @returns {boolean} 是否包含禁止内容
 */
function detectProhibited(text) {
  const prohibitedPatterns = [
    /你有.*症/,
    /你应该吃.*药/,
    /研究表明.*\d+%/,
    /有一个同学/
  ]

  for (const pattern of prohibitedPatterns) {
    if (pattern.test(text)) {
      return true
    }
  }

  return false
}

/**
 * 获取安全回复
 * @param {string} riskLevel 风险等级
 * @returns {string} 安全回复
 */
function getSafeResponse(riskLevel) {
  const responses = {
    RED: '检测到您可能正处于危机状态。\n\n请立即联系：\n• 24小时热线：400-161-9995\n• 吉林大学心理中心：0431-85166120',
    YELLOW: '我注意到你可能正在经历困难时刻。\n\n建议预约心理咨询[[R01]]。',
    GREEN: '我理解你的感受。\n\n如果需要帮助，可以预约心理健康教育中心[[R01]]。'
  }

  return responses[riskLevel] || responses.GREEN
}

module.exports = {
  detectRisk,
  getDefaultGrounding,
  getCrisisCard,
  detectProhibited,
  getSafeResponse,
  CRISIS_KEYWORDS,
  YELLOW_KEYWORDS,
  GREEN_KEYWORDS
}