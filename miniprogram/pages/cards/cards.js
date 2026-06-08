// 技能卡页面
Page({
  data: {
    cards: [],
    categories: ['全部', '着陆', '边界', '任务'],
    currentCategory: '全部'
  },

  onLoad() {
    this.loadCards()
  },

  // 加载技能卡
  loadCards() {
    const cards = [
      // 着陆卡
      {
        id: 'B1_01',
        type: '着陆',
        name: '5-4-3-2-1 着陆法',
        content: '当情绪冲顶时，用感官把自己拉回当下：\n\n说出（或心里默念）：\n• 5个你能看到的东西\n• 4个你能摸到的东西\n• 3个你能听到的声音\n• 2个你能闻到的气味\n• 1个你能尝到的味道（或喝一口水代替）',
        when_to_use: '情绪突然崩溃、焦虑冲顶、感觉要失控时',
        color: '#52c41a'
      },
      {
        id: 'B1_02',
        type: '着陆',
        name: '冷刺激/窗口法',
        content: '• 打开窗户30秒，感受冷空气\n• 用冷水洗脸\n• 握一杯冰水直到杯壁变暖\n\n寒冷刺激可以激活副交感神经，帮你从"战斗/逃跑"状态稍微缓过来。',
        when_to_use: '恐慌发作、过度换气、情绪激动到无法思考时',
        color: '#52c41a'
      },
      {
        id: 'B1_03',
        type: '着陆',
        name: '安全带法则',
        content: '如果你现在给自己的痛苦打分 ≥ 8/10：\n\n别急着复盘、别急着解决问题。\n先做"安全动作"：\n• 联系一个信任的人\n• 打一个热线电话\n• 去一个有人的地方\n\n等痛苦降到 6/10 以下，再想下一步。',
        when_to_use: '痛苦感非常强烈、几乎撑不住时',
        color: '#52c41a'
      },

      // 边界脚本
      {
        id: 'B2_01',
        type: '边界',
        name: '延迟回复脚本',
        content: '当导师/老板的消息让你压力爆棚，试试：\n\n「老师，收到。我今天进度到[具体内容]；我明早10点前把[具体任务]发您确认，可以吗？」\n\n要点：\n• 确认收到（不让对方焦虑）\n• 说明当前进度（展示你在做）\n• 把时间框死（给自己喘息空间）',
        when_to_use: '导师消息频繁、感觉被追着跑时',
        color: '#1890ff'
      },
      {
        id: 'B2_02',
        type: '边界',
        name: '容量边界脚本',
        content: '当你已经超负荷，无法再接新任务：\n\n「最近负荷到上限了；这周我只能保证[A]和[B]；[C]可否延到下周或找其他同学帮忙？」\n\n要点：\n• 诚实说明状态（不装没事）\n• 给出能保证的（不让对方完全落空）\n• 提供选项（不把话说死）',
        when_to_use: '任务太多、感觉要崩溃时',
        color: '#1890ff'
      },
      {
        id: 'B2_03',
        type: '边界',
        name: '结束聊天脚本',
        content: '当聊天/沟通消耗你太多能量：\n\n「我先去[具体事情：跑实验/整理数据/吃饭]，晚点回您」\n\n要点：\n• 给出具体理由（不是敷衍）\n• 不承诺具体时间（避免给自己压力）\n• 主动结束（防止被无限吸走）',
        when_to_use: '感觉对话在消耗你、需要退出时',
        color: '#1890ff'
      },

      // 任务重启
      {
        id: 'B3_01',
        type: '任务',
        name: '2分钟启动法',
        content: '当你完全不想动、任务堆积到窒息：\n\n只承诺做2分钟：\n• 打开文档\n• 写3行\n• 跑通一个case\n\n2分钟后，你完全可以停下来。\n通常，一旦开始，就没那么难继续了。',
        when_to_use: '任务瘫痪、完全不想开始时',
        color: '#fa8c16'
      },
      {
        id: 'B3_02',
        type: '任务',
        name: '今天只做一件事',
        content: '当你的待办清单长得让你绝望：\n\n把列表砍到：\n「今天唯一必须：____」\n\n其他的事：\n• 可以做，但不必须\n• 可以延后\n• 可以不做\n\n一天结束，只要完成那一件事，今天就是成功的。',
        when_to_use: '任务太多、不知道从哪开始时',
        color: '#fa8c16'
      },
      {
        id: 'B3_03',
        type: '任务',
        name: '晚上关机仪式',
        content: '在睡前30分钟做：\n\n1. 关掉企业微信/邮件通知\n2. 把明天第一件事写在便签上\n3. 合上电脑（哪怕还有没做完的）\n\n目的：\n• 保护睡眠（不被消息打扰）\n• 明天不用纠结"从哪开始"\n• 允许自己休息（不完美也可以）',
        when_to_use: '晚上睡前还在想工作、睡眠质量差时',
        color: '#fa8c16'
      }
    ]

    this.setData({ cards })
  },

  // 切换分类
  switchCategory(e) {
    const category = e.currentTarget.dataset.category
    this.setData({ currentCategory: category })
  },

  // 查看卡片详情
  viewCard(e) {
    const card = e.currentTarget.dataset.card

    wx.showModal({
      title: card.name,
      content: card.content,
      showCancel: true,
      cancelText: '关闭',
      confirmText: '复制',
      success: (res) => {
        if (res.confirm) {
          wx.setClipboardData({
            data: card.content,
            success: () => {
              wx.showToast({
                title: '已复制',
                icon: 'success'
              })
            }
          })
        }
      }
    })
  },

  // 复制卡片内容
  copyCard(e) {
    const content = e.currentTarget.dataset.content

    wx.setClipboardData({
      data: content,
      success: () => {
        wx.showToast({
          title: '已复制到剪贴板',
          icon: 'success'
        })
      }
    })
  },

  // 过滤卡片
  getFilteredCards() {
    const { cards, currentCategory } = this.data

    if (currentCategory === '全部') {
      return cards
    }

    return cards.filter(c => c.type === currentCategory)
  },

  // 分享
  onShareAppMessage() {
    return {
      title: '心理健康自助技能卡',
      path: '/pages/cards/cards'
    }
  }
})