// 资源页面
const app = getApp()

Page({
  data: {
    resources: [],
    categories: ['全部', '咨询', '急诊', '热线'],
    currentCategory: '全部'
  },

  onLoad() {
    this.loadResources()
  },

  // 加载资源
  loadResources() {
    const resources = [
      {
        id: 'R01',
        name: '吉林大学心理健康教育中心',
        phone: '0431-85166120',
        address: '长春市朝阳区前进大街2699号吉林大学中心校区',
        hours: '周一至周五 8:30-11:30 / 14:00-17:00',
        how_to: '网上预约（吉林大学官网-心理健康教育中心）/现场登记',
        note: '免费服务，保密原则',
        tags: ['咨询', '校内'],
        color: '#1890ff'
      },
      {
        id: 'R02',
        name: '吉林大学校医院急诊',
        phone: '0431-85166120',
        phone_alt: '120',
        address: '吉林大学各校区校医院',
        note: '24小时急诊服务',
        tags: ['急诊', '校内'],
        color: '#dc3545'
      },
      {
        id: 'R03',
        name: '24小时心理援助热线（国家级）',
        phone: '400-161-9995',
        phone_alt: '12355',
        note: '希望24热线 / 共青团热线',
        tags: ['热线', '国家级'],
        color: '#52c41a'
      },
      {
        id: 'R04',
        name: '长春市心理危机干预热线',
        phone: '0431-89685000',
        hours: '24小时',
        note: '本地心理危机干预',
        tags: ['热线', '本地'],
        color: '#fa8c16'
      },
      {
        id: 'R05',
        name: '长春市第六医院（精神卫生中心）',
        phone: '0431-82703999',
        address: '长春市朝阳区红旗街1118号',
        note: '精神卫生专业医院',
        tags: ['医院', '本地'],
        color: '#722ed1'
      },
      {
        id: 'R06',
        name: '各学院心理辅导员',
        note: '联系所在学院学工办',
        how_to: '每个学院都有负责心理工作的辅导员',
        tags: ['咨询', '校内'],
        color: '#13c2c2'
      }
    ]

    this.setData({ resources })
  },

  // 切换分类
  switchCategory(e) {
    const category = e.currentTarget.dataset.category
    this.setData({ currentCategory: category })
  },

  // 拨打电话
  callPhone(e) {
    const phone = e.currentTarget.dataset.phone
    wx.makePhoneCall({
      phoneNumber: phone,
      success: () => {
        console.log('拨打成功:', phone)
      }
    })
  },

  // 复制地址
  copyAddress(e) {
    const address = e.currentTarget.dataset.address
    wx.setClipboardData({
      data: address,
      success: () => {
        wx.showToast({
          title: '地址已复制',
          icon: 'success'
        })
      }
    })
  },

  // 导航到地址
  navigateTo(e) {
    const address = e.currentTarget.dataset.address
    wx.openLocation({
      latitude: 43.8,
      longitude: 125.3,
      name: address,
      address: address
    })
  },

  // 分享资源
  shareResource(e) {
    const resource = e.currentTarget.dataset.resource
    wx.showShareMenu({
      withShareTicket: true,
      menus: ['shareAppMessage', 'shareTimeline']
    })
  },

  // 过滤资源
  getFilteredResources() {
    const { resources, currentCategory } = this.data

    if (currentCategory === '全部') {
      return resources
    }

    return resources.filter(r => r.tags.includes(currentCategory))
  },

  onShareAppMessage() {
    return {
      title: '吉林大学心理健康资源',
      path: '/pages/resources/resources'
    }
  }
})