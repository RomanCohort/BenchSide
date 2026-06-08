// 本地存储封装

const STORAGE_KEYS = {
  DISCLAIMER_ACCEPTED: 'disclaimerAccepted',
  USER_SETTINGS: 'userSettings',
  TRUSTED_CONTACTS: 'trustedContacts',
  CHAT_HISTORY: 'chatHistory',
  FIRST_USE: 'firstUse'
}

/**
 * 保存声明接受状态
 * @param {boolean} accepted 是否接受
 */
function saveDisclaimerAccepted(accepted) {
  wx.setStorageSync(STORAGE_KEYS.DISCLAIMER_ACCEPTED, accepted)
}

/**
 * 获取声明接受状态
 * @returns {boolean} 是否已接受
 */
function getDisclaimerAccepted() {
  return wx.getStorageSync(STORAGE_KEYS.DISCLAIMER_ACCEPTED) || false
}

/**
 * 保存用户设置
 * @param {object} settings 设置对象
 */
function saveUserSettings(settings) {
  wx.setStorageSync(STORAGE_KEYS.USER_SETTINGS, settings)
}

/**
 * 获取用户设置
 * @returns {object} 设置对象
 */
function getUserSettings() {
  return wx.getStorageSync(STORAGE_KEYS.USER_SETTINGS) || {}
}

/**
 * 保存信任联系人
 * @param {array} contacts 联系人列表
 */
function saveTrustedContacts(contacts) {
  wx.setStorageSync(STORAGE_KEYS.TRUSTED_CONTACTS, contacts)
}

/**
 * 获取信任联系人
 * @returns {array} 联系人列表
 */
function getTrustedContacts() {
  return wx.getStorageSync(STORAGE_KEYS.TRUSTED_CONTACTS) || []
}

/**
 * 添加信任联系人
 * @param {object} contact 联系人 {name, phone}
 */
function addTrustedContact(contact) {
  const contacts = getTrustedContacts()
  contacts.push(contact)
  saveTrustedContacts(contacts)
}

/**
 * 保存聊天历史
 * @param {array} history 聊天历史
 */
function saveChatHistory(history) {
  wx.setStorageSync(STORAGE_KEYS.CHAT_HISTORY, history)
}

/**
 * 获取聊天历史
 * @returns {array} 聊天历史
 */
function getChatHistory() {
  return wx.getStorageSync(STORAGE_KEYS.CHAT_HISTORY) || []
}

/**
 * 清空聊天历史
 */
function clearChatHistory() {
  wx.removeStorageSync(STORAGE_KEYS.CHAT_HISTORY)
}

/**
 * 清空所有数据
 */
function clearAll() {
  wx.clearStorageSync()
}

module.exports = {
  saveDisclaimerAccepted,
  getDisclaimerAccepted,
  saveUserSettings,
  getUserSettings,
  saveTrustedContacts,
  getTrustedContacts,
  addTrustedContact,
  saveChatHistory,
  getChatHistory,
  clearChatHistory,
  clearAll,
  STORAGE_KEYS
}