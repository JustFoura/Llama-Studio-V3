const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
  getConfig: () => ipcRenderer.invoke('config:get'),
  getDefaults: () => ipcRenderer.invoke('config:defaults'),
  saveConfig: (cfg) => ipcRenderer.invoke('config:save', cfg),
  startServer: (cfg) => ipcRenderer.invoke('server:start', cfg),
  stopServer: () => ipcRenderer.invoke('server:stop'),
  serverRunning: () => ipcRenderer.invoke('server:running'),
  listModels: (dir) => ipcRenderer.invoke('models:list', dir),
  pickFile: (opts) => ipcRenderer.invoke('dialog:open', opts),
  listPresets: () => ipcRenderer.invoke('presets:list'),
  savePreset: (name, cfg) => ipcRenderer.invoke('presets:save', name, cfg),
  deletePreset: (name) => ipcRenderer.invoke('presets:delete', name),
  onLog: (cb) => ipcRenderer.on('server:log', (_e, p) => cb(p)),
  onStatus: (cb) => ipcRenderer.on('server:status', (_e, p) => cb(p)),
  onConfigChanged: (cb) => ipcRenderer.on('config:changed', (_e, p) => cb(p))
});