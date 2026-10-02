const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const { spawn, exec } = require('child_process');
const http = require('http');
const fs = require('fs');
const path = require('path');

const APP_ROOT = __dirname;

function configDir() {
  return app.isPackaged ? app.getPath('userData') : APP_ROOT;
}
const CONFIG_PATH = path.join(configDir(), 'config.json');
const PRESETS_PATH = path.join(configDir(), 'presets.json');

let mainWindow = null;
let serverProc = null;
let loadedModelPath = '';

function defaultServerPath() {
  if (!app.isPackaged) return path.resolve(APP_ROOT, '..', 'llama.cpp', 'build', 'bin', 'llama-server.exe');
  return path.join(process.resourcesPath, 'bin', 'llama-server.exe');
}

function defaultModelsDir() {
  if (!app.isPackaged) {
    const dev = path.join(APP_ROOT, 'models');
    const alt = path.join(APP_ROOT, 'dist', 'models');
    if (fs.existsSync(alt) && fs.readdirSync(alt).some(f => /\.(gguf|ggml|safe)$/i.test(f))) return alt;
    return dev;
  }
  if (process.env.PORTABLE_EXECUTABLE_DIR) return path.join(process.env.PORTABLE_EXECUTABLE_DIR, 'models');
  return path.join(path.dirname(process.execPath), 'models');
}

function defaultConfig() {
  return {
    apiAddress: 'http://127.0.0.1:1234',
    serverPath: '',
    modelsDir: '',
    model: '',
    hfModel: '',
    mmproj: '',
    context: 8192,
    ngl: 99,
    batchSize: 2048,
    ubatch: 512,
    threads: 0,
    threadsBatch: 0,
    flashAttention: true,
    cacheTypeK: 'q8_0',
    cacheTypeV: 'q8_0',
    parallel: 1,
    jinja: true,
    mtpEnabled: false,
    mtpNMax: 3,
    mtpModel: '',
    dflashEnabled: false,
    dflashModel: '',
    dflashNMax: 15,
    reasoning: 'auto',
    reasoningBudget: -1,
    reasoningEffort: '',
    preserveReasoning: false,
    temp: 1.0,
    topP: 0.95,
    topK: 40,
    minP: 0.05,
    repeatPenalty: 1.1,
    presencePenalty: 0.0,
    frequencyPenalty: 0.0,
    mmprojOffload: true,
    imageMinTokens: -1,
    imageMaxTokens: -1,
    mtmdBatchMaxTokens: 1024,
    extraArgs: ''
  };
}

function sanitizePaths(cfg) {
  if (!cfg.serverPath || !fs.existsSync(cfg.serverPath)) cfg.serverPath = defaultServerPath();
  if (!cfg.modelsDir) cfg.modelsDir = defaultModelsDir();
  if (!cfg.apiAddress) cfg.apiAddress = 'http://127.0.0.1:1234';
  return cfg;
}

function loadConfig() {
  let cfg;
  try {
    const raw = fs.readFileSync(CONFIG_PATH, 'utf8');
    cfg = { ...defaultConfig(), ...JSON.parse(raw) };
  } catch {
    cfg = defaultConfig();
  }
  return sanitizePaths(cfg);
}

function saveConfig(cfg) {
  fs.mkdirSync(configDir(), { recursive: true });
  fs.writeFileSync(CONFIG_PATH, JSON.stringify(cfg, null, 2));
}

function parseAddress(addr) {
  const m = /^(?:https?:\/\/)?([^:/]+)(?::(\d+))?(?:\/.*)?$/.exec((addr || '').trim());
  if (!m) return null;
  return { host: m[1], port: m[2] ? parseInt(m[2], 10) : 8080 };
}

function loadPresets() {
  try {
    return JSON.parse(fs.readFileSync(PRESETS_PATH, 'utf8')) || {};
  } catch {
    return {};
  }
}

function savePresets(presets) {
  fs.mkdirSync(configDir(), { recursive: true });
  fs.writeFileSync(PRESETS_PATH, JSON.stringify(presets, null, 2));
}

function internalApi(cfg) {
  const parsed = parseAddress(cfg.apiAddress);
  return `http://127.0.0.1:${parsed ? parsed.port : 1234}`;
}

function buildArgs(cfg) {
  const args = [];
  const parsed = parseAddress(cfg.apiAddress);
  const host = '127.0.0.1';
  const port = parsed ? parsed.port : 8080;

  if (cfg.hfModel.trim()) {
    args.push('-hf', cfg.hfModel.trim());
  } else if (cfg.model) {
    args.push('-m', cfg.model);
  }

  if (cfg.mmproj) args.push('--mmproj', cfg.mmproj);
  args.push('--host', host);
  args.push('--port', String(port));
  args.push('--metrics');
  args.push('-ngl', String(cfg.ngl));
  args.push('-c', String(cfg.context));
  args.push('-t', String(cfg.threads));
  args.push('-tb', String(cfg.threadsBatch));
  args.push('-b', String(cfg.batchSize));
  args.push('-ub', String(cfg.ubatch));
  args.push('-fa', cfg.flashAttention ? 'on' : 'off');
  args.push('--cache-type-k', cfg.cacheTypeK);
  args.push('--cache-type-v', cfg.cacheTypeV);
  args.push('-np', String(cfg.parallel));
  args.push('--temp', String(cfg.temp));
  args.push('--top-p', String(cfg.topP));
  args.push('--top-k', String(cfg.topK));
  args.push('--min-p', String(cfg.minP));
  const rp = cfg.repeatPenalty > 0 ? cfg.repeatPenalty : 1.0;
  args.push('--repeat-penalty', String(rp));
  args.push('--presence-penalty', String(cfg.presencePenalty));
  args.push('--frequency-penalty', String(cfg.frequencyPenalty));

  if (cfg.jinja) args.push('--jinja');

  if (cfg.mtpEnabled) {
    args.push('--spec-type', 'draft-mtp');
    if (cfg.mtpModel && cfg.mtpModel.trim()) args.push('--spec-draft-model', cfg.mtpModel.trim());
    args.push('--spec-draft-n-max', String(cfg.mtpNMax));
  }

  if (cfg.dflashEnabled) {
    args.push('--spec-type', 'draft-dflash');
    if (cfg.dflashModel && cfg.dflashModel.trim()) args.push('--spec-draft-model', cfg.dflashModel.trim());
    args.push('--spec-draft-n-max', String(cfg.dflashNMax));
  }

  if (cfg.reasoning === 'on' || cfg.reasoning === 'off') {
    args.push('--reasoning', cfg.reasoning);
  }
  if (cfg.reasoningBudget >= 0) args.push('--reasoning-budget', String(cfg.reasoningBudget));
  if (cfg.reasoningEffort && cfg.reasoningEffort.trim()) {
    args.push('--chat-template-kwargs', JSON.stringify({ reasoning_effort: cfg.reasoningEffort.trim() }));
  }
  if (cfg.preserveReasoning) args.push('--reasoning-preserve');

  if (cfg.mmprojOffload === false) args.push('--no-mmproj-offload');
  if (cfg.imageMinTokens > 0) args.push('--image-min-tokens', String(cfg.imageMinTokens));
  if (cfg.imageMaxTokens > 0) args.push('--image-max-tokens', String(cfg.imageMaxTokens));
  if (cfg.mtmdBatchMaxTokens > 0) args.push('--mtmd-batch-max-tokens', String(cfg.mtmdBatchMaxTokens));

  if (cfg.extraArgs && cfg.extraArgs.trim()) {
    // naive split; users should quote values containing spaces
    args.push(...splitArgs(cfg.extraArgs));
  }

  return args;
}

function splitArgs(str) {
  const args = [];
  const re = /"([^"]*)"|'([^']*)'|(\S+)/g;
  let m;
  while ((m = re.exec(str))) {
    args.push(m[1] !== undefined ? m[1] : m[2] !== undefined ? m[2] : m[3]);
  }
  return args;
}

function broadcast(channel, payload) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send(channel, payload);
  }
}

function spawnServer(cfg) {
  const args = buildArgs(cfg);
  const cmd = cfg.serverPath;

  const mi = args.indexOf('-m');
  loadedModelPath = mi >= 0 ? args[mi + 1] : '';

  if (!fs.existsSync(cmd)) {
    broadcast('server:log', { kind: 'error', text: `llama-server.exe not found at:\n${cmd}\nCheck 'serverPath' in settings.` });
    return;
  }

  const bat = ['', `$ ${cmd}`].concat(args.map(a => (a.startsWith('-') ? a : `"${a}"`))).join(' \r\n');
  broadcast('server:log', { kind: 'info', text: bat });

  try {
    serverProc = spawn(cmd, args, {
      stdio: ['ignore', 'pipe', 'pipe'],
      windowsHide: false
    });
  } catch (err) {
    broadcast('server:log', { kind: 'error', text: String(err.message || err) });
    return;
  }

  const append = (kind) => (data) => {
    const text = data.toString('utf8');
    repr(text).split('\n').forEach((line) => {
      broadcast('server:log', { kind: kind, text: line });
    });
  };

  serverProc.stdout.on('data', append('stdout'));
  serverProc.stderr.on('data', append('stderr'));

  serverProc.on('spawn', () => broadcast('server:status', { state: 'started', pid: serverProc.pid }));
  serverProc.on('exit', (code, signal) => {
    serverProc = null;
    loadedModelPath = '';
    broadcast('server:status', { state: 'stopped', code, signal });
    broadcast('server:log', { kind: 'info', text: `[server exited code=${code} signal=${signal}]` });
  });
  serverProc.on('error', (err) => {
    serverProc = null;
    broadcast('server:log', { kind: 'error', text: String(err.message || err) });
    broadcast('server:status', { state: 'stopped', code: -1 });
  });
}

function repr(text) {
  // crude ANSI parser - just trim the escape codes for cleaner logs
  return text.replace(/\x1b\[[0-9;]*m/g, '');
}

function stopServer() {
  if (!serverProc) return 'not-running';
  const pid = serverProc.pid;
  return new Promise((resolve) => {
    serverProc.once('exit', () => resolve('stopped'));
    try {
      serverProc.kill();
    } catch {}
    // force kill tree on Windows after a grace period
    setTimeout(() => {
      try { exec(`taskkill /PID ${pid} /T /F`, () => {}); } catch {}
    }, 1500);
  });
}

function listGgufs(dir) {
  if (!dir || !fs.existsSync(dir)) return [];
  let out = [];
  try {
    for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
      const p = path.join(dir, e.name);
      if (e.isDirectory()) {
        out = out.concat(listGgufs(p));
      } else if (/\.(gguf|ggml|safe)$/i.test(e.name)) {
        out.push(p);
      }
    }
  } catch {}
  return out;
}

// ---------- Phone control panel (browse + switch models) ----------
let controlServer = null;

function findMmprojFor(modelPath) {
  const dir = path.dirname(modelPath);
  const stem = path.basename(modelPath).replace(/\.gguf$/i, '').toLowerCase();
  const cands = [];
  try {
    for (const f of fs.readdirSync(dir)) {
      if (/mmproj/i.test(f) && /\.gguf$/i.test(f)) cands.push(f);
    }
  } catch {}
  if (!cands.length) return '';
  if (cands.length === 1) return path.join(dir, cands[0]);
  const toks = stem.split(/[^a-z0-9]+/i).filter(t => t.length >= 3);
  let best = '';
  let bestScore = 0;
  for (const c of cands) {
    const cl = c.toLowerCase();
    let score = 0;
    for (const t of toks) if (cl.includes(t)) score++;
    if (score > bestScore) { bestScore = score; best = c; }
  }
  if (!best) return '';
  return path.join(dir, best);
}

function currentPresetName(cfg) {
  const presets = loadPresets();
  for (const name of Object.keys(presets)) {
    if (presets[name].model && cfg.model && presets[name].model === cfg.model) return name;
  }
  return '';
}

function fetchServerSpeed() {
  return new Promise((resolve) => {
    if (!serverProc) return resolve(null);
    const req = http.get(`${internalApi(loadConfig())}/slots`, (res) => {
      let d = '';
      res.on('data', c => { d += c; });
      res.on('end', () => {
        try {
          const j = JSON.parse(d);
          const t = j && j[0] && j[0].timings;
          resolve(t ? { prompt: t.prompt_per_second || 0, gen: t.predicted_per_second || 0 } : null);
        } catch { resolve(null); }
      });
    });
    req.on('error', () => resolve(null));
    req.setTimeout(2500, () => { req.destroy(); resolve(null); });
  });
}

async function controlStatus() {
  const cfg = loadConfig();
  const presets = loadPresets();
  const models = listGgufs(cfg.modelsDir).filter(m => !/mmproj/i.test(path.basename(m)));
  const parsed = parseAddress(cfg.apiAddress);
  const host = parsed ? parsed.host : '0.0.0.0';
  return {
    running: !!serverProc,
    host,
    apiAddress: cfg.apiAddress,
    currentModel: cfg.model,
    currentPreset: currentPresetName(cfg),
    speed: serverProc ? await fetchServerSpeed() : null,
    presets: Object.keys(presets).sort().map(name => ({
      name,
      model: presets[name].model || ''
    })),
    models: models.map(m => ({
      path: m,
      name: path.basename(m),
      sizeMB: Math.round(fs.statSync(m).size / 1048576)
    }))
  };
}

async function handleSwitch(target) {
  const cfg = sanitizePaths({ ...defaultConfig(), ...loadConfig() });
  if (target.startsWith('preset:')) {
    const name = target.slice(7);
    const p = loadPresets()[name];
    if (!p) return { ok: false, error: 'Preset not found' };
    Object.assign(cfg, sanitizePaths({ ...defaultConfig(), ...p }));
  } else if (target.startsWith('model:')) {
    const model = target.slice(6);
    if (!fs.existsSync(model)) return { ok: false, error: 'Model file missing: ' + model };
    cfg.model = model;
    cfg.hfModel = '';
    cfg.mmproj = findMmprojFor(model);
  } else {
    return { ok: false, error: 'Unknown target' };
  }
  if (!cfg.model || !fs.existsSync(cfg.model)) return { ok: false, error: 'No model selected' };
  if (!fs.existsSync(cfg.serverPath)) cfg.serverPath = defaultServerPath();
  if (serverProc) await stopServer();
  saveConfig(cfg);
  spawnServer(cfg);
  broadcast('config:changed', cfg);
  return { ok: true, model: cfg.model };
}

function controlHtml() {
  return `<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Llama Studio - Phone</title>
<style>
:root{--bg:#121318;--panel:#1d1f26;--panel2:#262933;--border:#363a46;--text:#e8eaef;--muted:#9aa1b0;--accent:#4f8cff;--ok:#3fb96b;--warn:#e5a33b;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif;padding:14px;}
h1{font-size:19px;margin:0 0 2px}.sub{color:var(--muted);font-size:12px;margin-bottom:12px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:12px 14px;margin-bottom:12px}
.speed{display:flex;gap:10px;font-size:13px}.chip{background:var(--panel2);border:1px solid var(--border);border-radius:8px;padding:6px 10px}
.chip b{font-size:17px;color:var(--accent)}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:600;margin-left:8px}
.badge.on{background:#17402a;color:var(--ok)}.badge.off{background:#3a3d4a;color:var(--muted)}
h2{font-size:13px;text-transform:uppercase;letter-spacing:1px;color:var(--muted);margin:14px 0 8px}
.item{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:10px 12px;border:1px solid var(--border);border-radius:10px;margin-bottom:8px;background:var(--panel2);width:100%;text-align:left;color:var(--text);font-size:14px;cursor:pointer}
.item.current{border-color:var(--accent)}
.item .meta{color:var(--muted);font-size:12px;word-break:break-all}
.item .go{color:var(--accent);font-weight:700}
.sel{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:8px;color:var(--text);font-size:14px;width:100%;margin-bottom:8px}
.msg{color:var(--warn);font-size:13px;min-height:18px}
button{font-family:inherit}
</style></head><body>
<h1>Llama Studio <span id="badge" class="badge off">offline</span></h1>
<div class="sub" id="sub">loading...</div>
<div class="card"><div class="speed">
<div class="chip">Prompt <b id="pp">--</b> t/s</div>
<div class="chip">Generate <b id="tg">--</b> t/s</div>
</div></div>
<h2>Presets (switch model + settings)</h2>
<div id="presets"></div>
<h2>All models</h2>
<div id="models"></div>
<div class="msg" id="msg"></div>
<script>
async function load(){
  try{
    const r=await fetch('/api/status');const s=await r.json();
    const b=document.getElementById('badge');b.textContent=s.running?'running':'offline';b.className='badge '+(s.running?'on':'off');
    document.getElementById('sub').textContent=s.running?('loaded: '+s.currentModel+'  |  '+s.apiAddress):('idle  |  '+s.apiAddress);
    document.getElementById('pp').textContent=s.speed&&s.speed.prompt?s.speed.prompt.toFixed(1):'--';
    document.getElementById('tg').textContent=s.speed&&s.speed.gen?s.speed.gen.toFixed(1):'--';
    const pl=document.getElementById('presets');pl.innerHTML='';
    s.presets.forEach(p=>{
      const d=document.createElement('button');d.className='item'+(p.name===s.currentPreset?' current':'');
      const nm=(p.model||'').split(/[\\\\/]/).pop()||'no model';
      const info=document.createElement('span');
      const title=document.createElement('b');title.textContent=p.name;
      const meta=document.createElement('div');meta.className='meta';meta.textContent=nm;
      const go=document.createElement('span');go.className='go';go.textContent='switch ▸';
      info.append(title,meta);d.append(info,go);
      d.onclick=async()=>{await post({preset:p.name});load();};
      pl.appendChild(d);
    });
    const ml=document.getElementById('models');ml.innerHTML='';
    s.models.forEach(m=>{
      const d=document.createElement('button');d.className='item'+(m.path===s.currentModel?' current':'');
      const info=document.createElement('span');
      const name=document.createElement('span');name.textContent=m.name;
      const meta=document.createElement('div');meta.className='meta';meta.textContent=(m.sizeMB/1024).toFixed(1)+' GB';
      const go=document.createElement('span');go.className='go';go.textContent='load ▸';
      info.append(name,meta);d.append(info,go);
      d.onclick=async()=>{await post({model:m.path});load();};
      ml.appendChild(d);
    });
  }catch(e){document.getElementById('sub').textContent='cannot reach server: '+e.message;}
}
async function post(body){
  const m=document.getElementById('msg');
  try{
    m.textContent='switching...';
    const r=await fetch('/api/switch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const j=await r.json();
    m.textContent=j.ok?('switched to '+j.model):('error: '+(j.error||'unknown'));
  }catch(e){m.textContent='error: '+e.message;}
}
setInterval(load,2000);load();
</script>
</body></html>`;
}

function sendRes(res, code, body, type) {
  res.writeHead(code, { 'Content-Type': type || 'text/html; charset=utf-8' });
  res.end(body);
}

function proxyTo(target) {
  return (req, res) => {
    const u = new URL(req.url, target);
    const opts = {
      hostname: u.hostname,
      port: u.port || 80,
      method: req.method,
      path: u.pathname + u.search,
      headers: Object.assign({}, req.headers, { host: u.host })
    };
    const pr = http.request(opts, (p) => {
      res.writeHead(p.statusCode, p.headers);
      p.pipe(res);
    });
    pr.on('error', (e) => {
      try { res.writeHead(502, { 'Content-Type': 'text/plain' }); res.end('llama-server not responding (' + e.message + ')'); } catch {}
    });
    req.on('error', () => { try { pr.abort(); } catch {} });
    req.pipe(pr);
  };
}

function modelEntry(id, vision, ownedBy) {
  const disp = vision ? (/vision/i.test(id) ? id : `${id}-vision`) : id;
  return {
    id: disp,
    object: 'model',
    created: 0,
    owned_by: ownedBy,
    capabilities: { vision: !!vision, chat: true, completion: true, embeddings: false },
    architecture: vision
      ? { input_modalities: ['text', 'image'], output_modalities: ['text'] }
      : { input_modalities: ['text'], output_modalities: ['text'] }
  };
}

function modelListJson() {
  const cfg = loadConfig();
  const presets = loadPresets();
  const data = [];
  for (const name of Object.keys(presets).sort()) {
    const mmproj = presets[name].mmproj;
    data.push(modelEntry(name, !!mmproj, 'preset'));
  }
  for (const m of listGgufs(cfg.modelsDir)) {
    const b = path.basename(m);
    if (/mmproj/i.test(b) || data.some(d => d.id === b)) continue;
    data.push(modelEntry(b, !!findMmprojFor(m), 'llama-studio'));
  }
  return { object: 'list', data };
}

function findModelByBasename(name) {
  for (const m of listGgufs(loadConfig().modelsDir)) {
    if (!/mmproj/i.test(path.basename(m)) && path.basename(m) === name) return m;
  }
  return '';
}

function resolveModelTarget(id) {
  if (!id) return null;
  const norm = id.replace(/-(vision|vl|vlm)$/i, '');
  const presets = loadPresets();
  if (presets[norm] && presets[norm].model) return { target: 'preset:' + norm, label: norm, modelPath: presets[norm].model };
  const m = findModelByBasename(norm);
  if (m) return { target: 'model:' + m, label: norm, modelPath: m };
  return null;
}

function waitForBackend(port, timeoutMs) {
  return new Promise((resolve) => {
    const t0 = Date.now();
    const probe = () => {
      if (Date.now() - t0 > (timeoutMs || 30000)) return resolve(false);
      const req = http.get({ hostname: '127.0.0.1', port, path: '/health' }, (r) => {
        r.resume();
        if (r.statusCode === 200) return resolve(true);
        setTimeout(probe, 500);
      });
      req.on('error', () => setTimeout(probe, 500));
    };
    probe();
  });
}

function handleModelRequest(cfg, req, res) {
  let body = '';
  req.on('data', c => { body += c; });
  req.on('end', async () => {
    let modelId = '';
    try { const j = JSON.parse(body); modelId = j.model || ''; } catch {}
    const r = resolveModelTarget(modelId);
    if (r && r.modelPath && r.modelPath !== loadedModelPath) {
      const sw = await handleSwitch(r.target);
      if (sw.ok) {
        broadcast('server:log', { kind: 'info', text: `[auto-load: ${r.label} -> ${sw.model}]` });
      }
    }
    const parsed = parseAddress(loadConfig().apiAddress);
    const port = parsed ? parsed.port : 1234;
    await waitForBackend(port, 30000);
    const u = new URL(req.url, internalApi(loadConfig()));
    const opts = {
      hostname: u.hostname, port: u.port, method: req.method, path: u.pathname + u.search,
      headers: Object.assign({}, req.headers, { host: u.host, 'content-length': Buffer.byteLength(body) })
    };
    const pr = http.request(opts, (p) => { res.writeHead(p.statusCode, p.headers); p.pipe(res); });
    pr.on('error', (e) => { try { res.writeHead(502, { 'Content-Type': 'text/plain' }); res.end('backend error: ' + e.message); } catch {} });
    pr.end(body);
  });
  req.on('error', () => { try { res.writeHead(400, { 'Content-Type': 'text/plain' }).end('bad request'); } catch {} });
}

function startControlServer() {
  const cfg = loadConfig();
  const parsed = parseAddress(cfg.apiAddress);
  const host = parsed ? parsed.host : '0.0.0.0';
  const port = parsed ? parsed.port : 8080;
  const target = internalApi(cfg);
  if (controlServer) { try { controlServer.close(); } catch {} }
  controlServer = null;
  // The local backend itself owns 127.0.0.1:<port>. The front door is only
  // needed for a distinct LAN/VPN bind address; binding both to loopback
  // would race for the same port and prevent the backend from starting.
  if (['127.0.0.1', 'localhost', '::1'].includes(host)) return;
  controlServer = http.createServer((req, res) => {
    const url = new URL(req.url, 'http://x');
    try {
      if (req.method === 'GET' && (url.pathname === '/' || url.pathname === '/control')) {
        return sendRes(res, 200, controlHtml());
      }
      if (req.method === 'GET' && url.pathname === '/api/status') {
        controlStatus().then(s => sendRes(res, 200, JSON.stringify(s), 'application/json'));
        return;
      }
      if (req.method === 'POST' && url.pathname === '/api/switch') {
        let body = '';
        req.on('data', c => { body += c; });
        req.on('end', () => {
          let target = null;
          try { const j = JSON.parse(body); target = j.preset ? 'preset:' + j.preset : (j.model ? 'model:' + j.model : null); } catch {}
          handleSwitch(target).then(r => sendRes(res, 200, JSON.stringify(r), 'application/json'));
        });
        return;
      }
      if (req.method === 'GET' && url.pathname === '/v1/models') {
        return sendRes(res, 200, JSON.stringify(modelListJson()), 'application/json');
      }
      if (req.method === 'POST' && ['/v1/chat/completions', '/v1/completions', '/v1/embeddings', '/v1/responses'].includes(url.pathname)) {
        return handleModelRequest(cfg, req, res);
      }
      return proxyTo(target)(req, res);
    } catch (e) {
      return sendRes(res, 500, 'server error: ' + e.message);
    }
  });
  controlServer.on('error', (e) => {
    broadcast('server:log', { kind: 'error', text: 'Front door on ' + host + ':' + port + ' error: ' + e.message });
  });
  controlServer.listen(port, host, () => {
    broadcast('server:log', { kind: 'info', text: `Phone/presets page: http://${host}:${port}  (proxies llama-server API)` });
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 900,
    minHeight: 600,
    title: 'Llama Studio',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    }
  });
  mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));
  mainWindow.on('closed', () => { mainWindow = null; });
}

ipcMain.handle('config:get', () => loadConfig());
ipcMain.handle('config:defaults', () => sanitizePaths(defaultConfig()));
ipcMain.handle('config:save', (e, cfg) => { saveConfig(cfg); });

ipcMain.handle('server:start', async (e, cfg) => {
  if (serverProc) return { ok: false, error: 'Server already running' };
  saveConfig(cfg);
  startControlServer();
  spawnServer(cfg);
  return { ok: true };
});

ipcMain.handle('server:stop', async () => ({ ok: true, result: await stopServer() }));
ipcMain.handle('server:running', () => !!serverProc);

ipcMain.handle('models:list', (e, dir) => listGgufs(dir || ''));
ipcMain.handle('dialog:open', async (e, { title, filter }) => {
  const res = await dialog.showOpenDialog(mainWindow, {
    title,
    properties: ['openFile'],
    filters: filter && /\.gguf$/i.test(filter) ? [{ name: 'GGUF', extensions: ['gguf'] }] : []
  });
  return res.canceled ? null : res.filePaths[0];
});

ipcMain.handle('presets:list', () => {
  const presets = loadPresets();
  const out = {};
  for (const k of Object.keys(presets)) {
    out[k] = sanitizePaths({ ...defaultConfig(), ...presets[k] });
  }
  return out;
});
ipcMain.handle('presets:save', (e, name, cfg) => {
  const presets = loadPresets();
  presets[name] = cfg;
  savePresets(presets);
  return true;
});
ipcMain.handle('presets:delete', (e, name) => {
  const presets = loadPresets();
  delete presets[name];
  savePresets(presets);
  return true;
});

app.whenReady().then(() => {
  fs.mkdirSync(defaultModelsDir(), { recursive: true });
  startControlServer();
  createWindow();
});
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
app.on('before-quit', () => { try { if (serverProc) exec('taskkill /PID ' + serverProc.pid + ' /T /F', () => {}); } catch {} });
