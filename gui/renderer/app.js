const $ = (id) => document.getElementById(id);

const FIELDS = [
  'serverPath', 'modelsDir', 'model', 'hfModel', 'mmproj', 'apiAddress',
  'ngl', 'context', 'threads', 'threadsBatch', 'batchSize', 'ubatch',
  'cacheTypeK', 'cacheTypeV', 'parallel', 'mtpNMax', 'mtpModel', 'dflashModel', 'dflashNMax',
  'reasoning', 'reasoningBudget', 'reasoningEffort', 'temp', 'topP', 'topK', 'minP',
  'repeatPenalty', 'presencePenalty', 'frequencyPenalty',
  'imageMinTokens', 'imageMaxTokens', 'mtmdBatchMaxTokens',
  'extraArgs'
];
const BOOL_FIELDS = ['flashAttention', 'jinja', 'mtpEnabled', 'dflashEnabled', 'preserveReasoning', 'mmprojOffload'];

const state = {
  config: null,
  serverRunning: false,
  logsVisible: true
};

const CACHE_TYPES = ['q8_0', 'f16', 'q4_0', 'q4_1', 'q5_0', 'q5_1', 'iq4_nl', 'iq4_xs', 'iq2_s', 'iq3_xxs'];

function num(id, fallback = 0) {
  const v = parseFloat($(id).value);
  return isNaN(v) ? fallback : v;
}
function bool(id) { return $(id).checked; }
function text(id) { return $(id).value.trim(); }
function int(id, fallback = 0) {
  const v = parseInt($(id).value, 10);
  return isNaN(v) ? fallback : v;
}

function readConfig() {
  const c = {};
  for (const f of FIELDS) c[f] = text(f);
  for (const f of BOOL_FIELDS) c[f] = bool(f);
  c.context = int('context', 8192);
  c.threads = int('threads', 0);
  c.threadsBatch = int('threadsBatch', 0);
  c.batchSize = int('batchSize', 2048);
  c.ubatch = int('ubatch', 512);
  c.ngl = int('ngl', 99);
  c.parallel = int('parallel', 1);
  c.mtpNMax = int('mtpNMax', 3);
  c.dflashNMax = int('dflashNMax', 15);
  c.reasoningBudget = int('reasoningBudget', -1);
  c.temp = num('temp', 1.0);
  c.topP = num('topP', 0.95);
  c.topK = int('topK', 40);
  c.minP = num('minP', 0.05);
  c.repeatPenalty = num('repeatPenalty', 1.1);
  c.presencePenalty = num('presencePenalty', 0);
  c.frequencyPenalty = num('frequencyPenalty', 0);
  c.imageMinTokens = int('imageMinTokens', -1);
  c.imageMaxTokens = int('imageMaxTokens', -1);
  c.mtmdBatchMaxTokens = int('mtmdBatchMaxTokens', 1024);
  return c;
}

function writeConfig(c) {
  if (!c) return;
  for (const f of FIELDS) {
    const el = $(f);
    if (!el) continue;
    if (el.tagName === 'SELECT' && ![...el.options].some(o => o.value === String(c[f]))) {
      el.add(new Option(c[f], c[f]));
    }
    el.value = c[f] == null ? '' : c[f];
  }
  for (const f of BOOL_FIELDS) $(f).checked = !!c[f];
}

function flash(msg, ok = true) {
  const f = $('presetFeedback');
  f.textContent = msg;
  f.className = 'feedback ' + (ok ? 'ok' : 'err');
  clearTimeout(flash._t);
  flash._t = setTimeout(() => { f.textContent = ''; f.className = 'feedback'; }, 3500);
}

async function refreshModels() {
  const dir = text('modelsDir');
  if (!dir) return;
  const list = await window.api.listModels(dir);
  const sel = $('model');
  const keep = sel.value;
  sel.innerHTML = '';
  if (list.length === 0) {
    sel.add(new Option('— no .gguf found —', ''));
    return;
  }
  for (const p of list) sel.add(new Option(p, p));
  if (keep && [...sel.options].some(o => o.value === keep)) sel.value = keep;
}

function appendLog({ kind, text: line }) {
  const log = $('log');
  const div = document.createElement('div');
  div.className = kind;
  div.textContent = line;
  log.appendChild(div);
  if (state.logsVisible) log.scrollTop = log.scrollHeight;
}

function setStatus(badge, detail) {
  const b = $('statusBadge');
  b.className = 'badge badge-' + (badge || 'idle');
  b.textContent = badge === 'running' ? 'Running' : badge === 'err' ? 'Error' : badge === 'starting' ? 'Starting…' : 'Idle';
  $('statusDetail').textContent = detail || '';
  state.serverRunning = badge === 'running';
  $('btnStart').disabled = badge === 'running' || badge === 'starting';
  $('btnStop').disabled = !(badge === 'running' || badge === 'starting');
}

function apiBase() {
  return (text('apiAddress') || 'http://127.0.0.1:8080').replace(/\/+$/, '');
}

async function updateMetrics() {
  try {
    const r = await fetch(`${apiBase()}/slots`, { cache: 'no-store' });
    const j = await r.json();
    const t = j && j[0] && j[0].timings;
    $('metricPrompt').textContent = t && t.prompt_per_second ? t.prompt_per_second.toFixed(1) : '--';
    $('metricGen').textContent = t && t.predicted_per_second ? t.predicted_per_second.toFixed(1) : '--';
  } catch {
    $('metricPrompt').textContent = '--';
    $('metricGen').textContent = '--';
  }
}

let healthTimer = null;
function startHealthPoll() {
  stopHealthPoll();
  healthTimer = setInterval(async () => {
    try {
      const r = await fetch(`${apiBase()}/api/status`, { cache: 'no-store' });
      if (r.ok) {
        const j = await r.json();
        if (j.currentModel) {
          const name = j.currentModel.split(/[\\/]/).pop();
          setStatus('running', `server up · model: ${name}`);
        }
      } else {
        setStatus('starting', 'server starting…');
      }
    } catch {
      const r2 = await fetch(`${apiBase()}/health`, { cache: 'no-store' }).catch(() => null);
      setStatus(r2 && r2.ok ? 'running' : 'starting', r2 && r2.ok ? 'server up' : 'not responding yet');
    }
    updateMetrics();
  }, 1500);
}
function stopHealthPoll() { if (healthTimer) { clearInterval(healthTimer); healthTimer = null; } }
function resetMetrics() { $('metricPrompt').textContent = '--'; $('metricGen').textContent = '--'; }

async function startServer() {
  if (state.serverRunning) return;
  const cfg = readConfig();
  await window.api.saveConfig(cfg);
  setStatus('starting', 'launching llama-server…');
  appendLog('info', '[starting llama-server…]');
  const res = await window.api.startServer(cfg);
  if (!res.ok) {
    setStatus('err', res.error);
    appendLog('error', res.error);
  } else {
    startHealthPoll();
  }
}

async function stopServer() {
  await window.api.stopServer();
  stopHealthPoll();
  resetMetrics();
  setStatus('idle', 'stopped');
  appendLog('info', '[server stopped by user]');
}

// ---------- Chat playground ----------
async function sendChat() {
  const box = $('chatInput');
  const msg = box.value.trim();
  if (!msg || !state.serverRunning) return;
  box.value = '';
  addChat('user', msg, 'You');
  chatCol.push({ role: 'user', content: msg });

  const assistantMsg = addChat('assistant', '', 'Thinking…');
  assistantMsg.querySelector('.body').textContent = '';
  try {
    const res = await fetch(`${apiBase()}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'local',
        messages: chatCol,
        stream: true,
        temperature: num('temp', 1.0),
        max_tokens: 512
      })
    });
    if (!res.ok) {
      const j = await res.json().catch(() => null);
      throw new Error((j && j.error && j.error.message) || `HTTP ${res.status}`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let reasoning = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const raw of lines) {
        const line = raw.trim();
        if (!line.startsWith('data:')) continue;
        const payload = line.slice(5).trim();
        if (payload === '[DONE]') continue;
        try {
          const j = JSON.parse(payload);
          const d = j.choices && j.choices[0] && j.choices[0].delta;
          if (d && d.reasoning_content) {
            reasoning += d.reasoning_content;
            assistantMsg.querySelector('.thinking').textContent = reasoning;
          }
          if (d && d.content) {
            assistantMsg.querySelector('.body').textContent += d.content;
          }
        } catch {}
      }
      assistantMsg.scrollIntoView({ block: 'end' });
    }
    chatCol.push({ role: 'assistant', content: assistantMsg.querySelector('.body').textContent });
    assistantMsg.classList.remove('typing');
  } catch (err) {
    assistantMsg.querySelector('.body').textContent = 'Error: ' + err.message;
    assistantMsg.classList.remove('typing');
  }
}

let chatCol = [];

function addChat(role, content, label) {
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + role;
  const meta = document.createElement('div');
  meta.className = 'meta';
  meta.textContent = label;
  const thinking = document.createElement('div');
  thinking.className = 'thinking';
  const body = document.createElement('div');
  body.className = 'body';
  body.textContent = content;
  wrap.appendChild(meta);
  if (role === 'assistant') wrap.appendChild(thinking);
  wrap.appendChild(body);
  $('chatMessages').appendChild(wrap);
  return wrap;
}

// ---------- Tabs ----------
function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
  document.getElementById('tab-logs').hidden = name !== 'logs';
  document.getElementById('tab-chat').hidden = name !== 'chat';
  document.getElementById('tab-info').hidden = name !== 'info';
  state.logsVisible = name === 'logs';
}

// ---------- Presets ----------
let presets = {};

async function refreshPresets() {
  presets = await window.api.listPresets();
  const sel = $('presetList');
  const keep = sel.value;
  sel.innerHTML = '';
  const names = Object.keys(presets).sort();
  if (names.length === 0) sel.add(new Option('— no presets —', ''));
  for (const n of names) sel.add(new Option(n, n));
  if (keep && names.includes(keep)) sel.value = keep;
}

async function savePreset() {
  const name = $('presetName').value.trim();
  if (!name) { flash('Type a preset name first.', false); $('presetName').focus(); return; }
  try {
    await window.api.savePreset(name, readConfig());
    $('presetName').value = '';
    await refreshPresets();
    $('presetList').value = name;
    flash(`Preset "${name}" saved.`);
    appendLog('info', `[preset "${name}" saved]`);
  } catch (err) {
    flash('Failed to save: ' + err.message, false);
  }
}

async function loadPreset() {
  const name = $('presetList').value;
  if (!name || !presets[name]) { flash('Select a preset to load.', false); return; }
  writeConfig(presets[name]);
  await refreshModels();
  flash(`Preset "${name}" loaded.`);
  appendLog('info', `[preset "${name}" loaded]`);
}

async function deletePreset() {
  const name = $('presetList').value;
  if (!name || !presets[name]) { flash('Select a preset to delete.', false); return; }
  await window.api.deletePreset(name);
  await refreshPresets();
  flash(`Preset "${name}" deleted.`);
  appendLog('info', `[preset "${name}" deleted]`);
}

async function resetDefaults() {
  const d = await window.api.getDefaults();
  writeConfig(d);
  await refreshModels();
  flash('All settings reset to recommended defaults.');
  appendLog('info', '[reset to defaults]');
}

// ---------- Tooltips ----------
function setupTooltips() {
  const tip = document.createElement('div');
  tip.id = 'tip';
  tip.className = 'tooltip';
  document.body.appendChild(tip);
  document.addEventListener('mouseover', (e) => {
    const t = e.target.closest('.tip');
    if (!t) return;
    tip.textContent = t.dataset.help || '';
    tip.style.display = 'block';
    const r = t.getBoundingClientRect();
    const tw = tip.offsetWidth;
    const th = tip.offsetHeight;
    let left = r.left;
    if (left + tw > window.innerWidth - 8) left = window.innerWidth - tw - 8;
    if (left < 8) left = 8;
    let top = r.bottom + 6;
    if (top + th > window.innerHeight - 8) top = r.top - th - 6;
    tip.style.left = left + 'px';
    tip.style.top = top + 'px';
  });
  document.addEventListener('mouseout', (e) => {
    if (e.target.closest('.tip')) tip.style.display = 'none';
  });
}

// ---------- Init ----------
async function init() {
  for (const t of CACHE_TYPES) {
    $('cacheTypeK').add(new Option(t, t));
    $('cacheTypeV').add(new Option(t, t));
  }

  document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => switchTab(t.dataset.tab)));
  $('btnStart').addEventListener('click', startServer);
  $('btnStop').addEventListener('click', stopServer);
  $('btnSave').addEventListener('click', async () => { await window.api.saveConfig(readConfig()); flash('Config saved.'); });
  $('btnBrowseModel').addEventListener('click', async () => {
    const p = await window.api.pickFile({ title: 'Select model GGUF', filter: '.gguf' });
    if (p) { $('model').value = p; }
  });
  $('btnBrowseMmproj').addEventListener('click', async () => {
    const p = await window.api.pickFile({ title: 'Select mmproj GGUF', filter: '.gguf' });
    if (p) $('mmproj').value = p;
  });
  $('btnBrowseMtpModel').addEventListener('click', async () => {
    const p = await window.api.pickFile({ title: 'Select MTP model GGUF', filter: '.gguf' });
    if (p) $('mtpModel').value = p;
  });
  $('btnBrowseDflash').addEventListener('click', async () => {
    const p = await window.api.pickFile({ title: 'Select DFlash model GGUF', filter: '.gguf' });
    if (p) $('dflashModel').value = p;
  });
  $('model').addEventListener('change', () => {});
  const specExclusive = (a, b) => {
    $(a).addEventListener('change', () => { if ($(a).checked) $(b).checked = false; });
    $(b).addEventListener('change', () => { if ($(b).checked) $(a).checked = false; });
  };
  specExclusive('mtpEnabled', 'dflashEnabled');
  $('modelsDir').addEventListener('change', refreshModels);
  $('btnSend').addEventListener('click', sendChat);
  $('chatInput').addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); } });
  $('btnPresetSave').addEventListener('click', savePreset);
  $('btnPresetLoad').addEventListener('click', loadPreset);
  $('btnPresetDelete').addEventListener('click', deletePreset);
  $('btnResetDefaults').addEventListener('click', resetDefaults);

  setupTooltips();

  window.api.onLog(appendLog);
  window.api.onStatus((s) => {
    if (s.state === 'started') {
      setStatus('starting', `pid ${s.pid}`);
    } else if (s.state === 'stopped') {
      stopHealthPoll();
      resetMetrics();
      setStatus('idle', `exited (${s.code})`);
    } else if (s.state === 'error') {
      stopHealthPoll();
      resetMetrics();
      setStatus('err', s.text);
    }
  });
  window.api.onConfigChanged((cfg) => {
    writeConfig(cfg);
    refreshModels();
    refreshPresets();
  });

  const cfg = await window.api.getConfig();
  state.config = cfg;
  writeConfig(cfg);
  await refreshPresets();
  await refreshModels();
}
document.addEventListener('DOMContentLoaded', init);
