/* Llama Studio — frontend logic */

const T = window.__TAURI__;

/* Browser-preview mode: when opened without Tauri (e.g. plain http server)
   fall back to no-ops + demo data so the design can be iterated in a browser. */
const invoke = T
  ? T.core.invoke
  : (() => {
      const mockPreview = (c) => {
        const q = (v) => (/[\s"]/.test(String(v)) ? `"${v}"` : v);
        if (c.engine === "sglang") {
          const model = c.hfModel || c.model || "<model>";
          return `${c.sglangPath || "python3"} -m sglang.launch_server --model-path ${q(model)} --host ${q(c.tailscaleOnly ? "<tailscale-ip>" : "127.0.0.1")} --port 1234 --context-length ${c.context} --max-running-requests ${c.parallel}`;
        }
        if (c.engine === "vllm") {
          const gguf = String(c.model || "").toLowerCase().endsWith(".gguf");
          const a = ["serve", q(c.model || "<model>")];
          a.push("--host", c.tailscaleOnly ? "<tailscale-ip>" : "127.0.0.1", "--port", "1234",
            "--max-model-len", c.context, "--max-num-seqs", c.parallel);
          if (gguf && c.mmproj)
            a.push("--model-loader-extra-config", JSON.stringify({ mm_proj: c.mmproj }));
          if (c.mtpEnabled)
            a.push("--speculative-config", JSON.stringify({ method: "mtp", num_speculative_tokens: c.mtpNMax }));
          else if (c.dflashEnabled && c.dflashModel)
            a.push("--speculative-config", JSON.stringify({ method: "draft_model", model: c.dflashModel, num_speculative_tokens: c.dflashNMax }));
          if (gguf && (c.ggufTokenizers || {})[c.model])
            a.push("--tokenizer", (c.ggufTokenizers || {})[c.model]);
          a.push("--generation-config", "vllm",
            "--override-generation-config", JSON.stringify({
              temperature: c.temp, top_p: c.topP, top_k: c.topK,
              min_p: c.minP, repetition_penalty: c.repeatPenalty,
            }));
          return "vllm " + a.map(String).map(q).join(" ");
        }
        const a = [];
        if (c.hfModel) a.push("-hf", c.hfModel);
        else if (c.model) a.push("-m", q(c.model));
        if (c.mmproj) a.push("--mmproj", q(c.mmproj));
        a.push("--host", c.tailscaleOnly ? "<tailscale-ip>" : "127.0.0.1", "--port", "1234", "--metrics");
        a.push("-ngl", c.ngl, "-c", c.context, "-t", c.threads, "-tb", c.threadsBatch);
        a.push("-b", c.batchSize, "-ub", c.ubatch, "-fa", c.flashAttention ? "on" : "off");
        a.push("--cache-type-k", c.cacheTypeK, "--cache-type-v", c.cacheTypeV, "-np", c.parallel);
        if (c.cacheReuse > 0) a.push("--cache-reuse", c.cacheReuse);
        a.push("--temp", c.temp, "--top-p", c.topP, "--top-k", c.topK, "--min-p", c.minP);
        a.push("--repeat-penalty", c.repeatPenalty, "--presence-penalty", c.presencePenalty, "--frequency-penalty", c.frequencyPenalty);
        if (c.jinja) a.push("--jinja");
        if (c.mtpEnabled) a.push("--spec-type", "draft-mtp", "--spec-draft-n-max", c.mtpNMax);
        else if (c.dflashEnabled) a.push("--spec-type", "draft-dflash", "--spec-draft-n-max", c.dflashNMax);
        return "llama-server.exe " + a.map(String).map(q).join(" ");
      };
      return async (cmd) => {
        switch (cmd) {
          case "get_config":
          case "get_defaults":
            return { ...demoConfig };
          case "list_models":
          case "list_vllm_models":
            return demoModels;
          case "preview_args":
            return mockPreview(cfg);
          case "find_mmproj":
            return null;
          case "gguf_tokenizer_info":
            return { name: "Qwen3 0.6B Instruct", base: "Qwen3-0.6B", tokenizer: null };
          case "server_running":
            return false;
          default:
            return null;
        }
      };
    })();
const listen = T ? T.event.listen : async () => {};
const getCurrentWindow = T
  ? T.window.getCurrentWindow
  : () => ({ close() {}, minimize() {}, toggleMaximize() {} });

if (!T) document.documentElement.classList.add("browser-preview");

const demoConfig = {
  apiAddress: "http://127.0.0.1:1234",
  apiKey: "",
  engine: "llamacpp",
  serverPath: "",
  vllmPath: "vllm",
  sglangPath: "python3",
  ggufTokenizers: {},
  modelsDir: "C:\\models",
  model: "C:\\models\\Qwen3.6-35B-A3B-Uncensored-Genesis-Final-APEX-Compact.gguf",
  hfModel: "",
  mmproj: "C:\\models\\mmproj-Qwen3.6-35B-A3B-Uncensored-Genesis-F16.gguf",
  context: 131072,
  ngl: 36,
  batchSize: 2048,
  ubatch: 2048,
  threads: 6,
  threadsBatch: 6,
  flashAttention: true,
  cacheTypeK: "q8_0",
  cacheTypeV: "q8_0",
  cacheReuse: 0,
  parallel: 1,
  jinja: true,
  mtpEnabled: false,
  mtpNMax: 3,
  mtpModel: "",
  dflashEnabled: false,
  dflashModel: "",
  dflashNMax: 15,
  reasoning: "on",
  reasoningBudget: -1,
  reasoningEffort: "",
  preserveReasoning: true,
  temp: 1.0,
  topP: 0.95,
  topK: 20,
  minP: 0.05,
  repeatPenalty: 1.1,
  presencePenalty: 0.0,
  frequencyPenalty: 0.0,
  mmprojOffload: false,
  imageMinTokens: 1024,
  imageMaxTokens: 1024,
  mtmdBatchMaxTokens: 1024,
  extraArgs: "",
};

const demoModels = [
  { name: "Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf", path: "C:\\models\\Gemma4-31B.gguf", sizeMb: 19000, isMmproj: false },
  { name: "LFM2.5-2.6B-BF16.gguf", path: "C:\\models\\LFM2.5.gguf", sizeMb: 2700, isMmproj: false },
  { name: "Muse-Glimmer-30B-UD-Q3_K_XL.gguf", path: "C:\\models\\Muse-Glimmer.gguf", sizeMb: 15500, isMmproj: false },
  { name: "Qwen3.6-35B-A3B-Uncensored-Genesis-Final-APEX-Compact.gguf", path: "C:\\models\\Qwen3.6-35B.gguf", sizeMb: 18600, isMmproj: false },
  { name: "Qwen3.8-27B-Q4_K_M.gguf", path: "C:\\models\\Qwen3.8-27B.gguf", sizeMb: 16400, isMmproj: false },
];

const CTX_STOPS = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1000000];

const OPTION_SETS = {
  cacheTypes: ["f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"],
  reasoningModes: ["auto", "on", "off"],
  efforts: [
    ["", "default"],
    ["low", "low"],
    ["medium", "medium"],
    ["high", "high"],
    ["max", "max"],
  ],
};

const $ = (id) => document.getElementById(id);

let cfg = {};
let serverState = "idle";
let argEditing = false;
let argBefore = "";
let saveTimer = null;

const argbar = $("argbar");
const argText = $("argText");
const ctxTrack = $("ctxTrack");
const ctxThumb = $("ctxThumb");
const ctxFill = $("ctxFill");
const ctxValue = $("ctxValue");
const ctxTicks = $("ctxTicks");
const modelSelect = $("model");
const presetList = $("presetList");
const presetName = $("presetName");
const logEl = $("log");
const logAutoscroll = $("logAutoscroll");
const statusChipText = $("statusChipText");
const statusTitle = $("statusTitle");
const statusSub = $("statusSub");
const btnPower = $("btnPower");
const btnPowerLabel = btnPower.querySelector(".btn-power-label");

const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
const basename = (p) => String(p).split(/[\\/]/).pop() || p;

/* ---------- settings sections ----------
   One column of accordions, one open at a time. Each header doubles as a
   summary: it shows the value that section currently holds, so the whole
   configuration is readable without opening anything.

   Whole sections that the current engine has nothing to say about carry
   [data-llama-only] / [data-vllm-only] and are hidden by CSS — no measuring
   required. Individual settings inside a section keep the same attributes. */

function switchSection(name) {
  document.querySelectorAll(".acc").forEach((a) =>
    a.classList.toggle("open", a.dataset.section === name)
  );
}

function wireSections() {
  document.querySelectorAll(".acc-head").forEach((h) =>
    h.addEventListener("click", () => {
      const acc = h.closest(".acc");
      switchSection(acc.classList.contains("open") ? "" : acc.dataset.section);
    })
  );
}

/* If the engine switch hides the section you're looking at, open one that
   still exists. offsetParent is safe here: a closed .acc is still laid out,
   so only display:none (the engine rule) reads as hidden. */
function ensureSectionVisible() {
  const open = document.querySelector(".acc.open");
  if (open && open.offsetParent !== null) return;
  const first = [...document.querySelectorAll(".acc")].find(
    (a) => a.offsetParent !== null
  );
  if (first) switchSection(first.dataset.section);
}

/* ---------- launch command ---------- */
/* Always on screen — no collapse. This is also where any flag the app
   doesn't recognise goes: type it here and it reaches the server. */

function wireCommandPanel() {
  const panel = $("argbar");
  if (!panel) return;
  argText.addEventListener("focus", () => {
    if (!argEditing) {
      argEditing = true;
      argBefore = argText.textContent;
    }
    panel.classList.add("editing");
  });
}

/* ---------- throughput dials ---------- */
/* Backend-specific log formats are normalized by src/perf.js. */

const DIAL_MAX = { pp: 500, decode: 60 };
let dialHasRates = false;

/* Plain ring meter, mirrored by the .dial-track / .dial-arc circles in
   index.html: r=20, so the full circumference is 2πr = 125.66. */
const DIAL_C = 125.66;

function setDial(which, value) {
  const arc = $(which === "pp" ? "arcPp" : "arcDec");
  const val = $(which === "pp" ? "valPp" : "valDec");
  const wrap = $(which === "pp" ? "dialPp" : "dialDec");
  if (!arc || !val || !wrap) return;
  if (value === null || !isFinite(value) || value <= 0) {
    arc.style.strokeDasharray = "0 " + DIAL_C;
    val.textContent = "—";
    wrap.classList.remove("live");
    return;
  }
  const frac = Math.max(0, Math.min(1, value / DIAL_MAX[which]));
  arc.style.strokeDasharray = (frac * DIAL_C).toFixed(2) + " " + DIAL_C;
  val.textContent =
    (value >= 100 ? Math.round(value) : value.toFixed(1)) + "/t";
  wrap.classList.add("live");
}

function readThroughput(text) {
  const rates = window.LlamaStudioPerf?.parseThroughput(text);
  if (!rates) return false;
  dialHasRates = true;
  if (rates.decode !== undefined) {
    setDial("pp", null);
    setDial("decode", rates.decode);
  } else if (rates.pp !== undefined) {
    setDial("decode", null);
    setDial("pp", rates.pp);
  }
  return true;
}

function resetDials() {
  setDial("pp", null);
  setDial("decode", null);
  dialHasRates = false;
}

/* ---------- readouts ---------- */

/* Mirrors cfg into the nav hints. Owns no state — everything is derived. */
function renderReadout() {
  const model = cfg.hfModel || cfg.model || "";
  const modelName = model ? basename(model) : "not set";
  const ctx = Number(cfg.context) || 0;
  const ctxText = ctx ? ctx.toLocaleString("en-US") : "—";

  const set = (id, text) => {
    const el = $(id);
    if (el) el.textContent = text;
  };
  set("navModel", modelName);
  set("navContext", ctxText);
  set("navLayers", currentEngine() !== "llamacpp" || cfg.ngl === undefined ? "—" : cfg.ngl >= 99 ? "all" : `${cfg.ngl} layers`);
  set("navTemp", cfg.temp === undefined ? "—" : String(cfg.temp));
  set("navReasoning", currentEngine() === "llamacpp" ? cfg.reasoning || "auto" : "server default");
  set("navVision", cfg.mmproj ? "encoder set" : "off");
  set("navServer", (() => {
    try { return new URL(cfg.apiAddress).host; } catch { return cfg.apiAddress || "—"; }
  })());
  set("navThroughput", (() => {
    if (currentEngine() === "sglang") return cfg.parallel > 1 ? `${cfg.parallel} requests` : "standard";
    if (cfg.mtpEnabled) return "MTP";
    if (cfg.dflashEnabled) return "DFlash";
    if (cfg.parallel > 1) return `${cfg.parallel} slots`;
    return "standard";
  })());
  set("bar1Title", modelName === "not set" ? "Llama Studio" : modelName);
  set("chatModel", modelName === "not set" ? "no model" : modelName);
}

/* ---------- persistence ---------- */

function save() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => invoke("save_config", { config: cfg }).catch(() => {}), 350);
}

/* ---------- field binding ---------- */

function populateOptionSets() {
  document.querySelectorAll("select[data-options]").forEach((sel) => {
    const opts = OPTION_SETS[sel.dataset.options] || [];
    sel.innerHTML = opts
      .map((o) => (Array.isArray(o)
        ? `<option value="${esc(o[0])}">${esc(o[1])}</option>`
        : `<option value="${esc(o)}">${esc(o)}</option>`))
      .join("");
  });
}

function bindFields() {
  document.querySelectorAll("[data-field]").forEach((el) => {
    const field = el.dataset.field;
    const ev = el.type === "checkbox" ? "change" : "input";
    el.addEventListener(ev, () => {
      let v;
      if (el.type === "checkbox") v = el.checked;
      else if (el.type === "number") {
        const n = Number(el.value);
        if (!Number.isFinite(n)) return;
        v = n;
      } else v = el.value;
      cfg[field] = v;
      if (v === true && field === "mtpEnabled") {
        cfg.dflashEnabled = false;
        const dflash = document.querySelector('[data-field="dflashEnabled"]');
        if (dflash) dflash.checked = false;
      } else if (v === true && field === "dflashEnabled") {
        cfg.mtpEnabled = false;
        const mtp = document.querySelector('[data-field="mtpEnabled"]');
        if (mtp) mtp.checked = false;
      }
      if (field === "modelsDir") refreshModels();
      if (!argEditing) refreshArgs();
      save();
    });
  });
}

function renderAll() {
  document.querySelectorAll("[data-field]").forEach((el) => {
    const field = el.dataset.field;
    if (!(field in cfg)) return;
    if (el.type === "checkbox") el.checked = !!cfg[field];
    else el.value = cfg[field] ?? "";
  });
  if (cfg.model && modelSelect.value !== cfg.model) modelSelect.value = cfg.model;
  syncEngine();
  renderContext();
  refreshArgs();
}

/* ---------- engine toggle ---------- */

const ENGINE_LABELS = {
  llamacpp: {
    model: "Model (.gguf)",
    modelTip: "The main language model, from your models folder.",
    modelPlaceholder: "",
    dir: "Models folder",
    dirTip: "Folder that holds your .gguf files.",
    dirPlaceholder: "folder containing .gguf files",
    server: "llama-server path",
    serverTip: "Path to llama-server (or llama-server.exe on Windows). Auto-detected; only change if you moved it.",
  },
  vllm: {
    model: "Model (GGUF / HF folder)",
    modelTip: "Same models folder as llama.cpp. A .gguf is served through the vLLM GGUF plugin - tokenizer and vision encoder are picked up automatically. HuggingFace-format folders work too.",
    dir: "Models folder",
    dirTip: "Folder that holds your models - .gguf files and/or HuggingFace-format folders.",
    dirPlaceholder: "folder containing models",
    server: "vLLM command",
    serverTip: "Command that launches the vLLM server. Usually just `vllm` from your PATH.",
  },
  sglang: {
    model: "HF model (folder or repo)",
    modelTip: "SGLang serves a local Hugging Face model folder or a Hugging Face repo ID.",
    dir: "Models folder",
    dirTip: "Folder containing local Hugging Face model directories.",
    dirPlaceholder: "folder containing HF model folders",
    server: "Python command",
    serverTip: "Python interpreter with SGLang installed; launches `python -m sglang.launch_server`.",
  },
};

function currentEngine() {
  return ["vllm", "sglang"].includes(cfg.engine) ? cfg.engine : "llamacpp";
}

function syncEngine() {
  const eng = currentEngine();
  document.body.dataset.engine = eng;
  document.querySelectorAll("#engineToggle [data-engine-opt]").forEach((b) =>
    b.classList.toggle("active", b.dataset.engineOpt === eng)
  );
  const L = ENGINE_LABELS[eng];
  const set = (id, key) => {
    const el = document.getElementById(id);
    if (el && L[key]) el.textContent = L[key];
  };
  set("modelLabel", "model");
  set("modelsDirLabel", "dir");
  set("serverPathLabel", "server");
  const tip = (id, key) => {
    const el = document.getElementById(id);
    if (el && L[key]) el.title = L[key];
  };
  tip("modelTip", "modelTip");
  tip("modelsDirTip", "dirTip");
  tip("serverPathTip", "serverTip");
  const dirInput = document.querySelector('[data-field="modelsDir"]');
  if (dirInput) dirInput.placeholder = L.dirPlaceholder;
  renderGgufTokenizer();
  renderReadout();
  ensureSectionVisible();
}

function wireEngineToggle() {
  document.querySelectorAll("#engineToggle [data-engine-opt]").forEach((b) => {
    b.addEventListener("click", () => {
      if (["launching", "starting", "running", "stopping"].includes(serverState)) return;
      if (currentEngine() === b.dataset.engineOpt) return;
      cfg.engine = b.dataset.engineOpt;
      syncEngine();
      refreshModels();
      refreshArgs();
      save();
    });
  });
}

/* ---------- all-in-one args ---------- */

async function refreshArgs() {
  if (argEditing) return;
  try {
    const text = await invoke("preview_args", { config: cfg });
    if (argEditing) return;
    argText.textContent = text;
  } catch {}
}

function wireArgbar() {
  argText.addEventListener("focus", () => {
    argEditing = true;
    argBefore = argText.textContent;
    argbar.classList.add("editing");
  });
  argText.addEventListener("blur", async () => {
    if (!argEditing) return;
    argEditing = false;
    argbar.classList.remove("editing");
    const text = argText.textContent.trim();
    if (!text || text === argBefore) {
      argText.textContent = argBefore;
      return;
    }
    try {
      cfg = await invoke("apply_args", { config: cfg, text });
      renderAll();
      save();
      refreshModels();
    } catch {
      argText.textContent = argBefore;
    }
  });
  argText.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      argText.blur();
    } else if (e.key === "Escape") {
      argText.textContent = argBefore;
      argText.blur();
    }
  });
  argText.addEventListener("paste", (e) => {
    e.preventDefault();
    const t = (e.clipboardData || window.clipboardData).getData("text");
    document.execCommand("insertText", false, t);
  });
}

/* ---------- context slider ---------- */

function nearestStopIdx(value) {
  let best = 0;
  let bestD = Infinity;
  CTX_STOPS.forEach((s, i) => {
    const d = Math.abs(s - value);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  });
  return best;
}

function buildContextTicks() {
  const n = CTX_STOPS.length;
  for (let i = 0; i < n; i++) {
    const t = document.createElement("div");
    t.className = "ctx-tick";
    t.style.left = (i / (n - 1)) * 100 + "%";
    ctxTicks.appendChild(t);
  }
  // spell the stops out under the rail — the slider alone doesn't say what
  // the positions actually cost
  const scale = $("ctxScale");
  if (scale && !scale.children.length) {
    CTX_STOPS.forEach((s, i) => {
      const el = document.createElement("span");
      // 1m, not 976k — the top stop is a round million, so label it as one
      el.textContent =
        s >= 1000000 ? s / 1000000 + "m" : s >= 1024 ? s / 1024 + "k" : String(s);
      const pct = (i / (n - 1)) * 100;
      el.style.left = pct + "%";
      // keep the end labels inside the box
      if (i === 0) el.style.transform = "none";
      if (i === n - 1) el.style.transform = "translateX(-100%)";
      scale.appendChild(el);
    });
  }
}

function renderContext() {
  const v = Number(cfg.context) || CTX_STOPS[1];
  const idx = nearestStopIdx(v);
  setContextPct(idx / (CTX_STOPS.length - 1));
  // Custom values (e.g. 200000) sit between stops: thumb snaps to the
  // nearest one, but the label shows the value actually configured.
  if (!CTX_STOPS.includes(v)) ctxValue.textContent = v.toLocaleString("en-US");
  renderReadout();
}

function setContextPct(pct) {
  const clamped = Math.max(0, Math.min(1, pct));
  ctxThumb.style.left = clamped * 100 + "%";
  ctxFill.style.width = clamped * 100 + "%";
  const idx = Math.round(clamped * (CTX_STOPS.length - 1));
  ctxValue.textContent = CTX_STOPS[idx].toLocaleString("en-US");
  [...ctxTicks.children].forEach((t, i) => t.classList.toggle("on", i <= idx));
}

function wireContextSlider() {
  buildContextTicks();
  let dragging = false;
  const pctFromEvent = (e) => {
    const r = ctxTrack.getBoundingClientRect();
    return (e.clientX - r.left) / r.width;
  };
  ctxTrack.addEventListener("pointerdown", (e) => {
    dragging = true;
    ctxTrack.setPointerCapture(e.pointerId);
    ctxTrack.classList.add("dragging");
    setContextPct(pctFromEvent(e));
  });
  ctxTrack.addEventListener("pointermove", (e) => {
    if (dragging) setContextPct(pctFromEvent(e));
  });
  const release = (e) => {
    if (!dragging) return;
    dragging = false;
    ctxTrack.classList.remove("dragging");
    const idx = Math.round(Math.max(0, Math.min(1, pctFromEvent(e))) * (CTX_STOPS.length - 1));
    cfg.context = CTX_STOPS[idx];
    renderContext();
    if (!argEditing) refreshArgs();
    save();
  };
  ctxTrack.addEventListener("pointerup", release);
  ctxTrack.addEventListener("pointercancel", release);
}

/* ---------- models ---------- */

/* Both engines share one model field: a .gguf (or, under vLLM, a
   HuggingFace-format folder) from the models folder. Only the dropdown's
   contents and the labels differ per engine. */
async function refreshModels() {
  const engine = currentEngine();
  const llamaCpp = engine === "llamacpp";
  const current = cfg.model || "";
  let entries = [];
  try {
    entries = await invoke(llamaCpp ? "list_models" : "list_vllm_models", {
      dir: cfg.modelsDir || "",
    });
  } catch {}
  const models = (entries || []).filter((m) =>
    !m.isMmproj && !(engine === "sglang" && /\.gguf$/i.test(m.path))
  );
  modelSelect.innerHTML = "";
  const inList = models.some((m) => m.path === current);
  const unsupportedSglangGguf = engine === "sglang" && /\.gguf$/i.test(current);
  if (current && !inList && !unsupportedSglangGguf) {
    const o = document.createElement("option");
    o.value = current;
    o.textContent = `${basename(current)}  ·  (current)`;
    modelSelect.appendChild(o);
  }
  models.forEach((m) => {
    const o = document.createElement("option");
    o.value = m.path;
    const size = m.sizeMb > 0 ? `  ·  ${(m.sizeMb / 1024).toFixed(1)} GB` : "";
    o.textContent = `${m.name}${size}`;
    modelSelect.appendChild(o);
  });
  modelSelect.value = current || "";
  renderReadout();
}

async function onModelPicked() {
  cfg.model = modelSelect.value || "";
  if (cfg.model) {
    cfg.hfModel = "";
    const hf = document.querySelector('[data-field="hfModel"]');
    if (hf) hf.value = "";
    if (currentEngine() !== "sglang") {
      try {
        const mm = await invoke("find_mmproj", { path: cfg.model });
        if (mm) {
          cfg.mmproj = mm;
          const mmEl = document.querySelector('[data-field="mmproj"]');
          if (mmEl) mmEl.value = mm;
        }
      } catch {}
    }
    await autoGgufTokenizer();
  }
  save();
  refreshArgs();
  renderReadout();
}

/* ---------- GGUF tokenizer (vLLM) ---------- */

function isGgufModel() {
  return (
    currentEngine() === "vllm" &&
    String(cfg.model || "").toLowerCase().endsWith(".gguf")
  );
}

function renderGgufTokenizer() {
  const wrap = $("ggufTokenizerField");
  if (!wrap) return;
  const on = isGgufModel();
  wrap.style.display = on ? "" : "none";
  if (on) {
    const map = cfg.ggufTokenizers || {};
    $("ggufTokenizer").value = map[cfg.model] || "";
  }
}

function wireGgufTokenizer() {
  $("ggufTokenizer").addEventListener("input", () => {
    const m = cfg.model;
    if (!m) return;
    cfg.ggufTokenizers = cfg.ggufTokenizers || {};
    cfg.ggufTokenizers[m] = $("ggufTokenizer").value.trim();
    save();
    if (!argEditing) refreshArgs();
  });
}

/* Fill in the --tokenizer for a freshly picked GGUF (vLLM only). The backend
   matches the GGUF metadata against the local HF cache / models folder, then
   falls back to a huggingface.co search (that search must run in the backend:
   HF's API blocks cross-origin calls from the UI). Remembered per model. */
async function autoGgufTokenizer() {
  if (!isGgufModel()) return;
  const path = cfg.model;
  cfg.ggufTokenizers = cfg.ggufTokenizers || {};
  if (cfg.ggufTokenizers[path]) return;
  try {
    const info = await invoke("gguf_tokenizer_info", { path, modelsDir: cfg.modelsDir || "" });
    if (info && info.tokenizer) {
      cfg.ggufTokenizers[path] = info.tokenizer;
      save();
      renderGgufTokenizer();
      if (!argEditing) refreshArgs();
    }
  } catch {}
}

/* ---------- presets ---------- */

let selectedPreset = "";

async function renderPresets() {
  let names = [];
  try {
    names = Object.keys(await invoke("get_presets")).sort();
  } catch {}
  const menu = $("presetMenu");
  menu.innerHTML = names.length
    ? names
        .map(
          (n) =>
            `<li role="option" tabindex="-1" aria-selected="${n === selectedPreset}" class="select-opt${n === selectedPreset ? " sel" : ""}" data-name="${esc(n)}">${esc(n)}</li>`
        )
        .join("")
    : '<li class="select-empty" role="presentation">No presets saved yet</li>';
  if (!names.includes(selectedPreset)) selectedPreset = "";
  $("presetListText").textContent = selectedPreset || "Preset…";
}

function flashButton(btn, text) {
  const old = btn.textContent;
  btn.textContent = text;
  setTimeout(() => (btn.textContent = old), 1200);
}

async function loadPreset(name) {
  if (!name) return;
  try {
    const p = await invoke("get_presets");
    if (p[name]) {
      cfg = p[name];
      renderAll();
      await refreshModels();
      save();
    }
  } catch {}
}

function wirePresets() {
  const menu = $("presetMenu");
  const options = () => [...menu.querySelectorAll(".select-opt")];
  const focusOption = (index) => {
    const items = options();
    if (items.length) items[(index + items.length) % items.length].focus();
  };
  presetList.addEventListener("click", (e) => {
    e.stopPropagation();
    togglePop(menu, presetList);
    if (!menu.hidden) focusOption(0);
  });
  presetList.addEventListener("keydown", (e) => {
    if (!["ArrowDown", "Enter", " "].includes(e.key)) return;
    e.preventDefault();
    if (menu.hidden) togglePop(menu, presetList);
    focusOption(0);
  });
  menu.addEventListener("keydown", (e) => {
    const items = options();
    const current = items.indexOf(document.activeElement);
    if (e.key === "Escape") {
      e.preventDefault();
      closePop(menu, presetList);
      presetList.focus();
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      focusOption(current + 1);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      focusOption(current < 0 ? items.length - 1 : current - 1);
    } else if ((e.key === "Enter" || e.key === " ") && current >= 0) {
      e.preventDefault();
      items[current].click();
    }
  });
  menu.addEventListener("click", async (e) => {
    e.stopPropagation();
    const opt = e.target.closest(".select-opt");
    if (!opt) return;
    selectedPreset = opt.dataset.name;
    closePop(menu, presetList);
    presetList.focus();
    await renderPresets();
    await loadPreset(selectedPreset);
  });
  $("btnPresetSave").addEventListener("click", async (e) => {
    const btn = e.currentTarget;
    const name = (presetName.value || selectedPreset || "").trim();
    if (!name) return;
    try {
      await invoke("save_preset", { name, config: cfg });
      presetName.value = "";
      selectedPreset = name;
      await renderPresets();
      flashButton(btn, "Saved ✓");
    } catch {}
  });
  $("btnPresetDelete").addEventListener("click", async (e) => {
    const btn = e.currentTarget;
    const name = selectedPreset;
    if (!name) return;
    try {
      await invoke("delete_preset", { name });
      selectedPreset = "";
      await renderPresets();
      flashButton(btn, "Deleted");
    } catch {}
  });
  $("btnResetDefaults").addEventListener("click", async () => {
    cfg = await invoke("get_defaults");
    renderAll();
    await refreshModels();
    save();
  });
}

/* ---------- browse buttons ---------- */

function wireBrowse() {
  const setField = (field, value) => {
    cfg[field] = value || "";
    const el = document.querySelector(`[data-field="${field}"]`);
    if (el) el.value = cfg[field];
    save();
    if (!argEditing) refreshArgs();
  };
  $("btnBrowseModel").addEventListener("click", async () => {
    const p = await invoke("browse_path", { kind: currentEngine() === "llamacpp" ? "gguf" : "dir" }).catch(() => null);
    if (!p) return;
    cfg.model = p;
    cfg.hfModel = "";
    const hf = document.querySelector('[data-field="hfModel"]');
    if (hf) hf.value = "";
    if (currentEngine() !== "sglang") {
      try {
        const mm = await invoke("find_mmproj", { path: p });
        if (mm) setField("mmproj", mm);
      } catch {}
    }
    await autoGgufTokenizer();
    await refreshModels();
    save();
    refreshArgs();
  });
  $("btnRefreshModels").addEventListener("click", refreshModels);
  $("btnBrowseModelsDir").addEventListener("click", async () => {
    const p = await invoke("browse_path", { kind: "dir" }).catch(() => null);
    if (!p) return;
    setField("modelsDir", p);
    refreshModels();
  });
  $("btnBrowseMmproj").addEventListener("click", async () => {
    const p = await invoke("browse_path", { kind: "gguf" }).catch(() => null);
    if (p) setField("mmproj", p);
  });
  $("btnBrowseMtp").addEventListener("click", async () => {
    const p = await invoke("browse_path", { kind: "gguf" }).catch(() => null);
    if (p) setField("mtpModel", p);
  });
  $("btnBrowseDflash").addEventListener("click", async () => {
    const kind = currentEngine() === "vllm" ? "dir" : "gguf";
    const p = await invoke("browse_path", { kind }).catch(() => null);
    if (p) setField("dflashModel", p);
  });
  $("btnBrowseServer").addEventListener("click", async () => {
    const p = await invoke("browse_path", { kind: "exe" }).catch(() => null);
    if (p) setField("serverPath", p);
  });
}

/* ---------- power (start / stop) ---------- */

const STATE_UI = {
  idle:     { chip: "Idle",    title: "Server idle",     btn: "Start", cls: "idle" },
  launching:{ chip: "Loading", title: "Starting server…", btn: "…", cls: "starting" },
  starting: { chip: "Loading", title: "Loading model…",  btn: "Stop",  cls: "starting" },
  stopping: { chip: "Stopping", title: "Stopping server…", btn: "…", cls: "starting" },
  running:  { chip: "Running", title: "Server running",  btn: "Stop",  cls: "running" },
  stopped:  { chip: "Idle",    title: "Server stopped",  btn: "Start", cls: "idle" },
  failed:   { chip: "Error",   title: "Couldn't start",  btn: "Start", cls: "failed" },
};

function applyServerState(state, detail) {
  serverState = state;
  const ui = STATE_UI[state] || STATE_UI.idle;
  document.body.dataset.server = ui.cls;
  statusChipText.textContent = ui.chip;
  statusTitle.textContent = ui.title;
  btnPowerLabel.textContent = ui.btn;
  // dials only mean something while a server is up
  if (state === "idle" || state === "stopped" || state === "failed") resetDials();
  if (state === "launching") statusSub.textContent = "Starting the inference process…";
  else if (state === "starting") statusSub.textContent = "Loading the model — press Stop to cancel startup.";
  else if (state === "stopping") statusSub.textContent = "Waiting for the server process to exit.";
  else if (state === "running") statusSub.textContent = detail || cfg.apiAddress;
  else statusSub.textContent = detail || "Pick a model, then press Start.";
  // the chat header tracks the same thing
  if (state === "running") setChatState("ready", "ok");
  else if (state === "launching" || state === "starting") setChatState("loading", "live");
  else if (state === "stopping") setChatState("stopping", "live");
  else if (state === "failed") setChatState("error", "bad");
  else setChatState("idle", "");
}

function wirePower() {
  btnPower.addEventListener("click", async () => {
    if (serverState === "launching" || serverState === "stopping") return;
    if (serverState === "starting" || serverState === "running") {
      applyServerState("stopping", "Stopping…");
      try {
        await invoke("stop_server");
      } catch (err) {
        applyServerState("failed", String(err));
      }
      return;
    }
    applyServerState("launching");
    try {
      const address = await invoke("start_server", { config: cfg });
      if (cfg.tailscaleOnly && address) {
        cfg.apiAddress = address;
        const addressField = document.querySelector('[data-field="apiAddress"]');
        if (addressField) addressField.value = address;
        await invoke("save_config", { config: cfg });
      }
    } catch (err) {
      applyServerState("failed", String(err));
    }
  });
}

function onServerStatus(p) {
  if (p.state === "starting") applyServerState("starting");
  else if (p.state === "running") applyServerState("running", `${cfg.apiAddress}  ·  pid ${p.pid ?? "?"}`);
  else if (p.state === "stopped") {
    const code = p.code ?? 0;
    applyServerState("idle", code === 0 ? "Server stopped." : `Server exited with code ${code}.`);
  }
}

/* ---------- logs ---------- */

let logQuery = "";

function applyLogFilters() {
  const filter = logEl.dataset.filter;
  const q = logQuery.toLowerCase();
  let shown = 0;
  logEl.querySelectorAll(".log-line").forEach((el) => {
    const kind = el.dataset.kind;
    const kindOk =
      filter === "all" ||
      (filter === "info" && (kind === "info" || kind === "error" || kind === "perf")) ||
      (filter === "stdout" && kind === "stdout") ||
      (filter === "error" && (kind === "error" || kind === "stderr"));
    const qOk = !q || el.textContent.toLowerCase().includes(q);
    const ok = kindOk && qOk;
    el.classList.toggle("hidden", !ok);
    if (ok) shown++;
  });
  $("logsCount").textContent =
    shown === logEl.children.length
      ? logEl.children.length + " lines"
      : shown + " / " + logEl.children.length + " lines";
}

function atLogBottom() {
  return logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 24;
}

function appendLog(kind, text) {
  if (logEl.querySelector(".log-empty")) logEl.innerHTML = "";
  // the perf line is the only one that moves the dials
  const isPerf = readThroughput(text);
  const div = document.createElement("div");
  div.className = "log-line " + kind + (isPerf ? " perf" : "");
  div.dataset.kind = isPerf ? "perf" : kind;
  div.textContent = text;
  logEl.appendChild(div);
  while (logEl.children.length > 2500) logEl.removeChild(logEl.firstChild);
  applyLogFilters();
  const visible = !div.classList.contains("hidden");
  if (visible && logAutoscroll.checked && atLogBottom()) {
    logEl.scrollTop = logEl.scrollHeight;
  }
  $("btnJumpLatest").hidden = atLogBottom();
}

function wireLogs() {
  document.querySelectorAll(".chipbtn").forEach((b) => {
    b.addEventListener("click", () => {
      document.querySelectorAll(".chipbtn").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      logEl.dataset.filter = b.dataset.filter;
      applyLogFilters();
      logEl.scrollTop = logEl.scrollHeight;
    });
  });
  $("logSearch").addEventListener("input", (e) => {
    logQuery = e.target.value.trim();
    applyLogFilters();
  });
  logEl.addEventListener("scroll", () => {
    $("btnJumpLatest").hidden = atLogBottom();
  });
  $("btnJumpLatest").addEventListener("click", () => {
    logEl.scrollTop = logEl.scrollHeight;
    $("btnJumpLatest").hidden = true;
  });
  logAutoscroll.addEventListener("change", () => {
    if (logAutoscroll.checked) logEl.scrollTop = logEl.scrollHeight;
  });
  $("btnClearLogs").addEventListener("click", () => {
    logEl.innerHTML = '<div class="log-empty">Logs cleared. Start the server and its output lands here.</div>';
    $("logsCount").textContent = "0 lines";
    $("btnJumpLatest").hidden = true;
  });
}

/* ---------- chat ----------
   Talks to whatever server the Settings page is running, over its
   OpenAI-compatible endpoint. No separate backend command is needed — all
   supported engines expose /v1/chat/completions. */

let chatHistory = [];
let chatAbort = null;
let chatEpoch = 0;

function chatEndpoint() {
  let base = String(cfg.apiAddress || "").trim().replace(/\/+$/, "");
  if (!base) return "";
  if (!/^https?:\/\//i.test(base)) base = "http://" + base;
  return base + "/v1/chat/completions";
}

function chatModelName() {
  return cfg.hfModel || cfg.model || "";
}

function setChatState(text, kind) {
  const el = $("chatState");
  el.textContent = text;
  el.dataset.kind = kind || "";
}

function scrollChat() {
  const log = $("chatLog");
  log.scrollTop = log.scrollHeight;
}

function addChatBubble(role, text) {
  const empty = $("chatEmpty");
  if (empty) empty.remove();
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;
  const who = document.createElement("span");
  who.className = "msg-role";
  who.textContent = role === "user" ? "you" : "model";
  const body = document.createElement("div");
  body.className = "msg-body";
  body.textContent = text;
  wrap.append(who, body);
  $("chatLog").appendChild(wrap);
  scrollChat();
  return body;
}

function addChatError(text) {
  const empty = $("chatEmpty");
  if (empty) empty.remove();
  const wrap = document.createElement("div");
  wrap.className = "msg error";
  wrap.textContent = text;
  $("chatLog").appendChild(wrap);
  scrollChat();
}

async function sendChat(text) {
  const body = text.trim();
  if (!body) return;
  const endpoint = chatEndpoint();
  if (!endpoint) return addChatError("No address set. Add one under Network.");
  if (serverState !== "running") return addChatError("Server isn't running. Press Start first.");

  addChatBubble("user", body);
  const reply = addChatBubble("model", "");
  reply.classList.add("streaming");

  chatHistory.push({ role: "user", content: body });
  const payload = { model: chatModelName() || "local", messages: chatHistory, stream: true };
  if (currentEngine() === "sglang") {
    Object.assign(payload, {
      temperature: cfg.temp,
      top_p: cfg.topP,
      presence_penalty: cfg.presencePenalty,
      frequency_penalty: cfg.frequencyPenalty,
      extra_body: {
        top_k: cfg.topK,
        min_p: cfg.minP,
        repetition_penalty: cfg.repeatPenalty,
      },
    });
  }

  const epoch = chatEpoch;
  const controller = new AbortController();
  chatAbort = controller;
  $("btnSend").disabled = true;
  $("btnStopChat").hidden = false;
  setChatState("generating", "live");

  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(cfg.apiKey ? { Authorization: `Bearer ${cfg.apiKey}` } : {}),
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (!res.ok) throw new Error("HTTP " + res.status);

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      // SSE permits CRLF as well as LF separators.
      const parts = buf.split(/\r?\n\r?\n/);
      buf = parts.pop();
      for (const part of parts) {
        const data = part.split(/\r?\n/)
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trimStart())
          .join("\n");
        if (!data || data === "[DONE]") continue;
        try {
          const delta = JSON.parse(data).choices?.[0]?.delta?.content;
          if (typeof delta === "string") reply.textContent += delta;
        } catch {}
      }
      scrollChat();
    }
    buf += dec.decode();
    if (buf.trim()) {
      const data = buf.split(/\r?\n/)
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (data && data !== "[DONE]") {
        try {
          const delta = JSON.parse(data).choices?.[0]?.delta?.content;
          if (typeof delta === "string") reply.textContent += delta;
        } catch {}
      }
    }
    if (epoch !== chatEpoch) return;
    reply.classList.remove("streaming");
    if (!reply.textContent) reply.textContent = "(no reply)";
    chatHistory.push({ role: "assistant", content: reply.textContent });
    resetDials();
    setChatState("idle", "");
  } catch (err) {
    if (epoch !== chatEpoch) return;
    reply.classList.remove("streaming");
    if (err.name === "AbortError") {
      if (!reply.textContent) reply.textContent = "(stopped)";
      chatHistory.push({ role: "assistant", content: reply.textContent });
      resetDials();
      setChatState("idle", "");
    } else {
      reply.remove();
      addChatError("Could not reach " + endpoint + " — " + err.message);
      setChatState("offline", "bad");
    }
  } finally {
    if (chatAbort === controller) {
      chatAbort = null;
      $("btnSend").disabled = false;
      $("btnStopChat").hidden = true;
    }
  }
}

function wireChat() {
  const form = $("chatForm");
  const input = $("chatInput");
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    if (chatAbort) return;
    const text = input.value;
    input.value = "";
    input.style.height = "auto";
    sendChat(text);
  });
  $("btnStopChat").addEventListener("click", () => chatAbort?.abort());
  $("btnClearChat").addEventListener("click", () => {
    chatEpoch += 1;
    chatHistory = [];
    chatAbort?.abort();
    $("chatLog").innerHTML =
      '<div class="chat-empty" id="chatEmpty"><p>Start the server, then send a message. Conversations use the model and address set under <b>Model</b> and <b>Network</b>.</p></div>';
    setChatState("idle", "");
  });
  // Enter sends, Shift+Enter breaks the line
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });
  // grow with the text instead of scrolling inside the box
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 160) + "px";
  });
}

/* ---------- popovers + window chrome ---------- */

/* One mechanism for the settings popover and the preset menu: open on the
   trigger, close on Escape or a click anywhere else, and never leave two open. */
function togglePop(pop, trigger) {
  const open = !pop.hidden;
  closeAllPops();
  if (open) return;
  pop.hidden = false;
  trigger.setAttribute("aria-expanded", "true");
  pop.classList.add("in");
}

function closePop(pop, trigger) {
  pop.hidden = true;
  pop.classList.remove("in");
  if (trigger) trigger.setAttribute("aria-expanded", "false");
}

function closeAllPops() {
  document.querySelectorAll(".pop:not([hidden]), .select-menu:not([hidden])").forEach((pop) => {
    pop.hidden = true;
    pop.classList.remove("in");
  });
  document.querySelectorAll('[aria-expanded="true"]').forEach((b) =>
    b.setAttribute("aria-expanded", "false")
  );
}

function wirePopovers() {
  $("btnSettings").addEventListener("click", (e) => {
    e.stopPropagation();
    togglePop($("settingsMenu"), $("btnSettings"));
  });
  // keep clicks inside the settings popover from closing it
  $("settingsMenu").addEventListener("click", (e) => e.stopPropagation());
  document.addEventListener("click", closeAllPops);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeAllPops();
  });
  $("uiVersion").addEventListener("click", (e) => {
    const b = e.target.closest("[data-ui]");
    if (!b || b.disabled) return;
    document.querySelectorAll("#uiVersion [data-ui]").forEach((x) =>
      x.classList.toggle("active", x === b)
    );
  });
}

function positionGlider() {
  const active = document.querySelector(".tab.active");
  const glider = $("tabGlider");
  if (!active) return;
  glider.style.left = active.offsetLeft + "px";
  glider.style.width = active.offsetWidth + "px";
}

function switchPage(page) {
  document.body.dataset.page = page;
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.page === page));
  document.querySelectorAll(".page").forEach((p) =>
    p.classList.toggle("page-active", p.id === "page-" + page)
  );
  positionGlider();
}

function wireTabs() {
  document.querySelectorAll(".tab").forEach((t) =>
    t.addEventListener("click", () => switchPage(t.dataset.page))
  );
  window.addEventListener("resize", positionGlider);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(positionGlider);
}

function wireWindowControls() {
  const win = getCurrentWindow();
  $("winClose").addEventListener("click", () => win.close());
}

/* ---------- init ---------- */

window.addEventListener("DOMContentLoaded", async () => {
  populateOptionSets();
  wireWindowControls();
  wireTabs();
  wireSections();
  wireCommandPanel();
  wireArgbar();
  wireContextSlider();
  wireEngineToggle();
  wirePower();
  wirePresets();
  wireBrowse();
  bindFields();
  wireGgufTokenizer();
  wireLogs();
  wireChat();
  wirePopovers();
  modelSelect.addEventListener("change", onModelPicked);

  listen("server-log", (e) => appendLog(e.payload.kind, e.payload.text));
  listen("server-status", (e) => onServerStatus(e.payload));
  listen("server-activity", (e) => {
    if (!e.payload.active && dialHasRates) resetDials();
  });

  try {
    cfg = await invoke("get_config");
  } catch {
    // backend hiccup at startup - don't run on an empty shell
    try {
      cfg = await invoke("get_defaults");
    } catch {
      cfg = {};
    }
  }
  await refreshModels();
  renderAll();
  renderPresets();
  applyServerState("idle");
  logEl.innerHTML = '<div class="log-empty">No logs yet — start the server and its output lands here.</div>';
  positionGlider();
});
