/* Dev-only: builds a screenshot harness from index.html so the real UI can be
   driven through states in a plain browser (no Tauri, no build step).
   Not part of the app — run from anywhere, writes app/src/_shot.html. */
const fs = require("fs");
const path = require("path");

const src = path.join(__dirname, "app", "src", "index.html");
const html = fs.readFileSync(src, "utf8");

const harness = `
<script>
// dev screenshot harness — drives the real UI through states
const P = new URLSearchParams(location.search).get("state") || "idle";
window.addEventListener("load", () => setTimeout(() => {
  try {
    if (P === "running" || P === "logs") {
      applyServerState("running", "http://127.0.0.1:1234  ·  pid 4211");
      setDial("pp", 198.2);
      setDial("decode", 37.4);
    }
    if (P === "logs") {
      const lines = [
        ["stdout", "llama_model_loader: loaded meta data with 22 key-value pairs"],
        ["stdout", "llm_load_tensors: offloading 36 repeating layers to GPU"],
        ["stdout", "llm_load_tensors: offloaded 36/36 layers to GPU"],
        ["stdout", "init: ggml_flash_attn_ext   = enabled"],
        ["stdout", "main: server is listening on http://0.0.0.0:1234"],
        ["info", "---\\nmodel loaded in 4.21 seconds"],
        ["stdout", "prompt eval time =   1024.32 ms /    512 tokens (  500.11 tokens per second)"],
        ["stdout", "eval time =        802.11 ms /    30 runs   (   37.41 tokens per second, 26.74 ms per token)"],
        ["stderr", "warning: no slot reuse after 3 idle cycles"],
        ["error", "Error: failed to load model: unknown quantization type 'Q9_K_XL'"],
      ];
      lines.forEach(([k, t]) => appendLog(k, t));
    }
    const page = P.split("-")[0];
    if (["logs", "chat", "settings"].includes(page)) switchPage(page);
    if (P.includes("sampling")) switchSection("sampling");
    if (P.includes("context")) switchSection("context");
    if (P.includes("vllm")) {
      cfg.engine = "vllm";
      syncEngine();
    }
    if (P.includes("open")) document.getElementById("argbar").classList.add("open");
    if (P.includes("menu")) {
      document.getElementById("settingsMenu").hidden = false;
      document.getElementById("settingsMenu").classList.add("in");
    }
    if (P.includes("dropdown")) {
      const m = document.getElementById("presetMenu");
      m.hidden = false; m.classList.add("in");
      m.innerHTML = ["qwen3-35b-a3b-q4", "qwen3-27b-dflash", "llama-3-8b-instruct", "gemma-31b-qat"]
        .map(n => '<li role="option" class="select-opt">' + n + "</li>").join("");
    }
    if (P.includes("chat")) {
      addChatBubble("user", "What does -ngl 36 mean for a 35B model?");
      addChatBubble("model", "It offloads 36 transformer layers to VRAM and leaves the rest on CPU. For a 35B at Q4 the GPU holds roughly the weights, so 36 will usually fit; if load fails, drop it and the model falls back to partial offload.");
      addChatBubble("user", "And if I raise it to 99?");
      addChatBubble("model", "That asks for every layer. If VRAM runs out the server fails to load instead of quietly slowing down — so set it to the highest number that still fits, and use 99 as a fast way to find the ceiling.");
    }
    positionGlider();
  } catch (e) { document.title = "ERR " + e.message; }
}, 300));
</script>
`;

fs.writeFileSync(path.join(__dirname, "app", "src", "_shot.html"), html.replace("</body>", harness + "</body>"));
console.log("harness written");
