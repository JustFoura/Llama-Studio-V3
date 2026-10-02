use crate::config::Config;
use std::net::{IpAddr, Ipv6Addr, SocketAddr};

pub fn validate_addr(addr: &str) -> Result<(), String> {
    let raw = addr.trim();
    if raw.is_empty() {
        return Err("Enter an API address such as http://127.0.0.1:1234.".into());
    }
    if raw
        .get(..8)
        .is_some_and(|scheme| scheme.eq_ignore_ascii_case("https://"))
    {
        return Err("Managed inference servers use HTTP. Enter an http:// API address.".into());
    }
    let rest = if raw
        .get(..7)
        .is_some_and(|scheme| scheme.eq_ignore_ascii_case("http://"))
    {
        &raw[7..]
    } else {
        raw
    };
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    let remainder = &rest[authority.len()..];
    if authority.is_empty() || authority.contains('@') {
        return Err("The API address must contain a host and must not include credentials.".into());
    }
    if !remainder.is_empty() && remainder != "/" {
        return Err("Enter only a host and port in the API address; URL paths and query strings are not supported.".into());
    }

    let (host, port) = if let Some(rest) = authority.strip_prefix('[') {
        let Some((host, suffix)) = rest.split_once(']') else {
            return Err("The IPv6 host must be enclosed in brackets.".into());
        };
        if host.parse::<Ipv6Addr>().is_err() {
            return Err("The API address contains an invalid IPv6 host.".into());
        }
        let port = if suffix.is_empty() {
            None
        } else {
            Some(
                suffix
                    .strip_prefix(':')
                    .ok_or("Invalid API address authority")?,
            )
        };
        (host, port)
    } else if authority.parse::<Ipv6Addr>().is_ok() {
        return Ok(());
    } else if let Some((host, port)) = authority.rsplit_once(':') {
        if host.contains(':') {
            return Err("IPv6 hosts with a port must be enclosed in brackets.".into());
        }
        (host, Some(port))
    } else {
        (authority, None)
    };

    if host.is_empty()
        || (!host.parse::<IpAddr>().is_ok()
            && !host
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_')))
    {
        return Err("The API address contains an invalid host.".into());
    }
    if let Some(port) = port {
        let parsed = port
            .parse::<u16>()
            .map_err(|_| "The API port must be a number between 1 and 65535.")?;
        if parsed == 0 {
            return Err("The API port must be between 1 and 65535.".into());
        }
    }
    Ok(())
}

/// Split an API URL into its bind host and port. Bracketed and bare IPv6
/// literals are supported; an omitted port retains the historical 8080
/// fallback for manually entered URLs.
pub fn parse_addr(addr: &str) -> (String, u16) {
    let raw = addr.trim();
    let authority = raw
        .strip_prefix("http://")
        .or_else(|| raw.strip_prefix("https://"))
        .unwrap_or(raw)
        .split(['/', '?', '#'])
        .next()
        .unwrap_or("");

    if authority.is_empty() {
        return ("127.0.0.1".into(), 8080);
    }
    if let Ok(socket) = authority.parse::<SocketAddr>() {
        return (socket.ip().to_string(), socket.port());
    }
    if let Ok(ip) = authority.parse::<Ipv6Addr>() {
        return (ip.to_string(), 8080);
    }
    if let Some(rest) = authority.strip_prefix('[') {
        if let Some((host, suffix)) = rest.split_once(']') {
            if let Ok(ip) = host.parse::<Ipv6Addr>() {
                let port = suffix
                    .strip_prefix(':')
                    .and_then(|value| value.parse::<u16>().ok())
                    .unwrap_or(8080);
                return (ip.to_string(), port);
            }
        }
    }
    match authority.rsplit_once(':') {
        Some((host, value)) if !host.contains(':') => match value.parse::<u16>() {
            Ok(port) => (host.to_string(), port),
            Err(_) => (authority.to_string(), 8080),
        },
        _ => (authority.to_string(), 8080),
    }
}

pub fn format_addr(host: &str, port: u16) -> String {
    if host.contains(':') && !host.starts_with('[') {
        format!("http://[{}]:{}", host, port)
    } else {
        format!("http://{}:{}", host, port)
    }
}

fn fmt_f64(v: f64) -> String {
    format!("{}", v)
}

/// Build llama-server arguments.
pub fn build(cfg: &Config) -> Vec<String> {
    let (host, _) = parse_addr(&cfg.api_address);
    build_with_host(
        cfg,
        if cfg.tailscale_only {
            "<tailscale-ip>"
        } else {
            &host
        },
    )
}

/// Build the server command with an explicit listener address. `start_server`
/// supplies the live Tailscale IP when Tailscale-only mode is enabled; preview
/// uses a placeholder so the command bar reflects the restricted bind mode.
pub fn build_with_host(cfg: &Config, host: &str) -> Vec<String> {
    match cfg.engine.as_str() {
        "vllm" => build_vllm(cfg, host),
        "sglang" => build_sglang(cfg, host),
        _ => build_llamacpp(cfg, host),
    }
}

/// SGLang's Python launcher accepts Hugging Face repositories and local
/// Hugging Face model directories. GGUF loading remains an explicit
/// extra-argument configuration because its support is model/quantization
/// dependent and not equivalent to llama.cpp's general GGUF support.
fn build_sglang(cfg: &Config, host: &str) -> Vec<String> {
    let mut args = vec![
        "-m".into(),
        "sglang.launch_server".into(),
        "--model-path".into(),
        sglang_model(cfg),
        "--host".into(),
        host.into(),
        "--port".into(),
        parse_addr(&cfg.api_address).1.to_string(),
        "--context-length".into(),
        cfg.context.to_string(),
        "--max-running-requests".into(),
        cfg.parallel.to_string(),
    ];
    if !cfg.api_key.is_empty() {
        args.extend(["--api-key".into(), cfg.api_key.clone()]);
    }
    if !cfg.extra_args.trim().is_empty() {
        args.extend(safe_extra_args(cfg));
    }
    args
}

pub fn sglang_model(cfg: &Config) -> String {
    if !cfg.hf_model.trim().is_empty() {
        cfg.hf_model.trim().to_string()
    } else if cfg.model.trim().to_lowercase().ends_with(".gguf") {
        String::new()
    } else {
        cfg.model.trim().to_string()
    }
}

// `vllm serve` arguments translate the shared model and speculative-decoding
// fields into vLLM's flag shapes. Sampling defaults go through the generation
// config because current vLLM versions no longer accept the legacy sampling
// CLI flags; presence/frequency penalties remain per-request only.

/// The model vLLM will serve, in llama.cpp's order: a HuggingFace repo from
/// "Or HuggingFace repo" wins over the locally picked model; the legacy
/// vLLM-only field is only consulted for presets saved by older builds.
pub fn vllm_model(cfg: &Config) -> String {
    if !cfg.hf_model.trim().is_empty() {
        cfg.hf_model.trim().to_string()
    } else if !cfg.model.trim().is_empty() {
        cfg.model.trim().to_string()
    } else {
        cfg.vllm_model.trim().to_string()
    }
}

fn build_vllm(cfg: &Config, host: &str) -> Vec<String> {
    let mut a: Vec<String> = Vec::new();
    let (_, port) = parse_addr(&cfg.api_address);

    a.push("serve".into());
    let model = vllm_model(cfg);
    let model_is_gguf = model.to_lowercase().ends_with(".gguf");
    if !model.is_empty() {
        a.push(model.clone());
    }

    a.push("--host".into());
    a.push(host.into());
    a.push("--port".into());
    a.push(port.to_string());
    if !cfg.api_key.is_empty() {
        a.push("--api-key".into());
        a.push(cfg.api_key.clone());
    }
    a.push("--max-model-len".into());
    a.push(cfg.context.to_string());
    a.push("--max-num-seqs".into());
    a.push(cfg.parallel.to_string());

    // Vision, same field as llama.cpp: for GGUF models the mmproj goes to the
    // gguf plugin through the loader's extra config (HF-format models bring
    // their own vision weights, so there is nothing to pass).
    if model_is_gguf && !cfg.mmproj.trim().is_empty() {
        a.push("--model-loader-extra-config".into());
        a.push(serde_json::json!({ "mm_proj": cfg.mmproj.trim() }).to_string());
    }

    // Speculative decoding, same switches as llama.cpp, translated into the
    // --speculative-config shape vLLM expects.
    if cfg.mtp_enabled {
        a.push("--speculative-config".into());
        a.push(
            serde_json::json!({
                "method": "mtp",
                "num_speculative_tokens": cfg.mtp_n_max,
            })
            .to_string(),
        );
    } else if cfg.dflash_enabled && !cfg.dflash_model.trim().is_empty() {
        a.push("--speculative-config".into());
        a.push(
            serde_json::json!({
                "method": "draft_model",
                "model": cfg.dflash_model.trim(),
                "num_speculative_tokens": cfg.dflash_n_max,
            })
            .to_string(),
        );
    }

    // GGUF models don't carry an HF tokenizer; vLLM needs the original
    // model's one (auto-detected + remembered per model, see gguf.rs).
    if model_is_gguf {
        if let Some(tok) = cfg
            .gguf_tokenizers
            .get(&model)
            .filter(|t| !t.trim().is_empty())
        {
            a.push("--tokenizer".into());
            a.push(tok.trim().to_string());
        }
    }
    // GUI sampling values are the server-wide defaults ("vllm" = ignore any
    // generation_config.json shipped with the model).
    a.push("--generation-config".into());
    a.push("vllm".into());
    let rp = if cfg.repeat_penalty > 0.0 {
        cfg.repeat_penalty
    } else {
        1.0
    };
    let gen = serde_json::json!({
        "temperature": cfg.temp,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
        "min_p": cfg.min_p,
        "repetition_penalty": rp,
    });
    a.push("--override-generation-config".into());
    a.push(gen.to_string());

    if !cfg.extra_args.trim().is_empty() {
        // Extra args are shared across engines, so llama.cpp-only flags can
        // ride along in a preset (its --metrics is always emitted and the
        // pill-bar parser used to park it in extra args). vLLM exits with
        // "unrecognized arguments" on those, and they have no vLLM flag to
        // translate to: /metrics and the model's chat template are the
        // default behavior there. Drop the known offenders.
        a.extend(
            safe_extra_args(cfg)
                .into_iter()
                .filter(|t| !LLAMA_ONLY_FLAGS.contains(&t.as_str())),
        );
    }

    a
}

/// llama.cpp-only, value-less flags that must never reach a vLLM command
/// line. Everything the GUI emits itself is parsed back onto its field by
/// apply_llamacpp; this is the safety net for presets saved before that.
const LLAMA_ONLY_FLAGS: &[&str] = &[
    "--metrics",
    "--jinja",
    "--no-jinja",
    "--reasoning-preserve",
    "--mmproj-offload",
    "--no-mmproj-offload",
];

/// Build the llama-server argument list. Mirrors V1's buildArgs() exactly.
pub fn build_llamacpp(cfg: &Config, host: &str) -> Vec<String> {
    let mut a: Vec<String> = Vec::new();
    let (_, port) = parse_addr(&cfg.api_address);

    if !cfg.hf_model.trim().is_empty() {
        a.push("-hf".into());
        a.push(cfg.hf_model.trim().to_string());
    } else if !cfg.model.is_empty() {
        a.push("-m".into());
        a.push(cfg.model.clone());
    }

    if !cfg.mmproj.is_empty() {
        a.push("--mmproj".into());
        a.push(cfg.mmproj.clone());
    }
    // Bind to the explicit API address. Localhost is the safe default;
    // choose a LAN address explicitly when remote access is intended.
    a.push("--host".into());
    a.push(host.into());
    a.push("--port".into());
    a.push(port.to_string());
    a.push("--metrics".into());
    if !cfg.api_key.is_empty() {
        a.push("--api-key".into());
        a.push(cfg.api_key.clone());
    }
    a.push("-ngl".into());
    a.push(cfg.ngl.to_string());
    a.push("-c".into());
    a.push(cfg.context.to_string());
    a.push("-t".into());
    a.push(cfg.threads.to_string());
    a.push("-tb".into());
    a.push(cfg.threads_batch.to_string());
    a.push("-b".into());
    a.push(cfg.batch_size.to_string());
    a.push("-ub".into());
    a.push(cfg.ubatch.to_string());
    a.push("-fa".into());
    a.push(if cfg.flash_attention { "on" } else { "off" }.into());
    a.push("--cache-type-k".into());
    a.push(cfg.cache_type_k.clone());
    a.push("--cache-type-v".into());
    a.push(cfg.cache_type_v.clone());
    // --cache-reuse stays opt-in: its one real-world run produced a runaway
    // re-processing loop after a tool call (never seen without the flag).
    // Plain prefix caching above already covers append-only conversations.
    if cfg.cache_reuse > 0 {
        a.push("--cache-reuse".into());
        a.push(cfg.cache_reuse.to_string());
    }
    a.push("-np".into());
    a.push(cfg.parallel.to_string());
    a.push("--temp".into());
    a.push(fmt_f64(cfg.temp));
    a.push("--top-p".into());
    a.push(fmt_f64(cfg.top_p));
    a.push("--top-k".into());
    a.push(cfg.top_k.to_string());
    a.push("--min-p".into());
    a.push(fmt_f64(cfg.min_p));
    let rp = if cfg.repeat_penalty > 0.0 {
        cfg.repeat_penalty
    } else {
        1.0
    };
    a.push("--repeat-penalty".into());
    a.push(fmt_f64(rp));
    a.push("--presence-penalty".into());
    a.push(fmt_f64(cfg.presence_penalty));
    a.push("--frequency-penalty".into());
    a.push(fmt_f64(cfg.frequency_penalty));

    if cfg.jinja {
        a.push("--jinja".into());
    }

    if cfg.mtp_enabled {
        a.push("--spec-type".into());
        a.push("draft-mtp".into());
        if !cfg.mtp_model.trim().is_empty() {
            a.push("--spec-draft-model".into());
            a.push(cfg.mtp_model.trim().to_string());
        }
        a.push("--spec-draft-n-max".into());
        a.push(cfg.mtp_n_max.to_string());
    } else if cfg.dflash_enabled {
        a.push("--spec-type".into());
        a.push("draft-dflash".into());
        if !cfg.dflash_model.trim().is_empty() {
            a.push("--spec-draft-model".into());
            a.push(cfg.dflash_model.trim().to_string());
        }
        a.push("--spec-draft-n-max".into());
        a.push(cfg.dflash_n_max.to_string());
    }

    if cfg.reasoning == "on" || cfg.reasoning == "off" {
        a.push("--reasoning".into());
        a.push(cfg.reasoning.clone());
    }
    if cfg.reasoning_budget >= 0 {
        a.push("--reasoning-budget".into());
        a.push(cfg.reasoning_budget.to_string());
    }
    if !cfg.reasoning_effort.trim().is_empty() {
        a.push("--chat-template-kwargs".into());
        a.push(serde_json::json!({ "reasoning_effort": cfg.reasoning_effort.trim() }).to_string());
    }
    if cfg.preserve_reasoning {
        a.push("--reasoning-preserve".into());
    }

    if !cfg.mmproj_offload {
        a.push("--no-mmproj-offload".into());
    }
    if cfg.image_min_tokens > 0 {
        a.push("--image-min-tokens".into());
        a.push(cfg.image_min_tokens.to_string());
    }
    if cfg.image_max_tokens > 0 {
        a.push("--image-max-tokens".into());
        a.push(cfg.image_max_tokens.to_string());
    }
    if cfg.mtmd_batch_max_tokens > 0 {
        a.push("--mtmd-batch-max-tokens".into());
        a.push(cfg.mtmd_batch_max_tokens.to_string());
    }

    if !cfg.extra_args.trim().is_empty() {
        a.extend(safe_extra_args(cfg));
    }

    a
}

/// In Tailscale-only mode the generated listener address is a security
/// boundary. Do not let custom CLI arguments append a second host or port and
/// accidentally widen or move that listener.
fn safe_extra_args(cfg: &Config) -> Vec<String> {
    let tokens = tokenize(&cfg.extra_args);
    if !cfg.tailscale_only {
        return tokens;
    }
    let mut out = Vec::with_capacity(tokens.len());
    let mut skip_value = false;
    for token in tokens {
        if skip_value {
            skip_value = false;
            continue;
        }
        if token == "--host" || token == "--port" {
            skip_value = true;
        } else if token.starts_with("--host=") || token.starts_with("--port=") {
            continue;
        } else {
            out.push(token);
        }
    }
    out
}

pub fn quote_arg(arg: &str) -> String {
    if arg.is_empty() || arg.contains(' ') || arg.contains('\t') || arg.contains('"') {
        format!("\"{}\"", arg.replace('\\', "\\\\").replace('"', "\\\""))
    } else {
        arg.to_string()
    }
}

pub fn redact_sensitive_args(args: &[String]) -> Vec<String> {
    let mut redacted = Vec::with_capacity(args.len());
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--api-key" {
            redacted.push(arg.clone());
            if i + 1 < args.len() {
                redacted.push("********".into());
                i += 1;
            }
        } else if arg.starts_with("--api-key=") {
            redacted.push("--api-key=********".into());
        } else {
            redacted.push(arg.clone());
        }
        i += 1;
    }
    redacted
}

fn join_args(args: &[String]) -> String {
    args.iter()
        .map(|arg| quote_arg(arg))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Human-readable full command line for the fat pill bar.
pub fn preview(cfg: &Config) -> String {
    let raw = match cfg.engine.as_str() {
        "vllm" => &cfg.vllm_path,
        "sglang" => &cfg.sglang_path,
        _ => &cfg.server_path,
    };
    let exe = std::path::Path::new(raw)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| raw.clone());
    let args: Vec<String> = redact_sensitive_args(&build(cfg))
        .iter()
        .map(|a| quote_arg(a))
        .collect();
    format!("{} {}", exe, args.join(" "))
}

/// Split a string into tokens, honouring "double" and 'single' quotes.
/// Inside double quotes, `\"` and `\\` are escapes (so the JSON values
/// emitted by quote_arg round-trip); a backslash outside quotes is literal.
pub fn tokenize(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_dq = false;
    let mut in_sq = false;
    let mut esc = false;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if esc {
            // Escaped characters are always literal and never split the token.
            cur.push(c);
            esc = false;
            continue;
        }
        match c {
            '\\' if in_dq => match chars.peek() {
                Some('"' | '\\') => esc = true,
                _ => cur.push('\\'),
            },
            '"' if !in_sq => {
                in_dq = !in_dq;
                if !in_dq && !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            '\'' if !in_dq => {
                in_sq = !in_sq;
                if !in_sq && !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            ' ' | '\t' | '\n' | '\r' if !in_dq && !in_sq => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            _ => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn as_i64(cfg: &mut Config, flag: &str, val: Option<String>) {
    if let Some(v) = val.and_then(|v| v.parse::<i64>().ok()) {
        match flag {
            "ngl" => cfg.ngl = v,
            "context" => cfg.context = v,
            "threads" => cfg.threads = v,
            "threads_batch" => cfg.threads_batch = v,
            "batch" => cfg.batch_size = v,
            "ubatch" => cfg.ubatch = v,
            "parallel" => cfg.parallel = v,
            "top_k" => cfg.top_k = v,
            "reasoning_budget" => cfg.reasoning_budget = v,
            "cache_reuse" => cfg.cache_reuse = v,
            "image_min" => cfg.image_min_tokens = v,
            "image_max" => cfg.image_max_tokens = v,
            "mtmd_batch" => cfg.mtmd_batch_max_tokens = v,
            "spec_n_max" => {
                cfg.mtp_n_max = v;
                cfg.dflash_n_max = v;
            }
            _ => {}
        }
    }
}

fn as_f64(cfg: &mut Config, flag: &str, val: Option<String>) {
    if let Some(v) = val.and_then(|v| v.parse::<f64>().ok()) {
        match flag {
            "temp" => cfg.temp = v,
            "top_p" => cfg.top_p = v,
            "min_p" => cfg.min_p = v,
            "repeat" => cfg.repeat_penalty = v,
            "presence" => cfg.presence_penalty = v,
            "frequency" => cfg.frequency_penalty = v,
            _ => {}
        }
    }
}

/// Parse an edited command line back onto the config. Known flags update
/// their field; unknown flags are collected into extraArgs so nothing the
/// user typed is ever lost.
pub fn apply_to(cfg: &mut Config, text: &str) {
    match cfg.engine.as_str() {
        "vllm" => apply_vllm(cfg, text),
        "sglang" => apply_sglang(cfg, text),
        _ => apply_llamacpp(cfg, text),
    }
}

fn apply_sglang(cfg: &mut Config, text: &str) {
    let toks = tokenize(text);
    let mut extra = Vec::new();
    let mut i = 0;
    let val = |toks: &[String], i: &mut usize| -> Option<String> {
        let next = toks.get(*i + 1)?;
        if next.starts_with('-') {
            return None;
        }
        *i += 1;
        Some(next.clone())
    };

    // The generated command is `python -m sglang.launch_server ...`.
    if i < toks.len() && !toks[i].starts_with('-') {
        i += 1;
    }
    if toks.get(i).map(String::as_str) == Some("-m") {
        i += 2;
    }
    while i < toks.len() {
        let flag = toks[i].clone();
        match flag.as_str() {
            "--model-path" | "--model" => {
                cfg.model = val(&toks, &mut i).unwrap_or_default();
                cfg.hf_model.clear();
            }
            "--api-key" => {
                if let Some(key) = toks.get(i + 1).cloned() {
                    i += 1;
                    if key != "********" {
                        cfg.api_key = key;
                    }
                }
            }
            "--host" => {
                if let Some(host) = val(&toks, &mut i) {
                    if !cfg.tailscale_only {
                        let (_, port) = parse_addr(&cfg.api_address);
                        cfg.api_address = format_addr(&host, port);
                    }
                }
            }
            "--port" => {
                if let Some(port) = val(&toks, &mut i).and_then(|v| v.parse::<u16>().ok()) {
                    let (host, _) = parse_addr(&cfg.api_address);
                    cfg.api_address = format_addr(&host, port);
                }
            }
            "--context-length" => as_i64(cfg, "context", val(&toks, &mut i)),
            "--max-running-requests" => as_i64(cfg, "parallel", val(&toks, &mut i)),
            _ => {
                extra.push(flag);
                if let Some(next) = toks.get(i + 1) {
                    if !next.starts_with('-') {
                        extra.push(next.clone());
                        i += 1;
                    }
                }
            }
        }
        i += 1;
    }
    cfg.extra_args = join_args(&extra);
}

/// Parse an edited `vllm serve ...` command line back onto the config.
fn apply_vllm(cfg: &mut Config, text: &str) {
    let toks = tokenize(text);
    let mut extra: Vec<String> = Vec::new();
    let mut i = 0usize;
    let val = |toks: &[String], i: &mut usize| -> Option<String> {
        let n = toks.get(*i + 1);
        match n {
            Some(v) if !v.starts_with('-') => {
                *i += 1;
                Some(v.clone())
            }
            _ => None,
        }
    };

    // Skip the exe name, then the "serve" subcommand.
    if i < toks.len() && !toks[i].starts_with('-') {
        i += 1;
    }
    if i < toks.len() && toks[i] == "serve" {
        i += 1;
    }
    // Positional model tag (a "--model <x>" form later overrides it).
    if i < toks.len() && !toks[i].starts_with('-') {
        cfg.model = toks[i].clone();
        cfg.hf_model.clear();
        i += 1;
    }

    while i < toks.len() {
        let t = toks[i].clone();
        match t.as_str() {
            "--model" | "-m" => {
                cfg.model = val(&toks, &mut i).unwrap_or_default();
                cfg.hf_model.clear();
            }
            "--api-key" => {
                if let Some(key) = toks.get(i + 1).cloned() {
                    i += 1;
                    if key != "********" {
                        cfg.api_key = key;
                    }
                }
            }
            "--host" => {
                if let Some(host) = val(&toks, &mut i) {
                    if !cfg.tailscale_only {
                        let (_, port) = parse_addr(&cfg.api_address);
                        cfg.api_address = format_addr(&host, port);
                    }
                }
            }
            "--port" => {
                if let Some(p) = val(&toks, &mut i).and_then(|v| v.parse::<u16>().ok()) {
                    let (host, _) = parse_addr(&cfg.api_address);
                    cfg.api_address = format_addr(&host, p);
                }
            }
            "--max-model-len" => as_i64(cfg, "context", val(&toks, &mut i)),
            "--max-num-seqs" => as_i64(cfg, "parallel", val(&toks, &mut i)),
            // mmproj for GGUF models: the picked field wins, the rest of the
            // loader config is kept verbatim in extra args.
            "--model-loader-extra-config" => {
                if let Some(raw) = val(&toks, &mut i) {
                    match serde_json::from_str::<serde_json::Value>(&raw) {
                        Ok(serde_json::Value::Object(m)) => {
                            let mut rest = serde_json::Map::new();
                            for (k, v) in m {
                                if k == "mm_proj" {
                                    if let Some(x) = v.as_str() {
                                        cfg.mmproj = x.to_string();
                                    }
                                } else {
                                    rest.insert(k, v);
                                }
                            }
                            if !rest.is_empty() {
                                extra.push(t.clone());
                                extra.push(serde_json::Value::Object(rest).to_string());
                            }
                        }
                        _ => {
                            extra.push(t.clone());
                            extra.push(raw);
                        }
                    }
                }
            }
            // Speculative decoding: the llama.cpp switches in vLLM clothing.
            // Only fully-known shapes land on the fields; anything else is
            // kept verbatim so exotic methods survive in extra args.
            "--speculative-config" => {
                if let Some(raw) = val(&toks, &mut i) {
                    let parsed = serde_json::from_str::<serde_json::Value>(&raw).ok();
                    if let Some(serde_json::Value::Object(m)) = parsed {
                        let method = m.get("method").and_then(|v| v.as_str()).unwrap_or("");
                        let n = m.get("num_speculative_tokens").and_then(|v| v.as_i64());
                        let known = matches!(method, "mtp" | "draft_model")
                            && m.len() == 2 + usize::from(method == "draft_model");
                        if known {
                            if method == "mtp" {
                                cfg.mtp_enabled = true;
                                cfg.dflash_enabled = false;
                                if let Some(n) = n {
                                    cfg.mtp_n_max = n;
                                }
                            } else {
                                cfg.dflash_enabled = true;
                                cfg.mtp_enabled = false;
                                if let Some(d) = m.get("model").and_then(|v| v.as_str()) {
                                    cfg.dflash_model = d.to_string();
                                }
                                if let Some(n) = n {
                                    cfg.dflash_n_max = n;
                                }
                            }
                        } else {
                            extra.push(t.clone());
                            extra.push(raw);
                        }
                    } else {
                        extra.push(t.clone());
                        extra.push(raw);
                    }
                }
            }
            "--generation-config" => {
                val(&toks, &mut i); // always "vllm" internally; ignored
            }
            "--override-generation-config" => {
                if let Some(raw) = val(&toks, &mut i) {
                    match serde_json::from_str::<serde_json::Value>(&raw) {
                        Ok(serde_json::Value::Object(m)) => {
                            // Known keys land on their config fields; anything
                            // else is kept verbatim so nothing is lost.
                            let mut rest = serde_json::Map::new();
                            for (k, v) in m {
                                match k.as_str() {
                                    "temperature" => {
                                        if let Some(x) = v.as_f64() {
                                            cfg.temp = x;
                                        }
                                    }
                                    "top_p" => {
                                        if let Some(x) = v.as_f64() {
                                            cfg.top_p = x;
                                        }
                                    }
                                    "top_k" => {
                                        if let Some(x) = v.as_i64() {
                                            cfg.top_k = x;
                                        }
                                    }
                                    "min_p" => {
                                        if let Some(x) = v.as_f64() {
                                            cfg.min_p = x;
                                        }
                                    }
                                    "repetition_penalty" => {
                                        if let Some(x) = v.as_f64() {
                                            cfg.repeat_penalty = x;
                                        }
                                    }
                                    _ => {
                                        rest.insert(k, v);
                                    }
                                }
                            }
                            if !rest.is_empty() {
                                extra.push(t.clone());
                                extra.push(serde_json::Value::Object(rest).to_string());
                            }
                        }
                        _ => {
                            extra.push(t.clone());
                            extra.push(raw);
                        }
                    }
                }
            }
            "--tokenizer" => {
                if let Some(t) = val(&toks, &mut i) {
                    let m = vllm_model(cfg);
                    if m.to_lowercase().ends_with(".gguf") {
                        cfg.gguf_tokenizers.insert(m, t);
                    } else {
                        extra.push("--tokenizer".into());
                        extra.push(t);
                    }
                }
            }
            // Legacy direct flags (pre-0.29 vLLM) still map onto the fields;
            // a rebuilt command line carries them in the generation config.
            "--temperature" => as_f64(cfg, "temp", val(&toks, &mut i)),
            "--top-p" => as_f64(cfg, "top_p", val(&toks, &mut i)),
            "--top-k" => as_i64(cfg, "top_k", val(&toks, &mut i)),
            "--min-p" => as_f64(cfg, "min_p", val(&toks, &mut i)),
            "--repetition-penalty" => as_f64(cfg, "repeat", val(&toks, &mut i)),
            "--presence-penalty" => as_f64(cfg, "presence", val(&toks, &mut i)),
            "--frequency-penalty" => as_f64(cfg, "frequency", val(&toks, &mut i)),
            _ => {
                // Unknown flag: keep it (and a bare-looking value) intact.
                extra.push(t.clone());
                if let Some(n) = toks.get(i + 1) {
                    if !n.starts_with('-') {
                        extra.push(n.clone());
                        i += 1;
                    }
                }
            }
        }
        i += 1;
    }
    cfg.extra_args = join_args(&extra);
}

/// Parse an edited llama-server command line back onto the config.
fn apply_llamacpp(cfg: &mut Config, text: &str) {
    let toks = tokenize(text);
    let mut extra: Vec<String> = Vec::new();
    let mut i = 0usize;
    // Skip the leading exe name ("llama-server", "llama-server.exe", ...)
    // so it never lands in extra_args.
    if i < toks.len() && !toks[i].starts_with('-') {
        i += 1;
    }
    let val = |toks: &[String], i: &mut usize| -> Option<String> {
        let n = toks.get(*i + 1);
        match n {
            Some(v) if !v.starts_with('-') => {
                *i += 1;
                Some(v.clone())
            }
            _ => None,
        }
    };

    while i < toks.len() {
        let t = toks[i].clone();
        match t.as_str() {
            "-m" | "--model" => {
                cfg.model = val(&toks, &mut i).unwrap_or_default();
                cfg.hf_model.clear();
            }
            "-hf" => {
                cfg.hf_model = val(&toks, &mut i).unwrap_or_default();
            }
            "--mmproj" => cfg.mmproj = val(&toks, &mut i).unwrap_or_default(),
            "--api-key" => {
                if let Some(key) = toks.get(i + 1).cloned() {
                    i += 1;
                    if key != "********" {
                        cfg.api_key = key;
                    }
                }
            }
            "--host" => {
                if let Some(host) = val(&toks, &mut i) {
                    if !cfg.tailscale_only {
                        let (_, port) = parse_addr(&cfg.api_address);
                        cfg.api_address = format_addr(&host, port);
                    }
                }
            }
            // Always emitted by build_llamacpp; consuming it here keeps it
            // from piling up in extra args (where it would later reach a
            // vLLM command line, which has no such flag).
            "--metrics" => {}
            "--port" => {
                if let Some(p) = val(&toks, &mut i).and_then(|v| v.parse::<u16>().ok()) {
                    let (host, _) = parse_addr(&cfg.api_address);
                    cfg.api_address = format_addr(&host, p);
                }
            }
            "-ngl" | "--n-gpu-layers" => as_i64(cfg, "ngl", val(&toks, &mut i)),
            "-c" | "--ctx-size" => as_i64(cfg, "context", val(&toks, &mut i)),
            "-t" | "--threads" => as_i64(cfg, "threads", val(&toks, &mut i)),
            "-tb" | "--threads-batch" => as_i64(cfg, "threads_batch", val(&toks, &mut i)),
            "-b" | "--batch-size" => as_i64(cfg, "batch", val(&toks, &mut i)),
            "-ub" | "--ubatch-size" => as_i64(cfg, "ubatch", val(&toks, &mut i)),
            "-np" | "--parallel" => as_i64(cfg, "parallel", val(&toks, &mut i)),
            "-fa" | "--flash-attn" => match toks.get(i + 1).map(|s| s.as_str()) {
                Some("on") => {
                    cfg.flash_attention = true;
                    i += 1;
                }
                Some("off") => {
                    cfg.flash_attention = false;
                    i += 1;
                }
                _ => cfg.flash_attention = true,
            },
            "--cache-type-k" => {
                cfg.cache_type_k = val(&toks, &mut i).unwrap_or(cfg.cache_type_k.clone())
            }
            "--cache-type-v" => {
                cfg.cache_type_v = val(&toks, &mut i).unwrap_or(cfg.cache_type_v.clone())
            }
            "--cache-reuse" => as_i64(cfg, "cache_reuse", val(&toks, &mut i)),
            "--temp" | "--temperature" => as_f64(cfg, "temp", val(&toks, &mut i)),
            "--top-p" => as_f64(cfg, "top_p", val(&toks, &mut i)),
            "--top-k" => as_i64(cfg, "top_k", val(&toks, &mut i)),
            "--min-p" => as_f64(cfg, "min_p", val(&toks, &mut i)),
            "--repeat-penalty" => as_f64(cfg, "repeat", val(&toks, &mut i)),
            "--presence-penalty" => as_f64(cfg, "presence", val(&toks, &mut i)),
            "--frequency-penalty" => as_f64(cfg, "frequency", val(&toks, &mut i)),
            "--jinja" => cfg.jinja = true,
            "--no-jinja" => cfg.jinja = false,
            "--reasoning" => cfg.reasoning = val(&toks, &mut i).unwrap_or_else(|| "auto".into()),
            "--reasoning-budget" => as_i64(cfg, "reasoning_budget", val(&toks, &mut i)),
            "--reasoning-preserve" => cfg.preserve_reasoning = true,
            "--chat-template-kwargs" => {
                if let Some(raw) = val(&toks, &mut i) {
                    match serde_json::from_str::<serde_json::Value>(&raw) {
                        Ok(v) => {
                            let effort: Option<String> = v
                                .get("reasoning_effort")
                                .and_then(|e| e.as_str())
                                .map(|s| s.to_string());
                            let rest = match v {
                                serde_json::Value::Object(mut m) => {
                                    m.remove("reasoning_effort");
                                    if m.is_empty() {
                                        None
                                    } else {
                                        Some(serde_json::Value::Object(m).to_string())
                                    }
                                }
                                _ => Some(raw.clone()),
                            };
                            if let Some(e) = effort {
                                cfg.reasoning_effort = e;
                            }
                            if let Some(r) = rest {
                                extra.push("--chat-template-kwargs".into());
                                extra.push(r);
                            }
                        }
                        Err(_) => {
                            extra.push(t.clone());
                            extra.push(raw);
                        }
                    }
                }
            }
            "--spec-type" => match val(&toks, &mut i).as_deref() {
                Some("draft-mtp") => {
                    cfg.mtp_enabled = true;
                    cfg.dflash_enabled = false;
                }
                Some("draft-dflash") => {
                    cfg.dflash_enabled = true;
                    cfg.mtp_enabled = false;
                }
                Some("none") | None => {
                    cfg.mtp_enabled = false;
                    cfg.dflash_enabled = false;
                }
                Some(other) => {
                    extra.push(t.clone());
                    extra.push(other.to_string());
                }
            },
            "--spec-draft-model" => {
                let m = val(&toks, &mut i).unwrap_or_default();
                cfg.mtp_model = m.clone();
                cfg.dflash_model = m;
            }
            "--spec-draft-n-max" => as_i64(cfg, "spec_n_max", val(&toks, &mut i)),
            "--mmproj-offload" => cfg.mmproj_offload = true,
            "--no-mmproj-offload" => cfg.mmproj_offload = false,
            "--image-min-tokens" => as_i64(cfg, "image_min", val(&toks, &mut i)),
            "--image-max-tokens" => as_i64(cfg, "image_max", val(&toks, &mut i)),
            "--mtmd-batch-max-tokens" => as_i64(cfg, "mtmd_batch", val(&toks, &mut i)),
            _ => {
                // Unknown flag: keep it (and a bare-looking value) intact.
                extra.push(t.clone());
                if let Some(n) = toks.get(i + 1) {
                    if !n.starts_with('-') {
                        extra.push(n.clone());
                        i += 1;
                    }
                }
            }
        }
        i += 1;
    }
    cfg.extra_args = join_args(&extra);
}

#[cfg(test)]
mod tests {
    #![allow(clippy::field_reassign_with_default)]

    use super::*;
    use crate::config::Config;

    #[test]
    fn parse_addr_handles_forms() {
        assert_eq!(
            parse_addr("http://192.0.2.1:1234"),
            ("192.0.2.1".into(), 1234)
        );
        assert_eq!(parse_addr("127.0.0.1:8080"), ("127.0.0.1".into(), 8080));
        assert_eq!(parse_addr("http://host/"), ("host".into(), 8080));
        assert_eq!(parse_addr("http://[::1]:9443/v1"), ("::1".into(), 9443));
        assert_eq!(parse_addr("http://::1"), ("::1".into(), 8080));
        assert_eq!(parse_addr(""), ("127.0.0.1".into(), 8080));
        assert_eq!(format_addr("::1", 9443), "http://[::1]:9443");
        assert!(validate_addr("http://127.0.0.1:1234").is_ok());
        assert!(validate_addr("http://[::1]:1234/").is_ok());
        assert!(validate_addr("https://localhost:1234").is_err());
        assert!(validate_addr("http://localhost:1234/v1").is_err());
        assert!(validate_addr("http://localhost:not-a-port").is_err());
    }

    #[test]
    fn llamacpp_args_mirror_v1() {
        let mut cfg = Config::default();
        cfg.engine = "llamacpp".into();
        cfg.api_address = "http://127.0.0.1:1234".into();
        cfg.model = "/m/x.gguf".into();
        let a = build(&cfg);
        assert_eq!(&a[..3], &["-m", "/m/x.gguf", "--host"]);
        assert!(a.contains(&"--jinja".to_string()));
        assert!(a.windows(2).any(|w| w[0] == "--port" && w[1] == "1234"));
    }

    #[test]
    fn bind_host_follows_api_address_and_defaults_to_loopback() {
        let mut cfg = Config::default();
        cfg.model = "/m/model.gguf".into();
        assert_eq!(cfg.api_address, "http://127.0.0.1:1234");
        let args = build(&cfg);
        assert!(args
            .windows(2)
            .any(|w| w[0] == "--host" && w[1] == "127.0.0.1"));

        cfg.api_address = "http://192.0.2.15:4321".into();
        let args = build(&cfg);
        assert!(args
            .windows(2)
            .any(|w| w[0] == "--host" && w[1] == "192.0.2.15"));
        assert!(args.windows(2).any(|w| w[0] == "--port" && w[1] == "4321"));
    }

    #[test]
    fn vllm_args_shape() {
        let mut cfg = Config::default();
        cfg.engine = "vllm".into();
        cfg.api_address = "http://127.0.0.1:1234".into();
        cfg.model = "user/repo".into();
        cfg.context = 16384;
        cfg.parallel = 2;
        let a = build(&cfg);
        assert_eq!(&a[..3], &["serve", "user/repo", "--host"]);
        assert!(a
            .windows(2)
            .any(|w| w[0] == "--max-model-len" && w[1] == "16384"));
        assert!(a
            .windows(2)
            .any(|w| w[0] == "--max-num-seqs" && w[1] == "2"));
        assert!(a.windows(2).any(|w| w[0] == "--port" && w[1] == "1234"));
        // vLLM 0.29 removed the sampling CLI flags; defaults ride on the
        // generation config instead.
        assert!(!a.contains(&"--temperature".to_string()));
        assert!(!a.contains(&"--top-p".to_string()));
        assert!(!a.contains(&"--repetition-penalty".to_string()));
        assert!(a
            .windows(2)
            .any(|w| w[0] == "--generation-config" && w[1] == "vllm"));
        let gi = a
            .iter()
            .position(|x| x == "--override-generation-config")
            .unwrap();
        assert!(a[gi + 1].contains("\"temperature\":1.0"));
        assert!(a[gi + 1].contains("\"repetition_penalty\":1.1"));
        // llama-only flags must not leak into vLLM command lines
        assert!(!a.contains(&"--jinja".to_string()));
        assert!(!a.contains(&"-ngl".to_string()));

        // "Or HuggingFace repo" wins over the local model, like -hf does
        cfg.hf_model = "org/repo".into();
        assert_eq!(&build(&cfg)[..2], &["serve", "org/repo"]);
        cfg.hf_model.clear();

        // legacy presets that only filled the old vLLM field still resolve
        let mut old = Config::default();
        old.engine = "vllm".into();
        old.vllm_model = "legacy/repo".into();
        assert_eq!(&build(&old)[..2], &["serve", "legacy/repo"]);
    }

    #[test]
    fn sglang_args_and_command_roundtrip() {
        let mut cfg = Config::default();
        cfg.engine = "sglang".into();
        cfg.api_address = "http://127.0.0.1:3456".into();
        cfg.model = "/models/Qwen Instruct".into();
        cfg.context = 32768;
        cfg.parallel = 4;
        cfg.extra_args = "--trust-remote-code".into();

        let args = build(&cfg);
        assert_eq!(
            &args[..10],
            &[
                "-m",
                "sglang.launch_server",
                "--model-path",
                "/models/Qwen Instruct",
                "--host",
                "127.0.0.1",
                "--port",
                "3456",
                "--context-length",
                "32768"
            ]
        );
        assert!(args
            .windows(2)
            .any(|w| w == ["--max-running-requests", "4"]));
        assert!(args.contains(&"--trust-remote-code".to_string()));

        cfg.hf_model = "org/model".into();
        assert_eq!(sglang_model(&cfg), "org/model");
        cfg.hf_model.clear();
        cfg.model = "/models/llama.gguf".into();
        assert!(sglang_model(&cfg).is_empty());
        cfg.model = "/models/Qwen Instruct".into();
        let preview_text = preview(&cfg);
        assert!(preview_text.starts_with("python3 -m sglang.launch_server"));

        apply_to(&mut cfg, "python3 -m sglang.launch_server --model-path /tmp/model --host 192.0.2.10 --port 7777 --context-length 8192 --max-running-requests 2 --trust-remote-code --tokenizer-path \"/models folder/tokenizer.json\"");
        assert_eq!(cfg.model, "/tmp/model");
        assert!(cfg.hf_model.is_empty());
        assert_eq!(cfg.api_address, "http://192.0.2.10:7777");
        assert_eq!(cfg.context, 8192);
        assert_eq!(cfg.parallel, 2);
        assert_eq!(
            cfg.extra_args,
            "--trust-remote-code --tokenizer-path \"/models folder/tokenizer.json\""
        );
        assert!(build(&cfg)
            .windows(2)
            .any(|w| w[0] == "--tokenizer-path" && w[1] == "/models folder/tokenizer.json"));
    }

    #[test]
    fn api_keys_reach_engines_but_are_redacted_and_roundtrip() {
        for engine in ["llamacpp", "vllm", "sglang"] {
            let mut cfg = Config::default();
            cfg.engine = engine.into();
            cfg.model = if engine == "llamacpp" {
                "/models/model.gguf".into()
            } else {
                "org/model".into()
            };
            cfg.api_key = "test-only-api-key".into();
            let command = build(&cfg);
            assert!(command
                .windows(2)
                .any(|w| w[0] == "--api-key" && w[1] == cfg.api_key));

            let preview_text = preview(&cfg);
            assert!(preview_text.contains("--api-key ********"));
            assert!(!preview_text.contains(&cfg.api_key));
            apply_to(&mut cfg, &preview_text);
            assert_eq!(cfg.api_key, "test-only-api-key");
        }
    }

    #[test]
    fn vllm_auto_equivalents() {
        // A GGUF with an mmproj next to it: same fields as llama.cpp, vLLM
        // flag shapes out the other end.
        let mut cfg = Config::default();
        cfg.engine = "vllm".into();
        cfg.model = "/m/Qwen3.8-27B-IQ4_XS.gguf".into();
        cfg.mmproj = "/m/mmproj-bf16.gguf".into();
        cfg.dflash_enabled = true;
        cfg.dflash_model = "/m/Qwen3.8-27B-DFlash2-Q8_0.gguf".into();
        cfg.dflash_n_max = 3;
        let a = build(&cfg);
        let mi = a
            .iter()
            .position(|x| x == "--model-loader-extra-config")
            .unwrap();
        assert!(a[mi + 1].contains("\"mm_proj\":\"/m/mmproj-bf16.gguf\""));
        let si = a.iter().position(|x| x == "--speculative-config").unwrap();
        assert!(a[si + 1].contains("\"method\":\"draft_model\""));
        assert!(a[si + 1].contains("\"model\":\"/m/Qwen3.8-27B-DFlash2-Q8_0.gguf\""));
        assert!(a[si + 1].contains("\"num_speculative_tokens\":3"));

        // MTP translates too, and the two switches are mutually exclusive
        let mut mtp = cfg.clone();
        mtp.dflash_enabled = false;
        mtp.mtp_enabled = true;
        mtp.mtp_n_max = 2;
        let a = build(&mtp);
        let si = a.iter().position(|x| x == "--speculative-config").unwrap();
        assert!(a[si + 1].contains("\"method\":\"mtp\""));
        assert!(a[si + 1].contains("\"num_speculative_tokens\":2"));

        // with the switches off / no mmproj, none of it is emitted
        let mut plain = Config::default();
        plain.engine = "vllm".into();
        plain.model = "/m/x.gguf".into();
        let a = build(&plain);
        for flag in [
            "--model-loader-extra-config",
            "--speculative-config",
            "--kv-cache-dtype",
            "--cpu-offload-gb",
            "--max-num-batched-tokens",
            "--attention-backend",
            "--enable-prefix-caching",
        ] {
            assert!(!a.contains(&flag.to_string()), "{} leaked", flag);
        }

        // and a pill-bar edit of the new flags lands back on the fields
        let p = preview(&cfg);
        apply_to(&mut plain, &p);
        assert_eq!(plain.mmproj, "/m/mmproj-bf16.gguf");
        assert!(plain.dflash_enabled);
        assert!(!plain.mtp_enabled);
        assert_eq!(plain.dflash_model, "/m/Qwen3.8-27B-DFlash2-Q8_0.gguf");
        assert_eq!(plain.dflash_n_max, 3);

        // manual vLLM flags the GUI has no field for are kept, not lost
        apply_to(
            &mut plain,
            "vllm serve /m/x.gguf --kv-cache-dtype fp8 --quantization awq",
        );
        assert_eq!(plain.extra_args, "--kv-cache-dtype fp8 --quantization awq");
    }

    #[test]
    fn vllm_drops_llama_only_extra_flags() {
        // A llama.cpp preset's extra args leak --metrics (it rides along
        // because build_llamacpp always emits it); vLLM must not see it.
        let mut cfg = Config::default();
        cfg.engine = "vllm".into();
        cfg.model = "/m/x.gguf".into();
        cfg.extra_args = "--metrics --disable-log-stats".into();
        let a = build(&cfg);
        assert!(!a.contains(&"--metrics".to_string()));
        assert!(a.contains(&"--disable-log-stats".to_string()));

        // and a pill-bar edit on the llama.cpp side no longer accumulates it
        let mut llama = Config::default();
        llama.model = "/m/x.gguf".into();
        apply_to(&mut llama, "llama-server -m /m/x.gguf --metrics");
        assert_eq!(llama.extra_args, "");
        // llama.cpp itself still gets its --metrics
        assert!(build(&llama).contains(&"--metrics".to_string()));
    }

    #[test]
    fn vllm_roundtrip() {
        let mut cfg = Config::default();
        cfg.engine = "vllm".into();
        let text = preview(&cfg);
        assert!(text.starts_with("vllm serve"));
        apply_to(&mut cfg, "vllm serve qwen/authors/Qwen3 --host 0.0.0.0 --port 9999 --max-model-len 32768 --override-generation-config \"{\\\"temperature\\\": 0.7}\" --quantization awq --extra-flag");
        assert_eq!(cfg.model, "qwen/authors/Qwen3");
        assert!(cfg.hf_model.is_empty());
        assert_eq!(parse_addr(&cfg.api_address).1, 9999);
        assert_eq!(cfg.context, 32768);
        assert_eq!(cfg.temp, 0.7);
        assert_eq!(cfg.extra_args, "--quantization awq --extra-flag");
        // and rebuilding keeps the edited values
        let a = build(&cfg);
        assert!(a.contains(&"qwen/authors/Qwen3".to_string()));
        // editing the pill bar's own preview keeps the sampling values too
        let p = preview(&cfg);
        apply_to(&mut cfg, &p);
        assert_eq!(cfg.temp, 0.7);
        assert_eq!(cfg.repeat_penalty, 1.1);
    }

    #[test]
    fn vllm_gguf_tokenizer() {
        let mut cfg = Config::default();
        cfg.engine = "vllm".into();
        cfg.model = "/m/Qwen3-0.6B-Q8_0.gguf".into();
        cfg.gguf_tokenizers
            .insert("/m/Qwen3-0.6B-Q8_0.gguf".into(), "Qwen/Qwen3-0.6B".into());
        let a = build(&cfg);
        assert!(a
            .windows(2)
            .any(|w| w[0] == "--tokenizer" && w[1] == "Qwen/Qwen3-0.6B"));
        // editing the pill bar's --tokenizer lands back in the map
        let p = preview(&cfg);
        apply_to(&mut cfg, &p);
        assert_eq!(
            cfg.gguf_tokenizers
                .get("/m/Qwen3-0.6B-Q8_0.gguf")
                .map(String::as_str),
            Some("Qwen/Qwen3-0.6B")
        );
        // without a map entry the flag is omitted entirely
        let mut cfg2 = Config::default();
        cfg2.engine = "vllm".into();
        cfg2.model = "/m/x.gguf".into();
        assert!(!build(&cfg2).contains(&"--tokenizer".to_string()));
    }

    #[test]
    fn llamacpp_cache_reuse() {
        let mut cfg = Config::default();
        cfg.model = "/m/x.gguf".into();
        // Off by default: the KV-shift path is implicated in a runaway
        // re-processing session (its first and only real-world run).
        assert!(!build(&cfg).contains(&"--cache-reuse".to_string()));

        // opting in emits the flag
        cfg.cache_reuse = 256;
        assert!(build(&cfg)
            .windows(2)
            .any(|w| w[0] == "--cache-reuse" && w[1] == "256"));

        // and the pill bar round-trips it
        let p = preview(&cfg);
        apply_to(&mut cfg, &p);
        assert_eq!(cfg.cache_reuse, 256);
        apply_to(&mut cfg, "llama-server -m /m/x.gguf --cache-reuse 512");
        assert_eq!(cfg.cache_reuse, 512);
        assert!(build(&cfg)
            .windows(2)
            .any(|w| w[0] == "--cache-reuse" && w[1] == "512"));

        // 0 = off, flag omitted entirely
        cfg.cache_reuse = 0;
        assert!(!build(&cfg).contains(&"--cache-reuse".to_string()));
    }

    #[test]
    fn tokenize_handles_escapes() {
        assert_eq!(
            tokenize(r#"--x "{\"a\": 1}" y"#),
            vec!["--x", "{\"a\": 1}", "y"]
        );
        // backslashes outside quotes are literal (paths)
        assert_eq!(
            tokenize(r"C:\path to\model.gguf"),
            vec![r"C:\path", r"to\model.gguf"]
        );

        // Quoted Windows paths preserve slashes, spaces and quotes through
        // the command bar's display/edit round trip.
        let path = r#"C:\Models Folder\Qwen "Instruct"\model.gguf"#;
        assert_eq!(tokenize(&quote_arg(path)), vec![path]);
    }

    #[test]
    fn llama_roundtrip_keeps_unknown_flags() {
        let mut cfg = Config::default();
        cfg.model = "/m/x.gguf".into();
        apply_to(
            &mut cfg,
            "llama-server -m /m/y.gguf -c 4096 --something-new 5",
        );
        assert_eq!(cfg.model, "/m/y.gguf");
        assert_eq!(cfg.context, 4096);
        assert_eq!(cfg.extra_args, "--something-new 5");
    }
}
