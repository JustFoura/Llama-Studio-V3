mod args;
mod config;
mod gguf;
mod network;
mod server;

use config::Config;
use std::collections::BTreeMap;
use tauri::{AppHandle, Manager, State, WebviewWindow};

#[tauri::command]
fn get_config() -> Config {
    config::load_config()
}

#[tauri::command]
fn get_defaults() -> Config {
    config::sanitized_defaults()
}

#[tauri::command]
fn save_config(config: Config) -> Result<(), String> {
    let mut cfg = config;
    config::sanitize(&mut cfg);
    config::save_config(&cfg).map_err(|e| format!("Could not save settings: {}", e))
}

/// Live command line for the fat pill bar.
#[tauri::command]
fn preview_args(config: Config) -> String {
    let mut cfg = config;
    config::sanitize(&mut cfg);
    args::preview(&cfg)
}

/// Parse an edited command line back onto the saved config (does not save).
#[tauri::command]
fn apply_args(config: Config, text: String) -> Config {
    let mut cfg = config;
    args::apply_to(&mut cfg, &text);
    config::sanitize(&mut cfg);
    cfg
}

fn validate_server_config(cfg: &Config) -> Result<(), String> {
    if cfg
        .api_key
        .chars()
        .any(|c| c.is_ascii_control() || c.is_whitespace())
    {
        return Err("API keys cannot contain whitespace or control characters.".into());
    }
    if cfg.context <= 0 || cfg.parallel <= 0 {
        return Err("Context length and parallel requests must both be greater than zero.".into());
    }
    if !cfg.temp.is_finite() || cfg.temp < 0.0 {
        return Err("Temperature must be a finite, non-negative number.".into());
    }
    if !cfg.top_p.is_finite() || !(0.0..=1.0).contains(&cfg.top_p) {
        return Err("Top-P must be between 0 and 1.".into());
    }
    if !cfg.min_p.is_finite() || !(0.0..=1.0).contains(&cfg.min_p) || cfg.top_k < 0 {
        return Err("Top-K must be non-negative and Min-P must be between 0 and 1.".into());
    }
    if !cfg.repeat_penalty.is_finite()
        || cfg.repeat_penalty < 0.0
        || !(-2.0..=2.0).contains(&cfg.presence_penalty)
        || !(-2.0..=2.0).contains(&cfg.frequency_penalty)
    {
        return Err("Repeat penalty must be non-negative, and presence/frequency penalties must be between -2 and 2.".into());
    }

    match cfg.engine.as_str() {
        "vllm" => {
            let model = args::vllm_model(cfg);
            if model.is_empty() {
                return Err(
                    "Pick a model first (a GGUF from your models folder, or a Hugging Face repo)."
                        .into(),
                );
            }
            if cfg.mtp_enabled && !cfg.mtp_model.trim().is_empty() {
                return Err("vLLM MTP requires a head embedded in the target model; it cannot use a separate MTP head file.".into());
            }
            if cfg.mtp_enabled && cfg.mtp_n_max <= 0 {
                return Err("MTP draft token count must be greater than zero.".into());
            }
            if cfg.dflash_enabled && cfg.dflash_model.trim().is_empty() {
                return Err(
                    "Choose a Hugging Face/safetensors DFlash draft model before starting vLLM."
                        .into(),
                );
            }
            if cfg.dflash_enabled && cfg.dflash_n_max <= 0 {
                return Err("DFlash draft token count must be greater than zero.".into());
            }
            let map_ok = cfg
                .gguf_tokenizers
                .get(&model)
                .map(|tokenizer| !tokenizer.trim().is_empty())
                .unwrap_or(false);
            let has_tok_arg = args::tokenize(&cfg.extra_args)
                .iter()
                .any(|token| token == "--tokenizer");
            if model.to_lowercase().ends_with(".gguf") && !map_ok && !has_tok_arg {
                return Err("This GGUF needs a tokenizer and none was found automatically. Set 'GGUF tokenizer' in the Model section once (for example, Qwen/Qwen3-0.6B).".into());
            }
            if cfg.dflash_enabled && cfg.dflash_model.trim().to_lowercase().ends_with(".gguf") {
                return Err("vLLM cannot use a GGUF DFlash draft. Choose the draft's Hugging Face/safetensors repo or a local HF-format folder.".into());
            }
        }
        "sglang" => {
            let model = args::sglang_model(cfg);
            if model.is_empty() {
                return Err("Choose a local Hugging Face model folder or enter a Hugging Face repo ID for SGLang.".into());
            }
            if model.to_lowercase().ends_with(".gguf") {
                return Err("SGLang's model picker currently supports Hugging Face model folders and repos, not GGUF files.".into());
            }
        }
        _ => {
            if cfg.model.trim().is_empty() && cfg.hf_model.trim().is_empty() {
                return Err(
                    "Choose a GGUF model or enter a Hugging Face repo before starting llama.cpp."
                        .into(),
                );
            }
            if cfg.batch_size <= 0 || cfg.ubatch <= 0 {
                return Err(
                    "llama.cpp batch and micro-batch sizes must be greater than zero.".into(),
                );
            }
            if cfg.mtp_enabled {
                if cfg.mtp_n_max <= 0 {
                    return Err("MTP draft token count must be greater than zero.".into());
                }
                if !cfg.mmproj.trim().is_empty() || cfg.parallel > 1 {
                    return Err("llama.cpp MTP cannot be combined with a vision encoder or parallel requests.".into());
                }
            }
            if cfg.dflash_enabled {
                if cfg.dflash_model.trim().is_empty() {
                    return Err("Choose a DFlash draft model before starting llama.cpp.".into());
                }
                if cfg.dflash_n_max <= 0 {
                    return Err("DFlash draft token count must be greater than zero.".into());
                }
                if !cfg.mmproj.trim().is_empty() {
                    return Err("llama.cpp DFlash cannot be combined with a vision encoder.".into());
                }
            }
        }
    }
    Ok(())
}

#[tauri::command]
fn start_server(
    app: AppHandle,
    state: State<'_, server::ServerState>,
    config: Config,
) -> Result<String, String> {
    let mut cfg = config;
    config::sanitize(&mut cfg);
    args::validate_addr(&cfg.api_address)?;
    validate_server_config(&cfg)?;
    let (_, port) = args::parse_addr(&cfg.api_address);
    let listen_host = if cfg.tailscale_only {
        let ip = network::tailscale_ipv4()?;
        cfg.api_address = args::format_addr(&ip, port);
        ip
    } else {
        args::parse_addr(&cfg.api_address).0
    };
    let health_host = match listen_host.as_str() {
        "0.0.0.0" => "127.0.0.1".to_string(),
        "::" => "::1".to_string(),
        _ => listen_host.clone(),
    };
    if let Err(e) = config::save_config(&cfg) {
        server::emit_log(&app, "error", format!("Could not save settings: {}", e));
    }
    let built = args::build_with_host(&cfg, &listen_host);
    if cfg.engine == "vllm" {
        server::start(
            &app,
            &state,
            server::LaunchSpec {
                executable: cfg.vllm_path.clone(),
                args: built,
                health_host,
                port,
                api_key: cfg.api_key.clone(),
                startup_timeout: std::time::Duration::from_secs(1800),
                track_slot_activity: false,
            },
        )
    } else if cfg.engine == "sglang" {
        server::start(
            &app,
            &state,
            server::LaunchSpec {
                executable: cfg.sglang_path.clone(),
                args: built,
                health_host,
                port,
                api_key: cfg.api_key.clone(),
                startup_timeout: std::time::Duration::from_secs(1800),
                track_slot_activity: false,
            },
        )
    } else {
        server::start(
            &app,
            &state,
            server::LaunchSpec {
                executable: cfg.server_path.clone(),
                args: built,
                health_host,
                port,
                api_key: cfg.api_key.clone(),
                startup_timeout: std::time::Duration::from_secs(600),
                track_slot_activity: true,
            },
        )
    }
    .map(|()| cfg.api_address)
}

#[tauri::command]
fn stop_server(state: State<'_, server::ServerState>) -> Result<(), String> {
    server::stop(&state)
}

#[tauri::command]
fn server_running(state: State<'_, server::ServerState>) -> bool {
    state.running.load(std::sync::atomic::Ordering::SeqCst)
}

#[tauri::command]
fn list_models(dir: String) -> Vec<config::ModelEntry> {
    let mut out = Vec::new();
    config::list_ggufs(&dir, &mut out);
    out.sort_by_key(|a| a.name.to_lowercase());
    out
}

#[tauri::command]
fn list_vllm_models(dir: String) -> Vec<config::ModelEntry> {
    let mut out = Vec::new();
    config::list_hf_models(&dir, &mut out);
    out.sort_by_key(|a| a.name.to_lowercase());
    out
}

#[tauri::command]
fn find_mmproj(path: String) -> Option<String> {
    config::find_mmproj_for(&path)
}

/// Auto-detect a --tokenizer value for a GGUF under vLLM (local HF cache or
/// models folder, matched against the GGUF's embedded metadata).
#[tauri::command]
fn gguf_tokenizer_info(path: String, models_dir: String) -> gguf::GgufTokenizerSuggestion {
    gguf::suggest_tokenizer(std::path::Path::new(&path), &models_dir)
}

#[tauri::command]
fn get_presets() -> BTreeMap<String, Config> {
    config::load_presets()
        .into_iter()
        .map(|(name, mut preset)| {
            config::sanitize(&mut preset);
            (name, preset)
        })
        .collect()
}

#[tauri::command]
fn save_preset(name: String, config: Config) {
    let mut presets = config::load_presets();
    let mut config = config;
    config::sanitize(&mut config);
    presets.insert(name, config);
    config::save_presets(&presets);
}

#[tauri::command]
fn delete_preset(name: String) {
    let mut presets = config::load_presets();
    presets.remove(&name);
    config::save_presets(&presets);
}

#[tauri::command]
fn browse_path(window: WebviewWindow, kind: String) -> Option<String> {
    let mut dialog = rfd::FileDialog::new();
    if kind == "gguf" {
        dialog = dialog.add_filter("GGUF model", &["gguf", "ggml", "safe"]);
    } else if kind == "exe" {
        // Windows builds carry a .exe suffix; Linux binaries usually don't.
        if cfg!(windows) {
            dialog = dialog.add_filter("llama-server", &["exe"]);
        }
    }
    let picked = if kind == "dir" {
        dialog.set_parent(&window).pick_folder()
    } else {
        dialog.set_parent(&window).pick_file()
    };
    picked.map(|p| p.to_string_lossy().to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(server::ServerState::default())
        .setup(|app| {
            config::init_paths(app.handle());
            let _ = std::fs::create_dir_all(&config::paths().default_models);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_config,
            get_defaults,
            save_config,
            preview_args,
            apply_args,
            start_server,
            stop_server,
            server_running,
            list_models,
            list_vllm_models,
            find_mmproj,
            gguf_tokenizer_info,
            get_presets,
            save_preset,
            delete_preset,
            browse_path
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|_app, event| {
            if let tauri::RunEvent::Exit = event {
                // Make sure the child llama-server doesn't outlive the GUI.
                let pid = _app
                    .state::<server::ServerState>()
                    .pid
                    .lock()
                    .unwrap()
                    .take();
                server::force_kill(pid);
            }
        });
}
