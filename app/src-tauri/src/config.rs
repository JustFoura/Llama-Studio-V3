use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::OnceLock;
use tauri::Manager;

/// Every llama-server knob the UI knows about. Field names stay camelCase on
/// the wire so V1 (Electron) config.json / presets.json files load as-is.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Config {
    pub api_address: String,
    /// Optional bearer token used to protect the local inference API.
    pub api_key: String,
    /// Restrict the inference server listener to this machine's Tailscale IP.
    /// The Tailscale client must already be installed, running, and signed in.
    pub tailscale_only: bool,
    /// Which inference engine to launch: "llamacpp", "vllm", or "sglang".
    pub engine: String,
    pub server_path: String,
    /// Command or path for the vLLM server (usually just "vllm" from PATH).
    pub vllm_path: String,
    /// Python interpreter used to launch `python -m sglang.launch_server`.
    pub sglang_path: String,
    /// Legacy: the vLLM-only model field from before the engines shared one.
    /// Older presets/configs carry their model here; sanitize() migrates it
    /// onto `model` and new code only reads it as a last-resort fallback.
    pub vllm_model: String,
    /// Per-GGUF tokenizer repos/paths (GGUF path -> HF repo id or local dir),
    /// used for the --tokenizer flag vLLM needs for GGUF models.
    pub gguf_tokenizers: BTreeMap<String, String>,
    pub models_dir: String,
    pub model: String,
    pub hf_model: String,
    pub mmproj: String,
    pub context: i64,
    pub ngl: i64,
    pub batch_size: i64,
    pub ubatch: i64,
    pub threads: i64,
    pub threads_batch: i64,
    pub flash_attention: bool,
    pub cache_type_k: String,
    pub cache_type_v: String,
    /// llama.cpp KV chunk reuse via shifting (--cache-reuse): lets an edited
    /// prompt reuse KV cache from an earlier request instead of reprocessing
    /// everything. Min chunk size in tokens; 0 = off. Opt-in only: the one
    /// session that ever ran with it produced a runaway re-processing loop
    /// after a tool call, so it ships disabled.
    pub cache_reuse: i64,
    pub parallel: i64,
    pub jinja: bool,
    pub mtp_enabled: bool,
    pub mtp_n_max: i64,
    pub mtp_model: String,
    pub dflash_enabled: bool,
    pub dflash_model: String,
    pub dflash_n_max: i64,
    pub reasoning: String,
    pub reasoning_budget: i64,
    pub reasoning_effort: String,
    pub preserve_reasoning: bool,
    pub temp: f64,
    pub top_p: f64,
    pub top_k: i64,
    pub min_p: f64,
    pub repeat_penalty: f64,
    pub presence_penalty: f64,
    pub frequency_penalty: f64,
    pub mmproj_offload: bool,
    pub image_min_tokens: i64,
    pub image_max_tokens: i64,
    pub mtmd_batch_max_tokens: i64,
    pub extra_args: String,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            api_address: "http://127.0.0.1:1234".into(),
            api_key: String::new(),
            tailscale_only: false,
            engine: "llamacpp".into(),
            server_path: String::new(),
            vllm_path: "vllm".into(),
            sglang_path: if cfg!(windows) { "python" } else { "python3" }.into(),
            vllm_model: String::new(),
            gguf_tokenizers: BTreeMap::new(),
            models_dir: String::new(),
            model: String::new(),
            hf_model: String::new(),
            mmproj: String::new(),
            context: 8192,
            ngl: 99,
            batch_size: 2048,
            ubatch: 512,
            threads: 0,
            threads_batch: 0,
            flash_attention: true,
            cache_type_k: "q8_0".into(),
            cache_type_v: "q8_0".into(),
            // Off: the KV-shift path is implicated in a runaway re-processing
            // session (its first and only real-world run). Plain prefix
            // caching, which is always on, is the supported behavior.
            cache_reuse: 0,
            parallel: 1,
            jinja: true,
            mtp_enabled: false,
            mtp_n_max: 3,
            mtp_model: String::new(),
            dflash_enabled: false,
            dflash_model: String::new(),
            dflash_n_max: 15,
            reasoning: "auto".into(),
            reasoning_budget: -1,
            reasoning_effort: String::new(),
            preserve_reasoning: false,
            temp: 1.0,
            top_p: 0.95,
            top_k: 40,
            min_p: 0.05,
            repeat_penalty: 1.1,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            mmproj_offload: true,
            image_min_tokens: -1,
            image_max_tokens: -1,
            mtmd_batch_max_tokens: 1024,
            extra_args: String::new(),
        }
    }
}

pub struct AppPaths {
    pub config_dir: PathBuf,
    pub legacy_gui_dir: PathBuf,
    pub default_server: PathBuf,
    pub default_models: PathBuf,
}

static PATHS: OnceLock<AppPaths> = OnceLock::new();

fn dir_has_gguf(dir: &std::path::Path) -> bool {
    let mut visited = HashSet::new();
    dir_has_gguf_inner(dir, &mut visited)
}

fn dir_has_gguf_inner(dir: &std::path::Path, visited: &mut HashSet<PathBuf>) -> bool {
    let Ok(canonical) = fs::canonicalize(dir) else {
        return false;
    };
    if !visited.insert(canonical) {
        return false;
    }
    let Ok(rd) = fs::read_dir(dir) else {
        return false;
    };
    rd.flatten().any(|e| {
        let path = e.path();
        let Ok(metadata) = fs::metadata(&path) else {
            return false;
        };
        if metadata.is_dir() {
            dir_has_gguf_inner(&path, visited)
        } else {
            let name = e.file_name().to_string_lossy().to_lowercase();
            name.ends_with(".gguf") || name.ends_with(".ggml") || name.ends_with(".safe")
        }
    })
}

/// Find an executable name in $PATH (e.g. "llama-server" installed via pacman).
fn which(name: &str) -> Option<PathBuf> {
    let from_path = std::env::var("PATH").ok().and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|dir| dir.join(name))
            .find(|path| path.is_file())
    });
    from_path.or_else(|| {
        std::env::var(if cfg!(windows) { "USERPROFILE" } else { "HOME" })
            .ok()
            .map(|home| PathBuf::from(home).join(".local").join("bin").join(name))
            .filter(|path| path.is_file())
    })
}

pub fn init_paths(app: &tauri::AppHandle) {
    let is_dev = cfg!(debug_assertions);
    let server_exe = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    let paths = if is_dev {
        // .../app/src-tauri -> .../app -> repo root
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let app_dir = manifest.parent().unwrap().to_path_buf();
        let repo = app_dir.parent().unwrap().to_path_buf();
        let gui = repo.join("gui");

        // Reuse the V1 models folder if it has content so nothing "disappears".
        let candidates = [
            app_dir.join("dist").join("models"),
            gui.join("dist").join("models"),
            gui.join("models"),
            app_dir.join("models"),
        ];
        let models = candidates
            .iter()
            .find(|d| d.is_dir() && dir_has_gguf(d))
            .cloned()
            .unwrap_or_else(|| app_dir.join("models"));

        AppPaths {
            config_dir: app_dir,
            legacy_gui_dir: gui,
            default_server: repo
                .join("llama.cpp")
                .join("build")
                .join("bin")
                .join(server_exe),
            default_models: models,
        }
    } else {
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()));
        let portable = std::env::var("PORTABLE_EXECUTABLE_DIR")
            .ok()
            .map(PathBuf::from);
        let base = portable
            .clone()
            .or_else(|| exe_dir.clone())
            .unwrap_or_default();
        let config_dir = app
            .path()
            .app_config_dir()
            .unwrap_or_else(|_| base.join("config"));
        let default_models = portable
            .map(|dir| dir.join("models"))
            .or_else(|| app.path().app_data_dir().ok().map(|dir| dir.join("models")))
            .unwrap_or_else(|| base.join("models"));
        AppPaths {
            config_dir,
            legacy_gui_dir: base.join("gui"),
            default_server: base.join("bin").join(server_exe),
            default_models,
        }
    };
    let _ = PATHS.set(paths);
}

pub fn paths() -> &'static AppPaths {
    PATHS.get().expect("paths not initialized")
}

pub fn config_path() -> PathBuf {
    paths().config_dir.join("config.json")
}

fn restrict_private_file(path: &std::path::Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if path.exists() {
            fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
        }
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

fn ensure_private_dir(path: &std::path::Path) -> std::io::Result<()> {
    fs::create_dir_all(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

fn write_private_file(path: &std::path::Path, contents: &[u8]) -> std::io::Result<()> {
    let mut options = fs::OpenOptions::new();
    options.create(true).truncate(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    file.write_all(contents)?;
    restrict_private_file(path)
}

pub fn presets_path() -> PathBuf {
    paths().config_dir.join("presets.json")
}

pub fn load_config() -> Config {
    let _ = restrict_private_file(&config_path());
    let mut cfg = match fs::read_to_string(config_path()) {
        Ok(raw) => serde_json::from_str::<Config>(&raw).unwrap_or_default(),
        Err(_) => {
            // First V2 run: inherit the V1 Electron config so models,
            // presets and settings carry over seamlessly.
            let legacy = paths().legacy_gui_dir.join("config.json");
            fs::read_to_string(legacy)
                .ok()
                .and_then(|raw| serde_json::from_str::<Config>(&raw).ok())
                .unwrap_or_default()
        }
    };
    sanitize(&mut cfg);
    cfg
}

pub fn sanitize(cfg: &mut Config) {
    if !matches!(cfg.engine.as_str(), "vllm" | "sglang") {
        cfg.engine = "llamacpp".into();
    }
    if cfg.mtp_enabled {
        cfg.dflash_enabled = false;
    }
    // Both engines share one model field now. Carry an old config's
    // vLLM-only pick over so nothing "disappears" after an update.
    if cfg.vllm_model.trim().is_empty() {
        cfg.vllm_model.clear();
    } else if cfg.model.trim().is_empty() {
        cfg.model = cfg.vllm_model.trim().to_string();
    }
    if cfg.server_path.trim().is_empty() || !std::path::Path::new(&cfg.server_path).exists() {
        // Local build first, then an installed llama-server from $PATH
        // (e.g. Arch's `pacman -S llama.cpp` puts it in /usr/bin).
        cfg.server_path = match which("llama-server") {
            Some(p) => p.to_string_lossy().to_string(),
            None => paths().default_server.to_string_lossy().to_string(),
        };
    }
    if cfg.vllm_path.trim().is_empty() {
        cfg.vllm_path = "vllm".into();
    }
    if cfg.sglang_path.trim().is_empty() {
        cfg.sglang_path = if cfg!(windows) { "python" } else { "python3" }.into();
    }
    if cfg.models_dir.trim().is_empty() {
        cfg.models_dir = paths().default_models.to_string_lossy().to_string();
    }
    if !cfg.mmproj.trim().is_empty() && !std::path::Path::new(&cfg.mmproj).is_file() {
        cfg.mmproj = find_mmproj_for(&cfg.model).unwrap_or_default();
    }
    if cfg.api_address.trim().is_empty() {
        cfg.api_address = "http://127.0.0.1:1234".into();
    }
}

pub fn save_config(cfg: &Config) -> std::io::Result<()> {
    ensure_private_dir(&paths().config_dir)?;
    let json = serde_json::to_string_pretty(cfg).map_err(std::io::Error::other)?;
    write_private_file(&config_path(), json.as_bytes())
}

pub fn sanitized_defaults() -> Config {
    let mut cfg = Config::default();
    sanitize(&mut cfg);
    cfg
}

pub fn load_presets() -> BTreeMap<String, Config> {
    let read = |p: &PathBuf| -> Option<BTreeMap<String, Config>> {
        let _ = restrict_private_file(p);
        let raw = fs::read_to_string(p).ok()?;
        serde_json::from_str(&raw).ok()
    };
    read(&presets_path())
        .or_else(|| {
            // First V2 run: inherit V1 presets.
            read(&paths().legacy_gui_dir.join("presets.json"))
        })
        .unwrap_or_default()
}

pub fn save_presets(presets: &BTreeMap<String, Config>) {
    let _ = ensure_private_dir(&paths().config_dir);
    if let Ok(json) = serde_json::to_string_pretty(presets) {
        let _ = write_private_file(&presets_path(), json.as_bytes());
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelEntry {
    pub name: String,
    pub path: String,
    pub size_mb: u64,
    pub is_mmproj: bool,
}

fn is_model_file(name: &str) -> bool {
    let n = name.to_lowercase();
    n.ends_with(".gguf") || n.ends_with(".ggml") || n.ends_with(".safe")
}

pub fn list_ggufs(dir: &str, out: &mut Vec<ModelEntry>) {
    let mut visited = HashSet::new();
    list_ggufs_inner(std::path::Path::new(dir), out, &mut visited);
}

fn list_ggufs_inner(
    dir: &std::path::Path,
    out: &mut Vec<ModelEntry>,
    visited: &mut HashSet<PathBuf>,
) {
    let Ok(canonical) = fs::canonicalize(dir) else {
        return;
    };
    if !visited.insert(canonical) {
        return;
    }
    let Ok(rd) = fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        let Ok(metadata) = fs::metadata(&p) else {
            continue;
        };
        if metadata.is_dir() {
            list_ggufs_inner(&p, out, visited);
        } else if metadata.is_file() && is_model_file(&e.file_name().to_string_lossy()) {
            let size_mb = metadata.len() / (1024 * 1024);
            let name = e.file_name().to_string_lossy().to_string();
            let is_mmproj = name.to_lowercase().contains("mmproj");
            out.push(ModelEntry {
                name,
                path: p.to_string_lossy().to_string(),
                size_mb,
                is_mmproj,
            });
        }
    }
}

/// Scan for Hugging Face model directories and, for vLLM's optional GGUF
/// plugin, standalone `.gguf` files. The folder itself counts if it is already
/// a model directory. Canonical-path tracking prevents symlink cycles.
pub fn list_hf_models(dir: &str, out: &mut Vec<ModelEntry>) {
    let mut visited = HashSet::new();
    list_hf_models_inner(std::path::Path::new(dir), out, &mut visited);
}

fn list_hf_models_inner(
    root: &std::path::Path,
    out: &mut Vec<ModelEntry>,
    visited: &mut HashSet<PathBuf>,
) {
    let Ok(canonical) = fs::canonicalize(root) else {
        return;
    };
    if !visited.insert(canonical) {
        return;
    }
    if root.join("config.json").is_file() {
        out.push(ModelEntry {
            name: root
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| root.to_string_lossy().to_string()),
            path: root.to_string_lossy().to_string(),
            size_mb: 0,
            is_mmproj: false,
        });
        return;
    }
    let Ok(rd) = fs::read_dir(root) else { return };
    for e in rd.flatten() {
        let p = e.path();
        let Ok(metadata) = fs::metadata(&p) else {
            continue;
        };
        if !metadata.is_dir() {
            // GGUF files are servable by vLLM through vllm-gguf-plugin
            // (mmproj-named ones are vision encoders, skip them).
            let name = e.file_name().to_string_lossy().to_string();
            if metadata.is_file() && name.to_lowercase().ends_with(".gguf") {
                let is_mmproj = name.to_lowercase().contains("mmproj");
                if !is_mmproj {
                    let size_mb = metadata.len() / (1024 * 1024);
                    out.push(ModelEntry {
                        name,
                        path: p.to_string_lossy().to_string(),
                        size_mb,
                        is_mmproj,
                    });
                }
            }
            continue;
        }
        if p.join("config.json").is_file() {
            out.push(ModelEntry {
                name: p
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default(),
                path: p.to_string_lossy().to_string(),
                size_mb: 0,
                is_mmproj: false,
            });
        } else {
            list_hf_models_inner(&p, out, visited);
        }
    }
}

/// Best-effort pairing of a main model with an mmproj vision encoder that
/// lives next to it (same heuristic as V1).
pub fn find_mmproj_for(model_path: &str) -> Option<String> {
    let model = std::path::Path::new(model_path);
    let dir = model.parent()?;
    let stem = model.file_stem()?.to_string_lossy().to_lowercase();

    let mut cands: Vec<String> = Vec::new();
    let rd = fs::read_dir(dir).ok()?;
    for e in rd.flatten() {
        let n = e.file_name().to_string_lossy().to_lowercase();
        if n.contains("mmproj") && n.ends_with(".gguf") {
            cands.push(e.file_name().to_string_lossy().to_string());
        }
    }
    if cands.is_empty() {
        return None;
    }
    if cands.len() == 1 {
        return Some(dir.join(&cands[0]).to_string_lossy().to_string());
    }
    let toks: Vec<&str> = stem
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| t.len() >= 3)
        .collect();
    let mut best = String::new();
    let mut best_score = 0usize;
    for c in &cands {
        let cl = c.to_lowercase();
        let score = toks.iter().filter(|t| cl.contains(**t)).count();
        if score > best_score {
            best_score = score;
            best = c.clone();
        }
    }
    if best.is_empty() {
        None
    } else {
        Some(dir.join(best).to_string_lossy().to_string())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::field_reassign_with_default)]

    use super::*;

    #[test]
    fn defaults_bind_to_loopback_and_include_sglang_command() {
        let cfg = Config::default();
        assert_eq!(cfg.api_address, "http://127.0.0.1:1234");
        assert!(cfg.api_key.is_empty());
        assert_eq!(
            cfg.sglang_path,
            if cfg!(windows) { "python" } else { "python3" }
        );
        let old_config: Config = serde_json::from_str("{}").unwrap();
        assert_eq!(old_config.api_address, "http://127.0.0.1:1234");
        assert!(old_config.api_key.is_empty());
    }

    #[test]
    fn sanitize_keeps_sglang_and_resolves_conflicting_speculation() {
        let mut cfg = Config::default();
        cfg.engine = "sglang".into();
        cfg.server_path = std::env::current_exe()
            .unwrap()
            .to_string_lossy()
            .to_string();
        cfg.models_dir = std::env::temp_dir().to_string_lossy().to_string();
        cfg.mtp_enabled = true;
        cfg.dflash_enabled = true;
        cfg.api_address.clear();
        sanitize(&mut cfg);
        assert_eq!(cfg.engine, "sglang");
        assert_eq!(cfg.api_address, "http://127.0.0.1:1234");
        assert!(cfg.mtp_enabled);
        assert!(!cfg.dflash_enabled);
    }

    #[test]
    fn sanitize_repairs_missing_mmproj() {
        let dir =
            std::env::temp_dir().join(format!("llama-studio-preset-repair-{}", std::process::id()));
        fs::remove_dir_all(&dir).ok();
        fs::create_dir_all(&dir).unwrap();
        let model = dir.join("qwen.gguf");
        let mmproj = dir.join("mmproj-qwen.gguf");
        fs::write(&model, b"model").unwrap();
        fs::write(&mmproj, b"projector").unwrap();

        let mut cfg = Config::default();
        cfg.server_path = std::env::current_exe()
            .unwrap()
            .to_string_lossy()
            .to_string();
        cfg.models_dir = dir.to_string_lossy().to_string();
        cfg.model = model.to_string_lossy().to_string();
        cfg.mmproj = dir
            .join("old-mmproj-name.gguf")
            .to_string_lossy()
            .to_string();
        sanitize(&mut cfg);

        assert_eq!(cfg.mmproj, mmproj.to_string_lossy().to_string());
        fs::remove_dir_all(dir).ok();
    }

    #[cfg(unix)]
    #[test]
    fn private_settings_permissions_exclude_other_users() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!(
            "llama-studio-private-config-{}",
            std::process::id()
        ));
        fs::remove_dir_all(&dir).ok();
        ensure_private_dir(&dir).unwrap();
        let file = dir.join("config.json");
        write_private_file(&file, b"{}\n").unwrap();
        assert_eq!(
            fs::metadata(&dir).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(
            fs::metadata(&file).unwrap().permissions().mode() & 0o777,
            0o600
        );
        fs::remove_dir_all(dir).ok();
    }

    #[cfg(unix)]
    #[test]
    fn model_scanners_follow_symlinks_without_looping() {
        let base =
            std::env::temp_dir().join(format!("llama-studio-model-scan-{}", std::process::id()));
        let gguf_root = base.join("gguf");
        let gguf_child = gguf_root.join("nested");
        let hf_root = base.join("hf");
        let hf_model = hf_root.join("model");
        fs::remove_dir_all(&base).ok();
        fs::create_dir_all(&gguf_child).unwrap();
        fs::create_dir_all(&hf_model).unwrap();
        fs::write(gguf_child.join("model.gguf"), b"test").unwrap();
        fs::write(hf_model.join("config.json"), b"{}").unwrap();
        std::os::unix::fs::symlink(&gguf_root, gguf_child.join("back")).unwrap();
        std::os::unix::fs::symlink(&hf_root, hf_model.join("back")).unwrap();

        let mut ggufs = Vec::new();
        list_ggufs(&gguf_root.to_string_lossy(), &mut ggufs);
        assert_eq!(ggufs.len(), 1);
        assert!(dir_has_gguf(&gguf_root));

        let mut hfs = Vec::new();
        list_hf_models(&hf_root.to_string_lossy(), &mut hfs);
        assert_eq!(hfs.len(), 1);
        fs::remove_dir_all(base).ok();
    }
}
