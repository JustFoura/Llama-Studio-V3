//! Minimal GGUF header reader + tokenizer suggestion for serving GGUFs
//! under vLLM. vLLM (via vllm-gguf-plugin) needs the *original* model's
//! HuggingFace tokenizer, which the GGUF itself doesn't carry — so we look
//! for a matching repo in the local HF cache, a local HF-format dir, and
//! let the frontend fall back to a huggingface.co search using the names
//! embedded in the GGUF metadata.

use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Scalar GGUF value types (GGUF v2/v3 spec).
const T_STRING: u32 = 8;
const T_ARRAY: u32 = 9;

const GENERAL_KEYS: &[&str] = &[
    "general.name",
    "general.basename",
    "general.size_label",
    "general.finetune",
    "general.organization",
    "general.architecture",
];

pub struct GgufMeta(pub HashMap<String, String>);

impl GgufMeta {
    /// Model-name keys for matching a tokenizer repo, most specific first:
    /// "Qwen3-0.6B" (basename + size_label), then general.name, then the
    /// bare basename. For finetunes the basename+size form points at the
    /// base model, which has the right tokenizer.
    pub fn name_variants(&self) -> Vec<String> {
        let mut out = Vec::new();
        if let (Some(b), Some(s)) = (
            self.0.get("general.basename"),
            self.0.get("general.size_label"),
        ) {
            out.push(format!("{}-{}", b, s));
        }
        if let Some(n) = self.0.get("general.name") {
            out.push(n.clone());
        }
        if let Some(b) = self.0.get("general.basename") {
            out.push(b.clone());
        }
        out
    }

    pub fn display_name(&self) -> Option<String> {
        self.0.get("general.name").cloned()
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GgufTokenizerSuggestion {
    /// general.name from the GGUF (may be None for exotic files).
    pub name: Option<String>,
    /// Primary matching key (basename + size_label), for HF search.
    pub base: Option<String>,
    /// A concrete --tokenizer value found locally: an HF repo id present in
    /// the local HF cache, or a local HF-format model dir path.
    pub tokenizer: Option<String>,
}

fn read_u32(r: &mut (impl Read + Seek)) -> Option<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).ok()?;
    Some(u32::from_le_bytes(b))
}

fn read_u64(r: &mut (impl Read + Seek)) -> Option<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).ok()?;
    Some(u64::from_le_bytes(b))
}

fn skip(r: &mut (impl Read + Seek), n: u64, file_len: u64) -> Option<()> {
    if n > (1 << 40) {
        return None; // implausible; don't seek anywhere weird
    }
    let end = r.stream_position().ok()?.checked_add(n)?;
    if end > file_len {
        return None;
    }
    r.seek(SeekFrom::Start(end)).ok()?;
    Some(())
}

fn read_string(r: &mut (impl Read + Seek), file_len: u64) -> Option<String> {
    let len = read_u64(r)?;
    if len > (1 << 22) {
        return None; // metadata strings are tiny; this is a corrupt header
    }
    if len > file_len.checked_sub(r.stream_position().ok()?)? {
        return None;
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf).ok()?;
    Some(String::from_utf8_lossy(&buf).into_owned())
}

fn skip_array(r: &mut (impl Read + Seek), file_len: u64, depth: u8) -> Option<()> {
    const MAX_ARRAY_DEPTH: u8 = 8;
    const MAX_ARRAY_ITEMS: u64 = 10_000_000;
    if depth >= MAX_ARRAY_DEPTH {
        return None;
    }
    let elem = read_u32(r)?;
    let count = read_u64(r)?;
    if count > MAX_ARRAY_ITEMS {
        return None;
    }
    match elem {
        T_STRING => {
            if count > file_len.checked_sub(r.stream_position().ok()?)? / 8 {
                return None;
            }
            for _ in 0..count {
                let len = read_u64(r)?;
                skip(r, len, file_len)?;
            }
            Some(())
        }
        T_ARRAY => {
            if count > file_len.checked_sub(r.stream_position().ok()?)? / 12 {
                return None;
            }
            for _ in 0..count {
                skip_array(r, file_len, depth + 1)?;
            }
            Some(())
        }
        0 | 1 | 7 => skip(r, count, file_len),
        2 | 3 => skip(r, count.checked_mul(2)?, file_len),
        4..=6 => skip(r, count.checked_mul(4)?, file_len),
        10..=12 => skip(r, count.checked_mul(8)?, file_len),
        _ => None,
    }
}

/// Read the `general.*` string metadata from a GGUF's header. Returns None
/// for non-GGUF or badly malformed files.
pub fn read_meta(path: &Path) -> Option<GgufMeta> {
    let file = std::fs::File::open(path).ok()?;
    let file_len = file.metadata().ok()?.len();
    let mut r = std::io::BufReader::new(file);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).ok()?;
    if &magic != b"GGUF" {
        return None;
    }
    let _version = read_u32(&mut r)?;
    let _tensor_count = read_u64(&mut r)?;
    let kv_count = read_u64(&mut r)?;
    if kv_count > 10_000 {
        return None;
    }

    let mut map: HashMap<String, String> = HashMap::new();
    for _ in 0..kv_count {
        let key = read_string(&mut r, file_len)?;
        let vtype = read_u32(&mut r)?;
        let mut sval = None;
        match vtype {
            T_STRING => sval = Some(read_string(&mut r, file_len)?),
            T_ARRAY => skip_array(&mut r, file_len, 0)?,
            0 | 1 | 7 => skip(&mut r, 1, file_len)?,
            2 | 3 => skip(&mut r, 2, file_len)?,
            4..=6 => skip(&mut r, 4, file_len)?,
            10..=12 => skip(&mut r, 8, file_len)?,
            _ => return None,
        }
        if let Some(s) = sval {
            if key.starts_with("general.") {
                map.insert(key, s);
            }
        }
        // general.* keys are written before the huge tokenizer arrays, so
        // once we have them all we can stop reading.
        if GENERAL_KEYS.iter().all(|k| map.contains_key(*k)) {
            break;
        }
    }
    Some(GgufMeta(map))
}

fn norm(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect()
}

/// HuggingFace-format model dirs under `dir` (dirs holding a config.json).
fn hf_model_dirs(dir: &str, out: &mut Vec<String>) {
    let mut visited = HashSet::new();
    hf_model_dirs_inner(Path::new(dir), out, &mut visited);
}

fn hf_model_dirs_inner(root: &Path, out: &mut Vec<String>, visited: &mut HashSet<PathBuf>) {
    let Ok(canonical) = std::fs::canonicalize(root) else {
        return;
    };
    if !visited.insert(canonical) {
        return;
    }
    if root.join("config.json").is_file() {
        out.push(root.to_string_lossy().to_string());
        return;
    }
    let Ok(rd) = std::fs::read_dir(root) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if std::fs::metadata(&p).map(|m| m.is_dir()).unwrap_or(false) {
            if p.join("config.json").is_file() {
                out.push(p.to_string_lossy().to_string());
            } else {
                hf_model_dirs_inner(&p, out, visited);
            }
        }
    }
}

fn hub_dir() -> Option<PathBuf> {
    if let Ok(h) = std::env::var("HF_HOME") {
        return Some(PathBuf::from(h).join("hub"));
    }
    let home_var = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
    let home = std::env::var(home_var).ok()?;
    Some(
        PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub"),
    )
}

/// Repo ids in the local HF cache that are actually downloaded (a snapshot
/// holding config.json). "models--Qwen--Qwen3-0.6B" -> "Qwen/Qwen3-0.6B".
fn hf_cache_repos() -> Vec<String> {
    let Some(hub) = hub_dir() else {
        return Vec::new();
    };
    let Ok(rd) = std::fs::read_dir(&hub) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let Some(rest) = name.strip_prefix("models--") else {
            continue;
        };
        let Some((org, repo)) = rest.split_once("--") else {
            continue;
        };
        let downloaded = std::fs::read_dir(e.path().join("snapshots"))
            .map(|snaps| {
                snaps
                    .flatten()
                    .any(|s| s.path().join("config.json").is_file())
            })
            .unwrap_or(false);
        if downloaded {
            out.push(format!("{}/{}", org, repo));
        }
    }
    out
}

/// Repo names that are quant/repack uploads rather than the base model —
/// they carry no HF tokenizer files, so --tokenizer would fail on them.
const QUANT_REPO_MARKERS: &[&str] = &[
    "-gguf", "-mlx", "-awq", "-gptq", "-fp8", "-fp4", "-nvfp4", "-int4", "-int8", "-exl2", "-i1",
    "imatrix", "-dqart",
];

fn is_quant_repo(id: &str) -> bool {
    let l = id.to_lowercase();
    QUANT_REPO_MARKERS.iter().any(|m| l.contains(m))
}

/// huggingface.co model search, run via curl (the webview can't: HF's API
/// only allows its own origin, so fetch() from the UI is blocked by CORS).
fn hf_search(query: &str) -> Vec<String> {
    let encoded: String = query
        .bytes()
        .map(|b| match b {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                (b as char).to_string()
            }
            _ => format!("%{:02X}", b),
        })
        .collect();
    let url = format!(
        "https://huggingface.co/api/models?limit=20&sort=downloads&direction=-1&search={}",
        encoded
    );
    let Ok(out) = std::process::Command::new("curl")
        .args(["-sS", "--max-time", "10", "-A", "llama-studio", "-L", &url])
        .output()
    else {
        return Vec::new();
    };
    if !out.status.success() {
        return Vec::new();
    }
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&String::from_utf8_lossy(&out.stdout))
    else {
        return Vec::new();
    };
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|m| m.get("id").and_then(|i| i.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

/// Pick the repo whose name segment best matches any of the normalized
/// variants: exact first, then prefix, then substring. Order of `ids`
/// (HF sorts by downloads) breaks ties.
fn pick_repo(ids: &[String], variants: &[String]) -> Option<String> {
    let clean: Vec<&String> = ids.iter().filter(|id| !is_quant_repo(id)).collect();
    let seg = |id: &str| norm(id.rsplit('/').next().unwrap_or(id));
    for pass in 0..3 {
        for v in variants {
            if v.is_empty() {
                continue;
            }
            if let Some(id) = clean.iter().find(|&&id| match pass {
                0 => seg(id) == *v,
                1 => seg(id).starts_with(v.as_str()),
                _ => seg(id).contains(v.as_str()),
            }) {
                return Some((*id).clone());
            }
        }
    }
    None
}

/// Filename-derived search keys, for GGUFs with junk metadata: strip
/// trailing quant tokens ("...-Q4_K_M", "...-BF16").
fn filename_variants(path: &Path) -> Vec<String> {
    let Some(stem) = path.file_stem().map(|s| s.to_string_lossy().to_string()) else {
        return Vec::new();
    };
    let is_quant = |t: &str| {
        let u = t.to_uppercase();
        matches!(
            u.as_str(),
            "F16" | "BF16" | "FP16" | "FP8" | "FP32" | "GGUF"
        ) || ((u.starts_with('Q') || u.starts_with("IQ") || u.starts_with("UD"))
            && u.chars().any(|c| c.is_ascii_digit()))
    };
    let mut parts: Vec<&str> = stem.split('-').collect();
    while parts.len() > 1 && is_quant(parts.last().unwrap()) {
        parts.pop();
    }
    let truncated = parts.join("-");
    let mut out = vec![stem.clone()];
    if truncated != stem {
        out.push(truncated);
    }
    out
}

/// Search huggingface.co for the tokenizer repo, trying each query until
/// one yields a plausible base-model repo.
fn search_hf_tokenizer(queries: &[String]) -> Option<String> {
    let mut seen = std::collections::HashSet::new();
    for q in queries.iter().filter(|q| !q.trim().is_empty()).take(4) {
        // Placeholder metadata names ("Hf_Format") and other digit-less
        // strings only return noise from the search.
        if !q.chars().any(|c| c.is_ascii_digit()) {
            continue;
        }
        if !seen.insert(q.clone()) {
            continue;
        }
        let ids = hf_search(q);
        if ids.is_empty() {
            continue;
        }
        let want = norm(q);
        if let Some(hit) = pick_repo(&ids, std::slice::from_ref(&want)) {
            return Some(hit);
        }
        // exact failed; a prefix/substring hit on this query is still
        // better than moving to a less specific query
        if let Some(hit) = pick_repo(
            &ids,
            &[want.clone(), norm(q.split('-').next().unwrap_or(q))],
        ) {
            return Some(hit);
        }
    }
    None
}

/// Suggest a --tokenizer value for a GGUF: a cached HF repo id or local
/// HF-format dir whose name matches the GGUF's embedded metadata, falling
/// back to a huggingface.co search (metadata first, then the filename).
pub fn suggest_tokenizer(path: &Path, models_dir: &str) -> GgufTokenizerSuggestion {
    let meta = read_meta(path);
    let variants: Vec<String> = meta
        .as_ref()
        .map(|m| m.name_variants())
        .unwrap_or_default()
        .into_iter()
        .map(|v| norm(&v))
        .filter(|v| !v.is_empty())
        .collect();

    let mut tokenizer = None;
    if !variants.is_empty() {
        // Cached repo ids first: stable, and vLLM resolves them offline.
        for repo in hf_cache_repos() {
            let seg = norm(repo.rsplit('/').next().unwrap_or(""));
            if variants.contains(&seg) {
                tokenizer = Some(repo);
                break;
            }
        }
        if tokenizer.is_none() {
            let mut dirs = Vec::new();
            hf_model_dirs(models_dir, &mut dirs);
            for d in dirs {
                let seg = norm(
                    &Path::new(&d)
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default(),
                );
                if variants.contains(&seg) {
                    tokenizer = Some(d);
                    break;
                }
            }
        }
    }

    // Online fallback: the GGUF's own names, then the filename (some
    // GGUFs carry junk like general.name = "Hf_Format").
    if tokenizer.is_none() {
        let mut queries: Vec<String> = meta.as_ref().map(|m| m.name_variants()).unwrap_or_default();
        queries.extend(filename_variants(path));
        tokenizer = search_hf_tokenizer(&queries);
    }

    let variants_raw = meta.map(|m| {
        let base = m.name_variants().into_iter().next();
        (m.display_name(), base)
    });
    let (name, base) = variants_raw.unwrap_or((None, None));
    GgufTokenizerSuggestion {
        name,
        base,
        tokenizer,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_kv_string(b: &mut Vec<u8>, k: &str, v: &str) {
        b.extend_from_slice(&(k.len() as u64).to_le_bytes());
        b.extend_from_slice(k.as_bytes());
        b.extend_from_slice(&T_STRING.to_le_bytes());
        b.extend_from_slice(&(v.len() as u64).to_le_bytes());
        b.extend_from_slice(v.as_bytes());
    }

    #[test]
    fn parses_general_meta_and_skips_arrays() {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&0u64.to_le_bytes()); // tensor count
        b.extend_from_slice(&3u64.to_le_bytes()); // kv count
                                                  // a tokenizer-style string array first, so the parser must skip it
        let k = b"tokenizer.ggml.tokens";
        b.extend_from_slice(&(k.len() as u64).to_le_bytes());
        b.extend_from_slice(k);
        b.extend_from_slice(&T_ARRAY.to_le_bytes());
        b.extend_from_slice(&T_STRING.to_le_bytes());
        b.extend_from_slice(&2u64.to_le_bytes());
        for s in ["a", "bb"] {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        }
        put_kv_string(&mut b, "general.basename", "Qwen3");
        put_kv_string(&mut b, "general.size_label", "0.6B");
        let p = std::env::temp_dir().join("llama-studio-gguf-test.gguf");
        std::fs::write(&p, &b).unwrap();
        let m = read_meta(&p).expect("parses");
        assert_eq!(
            m.0.get("general.basename").map(String::as_str),
            Some("Qwen3")
        );
        let v = m.name_variants();
        assert_eq!(v[0], "Qwen3-0.6B");
        assert_eq!(v[1], "Qwen3");
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn rejects_non_gguf() {
        let p = std::env::temp_dir().join("llama-studio-not-gguf.bin");
        std::fs::write(&p, b"not a gguf file at all........").unwrap();
        assert!(read_meta(&p).is_none());
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn rejects_untrusted_array_counts_without_overflow_or_panic() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        let key = b"tokenizer.ggml.tokens";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key);
        bytes.extend_from_slice(&T_ARRAY.to_le_bytes());
        bytes.extend_from_slice(&4u32.to_le_bytes()); // uint32 elements
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());

        let path = std::env::temp_dir().join(format!(
            "llama-studio-malformed-gguf-{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, bytes).unwrap();
        assert!(read_meta(&path).is_none());
        std::fs::remove_file(path).ok();
    }

    /// Real-file sanity check (run explicitly: cargo test -- --ignored).
    #[test]
    #[ignore]
    fn real_gguf_suggestion() {
        let mut hits = glob();
        let p = hits.pop().expect("no cached Qwen3 GGUF found");
        let s = suggest_tokenizer(Path::new(&p), "");
        println!(
            "name = {:?} base = {:?} tokenizer = {:?}",
            s.name, s.base, s.tokenizer
        );
        assert_eq!(s.tokenizer.as_deref(), Some("Qwen/Qwen3-0.6B"));
    }

    /// The models the user actually serves (run explicitly: cargo test
    /// -- --ignored --nocapture). Needs network for the search fallback.
    #[test]
    #[ignore]
    fn real_models_tokenizers() {
        let models = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("models");
        let cases = [
            (
                "Qwen3.8/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf",
                "Qwen/Qwen3.8-27B",
            ),
            ("Qwen3.8/Qwen3.8-27B-DFlash2-Q8_0.gguf", "Qwen/Qwen3.8-27B"),
            (
                "Nex N2.5/Nex-N2.5-mini-APEX-I-Compact.gguf",
                "nex-agi/Nex-N2.5-mini",
            ),
            ("Spark X2.5/Spark-X2.5-4B.gguf", "XHToken/Spark-X2.5-4B"),
        ];
        for (rel, want) in cases {
            let p = models.join(rel);
            let s = suggest_tokenizer(&p, models.to_str().unwrap());
            println!("{} -> {:?}", rel, s.tokenizer);
            assert_eq!(
                s.tokenizer.as_deref(),
                Some(want),
                "wrong match for {}",
                rel
            );
        }
    }

    #[test]
    fn filename_variants_strip_quants() {
        assert_eq!(
            filename_variants(Path::new("/m/Qwen3.8-27B-DFlash2-Q4_K_M.gguf")),
            vec![
                "Qwen3.8-27B-DFlash2-Q4_K_M".to_string(),
                "Qwen3.8-27B-DFlash2".to_string()
            ]
        );
        // sizes are not quant tokens
        assert_eq!(
            filename_variants(Path::new("/m/Spark-X2.5-4B.gguf")),
            vec!["Spark-X2.5-4B".to_string()]
        );
    }

    #[test]
    fn pick_repo_prefers_clean_exact_matches() {
        let ids: Vec<String> = [
            "unsloth/Qwen3.8-27B-GGUF",
            "Qwen/Qwen3.8-27B-FP8",
            "Qwen/Qwen3.8-27B",
            "lmstudio-community/Qwen3.8-27B-MLX-4bit",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(
            pick_repo(&ids, &[norm("Qwen3.8-27B")]).as_deref(),
            Some("Qwen/Qwen3.8-27B")
        );
    }

    fn glob() -> Vec<String> {
        let home = std::env::var("HOME").unwrap_or_default();
        let snapshots = PathBuf::from(home)
            .join(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B-GGUF/snapshots");
        let mut out = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&snapshots) {
            for s in rd.flatten() {
                if let Ok(files) = std::fs::read_dir(s.path()) {
                    for f in files.flatten() {
                        let is_gguf = f
                            .path()
                            .extension()
                            .map(|e| e == std::ffi::OsStr::new("gguf"))
                            .unwrap_or(false);
                        if is_gguf {
                            out.push(f.path().to_string_lossy().to_string());
                        }
                    }
                }
            }
        } else {
            eprintln!("snapshots dir not readable: {}", snapshots.display());
        }
        out
    }
}
