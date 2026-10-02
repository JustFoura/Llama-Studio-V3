use serde::Serialize;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager, State};

pub struct ServerState {
    pub pid: Mutex<Option<u32>>,
    /// One "running" slot: claimed atomically by start(), handed back by
    /// stop() and by the exit watcher. If this ever got stuck true while no
    /// process exists, Start would be blocked forever with a bogus
    /// "already running" - so every exit path must release it.
    pub running: Arc<AtomicBool>,
    /// Bumped on stop so stale health-poll threads give up.
    pub gen: Arc<AtomicU64>,
}

pub struct LaunchSpec {
    pub executable: String,
    pub args: Vec<String>,
    pub health_host: String,
    pub port: u16,
    pub api_key: String,
    pub startup_timeout: Duration,
    pub track_slot_activity: bool,
}

impl Default for ServerState {
    fn default() -> Self {
        ServerState {
            pid: Mutex::new(None),
            running: Arc::new(AtomicBool::new(false)),
            gen: Arc::new(AtomicU64::new(0)),
        }
    }
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StatusEvent {
    pub state: String, // starting | running | stopped | failed
    pub pid: Option<u32>,
    pub code: Option<i32>,
}

#[derive(Clone, Serialize)]
pub struct LogEvent {
    pub kind: String, // info | stdout | stderr | error
    pub text: String,
}

#[derive(Clone, Serialize)]
pub struct ActivityEvent {
    pub active: bool,
}

pub fn emit_log(app: &AppHandle, kind: &str, text: impl Into<String>) {
    let _ = app.emit(
        "server-log",
        LogEvent {
            kind: kind.into(),
            text: text.into(),
        },
    );
}

pub fn emit_status(app: &AppHandle, state: &str, pid: Option<u32>, code: Option<i32>) {
    let _ = app.emit(
        "server-status",
        StatusEvent {
            state: state.into(),
            pid,
            code,
        },
    );
}

/// Resolve a command to a full path. Paths (containing a separator) are taken
/// as-is if they exist; bare names are searched in $PATH (so "vllm" works
/// without an absolute path) plus ~/.local/bin, which desktop-launched apps
/// often don't inherit in $PATH.
pub fn resolve_command(exe: &str) -> Option<PathBuf> {
    let p = PathBuf::from(exe);
    if exe.contains('/') || exe.contains('\\') {
        return if p.exists() { Some(p) } else { None };
    }
    let mut dirs: Vec<PathBuf> =
        std::env::split_paths(&std::env::var("PATH").unwrap_or_default()).collect();
    if let Ok(home) = std::env::var(if cfg!(windows) { "USERPROFILE" } else { "HOME" }) {
        let local_bin = PathBuf::from(home).join(".local").join("bin");
        if !dirs.contains(&local_bin) {
            dirs.push(local_bin);
        }
    }
    for dir in &dirs {
        let cand = dir.join(exe);
        if cand.is_file() {
            return Some(cand);
        }
        #[cfg(windows)]
        for ext in [".exe", ".cmd", ".bat"] {
            let with_ext = dir.join(format!("{}{}", exe, ext));
            if with_ext.is_file() {
                return Some(with_ext);
            }
        }
    }
    None
}

/// Crude ANSI strip - llama-server colors its logs, we want clean text.
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' && chars.peek() == Some(&'[') {
            chars.next();
            for n in chars.by_ref() {
                if n.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn stream_pipe(app: AppHandle, pipe: impl Read + Send + 'static, kind: &'static str) {
    let mut reader = BufReader::new(pipe);
    let mut buf = Vec::new();
    loop {
        buf.clear();
        match reader.read_until(b'\n', &mut buf) {
            Ok(0) => break,
            Ok(_) => {
                let line = String::from_utf8_lossy(&buf);
                for l in strip_ansi(&line).trim_end_matches(['\r', '\n']).split('\n') {
                    if !l.trim().is_empty() {
                        emit_log(&app, kind, l.to_string());
                    }
                }
            }
            Err(_) => break,
        }
    }
}

fn health_ok(host: &str, port: u16, api_key: &str) -> bool {
    let Ok(mut s) = TcpStream::connect((host, port)) else {
        return false;
    };
    let _ = s.set_read_timeout(Some(Duration::from_secs(2)));
    let authorization = if api_key.is_empty() {
        String::new()
    } else {
        format!("Authorization: Bearer {}\r\n", api_key)
    };
    let request = format!(
        "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n{}Connection: close\r\n\r\n",
        authorization
    );
    if s.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut buf = String::new();
    if s.read_to_string(&mut buf).is_err() {
        return false;
    }
    buf.starts_with("HTTP/1.1 200") || buf.starts_with("HTTP/1.0 200")
}

fn parse_slot_activity(body: &str) -> Option<bool> {
    let json: serde_json::Value = serde_json::from_str(body).ok()?;
    let slots = json
        .as_array()
        .or_else(|| json.get("slots").and_then(serde_json::Value::as_array))?;
    let states: Vec<bool> = slots
        .iter()
        .filter_map(|slot| {
            slot.get("is_processing")
                .and_then(serde_json::Value::as_bool)
        })
        .collect();
    if states.is_empty() {
        None
    } else {
        Some(states.into_iter().any(|active| active))
    }
}

fn slot_activity(host: &str, port: u16, api_key: &str) -> Option<bool> {
    let mut stream = TcpStream::connect((host, port)).ok()?;
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let authorization = if api_key.is_empty() {
        String::new()
    } else {
        format!("Authorization: Bearer {}\r\n", api_key)
    };
    let request = format!(
        "GET /slots HTTP/1.1\r\nHost: 127.0.0.1\r\n{}Connection: close\r\n\r\n",
        authorization
    );
    stream.write_all(request.as_bytes()).ok()?;
    let mut response = String::new();
    stream.read_to_string(&mut response).ok()?;
    let body = response.split_once("\r\n\r\n")?.1;
    parse_slot_activity(body)
}

pub fn start(
    app: &AppHandle,
    state: &State<'_, ServerState>,
    launch: LaunchSpec,
) -> Result<(), String> {
    let LaunchSpec {
        executable,
        args,
        health_host,
        port,
        api_key,
        startup_timeout,
        track_slot_activity,
    } = launch;
    // Resolve before claiming the slot, so a missing binary can't leave
    // "running" set (that used to wedge Start until the GUI was restarted).
    let resolved = resolve_command(&executable).ok_or_else(|| {
        format!(
            "Server command not found: {}\nSet the command in Server settings or install the selected backend and ensure it is on PATH.",
            executable
        )
    })?;

    // Atomic claim so two Start clicks can't both spawn.
    if state.running.swap(true, Ordering::SeqCst) {
        return Err(match *state.pid.lock().unwrap() {
            Some(p) => format!("Server is already running (pid {}). Stop it first.", p),
            None => "A server start is already in progress.".into(),
        });
    }

    let display = format!(
        "$ {} {}",
        resolved.display(),
        crate::args::redact_sensitive_args(&args)
            .iter()
            .map(|a| crate::args::quote_arg(a))
            .collect::<Vec<_>>()
            .join(" ")
    );
    emit_log(app, "info", display);

    let mut cmd = Command::new(&resolved);
    cmd.args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // Own process group so Stop can take down the whole tree -
        // vLLM spawns engine worker processes that ignore plain SIGTERM to the parent.
        cmd.process_group(0);
    }
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            state.running.store(false, Ordering::SeqCst);
            return Err(format!("Failed to start {}: {}", resolved.display(), e));
        }
    };

    let pid = child.id();
    *state.pid.lock().unwrap() = Some(pid);
    state.running.store(true, Ordering::SeqCst);
    let my_gen = state.gen.fetch_add(1, Ordering::SeqCst) + 1;
    emit_status(app, "starting", Some(pid), None);

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    let app1 = app.clone();
    std::thread::spawn(move || {
        if let Some(p) = stdout {
            stream_pipe(app1, p, "stdout");
        }
    });
    let app2 = app.clone();
    std::thread::spawn(move || {
        if let Some(p) = stderr {
            stream_pipe(app2, p, "stderr");
        }
    });

    // Exit watcher.
    let app3 = app.clone();
    std::thread::spawn(move || {
        let status = child.wait();
        // Only the process that still owns the slot may publish a stop event
        // or invalidate its health poll. Stop keeps the slot claimed until the
        // child exits, preventing overlapping servers during a quick restart.
        let owns_slot = {
            let st = app3.state::<ServerState>();
            let mut slot = st.pid.lock().unwrap();
            if *slot == Some(pid) {
                *slot = None;
                st.running.store(false, Ordering::SeqCst);
                st.gen.fetch_add(1, Ordering::SeqCst);
                true
            } else {
                false
            }
        };
        let code: Option<i32> = status.ok().and_then(|s| s.code());
        if owns_slot {
            emit_status(&app3, "stopped", None, code);
            emit_log(
                &app3,
                "info",
                format!(
                    "[server exited code={}]",
                    code.map(|c| c.to_string())
                        .unwrap_or_else(|| "?".to_string())
                ),
            );
        }
    });

    // Health poll: big models can take minutes to load.
    let app4 = app.clone();
    let health_gen = state.gen.clone();
    let running = state.running.clone();
    std::thread::spawn(move || {
        let t0 = std::time::Instant::now();
        loop {
            if health_gen.load(Ordering::SeqCst) != my_gen || !running.load(Ordering::SeqCst) {
                return;
            }
            if t0.elapsed() > startup_timeout {
                emit_log(
                    &app4,
                    "error",
                    format!(
                        "[health check timed out after {} min]",
                        startup_timeout.as_secs() / 60
                    ),
                );
                return;
            }
            if health_ok(&health_host, port, &api_key) {
                emit_status(&app4, "running", Some(pid), None);
                break;
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        if !track_slot_activity {
            return;
        }
        let mut last_active = None;
        loop {
            if health_gen.load(Ordering::SeqCst) != my_gen || !running.load(Ordering::SeqCst) {
                return;
            }
            if let Some(active) = slot_activity(&health_host, port, &api_key) {
                if last_active != Some(active) || !active {
                    let _ = app4.emit("server-activity", ActivityEvent { active });
                    last_active = Some(active);
                }
            }
            std::thread::sleep(Duration::from_millis(750));
        }
    });

    Ok(())
}

/// Kill the whole process tree. On Unix the child runs in its own process
/// group (see `process_group(0)`), so signaling the group catches engine
/// worker subprocesses too (vLLM).
fn kill_tree(pid: u32, graceful: bool) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let _ = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .creation_flags(0x0800_0000) // CREATE_NO_WINDOW
            .output();
    }
    #[cfg(unix)]
    {
        let pgid = pid as i32;
        unsafe {
            if graceful {
                libc::killpg(pgid, libc::SIGTERM);
                // Escalate to SIGKILL if it ignores SIGTERM.
                std::thread::spawn(move || {
                    for _ in 0..6 {
                        std::thread::sleep(Duration::from_millis(500));
                        if libc::kill(pgid, libc::SIGKILL) != 0 {
                            return; // already gone
                        }
                    }
                });
            } else {
                libc::killpg(pgid, libc::SIGKILL);
            }
        }
    }
}

pub fn stop(state: &State<'_, ServerState>) -> Result<(), String> {
    let pid = *state.pid.lock().unwrap();
    match pid {
        Some(pid) => {
            state.gen.fetch_add(1, Ordering::SeqCst);
            kill_tree(pid, true);
            Ok(())
        }
        None => Err("Server is not running".into()),
    }
}

/// Best-effort kill on app exit (no State available there). Hard-kill:
/// nobody is listening for a graceful shutdown at this point.
pub fn force_kill(pid: Option<u32>) {
    if let Some(pid) = pid {
        kill_tree(pid, false);
    }
}

#[cfg(test)]
mod tests {
    use super::parse_slot_activity;

    #[test]
    fn parses_llama_slot_activity_and_aggregates_parallel_slots() {
        assert_eq!(
            parse_slot_activity(r#"[{"id":0,"is_processing":true}]"#),
            Some(true)
        );
        assert_eq!(
            parse_slot_activity(r#"{"slots":[{"is_processing":false},{"is_processing":true}]}"#),
            Some(true)
        );
        assert_eq!(
            parse_slot_activity(r#"[{"id":0,"is_processing":false}]"#),
            Some(false)
        );
        assert_eq!(parse_slot_activity(r#"[{"id":0}]"#), None);
    }
}
