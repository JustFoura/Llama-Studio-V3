#!/usr/bin/env bash
# ------------------------------------------------------------
# Llama Studio (Tauri) launcher for Linux.
#  - rebuilds automatically when sources are newer than the binary,
#    so a desktop entry can just point at this script and always
#    launch the current code
#  - first run (or big changes) takes a few minutes; otherwise instant
#  - the frontend is embedded into the binary at build time, which is
#    why the staleness check below watches app/src too
#  - the Windows launcher is start.cmd
# ------------------------------------------------------------
set -e
cd "$(dirname "$0")/app/src-tauri"
export PATH="$HOME/.cargo/bin:$PATH"
BIN=target/debug/llama-studio

needs_build() {
  [ -x "$BIN" ] || return 0
  # Rust code, the embedded frontend, icons or the Tauri config newer
  # than the binary?
  [ -n "$(find src ../src build.rs Cargo.toml tauri.conf.json capabilities icons \
        -newer "$BIN" -print -quit 2>/dev/null)" ]
}

if needs_build; then
  echo "Sources changed since last build - rebuilding..."
  # cargo itself only reruns on Rust changes; touch build.rs so edited
  # frontend files get re-embedded too (generate_context! expands them
  # into the crate, so the crate must recompile).
  touch build.rs
  if ! cargo build; then
    notify-send -u critical -a "Llama Studio" "Build failed - start.sh from a terminal for the error" 2>/dev/null || true
    exit 1
  fi
fi

exec "$BIN"
