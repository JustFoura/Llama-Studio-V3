# Llama Studio desktop app

The current app uses Tauri 2, a Rust process manager, and a static HTML/CSS/JS
frontend.

## Development

```sh
npm ci
npm run tauri dev
```

## Build

```sh
npm run tauri -- build --bundles deb
```

The frontend is embedded from `src/` during the build. Linux users can launch
the repository build with `../start.sh`, which rebuilds after source changes.

## Layout

- `src/` — frontend and bundled fonts
- `src-tauri/src/config.rs` — settings, presets, model discovery, user paths
- `src-tauri/src/args.rs` — backend command generation and command-bar parsing
- `src-tauri/src/server.rs` — process lifecycle, logs, and `/health` polling
- `src-tauri/src/gguf.rs` — bounded GGUF metadata reader and vLLM tokenizer lookup
- `src-tauri/src/network.rs` — fail-closed Tailscale address discovery

User configuration is stored outside the source tree in release builds. During
development it lives under `app/`; config and preset files are git-ignored.

See the root [README](../README.md) for supported backends, setup, networking,
and package builds.

## Tests

```sh
cargo test --manifest-path src-tauri/Cargo.toml
npm run test:ui
node --check src/main.js
```
