# Contributing

## Development setup

Install Rust (stable), Node.js, and the Tauri system dependencies listed in the
README. Then run:

```sh
cd app
npm ci
npm run tauri dev
```

Run the Rust regression suite before submitting changes:

```sh
cd app/src-tauri
cargo test
```

The desktop app embeds the static frontend at build time. Launch it through
`./start.sh` on Linux or `start.cmd` on Windows so source changes are reflected
in the executable.

## Adding an inference engine

Engine-specific options belong in the Rust argument builder and the engine
capability controls in `app/src/index.html` / `app/src/main.js`. Keep shared
OpenAI-compatible chat behavior in one client path, and document any options
that cannot be translated rather than silently ignoring them. Add tests for
argument generation, command-bar round trips, model selection, and validation.

Do not commit local `config.json`, presets, model files, tokens, or benchmark
outputs containing personal machine paths. User config and model directories
are ignored by git.
