# Llama Studio

Llama Studio is a desktop control panel for running local language-model
servers. The active application is the Tauri 2 app in `app/`. It launches a
selected backend and connects its built-in chat to the backend's OpenAI-style
`/v1/chat/completions` endpoint.

## Backends

| Backend | Models | Launch command | Notes |
| --- | --- | --- | --- |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | GGUF files or a Hugging Face repo | `llama-server` | Full llama.cpp settings, vision/mmproj, MTP, and DFlash support. |
| [vLLM](https://docs.vllm.ai/) | Hugging Face model directory/repo; GGUF through the separate [GGUF plugin](https://github.com/vllm-project/vllm-gguf-plugin) | `vllm serve` | GGUF support requires a compatible plugin install and tokenizer. |
| [SGLang](https://docs.sglang.io/) | Hugging Face model directory/repo | `python -m sglang.launch_server` | HF-format models only in the model picker. SGLang-specific options can be entered in Extra args. |

Settings shared by the UI are translated only where the selected engine has a
documented equivalent. GPU layers, KV formats, vision encoders, and speculative
decoding are not assumed to behave identically across engines.
The desktop app runs on Windows and Linux, but backend GPU/runtime support is
backend-specific. Check the [vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/)
and [SGLang installation guide](https://docs.sglang.io/get_started/install.html)
for the supported OS, Python, accelerator, and driver combinations.

## Requirements

- Windows 10/11 or Linux
- Rust stable and Node.js 22+ for development/builds
- Tauri platform dependencies: on Linux, WebKitGTK 4.1, GTK 3, and the standard
  Tauri build prerequisites; on Windows, the MSVC C++ build tools and WebView2
- At least one backend installed separately. Backend GPU/runtime requirements
  are determined by llama.cpp, vLLM, or SGLang; see their official installation
  guides before installing them.

## Run

Install a backend, clone this repository, then:

```sh
# Linux: build if needed, then launch
./start.sh

# Windows: build if needed, then launch
start.cmd
```

For development with live Tauri tooling:

```sh
cd app
npm ci
npm run tauri dev
```

The development model directory defaults to `app/models`; installed builds
store models in the user's application-data directory. Set a model in the UI or
enter a Hugging Face repo ID. The first launch of a remote model may download
weights.

## Network access

The default API address is `http://127.0.0.1:1234`, so the server is local-only.
To allow another device, explicitly set the API address to an interface on this
computer and set an API key in Network settings. Tailscale-only mode binds to
the computer's active Tailscale IPv4 address. Network access is also controlled
by your firewall/tailnet policy; do not expose an unauthenticated inference
server to an untrusted network.

## Build packages

```sh
cd app
npm ci
npm run tauri -- build --bundles deb
```

Tauri's Linux targets are configured for DEB, RPM, and AppImage; AppImage builds
also require a working `linuxdeploy` environment. Windows builds use the NSIS
target configured in `app/src-tauri/tauri.conf.json`.

## Tests

```sh
cargo test --manifest-path app/src-tauri/Cargo.toml
node --check app/src/main.js
npm run test:ui --prefix app
```

The Rust tests cover backend arguments, command-bar parsing, model metadata,
and malformed GGUF headers. Live inference tests require a separately installed
backend and model and are not part of the default test suite.

## Legacy app

`gui/` contains the original Windows Electron interface and `start-v1.cmd`
launches it. New development belongs in `app/` unless a change specifically
targets that legacy interface.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
The bundled IBM Plex fonts remain under the SIL Open Font License; see
[`app/src/fonts/LICENSE-IBM-Plex.txt`](app/src/fonts/LICENSE-IBM-Plex.txt).
