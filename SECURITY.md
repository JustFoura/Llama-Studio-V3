# Security policy

The API server binds to loopback by default. An optional API key enables bearer
authentication for the active llama.cpp, vLLM, and SGLang servers. Choosing a
LAN address or enabling Tailscale access still requires a suitable firewall or
tailnet policy. The legacy Electron phone-control page in `gui/` has no
authentication; run it only on a trusted interface/network.

Please report security vulnerabilities privately to
[JustFoura](mailto:JustFoura.dev@proton.me) before making them public. Include
affected versions and a minimal reproduction when possible.
