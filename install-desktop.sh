#!/usr/bin/env bash
# ------------------------------------------------------------
# Install a Linux desktop entry for Llama Studio, so it shows up in
# the desktop launcher (Omarchy: Super+Space / Walker, GNOME, KDE, ...).
#
# The entry points at ./start.sh with absolute paths. start.sh rebuilds
# automatically whenever sources changed, so the launcher entry never
# needs updating - edit code, click the icon, get the new build.
#
# Re-run this script if you move the repository to another path.
# ------------------------------------------------------------
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
ICON="$ROOT/app/src-tauri/icons/icon.png"
[ -f "$ICON" ] || ICON=application-x-executable

mkdir -p "$HOME/.local/share/applications"
cat > "$HOME/.local/share/applications/llama-studio.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Llama Studio
Comment=Local llama.cpp, vLLM, and SGLang desktop client
Exec=$ROOT/start.sh
Path=$ROOT
Icon=$ICON
Terminal=false
Categories=Development;
Keywords=llama;llm;vllm;ai;server;gguf;
StartupWMClass=llama-studio
StartupNotify=false
EOF

chmod +x "$ROOT/start.sh"
update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true

if command -v desktop-file-validate >/dev/null; then
  desktop-file-validate "$HOME/.local/share/applications/llama-studio.desktop"
fi

echo "Installed: ~/.local/share/applications/llama-studio.desktop"
echo "It launches $ROOT/start.sh (auto-rebuilds when sources changed)."
