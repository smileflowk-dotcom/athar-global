#!/usr/bin/env bash
set -u

echo "== Nebius MCP setup for Codex (WSL2 Ubuntu) =="

echo "[1/6] Version checks"
(codex --version || true)
(python3 --version || true)
(python --version || true)
(uv --version || true)
(uvx --version || true)
(nebius version || true)

echo "[2/6] Check Python >= 3.13"
PYBIN=""
if command -v python3 >/dev/null 2>&1; then PYBIN=python3; elif command -v python >/dev/null 2>&1; then PYBIN=python; fi
if [ -z "$PYBIN" ]; then
  echo "Python not found. Install Python 3.13+ in WSL, then rerun."
  exit 2
fi
$PYBIN - <<'PY'
import sys
if sys.version_info < (3, 13):
    print(f"Python {sys.version.split()[0]} detected; Nebius MCP requires Python 3.13+.")
    raise SystemExit(3)
print("Python version OK:", sys.version.split()[0])
PY

echo "[3/6] Ensure uv/uvx"
if ! command -v uvx >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
uv --version
uvx --version

echo "[4/6] Ensure Nebius CLI"
if ! command -v nebius >/dev/null 2>&1; then
  curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
  export PATH="$HOME/.nebius/bin:$HOME/.local/bin:$PATH"
fi
nebius version || { echo "Nebius CLI still not found; restart WSL shell and rerun."; exit 4; }

echo "[5/6] Check Nebius profile"
nebius profile list || true
if ! nebius profile active >/dev/null 2>&1; then
  echo "No active Nebius profile. Starting interactive login/profile creation..."
  nebius profile create
fi
nebius profile active

echo "[6/6] Configure Nebius MCP for Codex"
codex mcp add nebius \
  --env SAFE_MODE=true \
  -- uvx \
  --refresh-package nebius-mcp-server \
  "nebius-mcp-server@git+https://github.com/nebius/mcp-server@main"

echo
echo "Configured MCP servers:"
codex mcp list
echo
echo "DONE. Restart Codex so it connects to Nebius MCP."
