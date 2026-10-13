#!/usr/bin/env bash
#
# setup.sh — one-shot local setup for the Class Bot service (macOS / Linux).
#
# What it does, in order:
#   1. finds a Python 3.10+ interpreter
#   2. creates a virtual environment in ./.venv
#   3. installs requirements.txt into it
#   4. creates .env from .env.example on first run (never overwrites yours)
#   5. runs a smoke import so a broken install fails here, not at first request
#
# It deliberately does NOT start Postgres or the server — those are one command
# each and are spelled out in RUNNING.md. Re-running this script is safe.
#
# Usage:
#   ./setup.sh
#
set -euo pipefail

cd "$(dirname "$0")"

BOLD=$'\033[1m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; RESET=$'\033[0m'
step() { printf '\n%s==> %s%s\n' "$BOLD" "$1" "$RESET"; }
ok()   { printf '%s✓ %s%s\n' "$GREEN" "$1" "$RESET"; }
warn() { printf '%s! %s%s\n' "$YELLOW" "$1" "$RESET"; }
die()  { printf '%s✗ %s%s\n' "$RED" "$1" "$RESET" >&2; exit 1; }

# ── 1. Find a suitable Python ────────────────────────────────────────────────
step "Looking for Python 3.10+"
PYTHON=""
for cand in python3.13 python3.12 python3.11 python3.10 python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then
    if "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 10) else 1)' 2>/dev/null; then
      PYTHON="$cand"
      break
    fi
  fi
done
[ -n "$PYTHON" ] || die "No Python 3.10+ found. Install it (e.g. \`brew install python@3.12\`) and re-run."
ok "Using $($PYTHON --version) at $(command -v "$PYTHON")"

# ── 2. Virtual environment ───────────────────────────────────────────────────
step "Creating virtual environment in ./.venv"
if [ -d .venv ]; then
  ok ".venv already exists — reusing it"
else
  "$PYTHON" -m venv .venv
  ok "Created .venv"
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# ── 3. Dependencies ──────────────────────────────────────────────────────────
step "Installing dependencies (this can take a minute)"
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt
ok "Dependencies installed"

# ── 4. .env ──────────────────────────────────────────────────────────────────
step "Preparing .env"
if [ -f .env ]; then
  ok ".env already exists — leaving it untouched"
else
  cp .env.example .env
  ok "Created .env from .env.example"
  warn "Edit .env before going to production:"
  warn "  • API_KEYS — replace the *-change-me keys (Odoo and frontend need different keys)"
  warn "  • GOOGLE_CLOUD_PROJECT — your GCP project, or set LLM_PROVIDER=null to run without a model"
fi

# ── 5. Smoke import ──────────────────────────────────────────────────────────
step "Verifying the app imports"
python -c "import app.main; print('app.main imported OK')" \
  && ok "Import check passed" \
  || die "The app failed to import — see the traceback above."

# ── Done ─────────────────────────────────────────────────────────────────────
step "Setup complete"
cat <<'NEXT'
Next steps (details in RUNNING.md):

  1. Activate the venv in each new shell:
       source .venv/bin/activate

  2. Start PostgreSQL 16 (port 5433) with the schema + corpus preloaded:
       docker compose up -d

  3. Run the API:
       uvicorn app.main:app --reload --port 8000
     then open http://127.0.0.1:8000/docs

  4. (Optional) run the tests — no database needed:
       pytest

To demo without a GCP project, set LLM_PROVIDER=null in .env: search still works,
responses just come back with state="degraded" (no generated paragraph).
NEXT
