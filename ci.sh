#!/usr/bin/env bash
#
# The gate for this repo. Run it before committing.
#
# Usage: ./ci.sh [pytest args]
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

step() { printf '\n=== %s ===\n' "$1"; shift; "$@"; }

# -P keeps the snapshot folder off sys.path: run from inside it, its
# `types/` package would shadow the standard library's.
step "sema snapshot staging check" uv run python -P \
  src/bruce/sema/check_staging.py src/bruce/sema_seed_request.yaml \
  --deploy-branch main
step "pytest" uv run pytest -q "$@"

printf '\nAll checks passed.\n'
