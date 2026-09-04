#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3.12}"
VENV_ROOT="${MECH_VENV_ROOT:-$REPO_ROOT/.venv}"
EXPECTED_COMMIT="${MECH_EXPECTED_COMMIT:?set MECH_EXPECTED_COMMIT to the reviewed 40-hex harness commit}"
EXPECTED_MODEL_REVISION="2bfc9ae6e89079a5cc7ed2a68436010d91a3d289"
CONFIG="$REPO_ROOT/experiments/self_repair/mechanistic/config/mechanistic.json"
REQUIREMENTS="$REPO_ROOT/experiments/self_repair/requirements-mechanistic.txt"

if [[ ! "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "MECH_EXPECTED_COMMIT must be a lowercase 40-hex Git commit." >&2
  exit 2
fi
if [[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" != "$EXPECTED_COMMIT" ]]; then
  echo "Checkout does not match MECH_EXPECTED_COMMIT; refusing environment setup." >&2
  exit 2
fi
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked working tree is dirty; refusing an unbound RunPod environment." >&2
  exit 2
fi
if [[ ! -f "$CONFIG" || ! -f "$REQUIREMENTS" ]]; then
  echo "Mechanistic config or locked requirements are missing." >&2
  exit 2
fi

"$PYTHON_BOOTSTRAP" - <<'PY'
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"Python 3.12 is required, found {sys.version.split()[0]}")
PY

"$PYTHON_BOOTSTRAP" -m venv "$VENV_ROOT"
"$VENV_ROOT/bin/python" -m pip install --upgrade pip wheel
"$VENV_ROOT/bin/python" -m pip install -r "$REQUIREMENTS"
"$VENV_ROOT/bin/python" -m pip install -e "$REPO_ROOT/moshi" --no-deps
"$VENV_ROOT/bin/python" -m pip check

"$VENV_ROOT/bin/python" - "$CONFIG" "$EXPECTED_MODEL_REVISION" <<'PY'
import json
import pathlib
import sys

config_path = pathlib.Path(sys.argv[1])
expected_revision = sys.argv[2]
config = json.loads(config_path.read_text(encoding="utf-8"))
model = config.get("model", {})
if model.get("repo") != "kyutai/moshiko-pytorch-bf16":
    raise SystemExit("mechanistic config has the wrong model repository")
if model.get("revision") != expected_revision:
    raise SystemExit("mechanistic config has the wrong pinned model revision")
if (model.get("layers"), model.get("heads"), model.get("hidden_size")) != (32, 32, 4096):
    raise SystemExit("mechanistic config has the wrong Moshiko shape contract")
print(f"RunPod environment ready for {model['repo']}@{expected_revision}")
PY

echo "Environment: $VENV_ROOT"
echo "No checkpoint was downloaded and no paid scan was started."
