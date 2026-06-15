#!/usr/bin/env bash
set -euo pipefail

readonly RESOURCE_KEY="medical_desert_planner"
readonly CONFIG_ERROR_EXIT=2

python_bin="${PYTHON_BIN:-python3}"
target="${TARGET:-dev}"
profile="${DATABRICKS_CONFIG_PROFILE:-${PROFILE:-}}"
warehouse_id="${WAREHOUSE_ID:-}"
app_name="${APP_NAME:-}"
run_mypy="${RUN_MYPY:-1}"

if [[ -x ".venv/bin/python" && -z "${PYTHON_BIN:-}" ]]; then
  python_bin=".venv/bin/python"
fi

if [[ -z "$warehouse_id" ]]; then
  echo "Set WAREHOUSE_ID to the SQL warehouse ID used by the app." >&2
  exit "$CONFIG_ERROR_EXIT"
fi

if [[ -z "$profile" && -z "${DATABRICKS_HOST:-}" ]]; then
  echo "Set PROFILE/DATABRICKS_CONFIG_PROFILE for local auth or DATABRICKS_HOST for CI auth." >&2
  exit "$CONFIG_ERROR_EXIT"
fi

if [[ -z "$app_name" ]]; then
  case "$target" in
    dev)
      app_name="dev-medical-desert-planner"
      ;;
    prod)
      app_name="medical-desert-planner"
      ;;
    *)
      echo "Set APP_NAME for custom target '$target'." >&2
      exit "$CONFIG_ERROR_EXIT"
      ;;
  esac
fi

auth_args=()
if [[ -n "$profile" ]]; then
  auth_args+=(--profile "$profile")
fi

bundle_args=(
  --target "$target"
  --var "warehouse_id=$warehouse_id"
  "${auth_args[@]}"
)

"$python_bin" -m compileall src tools

if [[ "$run_mypy" != "0" ]] && "$python_bin" -m mypy --version >/dev/null 2>&1; then
  "$python_bin" -m mypy
elif [[ "$run_mypy" != "0" ]]; then
  echo "mypy is required for deploy. Install it with: $python_bin -m pip install mypy" >&2
  echo "To bypass only for emergency deploys, set RUN_MYPY=0." >&2
  exit "$CONFIG_ERROR_EXIT"
else
  echo "Skipping mypy because RUN_MYPY=0." >&2
fi

databricks bundle validate --strict "${bundle_args[@]}"

if databricks apps get "$app_name" "${auth_args[@]}" >/dev/null 2>&1; then
  bind_output="$(mktemp)"
  if ! databricks bundle deployment bind "$RESOURCE_KEY" "$app_name" \
    --auto-approve \
    "${bundle_args[@]}" >"$bind_output" 2>&1; then
    if ! grep -q "Resource already managed" "$bind_output"; then
      cat "$bind_output" >&2
      rm -f "$bind_output"
      exit "$CONFIG_ERROR_EXIT"
    fi
  fi
  rm -f "$bind_output"
fi

databricks bundle deploy --auto-approve "${bundle_args[@]}"
databricks bundle run "$RESOURCE_KEY" --restart "${bundle_args[@]}"
databricks apps get "$app_name" "${auth_args[@]}" -o json >/dev/null

echo "Deployment complete for Databricks App: $app_name"
