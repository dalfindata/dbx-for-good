#!/usr/bin/env bash
set -euo pipefail

profile="${DATABRICKS_CONFIG_PROFILE:-${PROFILE:-}}"
app_name="${APP_NAME:-dev-medical-desert-planner}"
principal="${APP_SERVICE_PRINCIPAL:-}"

auth_args=()
if [[ -n "$profile" ]]; then
  auth_args+=(--profile "$profile")
fi

echo "App details:"
databricks apps get "$app_name" "${auth_args[@]}" -o json

cat <<'EOF'

Grant template:

Use the service_principal_client_id/applicationId from the app details above.
For the dev app this is:

  fb4754d9-7dd1-4e21-a5d9-9bcf63068a05

Ask a catalog owner/admin to run these statements in the Databricks SQL editor:

EOF

if [[ -z "$principal" ]]; then
  principal="fb4754d9-7dd1-4e21-a5d9-9bcf63068a05"
fi

cat <<EOF
GRANT USE CATALOG
ON CATALOG databricks_virtue_foundation_dataset_dais_2026
TO \`$principal\`;

GRANT USE SCHEMA
ON SCHEMA databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset
TO \`$principal\`;

GRANT SELECT
ON TABLE databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset.facilities
TO \`$principal\`;

GRANT SELECT
ON TABLE databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset.nfhs_5_district_health_indicators
TO \`$principal\`;

GRANT SELECT
ON TABLE databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset.india_post_pincode_directory
TO \`$principal\`;
EOF
