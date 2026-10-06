#!/usr/bin/env sh
set -eu

worker_enabled=$(printf '%s' "${BATCH_WORKER_ENABLED:-false}" | tr '[:upper:]' '[:lower:]')
case "$worker_enabled" in
  1|t|true|y|yes|on) ;;
  *) exit 0 ;;
esac

if [ "${AUTH_MODE:-single}" != "tenant" ] || [ "${TENANT_CONFIG_SOURCE:-environment}" != "database" ]; then
  printf '%s\n' 'batch-worker requires AUTH_MODE=tenant and TENANT_CONFIG_SOURCE=database.' >&2
  exit 1
fi

exec flask batch-worker
