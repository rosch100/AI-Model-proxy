#!/usr/bin/env sh
set -eu

if [ "${BATCH_WORKER_ENABLED:-false}" != "true" ]; then
  exit 0
fi

if [ "${AUTH_MODE:-single}" != "tenant" ] || [ "${TENANT_CONFIG_SOURCE:-environment}" != "database" ]; then
  printf '%s\n' 'batch-worker requires AUTH_MODE=tenant and TENANT_CONFIG_SOURCE=database.' >&2
  exit 1
fi

exec flask batch-worker
