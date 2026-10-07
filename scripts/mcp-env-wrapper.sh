#!/usr/bin/env bash
# Sanitize AZURE_* env vars, then exec the real MCP server command.
# Usage: mcp-env-wrapper.sh <command> [args...]
# stdout belongs to the MCP stdio protocol: write diagnostics to stderr only,
# and never print values.
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "mcp-env-wrapper: missing command" >&2
  exit 64
fi

clean() {
  local v=$1
  v=${v//$'\r'/}
  v=${v//$'\n'/}
  v="${v#"${v%%[![:space:]]*}"}"   # trim leading whitespace
  v="${v%"${v##*[![:space:]]}"}"   # trim trailing whitespace
  if [[ ${#v} -ge 2 && ( $v == \"*\" || $v == \'*\' ) ]]; then
    v=${v:1:${#v}-2}               # strip one pair of surrounding quotes
  fi
  printf '%s' "$v"
}

guid='^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'

for name in AZURE_CLIENT_ID AZURE_TENANT_ID AZURE_SUBSCRIPTION_ID AZURE_CLIENT_SECRET; do
  [ -n "${!name+x}" ] || continue
  value=$(clean "${!name}")
  if [ -z "$value" ]; then
    unset "$name"
    echo "mcp-env-wrapper: $name is empty; unset" >&2
    continue
  fi
  # None of these values legitimately contain whitespace, but pasted secrets have arrived with notes
  # appended after the value. Keep only the first token, as login.sh does.
  if [[ $value == *[[:space:]]* ]]; then
    value=${value%%[[:space:]]*}
    echo "mcp-env-wrapper: $name contained whitespace; kept only the first token" >&2
  fi
  export "$name=$value"
  if [ "$name" != AZURE_CLIENT_SECRET ] && ! [[ $value =~ $guid ]]; then
    echo "mcp-env-wrapper: $name is not a GUID (check for stray characters)" >&2
  fi
done

exec "$@"
