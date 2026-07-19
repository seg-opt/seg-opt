#!/usr/bin/env bash
set -a; . <(tr -d '\r' < "$(dirname "$0")/.env"); set +a
key=$(mktemp); chmod 600 "$key"; printf '%s\n' "$SSH_KEY" > "$key"
ssh -i "$key" "$USERNAME@$CLUSTER_ADDRESS"
rm -f "$key"
