#!/usr/bin/env bash
# Full deploy: provision EC2 → DNS → setup gateway + Caddy.
set -euo pipefail

export AWS_PROFILE="${AWS_PROFILE:-mine}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REMOTE="${SUPERGLUE_DEPLOY_REMOTE:-}"

if [[ -z "${REMOTE}" ]]; then
  echo "=== Step 1: Provision EC2 ==="
  if "${SCRIPT_DIR}/provision-ec2.sh"; then
    # shellcheck source=/dev/null
    source "${HOME}/.superglue/ec2-instance.env"
    REMOTE="${SUPERGLUE_EC2_USER:-ubuntu}@${SUPERGLUE_EC2_PUBLIC_IP}"
    echo "Waiting 30s for SSH..."
    sleep 30
    for i in $(seq 1 20); do
      if ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new \
        "${REMOTE}" "echo ok" >/dev/null 2>&1; then
        break
      fi
      sleep 5
    done
  else
    echo ""
    echo "EC2 provision failed (likely missing IAM permissions)."
    echo "Attach deploy/iam-policy.json to your IAM user, then re-run."
    echo ""
    if [[ -n "${SUPERGLUE_FALLBACK_REMOTE:-}" ]]; then
      REMOTE="${SUPERGLUE_FALLBACK_REMOTE}"
      echo "Using fallback remote: ${REMOTE}"
    else
      echo "Or set SUPERGLUE_DEPLOY_REMOTE=user@host to deploy to an existing server."
      exit 1
    fi
  fi
fi

REMOTE_HOST="${REMOTE#*@}"
if [[ -z "${SUPERGLUE_EC2_PUBLIC_IP:-}" ]]; then
  export SUPERGLUE_EC2_PUBLIC_IP="${REMOTE_HOST}"
fi

echo ""
echo "=== Step 2: DNS (gateway.myharn.sh) ==="
"${SCRIPT_DIR}/setup-dns.sh" "${SUPERGLUE_EC2_PUBLIC_IP}"

echo ""
echo "=== Step 3: Setup server on ${REMOTE} ==="
"${SCRIPT_DIR}/setup-server.sh" "${REMOTE}"

echo ""
echo "=== Done ==="
if [[ -f "${HOME}/.superglue/gateway.env" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/.superglue/gateway.env"
  echo "Gateway URL: ${SUPERGLUE_GATEWAY_URL}"
  echo "Operator env: ${HOME}/.superglue/gateway.env"
fi
