#!/usr/bin/env bash
# Upsert Route53 A record for the Superglue gateway, or print Cloudflare/DNS instructions.
# Public traffic is expected to go through Cloudflare proxy; the A record's
# content/origin is the Elastic IP.
set -euo pipefail

export AWS_PROFILE="${AWS_PROFILE:-mine}"

DOMAIN="${SUPERGLUE_GATEWAY_DOMAIN:-gateway.myharn.sh}"
ZONE_NAME="${SUPERGLUE_DNS_ZONE:-myharn.sh}"
STATE_FILE="${HOME}/.superglue/ec2-instance.env"

if [[ -f "${STATE_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${STATE_FILE}"
fi

EIP="${SUPERGLUE_EC2_PUBLIC_IP:-}"
REGION="${SUPERGLUE_EC2_REGION:-${AWS_REGION:-us-west-2}}"

if [[ -z "${EIP}" && -n "${1:-}" ]]; then
  EIP="${1}"
fi

if [[ -z "${EIP}" ]]; then
  echo "error: SUPERGLUE_EC2_PUBLIC_IP not set; run provision-ec2.sh first or pass the Elastic IP as an argument" >&2
  exit 1
fi

echo "Target origin: ${DOMAIN} -> ${EIP} (via Cloudflare proxy)"

ZONE_ID="${SUPERGLUE_ROUTE53_ZONE_ID:-}"
if [[ -z "${ZONE_ID}" ]]; then
  ZONE_ID="$(aws route53 list-hosted-zones-by-name \
    --dns-name "${ZONE_NAME}." \
    --query "HostedZones[?Name=='${ZONE_NAME}.'].Id" \
    --output text 2>/dev/null | head -1 | sed 's|/hostedzone/||' || true)"
fi

if [[ -z "${ZONE_ID}" || "${ZONE_ID}" == "None" ]]; then
  echo ""
  echo "No Route53 hosted zone found for ${ZONE_NAME} (expected: manage DNS in Cloudflare)."
  echo "In Cloudflare DNS for ${ZONE_NAME}:"
  echo ""
  echo "  Name:    gateway  (or ${DOMAIN})"
  echo "  Type:    A"
  echo "  Content: ${EIP}"
  echo "  Proxy:   Proxied (orange cloud)"
  echo ""
  echo "SSL/TLS: set encryption mode to Full (not Flexible, not Full strict)."
  echo "Caddy on the origin uses an internal cert; Full strict needs an Origin CA cert."
  echo ""
  echo "After the record is set, continue with setup-server.sh."
  exit 0
fi

echo "Using Route53 hosted zone ${ZONE_ID} (${ZONE_NAME})"

CHANGE_BATCH="$(cat <<EOF
{
  "Comment": "Superglue gateway ${DOMAIN}",
  "Changes": [
    {
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "${DOMAIN}",
        "Type": "A",
        "TTL": 60,
        "ResourceRecords": [{"Value": "${EIP}"}]
      }
    }
  ]
}
EOF
)"

CHANGE_ID="$(aws route53 change-resource-record-sets \
  --hosted-zone-id "${ZONE_ID}" \
  --change-batch "${CHANGE_BATCH}" \
  --query 'ChangeInfo.Id' \
  --output text)"

echo "Waiting for DNS change ${CHANGE_ID}..."
aws route53 wait resource-record-sets-changed --id "${CHANGE_ID}"

echo ""
echo "DNS ready: ${DOMAIN} -> ${EIP}"
echo "Verify: dig +short ${DOMAIN}"
