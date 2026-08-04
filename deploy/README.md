# Superglue gateway on AWS

Public HTTPS LLM gateway at **https://gateway.myharn.sh**.

## Architecture

- EC2 (Ubuntu 24.04, `us-west-2`) + Elastic IP
- `superglue gateway serve` on `127.0.0.1:8080` (systemd)
- Caddy terminates TLS and reverse-proxies to the gateway
- SQLite at `~/superglue/superglue-gateway.db`

## DNS

After provision you need one record (scripts print the Elastic IP):

| Name | Type | Value |
|------|------|-------|
| `gateway.myharn.sh` | A | `<Elastic IP>` |

- **Route53** (`myharn.sh` hosted zone): `./deploy/setup-dns.sh` upserts it.
- **Other DNS**: create the A record yourself. Cloudflare: DNS-only (grey cloud).

Ports **80** and **443** must reach the instance for Let's Encrypt.

## Prerequisites

- AWS CLI credentials
- IAM permissions from [`iam-policy.json`](iam-policy.json)
- SSH public key at `~/.ssh/default.pub` (override with `SUPERGLUE_EC2_PUBLIC_KEY`)
- At least one provider key locally (`OPENAI_API_KEY`, etc.)

## Deploy

```bash
cd projects/superglue
./deploy/deploy.sh
```

Or step by step:

```bash
./deploy/provision-ec2.sh   # prints Elastic IP + writes ~/.superglue/ec2-instance.env
./deploy/setup-dns.sh       # Route53 upsert, or prints manual A-record instructions
./deploy/setup-server.sh    # build, install binary + Caddy, start services
```

Redeploy to an existing host:

```bash
SUPERGLUE_DEPLOY_REMOTE=ubuntu@<ip> ./deploy/deploy.sh
# or
./deploy/setup-server.sh ubuntu@<ip>
```

## Verify

```bash
curl -sf https://gateway.myharn.sh/health
source ~/.superglue/gateway.env

# Remote admin CLI
superglue gateway --url "$SUPERGLUE_GATEWAY_URL" user list
```

## Create a virtual key

```bash
source ~/.superglue/gateway.env

BUDGET=$(curl -sS -X POST "${SUPERGLUE_GATEWAY_URL}/v1/budgets" \
  -H "X-Superglue-Key: Bearer ${GATEWAY_MASTER_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{"max_budget": 50.0, "duration_sec": 2592000, "enforce": true}')
echo "${BUDGET}"

# Use budget_id from the response, then:
curl -sS -X POST "${SUPERGLUE_GATEWAY_URL}/v1/users" \
  -H "X-Superglue-Key: Bearer ${GATEWAY_MASTER_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{"user_id": "me", "alias": "Me", "budget_id": "<budget-id>"}'

curl -sS -X POST "${SUPERGLUE_GATEWAY_URL}/v1/keys" \
  -H "X-Superglue-Key: Bearer ${GATEWAY_MASTER_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{"name": "default", "user_id": "me", "allowed_models": ["openai:*", "anthropic:*"]}'
```

Store the returned plaintext key; it is shown once.

## Env overrides

| Variable | Default |
|----------|---------|
| `AWS_REGION` | `us-west-2` |
| `SUPERGLUE_GATEWAY_DOMAIN` | `gateway.myharn.sh` |
| `SUPERGLUE_EC2_INSTANCE_TYPE` | `t3.small` |
| `SUPERGLUE_ROUTE53_ZONE_ID` | auto-detect `myharn.sh` |
| `GATEWAY_MASTER_KEY` | generated if unset |
