# Superglue gateway on AWS

Public HTTPS LLM gateway at **https://gateway.myharn.sh**.

## Architecture

- Cloudflare proxies `gateway.myharn.sh` and terminates public HTTPS
- EC2 (Ubuntu 24.04, `us-west-2`) + Elastic IP as the origin
- `superglue gateway serve` on `127.0.0.1:8080` (systemd)
- Caddy reverse-proxies with `tls internal` (Cloudflare SSL mode **Full**)
- SQLite at `~/superglue/superglue-gateway.db`

## DNS

After provision, point Cloudflare at the Elastic IP (scripts print it):

| Name | Type | Content | Proxy |
|------|------|---------|-------|
| `gateway` | A | `<Elastic IP>` | Proxied (orange cloud) |

- **Cloudflare** (default for `myharn.sh`): `./deploy/setup-dns.sh` prints the record to create.
- **Route53** (if you have a hosted zone): `./deploy/setup-dns.sh` upserts an A record (still put Cloudflare in front if you use it for TLS).

Cloudflare SSL/TLS encryption mode must be **Full** (not Flexible). Full (strict) needs a Cloudflare Origin CA cert instead of Caddy `tls internal`.

Ports **80** and **443** must reach the instance from Cloudflare.

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

## Redeploy hygiene

- **Preserve `GATEWAY_MASTER_KEY`** — export it before redeploy and pass the same value into `setup-server.sh` (otherwise a new key is generated and harn admin provisioning breaks until `~/.harn/server.env` is updated).
- **Backup** `~/superglue/superglue-gateway.db` on the gateway host before wiping the instance or deleting the DB. Virtual keys in harn `auth.db` become invalid when the gateway DB is recreated.
- **Cloudflare allowlist** — the harn server EC2 calls this gateway over the public URL. Allow its Elastic IP in Cloudflare WAF (see [`projects/harn/deploy/README.md`](../../harn/deploy/README.md)) or server-side model listing / LLM calls may fail with error 1010.

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
