#!/usr/bin/env bash
# Configure Superglue gateway + Caddy on a remote Ubuntu host.
# Public TLS is terminated at Cloudflare (proxied). Caddy uses an internal
# cert for origin HTTPS — set Cloudflare SSL/TLS mode to Full (not Flexible).
# Usage: ./setup-server.sh [user@host] [--local-binary /path/to/superglue]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUPERGLUE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MONOREPO_ROOT="$(cd "${SUPERGLUE_DIR}/../.." && pwd)"

REMOTE=""
LOCAL_BINARY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --local-binary)
      LOCAL_BINARY="$2"
      shift 2
      ;;
    -*)
      echo "error: unknown option: $1" >&2
      echo "usage: $0 [user@host] [--local-binary /path/to/superglue]" >&2
      exit 1
      ;;
    *)
      REMOTE="$1"
      shift
      ;;
  esac
done

DOMAIN="${SUPERGLUE_GATEWAY_DOMAIN:-gateway.myharn.sh}"

if [[ -z "${REMOTE}" ]]; then
  if [[ -f "${HOME}/.superglue/ec2-instance.env" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/.superglue/ec2-instance.env"
    REMOTE="${SUPERGLUE_EC2_USER:-ubuntu}@${SUPERGLUE_EC2_PUBLIC_IP}"
    DOMAIN="${SUPERGLUE_GATEWAY_DOMAIN:-${DOMAIN}}"
  else
    echo "usage: $0 [user@host] [--local-binary /path/to/superglue]" >&2
    exit 1
  fi
fi

REMOTE_USER="${REMOTE%%@*}"
REMOTE_HOST="${REMOTE#*@}"

echo "Setting up Superglue gateway on ${REMOTE} (domain ${DOMAIN}, Cloudflare origin)..."

MASTER_KEY="${GATEWAY_MASTER_KEY:-}"
if [[ -z "${MASTER_KEY}" ]]; then
  MASTER_KEY="$(openssl rand -base64 32 | tr -d '/+=' | head -c 40)"
  echo "Generated GATEWAY_MASTER_KEY (saved to remote env + local state)"
else
  echo "Using existing GATEWAY_MASTER_KEY from environment"
fi

# Collect provider keys from local env / monorepo .env files (never print)
for f in \
  "${SUPERGLUE_DIR}/.env" \
  "${MONOREPO_ROOT}/projects/porque/.env" \
  "${MONOREPO_ROOT}/projects/harn/.env" \
  "${HOME}/.superglue/server.env"; do
  if [[ -f "${f}" ]]; then
    # shellcheck disable=SC1090
    set -a
    # shellcheck source=/dev/null
    source "${f}"
    set +a
  fi
done

OPENAI_KEY="${OPENAI_API_KEY:-}"
ANTHROPIC_KEY="${ANTHROPIC_API_KEY:-}"
XAI_KEY="${XAI_API_KEY:-}"
GROQ_KEY="${GROQ_API_KEY:-}"

if [[ -z "${OPENAI_KEY}" && -z "${ANTHROPIC_KEY}" && -z "${XAI_KEY}" && -z "${GROQ_KEY}" ]]; then
  echo "error: no provider API keys found; set OPENAI_API_KEY (or another provider) locally" >&2
  exit 1
fi

if [[ -z "${LOCAL_BINARY}" ]]; then
  echo "Building superglue release binary with gateway feature..."
  cargo build --release --features gateway --manifest-path "${SUPERGLUE_DIR}/Cargo.toml" --bin superglue
  LOCAL_BINARY="${SUPERGLUE_DIR}/target/release/superglue"
fi
if [[ ! -x "${LOCAL_BINARY}" ]]; then
  echo "error: binary not found or not executable: ${LOCAL_BINARY}" >&2
  exit 1
fi

resolve_a() {
  local name="$1"
  if command -v dig >/dev/null 2>&1; then
    dig +short "${name}" A 2>/dev/null | head -1 || true
  else
    getent ahostsv4 "${name}" 2>/dev/null | awk '{print $1; exit}' || true
  fi
}

# Cloudflare proxy: public A records are Cloudflare IPs, not the EIP.
RESOLVED="$(resolve_a "${DOMAIN}")"
if [[ -z "${RESOLVED}" ]]; then
  echo "warning: ${DOMAIN} does not resolve yet; ensure Cloudflare has a proxied A record -> ${REMOTE_HOST}" >&2
else
  echo "DNS OK: ${DOMAIN} -> ${RESOLVED} (Cloudflare edge; origin ${REMOTE_HOST})"
fi

echo "Installing system packages on remote..."
ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "${REMOTE}" bash -s <<REMOTE_DEPS
set -euo pipefail
sudo apt-get update -qq
sudo apt-get install -y -qq curl ca-certificates debian-keyring debian-archive-keyring apt-transport-https

if ! command -v caddy >/dev/null 2>&1; then
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  sudo apt-get update -qq
  sudo apt-get install -y -qq caddy
fi
REMOTE_DEPS

echo "Copying superglue binary..."
ssh "${REMOTE}" "sudo mkdir -p /usr/local/bin"
scp -q "${LOCAL_BINARY}" "${REMOTE}:/tmp/superglue"
ssh "${REMOTE}" "sudo install -m 755 /tmp/superglue /usr/local/bin/superglue && rm /tmp/superglue"

REMOTE_HOME="/home/${REMOTE_USER}"
if [[ "${REMOTE_USER}" == "root" ]]; then
  REMOTE_HOME="/root"
fi
DATA_DIR="${REMOTE_HOME}/superglue"
ENV_DIR="${REMOTE_HOME}/.superglue"

echo "Writing data dir, env, Caddyfile, and systemd unit..."
ssh "${REMOTE}" "mkdir -p ${DATA_DIR} ${ENV_DIR} && chmod 700 ${ENV_DIR}"

LOCAL_ENV_TMP="$(mktemp)"
chmod 600 "${LOCAL_ENV_TMP}"
cat > "${LOCAL_ENV_TMP}" <<EOF
GATEWAY_MASTER_KEY=${MASTER_KEY}
OPENAI_API_KEY=${OPENAI_KEY}
ANTHROPIC_API_KEY=${ANTHROPIC_KEY}
XAI_API_KEY=${XAI_KEY}
GROQ_API_KEY=${GROQ_KEY}
RUST_LOG=info
EOF
scp -q "${LOCAL_ENV_TMP}" "${REMOTE}:${ENV_DIR}/server.env"
rm -f "${LOCAL_ENV_TMP}"
ssh "${REMOTE}" "chmod 600 ${ENV_DIR}/server.env"

# Caddyfile: internal TLS for Cloudflare Full (edge terminates public HTTPS)
ssh "${REMOTE}" bash -s <<REMOTE_CADDY
set -euo pipefail
sudo tee /etc/caddy/Caddyfile >/dev/null <<'EOF'
${DOMAIN} {
	tls internal
	reverse_proxy 127.0.0.1:8080
}
EOF
sudo systemctl enable caddy
sudo systemctl restart caddy
REMOTE_CADDY

scp -q "${SCRIPT_DIR}/superglue-gateway.service" "${REMOTE}:/tmp/superglue-gateway.service"
ssh "${REMOTE}" bash -s <<REMOTE_SYSTEMD
set -euo pipefail
sudo sed "s/User=ubuntu/User=${REMOTE_USER}/; s|/home/ubuntu|${REMOTE_HOME}|g" \
  /tmp/superglue-gateway.service | sudo tee /etc/systemd/system/superglue-gateway.service >/dev/null
rm /tmp/superglue-gateway.service
sudo systemctl daemon-reload
sudo systemctl enable superglue-gateway
sudo systemctl restart superglue-gateway
REMOTE_SYSTEMD

echo "Waiting for local health check..."
for i in $(seq 1 30); do
  if ssh "${REMOTE}" "curl -sf http://127.0.0.1:8080/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

if ! ssh "${REMOTE}" "curl -sf http://127.0.0.1:8080/health" >/dev/null 2>&1; then
  echo "error: gateway health check failed" >&2
  ssh "${REMOTE}" "sudo journalctl -u superglue-gateway -n 40 --no-pager" >&2 || true
  exit 1
fi

echo "Waiting for public HTTPS health..."
HTTPS_OK=0
for i in $(seq 1 36); do
  if curl -sf "https://${DOMAIN}/health" >/dev/null 2>&1; then
    HTTPS_OK=1
    break
  fi
  sleep 5
done

# Persist master key locally for operator use
mkdir -p "${HOME}/.superglue"
LOCAL_ENV="${HOME}/.superglue/gateway.env"
cat > "${LOCAL_ENV}" <<EOF
export SUPERGLUE_GATEWAY_URL=https://${DOMAIN}
export SUPERGLUE_GATEWAY_DOMAIN=${DOMAIN}
export GATEWAY_MASTER_KEY=${MASTER_KEY}
export SUPERGLUE_REMOTE=${REMOTE}
EOF
chmod 600 "${LOCAL_ENV}"

echo ""
echo "Gateway configured on ${REMOTE}"
echo "  Local health: OK (http://127.0.0.1:8080/health)"
if [[ "${HTTPS_OK}" -eq 1 ]]; then
  echo "  Public URL:   https://${DOMAIN}/health OK"
else
  echo "  Public URL:   https://${DOMAIN}/health not ready yet"
  echo "  Check:        Cloudflare proxied A -> ${REMOTE_HOST}, SSL/TLS mode Full"
  echo "  Debug:        ssh ${REMOTE} 'sudo journalctl -u caddy -n 40 --no-pager'"
fi
echo "  Operator env: ${LOCAL_ENV}"
echo ""
echo "Create a virtual key (example):"
echo "  source ${LOCAL_ENV}"
echo "  curl -sS -X POST \"\${SUPERGLUE_GATEWAY_URL}/v1/budgets\" \\"
echo "    -H \"X-Superglue-Key: Bearer \${GATEWAY_MASTER_KEY}\" \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"max_budget\": 50.0, \"duration_sec\": 2592000, \"enforce\": true}'"
