#!/usr/bin/env bash
# Configure Superglue gateway + Caddy on a remote Ubuntu host.
# Public TLS is terminated at Cloudflare (proxied). Caddy uses an internal
# cert for origin HTTPS — set Cloudflare SSL/TLS mode to Full (not Flexible).
# Redeploys reuse GATEWAY_MASTER_KEY from ~/.superglue/gateway.env (or the
# remote host) and sync it into Harn (~/.harn/server.env and the Harn EC2 host).
# Usage: ./setup-server.sh [user@host] [--local-binary /path/to/superglue]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUPERGLUE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MONOREPO_ROOT="$(cd "${SUPERGLUE_DIR}/../.." && pwd)"

REMOTE="${SUPERGLUE_DEPLOY_REMOTE:-}"
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
INTERNAL_HTTP_PORT="${SUPERGLUE_INTERNAL_HTTP_PORT:-80}"
INTERNAL_ALLOW_IP="${SUPERGLUE_INTERNAL_ALLOW_IP:-54.202.41.32}"

echo "Setting up Superglue gateway on ${REMOTE} (domain ${DOMAIN}, Cloudflare origin)..."

# Reuse the live master key. A stale GATEWAY_MASTER_KEY in the shell must not
# rotate the gateway and break Harn admin. Generate only on first install.
MASTER_KEY=""
if [[ -f "${HOME}/.superglue/gateway.env" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/.superglue/gateway.env"
  MASTER_KEY="${GATEWAY_MASTER_KEY:-}"
fi
if [[ -z "${MASTER_KEY}" ]]; then
  MASTER_KEY="$(
    ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "${REMOTE}" \
      "awk -F= '/^GATEWAY_MASTER_KEY=/{print substr(\$0, index(\$0,\"=\")+1)}' ~/.superglue/server.env 2>/dev/null" \
      || true
  )"
  MASTER_KEY="${MASTER_KEY//$'\r'/}"
  MASTER_KEY="${MASTER_KEY//$'\n'/}"
fi
if [[ -z "${MASTER_KEY}" && -n "${GATEWAY_MASTER_KEY:-}" ]]; then
  MASTER_KEY="${GATEWAY_MASTER_KEY}"
fi
if [[ -z "${MASTER_KEY}" ]]; then
  MASTER_KEY="$(openssl rand -base64 32 | tr -d '/+=' | head -c 40)"
  echo "Generated GATEWAY_MASTER_KEY (first-time install; saved to remote env + local state)"
else
  echo "Reusing existing GATEWAY_MASTER_KEY"
fi

# Keep existing remote provider keys, then overlay local env files (never print).
plausible_key() {
  local k="${1:-}"
  [[ ${#k} -ge 16 && "${k}" != "sk-..." ]]
}

pick_key() {
  local local_val="${1:-}"
  local remote_val="${2:-}"
  if plausible_key "${local_val}"; then
    printf '%s' "${local_val}"
  else
    printf '%s' "${remote_val}"
  fi
}

REMOTE_ENV_TMP="$(mktemp)"
if scp -q "${REMOTE}:${REMOTE_HOME:-/home/${REMOTE_USER}}/.superglue/server.env" "${REMOTE_ENV_TMP}" 2>/dev/null; then
  set -a
  # shellcheck source=/dev/null
  source "${REMOTE_ENV_TMP}"
  set +a
fi
rm -f "${REMOTE_ENV_TMP}"
REMOTE_OPENAI="${OPENAI_API_KEY:-}"
REMOTE_ANTHROPIC="${ANTHROPIC_API_KEY:-}"
REMOTE_XAI="${XAI_API_KEY:-}"
REMOTE_GROQ="${GROQ_API_KEY:-}"
REMOTE_OPENROUTER="${OPENROUTER_API_KEY:-${OPENROUTER_GATEWAY_KEY:-}}"
REMOTE_RUNINFRA="${RUNINFRA_GATEWAY_KEY:-}"
REMOTE_VERCEL="${VERCEL_GATEWAY_KEY:-${AI_GATEWAY_API_KEY:-}}"
REMOTE_TYPESAFE="${TYPESAFE_API_KEY:-}"

for f in \
  "${SUPERGLUE_DIR}/.env" \
  "${MONOREPO_ROOT}/.env" \
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

OPENAI_KEY="$(pick_key "${OPENAI_API_KEY:-}" "${REMOTE_OPENAI}")"
ANTHROPIC_KEY="$(pick_key "${ANTHROPIC_API_KEY:-}" "${REMOTE_ANTHROPIC}")"
XAI_KEY="$(pick_key "${XAI_API_KEY:-}" "${REMOTE_XAI}")"
GROQ_KEY="$(pick_key "${GROQ_API_KEY:-}" "${REMOTE_GROQ}")"
OPENROUTER_KEY="$(pick_key "${OPENROUTER_API_KEY:-${OPENROUTER_GATEWAY_KEY:-}}" "${REMOTE_OPENROUTER}")"
RUNINFRA_KEY="$(pick_key "${RUNINFRA_GATEWAY_KEY:-}" "${REMOTE_RUNINFRA}")"
VERCEL_KEY="$(pick_key "${VERCEL_GATEWAY_KEY:-${AI_GATEWAY_API_KEY:-}}" "${REMOTE_VERCEL}")"
TYPESAFE_KEY="$(pick_key "${TYPESAFE_API_KEY:-}" "${REMOTE_TYPESAFE}")"
DEFAULT_ALLOWED_MODELS="openai:*,anthropic:*,xai:*,groq:*,openrouter:*,runinfra:*,vercel:*,typesafe:*"
CAPTURE_BUCKET="${SUPERGLUE_CAPTURE_S3_BUCKET:-superglue-gateway-capture}"
CAPTURE_PREFIX="${SUPERGLUE_CAPTURE_S3_PREFIX:-gateway-capture}"
CAPTURE_REGION="${AWS_REGION:-us-west-2}"

if [[ -z "${OPENAI_KEY}" && -z "${ANTHROPIC_KEY}" && -z "${XAI_KEY}" && -z "${GROQ_KEY}" && -z "${OPENROUTER_KEY}" && -z "${RUNINFRA_KEY}" && -z "${VERCEL_KEY}" ]]; then
  echo "error: no provider API keys found; set OPENAI_API_KEY (or another provider) locally" >&2
  exit 1
fi

configured_providers=()
[[ -n "${OPENAI_KEY}" ]] && configured_providers+=(openai)
[[ -n "${ANTHROPIC_KEY}" ]] && configured_providers+=(anthropic)
[[ -n "${XAI_KEY}" ]] && configured_providers+=(xai)
[[ -n "${GROQ_KEY}" ]] && configured_providers+=(groq)
[[ -n "${OPENROUTER_KEY}" ]] && configured_providers+=(openrouter)
[[ -n "${RUNINFRA_KEY}" ]] && configured_providers+=(runinfra)
[[ -n "${VERCEL_KEY}" ]] && configured_providers+=(vercel)
[[ -n "${TYPESAFE_KEY}" ]] && configured_providers+=(typesafe)
echo "Provider keys to install: ${configured_providers[*]}"
echo "Capture: s3://${CAPTURE_BUCKET}/${CAPTURE_PREFIX} (region ${CAPTURE_REGION})"

if [[ -z "${LOCAL_BINARY}" ]]; then
  echo "Building superglue release binary with gateway feature..."
  cargo build --release --features gateway,capture --manifest-path "${SUPERGLUE_DIR}/Cargo.toml" --bin superglue
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
OPENROUTER_API_KEY=${OPENROUTER_KEY}
RUNINFRA_GATEWAY_KEY=${RUNINFRA_KEY}
VERCEL_GATEWAY_KEY=${VERCEL_KEY}
TYPESAFE_API_KEY=${TYPESAFE_KEY}
RUST_LOG=info
AWS_REGION=${CAPTURE_REGION}
SUPERGLUE_CAPTURE_S3_BUCKET=${CAPTURE_BUCKET}
SUPERGLUE_CAPTURE_S3_PREFIX=${CAPTURE_PREFIX}
EOF
scp -q "${LOCAL_ENV_TMP}" "${REMOTE}:${ENV_DIR}/server.env"
rm -f "${LOCAL_ENV_TMP}"
ssh "${REMOTE}" "chmod 600 ${ENV_DIR}/server.env"

# Caddyfile: public HTTPS for Cloudflare Full. Port 80 is Harn-only HTTP
# (that SG port is already open). Other clients get HTTPS redirect.
CADDY_TMP="$(mktemp)"
cat > "${CADDY_TMP}" <<EOF
{
	auto_https disable_redirects
}

${DOMAIN} {
	tls internal
	reverse_proxy 127.0.0.1:8080
}

:${INTERNAL_HTTP_PORT} {
	@harn remote_ip ${INTERNAL_ALLOW_IP}
	handle @harn {
		reverse_proxy 127.0.0.1:8080
	}
	handle {
		redir https://${DOMAIN}{uri} permanent
	}
}
EOF
scp -q "${CADDY_TMP}" "${REMOTE}:/tmp/Caddyfile"
rm -f "${CADDY_TMP}"
ssh "${REMOTE}" bash -s <<REMOTE_CADDY
set -euo pipefail
# Caddy runs as user caddy. scp + mv keeps the ubuntu 0600 temp file, so
# install as root:caddy 0640 (Debian package layout).
sudo install -m 640 -o root -g caddy /tmp/Caddyfile /etc/caddy/Caddyfile
rm -f /tmp/Caddyfile
sudo systemctl enable caddy
if ! sudo systemctl restart caddy; then
  sudo journalctl -u caddy -n 40 --no-pager >&2 || true
  exit 1
fi
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

upsert_env_kv() {
  local file="$1"
  local key="$2"
  local value="$3"
  local tmp found=0 line
  mkdir -p "$(dirname "${file}")"
  tmp="$(mktemp)"
  chmod 600 "${tmp}"
  if [[ -f "${file}" ]]; then
    while IFS= read -r line || [[ -n "${line}" ]]; do
      if [[ "${line}" == "${key}="* ]]; then
        printf '%s=%s\n' "${key}" "${value}"
        found=1
      elif [[ "${line}" == "export ${key}="* ]]; then
        printf 'export %s=%s\n' "${key}" "${value}"
        found=1
      else
        printf '%s\n' "${line}"
      fi
    done < "${file}" > "${tmp}"
  fi
  if [[ "${found}" -eq 0 ]]; then
    printf '%s=%s\n' "${key}" "${value}" >> "${tmp}"
  fi
  mv "${tmp}" "${file}"
  chmod 600 "${file}"
}

sync_harn_gateway_env() {
  local gateway_url="https://${DOMAIN}"
  local origin_url="http://${REMOTE_HOST}"
  if [[ "${INTERNAL_HTTP_PORT}" != "80" ]]; then
    origin_url="http://${REMOTE_HOST}:${INTERNAL_HTTP_PORT}"
  fi
  local harn_env="${HOME}/.harn/server.env"
  echo "Syncing GATEWAY_MASTER_KEY into ${harn_env}"
  # Local cargo-run Harn is not on INTERNAL_ALLOW_IP. Origin :80 301s those
  # clients to Cloudflare and /v1/models then fails JSON parse.
  upsert_env_kv "${harn_env}" SUPERGLUE_GATEWAY_URL "${gateway_url}"
  upsert_env_kv "${harn_env}" SUPERGLUE_GATEWAY_ORIGIN_URL "${gateway_url}"
  upsert_env_kv "${harn_env}" GATEWAY_MASTER_KEY "${MASTER_KEY}"
  upsert_env_kv "${harn_env}" OPENAI_BASE_URL "${gateway_url}"
  upsert_env_kv "${harn_env}" HARN_GATEWAY_ALLOWED_MODELS \
    "${DEFAULT_ALLOWED_MODELS}"

  if systemctl --user is-active --quiet harn-server 2>/dev/null; then
    echo "Restarting local systemd user unit harn-server"
    systemctl --user restart harn-server
  fi

  local harn_remote=""
  if [[ -f "${HOME}/.harn/ec2-instance.env" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/.harn/ec2-instance.env"
    if [[ -n "${HARN_EC2_PUBLIC_IP:-}" ]]; then
      harn_remote="${HARN_EC2_USER:-ubuntu}@${HARN_EC2_PUBLIC_IP}"
    fi
  fi
  if [[ -z "${harn_remote}" && -f "${HOME}/.harn/remote-server.env" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/.harn/remote-server.env"
    harn_remote="${HARN_REMOTE:-}"
    if [[ -z "${harn_remote}" && -n "${HARN_REMOTE_HOST:-}" ]]; then
      harn_remote="${HARN_REMOTE_USER:-ubuntu}@${HARN_REMOTE_HOST}"
    fi
  fi
  if [[ -z "${harn_remote}" ]]; then
    echo "note: no Harn remote host in ~/.harn/ec2-instance.env or ~/.harn/remote-server.env"
    return
  fi

  echo "Syncing GATEWAY_MASTER_KEY to Harn host ${harn_remote}"
  if ! ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "${harn_remote}" bash -s <<REMOTE_HARN_ENV
set -euo pipefail
FILE="\$HOME/.harn/server.env"
mkdir -p "\$HOME/.harn"
touch "\$FILE"
chmod 600 "\$FILE"
upsert() {
  local key="\$1"
  local value="\$2"
  local tmp found=0 line
  tmp="\$(mktemp)"
  chmod 600 "\$tmp"
  while IFS= read -r line || [[ -n "\$line" ]]; do
    if [[ "\$line" == "\${key}="* ]]; then
      printf '%s=%s\n' "\$key" "\$value"
      found=1
    else
      printf '%s\n' "\$line"
    fi
  done < "\$FILE" > "\$tmp"
  if [[ "\$found" -eq 0 ]]; then
    printf '%s=%s\n' "\$key" "\$value" >> "\$tmp"
  fi
  mv "\$tmp" "\$FILE"
  chmod 600 "\$FILE"
}
upsert SUPERGLUE_GATEWAY_URL '${gateway_url}'
upsert SUPERGLUE_GATEWAY_ORIGIN_URL '${origin_url}'
upsert GATEWAY_MASTER_KEY '${MASTER_KEY}'
upsert OPENAI_BASE_URL '${origin_url}'
upsert HARN_GATEWAY_ALLOWED_MODELS '${DEFAULT_ALLOWED_MODELS}'
if systemctl is-active --quiet harn-server 2>/dev/null; then
  sudo systemctl restart harn-server
fi
REMOTE_HARN_ENV
  then
    echo "warning: could not sync GATEWAY_MASTER_KEY to Harn host ${harn_remote}" >&2
  else
    echo "Synced Harn host ${harn_remote}"
  fi
}

sync_harn_gateway_env

echo ""
echo "Gateway configured on ${REMOTE}"
echo "  Local health: OK (http://127.0.0.1:8080/health)"
echo "  Harn origin:  http://${REMOTE_HOST}:${INTERNAL_HTTP_PORT} (allow ${INTERNAL_ALLOW_IP})"
if [[ "${HTTPS_OK}" -eq 1 ]]; then
  echo "  Public URL:   https://${DOMAIN}/health OK"
else
  echo "  Public URL:   https://${DOMAIN}/health not ready yet"
  echo "  Check:        Cloudflare proxied A -> ${REMOTE_HOST}, SSL/TLS mode Full"
  echo "  Debug:        ssh ${REMOTE} 'sudo journalctl -u caddy -n 40 --no-pager'"
fi
echo "  Operator env: ${LOCAL_ENV}"
echo "  Harn env:     ${HOME}/.harn/server.env (synced; remote Harn restarted when reachable)"
echo "  If local Harn is cargo run, restart it in a clean shell so it loads ~/.harn/server.env"
echo ""
echo "Create a virtual key (example):"
echo "  source ${LOCAL_ENV}"
echo "  curl -sS -X POST \"\${SUPERGLUE_GATEWAY_URL}/v1/budgets\" \\"
echo "    -H \"X-Superglue-Key: Bearer \${GATEWAY_MASTER_KEY}\" \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"max_budget\": 50.0, \"duration_sec\": 2592000, \"enforce\": true}'"
