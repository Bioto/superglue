#!/usr/bin/env bash
# Launch an Ubuntu 24.04 EC2 instance for the Superglue LLM gateway (public HTTPS).
set -euo pipefail

REGION="${AWS_REGION:-us-west-2}"
INSTANCE_TYPE="${SUPERGLUE_EC2_INSTANCE_TYPE:-t3.small}"
VOLUME_SIZE="${SUPERGLUE_EC2_VOLUME_GB:-20}"
KEY_NAME="${SUPERGLUE_EC2_KEY_NAME:-superglue-deploy}"
SG_NAME="${SUPERGLUE_EC2_SG_NAME:-superglue-gateway-sg}"
INSTANCE_NAME="${SUPERGLUE_EC2_INSTANCE_NAME:-superglue-gateway}"
DOMAIN="${SUPERGLUE_GATEWAY_DOMAIN:-gateway.myharn.sh}"

MY_IP="$(curl -sf ifconfig.me || curl -sf icanhazip.com)"
if [[ -z "${MY_IP}" ]]; then
  echo "error: could not determine public IP for security group" >&2
  exit 1
fi

echo "Region: ${REGION}"
echo "Your IP: ${MY_IP}/32"
echo "Domain:  ${DOMAIN}"

# Ubuntu 24.04 Noble (us-west-2). Override with SUPERGLUE_EC2_AMI if needed.
AMI_ID="${SUPERGLUE_EC2_AMI:-ami-0ac74609c6396bed3}"

if ! aws sts get-caller-identity --region "${REGION}" >/dev/null 2>&1; then
  echo "error: AWS credentials not configured" >&2
  exit 1
fi

PUB_KEY="${SUPERGLUE_EC2_PUBLIC_KEY:-${HOME}/.ssh/default.pub}"
if [[ ! -f "${PUB_KEY}" ]]; then
  echo "error: SSH public key not found at ${PUB_KEY}" >&2
  exit 1
fi

echo "Ensuring key pair ${KEY_NAME}..."
if ! aws ec2 describe-key-pairs --key-names "${KEY_NAME}" --region "${REGION}" >/dev/null 2>&1; then
  aws ec2 import-key-pair \
    --key-name "${KEY_NAME}" \
    --public-key-material "fileb://${PUB_KEY}" \
    --region "${REGION}"
fi

VPC_ID="$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text --region "${REGION}")"
if [[ "${VPC_ID}" == "None" || -z "${VPC_ID}" ]]; then
  echo "error: no default VPC in ${REGION}" >&2
  exit 1
fi

SG_ID="$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=${SG_NAME}" "Name=vpc-id,Values=${VPC_ID}" \
  --query 'SecurityGroups[0].GroupId' --output text --region "${REGION}" 2>/dev/null || true)"

if [[ -z "${SG_ID}" || "${SG_ID}" == "None" ]]; then
  echo "Creating security group ${SG_NAME}..."
  SG_ID="$(aws ec2 create-security-group \
    --group-name "${SG_NAME}" \
    --description "Superglue gateway - SSH + HTTP/HTTPS for Caddy" \
    --vpc-id "${VPC_ID}" \
    --query 'GroupId' --output text --region "${REGION}")"
  aws ec2 authorize-security-group-ingress \
    --group-id "${SG_ID}" \
    --protocol tcp --port 22 --cidr "${MY_IP}/32" \
    --region "${REGION}"
  aws ec2 authorize-security-group-ingress \
    --group-id "${SG_ID}" \
    --protocol tcp --port 80 --cidr "0.0.0.0/0" \
    --region "${REGION}"
  aws ec2 authorize-security-group-ingress \
    --group-id "${SG_ID}" \
    --protocol tcp --port 443 --cidr "0.0.0.0/0" \
    --region "${REGION}"
  echo "Security group ${SG_ID}: SSH from ${MY_IP}/32; HTTP/HTTPS open"
else
  echo "Using existing security group ${SG_ID}"
fi

echo "Launching ${INSTANCE_TYPE} instance..."
INSTANCE_ID="$(aws ec2 run-instances \
  --image-id "${AMI_ID}" \
  --instance-type "${INSTANCE_TYPE}" \
  --key-name "${KEY_NAME}" \
  --security-group-ids "${SG_ID}" \
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${VOLUME_SIZE},\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${INSTANCE_NAME}}]" \
  --query 'Instances[0].InstanceId' --output text --region "${REGION}")"

echo "Waiting for instance ${INSTANCE_ID}..."
aws ec2 wait instance-running --instance-ids "${INSTANCE_ID}" --region "${REGION}"

echo "Allocating Elastic IP..."
ALLOC_JSON="$(aws ec2 allocate-address --domain vpc --region "${REGION}" --output json)"
ALLOC_ID="$(echo "${ALLOC_JSON}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["AllocationId"])')"
EIP="$(echo "${ALLOC_JSON}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["PublicIp"])')"

aws ec2 associate-address \
  --instance-id "${INSTANCE_ID}" \
  --allocation-id "${ALLOC_ID}" \
  --region "${REGION}" >/dev/null

aws ec2 create-tags \
  --resources "${ALLOC_ID}" \
  --tags "Key=Name,Value=${INSTANCE_NAME}" \
  --region "${REGION}" >/dev/null || true

STATE_FILE="${HOME}/.superglue/ec2-instance.env"
mkdir -p "${HOME}/.superglue"
cat > "${STATE_FILE}" <<EOF
SUPERGLUE_EC2_INSTANCE_ID=${INSTANCE_ID}
SUPERGLUE_EC2_PUBLIC_IP=${EIP}
SUPERGLUE_EC2_EIP_ALLOC_ID=${ALLOC_ID}
SUPERGLUE_EC2_REGION=${REGION}
SUPERGLUE_EC2_KEY_NAME=${KEY_NAME}
SUPERGLUE_EC2_USER=ubuntu
SUPERGLUE_GATEWAY_DOMAIN=${DOMAIN}
EOF

echo ""
echo "EC2 instance ready:"
echo "  Instance ID: ${INSTANCE_ID}"
echo "  Elastic IP:  ${EIP}"
echo "  State file:  ${STATE_FILE}"
echo ""
echo "DNS (required for HTTPS):"
echo "  Create an A record: ${DOMAIN} -> ${EIP}"
echo "  Or run: ./deploy/setup-dns.sh"
echo ""
echo "Next: ./deploy/setup-dns.sh && ./deploy/setup-server.sh ubuntu@${EIP}"
