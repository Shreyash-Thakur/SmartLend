#!/bin/bash
# First-boot self-configuration: everything we typed over SSH in Phase 2,
# automated. Runs once as root; log lives at /var/log/cloud-init-output.log
# on the instance if boot debugging is ever needed.
set -euxo pipefail

# --- Docker -----------------------------------------------------------------
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io
systemctl enable --now docker

# --- 2G swap: the overnight-wedge lesson, baked in --------------------------
fallocate -l 2G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab

# --- AWS CLI (for the ECR login; credentials come from the instance role) ---
snap install aws-cli --classic

# --- Pull and run the app ----------------------------------------------------
aws ecr get-login-password --region ${region} \
  | docker login --username AWS --password-stdin $(echo ${ecr_image} | cut -d/ -f1)
docker pull ${ecr_image}
docker run -d --name smartlend -p 80:8000 --restart unless-stopped \
  -e SMARTLEND_DATABASE_URL="${db_url}" \
  ${ecr_image}
