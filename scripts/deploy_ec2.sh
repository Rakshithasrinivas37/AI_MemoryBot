#!/usr/bin/env bash
# Run as root on EC2: sudo bash deploy_ec2.sh ghcr.io/owner/repo@sha256:...
set -euo pipefail
image="${1:?An immutable GHCR image digest is required}"
if [[ ! "$image" =~ ^ghcr\.io/[a-z0-9._/-]+@sha256:[a-f0-9]{64}$ ]]; then
  echo 'Expected a GHCR image pinned by SHA256 digest' >&2
  exit 1
fi

# Serialize deployments on the host, including deployments from other clients.
exec 9>/var/lock/memorybot-deploy.lock
flock -n 9 || { echo 'Another MemoryBot deployment is running' >&2; exit 1; }
docker info >/dev/null
if docker container inspect memorybot-rollback >/dev/null 2>&1; then
  echo 'A previous rollback container exists; inspect it before deploying again.' >&2
  exit 1
fi

# Pull first: a registry failure must not stop the existing application.
docker pull "$image"
old_saved=false
old_running=false
new_attempted=false
rollback() {
  result=$?
  trap - EXIT
  if [ "$result" -ne 0 ]; then
    echo 'Deployment failed; restoring the previous container.' >&2
    if [ "$new_attempted" = true ]; then
      docker logs --tail 100 memorybot >&2 || true
      docker rm -f memorybot >/dev/null || true
    fi
    if [ "$old_saved" = true ]; then
      if docker rename memorybot-rollback memorybot; then
        if [ "$old_running" = true ]; then
          docker start memorybot || echo 'ERROR: Could not restart previous container' >&2
        fi
      else
        echo 'ERROR: Previous container remains as memorybot-rollback; manual recovery needed' >&2
      fi
    fi
  fi
  exit "$result"
}
trap rollback EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if docker container inspect memorybot >/dev/null 2>&1; then
  old_running=$(docker inspect --format '{{.State.Running}}' memorybot)
  docker rename memorybot memorybot-rollback
  old_saved=true
  docker stop -t 60 memorybot-rollback >/dev/null
fi

new_attempted=true
docker run -d --name memorybot --init --restart unless-stopped \
  --platform linux/amd64 \
  -p 127.0.0.1:8000:8000 \
  -v memorybot-data:/data \
  -v memorybot-models:/home/app/.cache/huggingface \
  "$image"

# The Dockerfile health check calls /health inside the new container.
healthy=false
for attempt in $(seq 1 60); do
  status=$(docker inspect --format '{{.State.Health.Status}}' memorybot)
  if [ "$status" = healthy ]; then
    healthy=true
    break
  fi
  if [ "$(docker inspect --format '{{.State.Running}}' memorybot)" != true ]; then
    break
  fi
  sleep 5
done
if [ "$healthy" != true ]; then
  echo 'New container did not become healthy' >&2
  exit 1
fi

# Commit the deployment before cleanup; a cleanup failure should not undo success.
trap - EXIT INT TERM
if [ "$old_saved" = true ]; then
  docker rm memorybot-rollback >/dev/null || echo 'Warning: remove the old rollback container before the next deployment' >&2
fi
echo "MemoryBot deployed successfully: $image"
