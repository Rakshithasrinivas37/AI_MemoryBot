#!/usr/bin/env bash
set -euo pipefail
image="${1:-memorybot:ci}"
container="memorybot-smoke-${RANDOM}"
cleanup() { docker logs "$container" || true; docker rm -f "$container" >/dev/null || true; }
trap cleanup EXIT
docker run -d --name "$container" "$image" >/dev/null
for attempt in $(seq 1 60); do
  status="$(docker inspect --format '{{.State.Health.Status}}' "$container")"
  if [ "$status" = healthy ]; then break; fi
  if [ "$(docker inspect --format '{{.State.Running}}' "$container")" != true ]; then exit 1; fi
  sleep 3
done
[ "$(docker inspect --format '{{.State.Health.Status}}' "$container")" = healthy ]
docker exec -i "$container" python - <<'PY'
import json
import os
import urllib.request
from pathlib import Path
assert os.getuid() != 0
base = 'http://127.0.0.1:8000'
assert json.load(urllib.request.urlopen(base + '/health')) == {'status': 'ok'}
assert b'MemoryBot' in urllib.request.urlopen(base + '/').read()
assert b'memory-detail' in urllib.request.urlopen(base + '/chat.html').read()
assert b'MEMORYBOT_API_BASE' in urllib.request.urlopen(base + '/api-config.js').read()
for directory in ['/data/chroma_db', '/data/evaluation', '/home/app/.cache/huggingface']:
    file = Path(directory) / '.smoke-test'
    file.write_text('ok')
    file.unlink()
print('PASS: healthy non-root container, frontend/API routing, writable data/cache')
PY
