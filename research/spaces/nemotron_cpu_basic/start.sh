#!/bin/sh
set -eu
# Private HF Space is the authentication boundary. No token is put in this image,
# argv, logs, or model URL; no public/protected Space deployment is permitted.
case "${H3_ASR_EXPIRES_AT:-}" in
  ''|*[!0-9]*) echo 'Missing immutable finite admission deadline'; exit 2 ;;
esac
remaining=$((H3_ASR_EXPIRES_AT - $(date +%s)))
if [ "$remaining" -le 0 ]; then
  echo 'Admission expired; no model load'
  exec sleep infinity
fi
if [ "$remaining" -gt 1800 ]; then remaining=1800; fi
set +e
timeout --signal=TERM --kill-after=10 "$remaining" \
  /opt/nemo/bin/nemo-speech --quiet serve \
  --host 0.0.0.0 --port 7860 --threads 2 --no-ui \
  --device cpu --asr-model /opt/models/nemotron.gguf \
  --asr.batching.enabled=false --asr.batching.state_arena_slots 1 \
  --asr.endpointing.enable=true --asr.endpointing.stop_history_eou_ms 800 \
  --max-upload-mb 2 --read-timeout 30 --write-timeout 30
status=$?
set -e
printf 'Finite ASR owner ended (exit=%s); no automatic model restart\n' "$status"
# Keep the container inert if HF tries to restart a failed main process. The
# launcher/operator pauses the Space after tests. No inference after timeout.
exec sleep infinity
