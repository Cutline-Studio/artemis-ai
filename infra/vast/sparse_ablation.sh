#!/usr/bin/env bash
# Matched-token ablation on one 4-GPU Vast box: 100m (GPUs 0,1) against 100m-s (GPUs 2,3), same data and seed.
# Runs as the container command. Needs env: DPH (box $/h), CAP_USD, MAX_HOURS, STEPS, and the kill switch pair
# ARTEMIS_KILLSWITCH_STATE_URL + ARTEMIS_KILLSWITCH_STATE_PUBKEY_B64 (base64 of the public key PEM). The watchdog destroys the box
# at the cap, the time limit, or 20 minutes after both arms finish. Logs go to stdout so the Vast log endpoint shows them.
set -u
W=/w; mkdir -p $W/runs; cd "$(dirname "$0")/../.."
START=$(date +%s)
destroy() {
  echo "WATCHDOG destroying instance: $1" >&2
  for i in 1 2 3 4 5; do
    python - <<PY && break
import os, urllib.request
r = urllib.request.Request(f"https://console.vast.ai/api/v0/instances/{os.environ['CONTAINER_ID']}/", method="DELETE",
                           headers={"Authorization": "Bearer " + os.environ["CONTAINER_API_KEY"]})
print(urllib.request.urlopen(r, timeout=30).read()[:200])
PY
    sleep 20
  done
}
watchdog() {
  local done_at=0
  while true; do
    now=$(date +%s); el=$((now - START))
    cost=$(python -c "print($el / 3600 * $DPH)")
    if python -c "import sys; sys.exit(0 if $cost >= $CAP_USD else 1)"; then destroy "cap reached (\$$cost)"; fi
    if [ $el -ge $((MAX_HOURS * 3600)) ]; then destroy "time limit"; fi
    if [ -f $W/DONE ] && [ $done_at -eq 0 ]; then done_at=$now; fi
    if [ $done_at -gt 0 ] && [ $((now - done_at)) -ge 1200 ]; then destroy "finished"; fi
    if [ $((el % 600)) -lt 60 ]; then echo "WATCHDOG elapsed ${el}s cost \$$cost"; fi
    sleep 60
  done
}
watchdog &

pip install -q tokenizers pyyaml pyarrow cryptography numpy 2>&1 | tail -1
printf '%s' "$ARTEMIS_KILLSWITCH_STATE_PUBKEY_B64" | base64 -d > $W/state.pub.pem
export ARTEMIS_KILLSWITCH_STATE_PUBKEY=$W/state.pub.pem
python -c "from artemis.killswitch import KillSwitch; print('KILLSWITCH', KillSwitch.from_env().check())" || { destroy "kill switch not running; not training"; exit 1; }

python infra/vast/prep_fineweb.py --out $W/data --shards ${SHARDS:-2} 2>&1 | sed -u 's/^/PREP /'
[ -f $W/data/manifest.json ] || { destroy "data prep failed"; exit 1; }

ARGS="--data $W/data --steps $STEPS --batch 4 --accum 6 --lr 3e-4 --warmup 450 --route-coef 0 --eval-every 500 --eval-batches 20 --ckpt-every 500 --seed 1337"
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port 29500 -m artemis.train --size 100m --out $W/runs/100m $ARGS 2>&1 | sed -u 's/^/A100M /' &
A=$!
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --master_port 29501 -m artemis.train --size 100m-s --out $W/runs/100m-s $ARGS 2>&1 | sed -u 's/^/B100MS /' &
B=$!
wait $A; wait $B
for r in 100m 100m-s; do echo "FINAL $r $(grep val_loss $W/runs/$r/metrics.jsonl | tail -1 | cut -c1-300)"; done
touch $W/DONE
wait
