#!/bin/bash
# CPU prototype of the neural core: corpus -> tokenizer -> shards -> three runs -> evaluations.
#   proto           Prime Core + 10 experts, routing supervision on labeled tokens
#   proto-unsup     same architecture, routing supervision off (control: do named experts specialize on their own?)
#   proto-dense     dense baseline with matched active weights per token, same tokens and schedule
# Each run stops halfway and resumes from its checkpoint, so resume is exercised for real.
set -euo pipefail
cd "$(dirname "$0")/../.."
R=${RUNS:-runs}; STEPS=${STEPS:-2000}; HALF=$((STEPS / 2))
COMMON="--data $R/data-proto --batch 16 --lr 1e-3 --warmup 100 --eval-every 250 --eval-batches 20 --ckpt-every 250"
if [ ! -f "$R/data-proto/manifest.json" ]; then
python3 - "$R" <<'PY'
import sys, sysconfig
from pathlib import Path
from artemis.corpus import write
from artemis.tokenizer import train_tokenizer
from artemis.data import prepare
r = Path(sys.argv[1])
src = write(r / "corpus", 3000, seed=0)
g = r / "corpus" / "general-python-stdlib.txt"
g.write_text("\n".join(p.read_text(errors="ignore") for p in sorted(Path(sysconfig.get_paths()["stdlib"]).glob("*.py"))[:80]))
src.append({"path": str(g), "license": "PSF-2.0", "origin": "CPython 3.11 standard library source, first 80 modules (unlabeled general text)"})
train_tokenizer([s["path"] for s in src], 4096, r / "tok-proto.json")
prepare(src, str(r / "tok-proto.json"), str(r / "data-proto"), val_fraction=0.05)
PY
fi
for run in proto proto-unsup proto-dense; do
  size=${run/-unsup/}; extra=""; [ "$run" = proto-unsup ] && extra="--route-coef 0"
  python3 -m artemis.train --size "$size" --out "$R/$run" --steps "$STEPS" --stop-at "$HALF" $COMMON $extra
  python3 -m artemis.train --size "$size" --out "$R/$run" --steps "$STEPS" $COMMON $extra
done
python3 -m artemis.evaluate --checkpoint "$R/proto/latest.pt" --data "$R/data-proto" --out "$R/proto/eval.json" --batches 40 --baseline "$R/proto-dense/latest.pt"
python3 -m artemis.evaluate --checkpoint "$R/proto-unsup/latest.pt" --data "$R/data-proto" --out "$R/proto-unsup/eval.json" --batches 40
echo done
