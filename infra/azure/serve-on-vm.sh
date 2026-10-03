#!/bin/bash
# Run on the GPU VM: serve the Artemis foundation plus one LoRA adapter per specialist brain with vLLM.
# Each brain is addressed by name (model="saturn", model="neptune", ...); the base model answers as "artemis".
# Models with expert layers are exported as Qwen2-MoE (artemis/export.py; logits verified equal to ours on CPU).
# Not yet verified on a GPU: vLLM loading this export, and LoRA adapters on it. Check both on the first GPU run.
set -euo pipefail
MODEL_DIR=${MODEL_DIR:-serve/artemis}
pip install -q vllm
MODULES=""
for b in apollo mercury venus earth mars jupiter saturn uranus neptune pluto; do
  [ -d "$MODEL_DIR/adapters/$b" ] && MODULES="$MODULES $b=$MODEL_DIR/adapters/$b"
done
exec vllm serve "$MODEL_DIR" --served-model-name artemis --tensor-parallel-size 8 \
  --enable-lora --max-loras 10 --max-lora-rank 64 ${MODULES:+--lora-modules $MODULES} \
  --host 0.0.0.0 --port 8000 --api-key "${ARTEMIS_INFERENCE_KEY:?set ARTEMIS_INFERENCE_KEY}"
