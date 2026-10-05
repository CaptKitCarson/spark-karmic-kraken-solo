# LIL Karmic Kraken beta on DGX Spark (arm64, sm_121a), single node

Qwen3.8-Flash-Next-NVFP4 (5500 checkpoint) on ONE Spark, TP1. Built and run 2026-10-03.

Pins: vLLM `integration/karmic-kraken-beta` @1286ae9, b12x beta @78ee52c, FlashInfer beta @dbd6238,
blackwell-llm-docker tag `karmic-kraken-beta-d449...` (@353efc6), NGC PyTorch 26.08 base (pinned digest).

## Files
- `build.sh` - fetches the sources at exact commits and builds `lil-kk-qwen38:arm64-sm121-beta20261003`, then the `-uring` layer. BUILD_JOBS=4 by default; about 2 h 15 min.
- `Dockerfile` - two-stage build: vLLM + InstantTensor wheels for 121a, then the runtime with LIL's b12x + FlashInfer wheels and the two KK patches from blackwell-llm-docker.
- `Dockerfile.uring` - thin layer on top: `liburing-dev` + `pkg-config`. b12x JIT-compiles its io_uring PLE disk loader at first start and fails without them. Tag it `...-beta20261003-uring`.
- `run-lil-5500-solo.sh` - generated from LIL `runtime.launcher --print-config` (profile qwen38-flash-next, TP1), then Spark overrides:
  `VLLM_PLE_TABLE_MEMORY=disk`, `--load-format safetensors`, KV 8 GiB, GMU 0.80, max 4 seqs, MTP 3.
  `--security-opt seccomp=unconfined` is required: Docker's default seccomp profile blocks io_uring.
  Expects the model already downloaded to `~/.cache/huggingface` (HF_HUB_OFFLINE=1); mounts this directory for the template and `~/kk-cache` for JIT caches.
- `qwen38fn-chat-template.jinja` - our patched chat template (mounted at /serving).
- `memguard-solo.sh` - optional watchdog: kills the container if MemFree drops under 2 GB. Unified memory can hard-lock a Spark before the OOM killer acts.

## Gotchas
- eugr's spark-vllm images fail on the 5500 PLE layout (`expected (2500012, 160), got (2500012, 80)`); you need the beta pins.
- First start compiles a lot of JIT kernels; the `/cache` mount keeps them between restarts.

## Use
```bash
./build.sh                       # ~2 h on a Spark
huggingface-cli download local-inference-lab/Qwen3.8-Flash-Next-NVFP4
./run-lil-5500-solo.sh           # OpenAI API on :8000, model name qwen3.8-flash-next
setsid nohup ./memguard-solo.sh >/dev/null 2>&1 &   # optional
```
Unofficial; all credit to Local Inference Lab for vLLM/b12x/FlashInfer. This is just how we got it running on one Spark.
