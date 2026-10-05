#!/bin/bash
# LIL Karmic Kraken beta (2026-10-03) + Qwen3.8-Flash-Next-NVFP4 5500 (6909a5b) on ONE Spark.
# Generated from LIL runtime.launcher --print-config (profile qwen38-flash-next, TP1); Spark overrides:
# disk PLE, safetensors loader, KV 8 GiB, GMU 0.80, 4 seqs, patched chat template.
# seccomp=unconfined: Docker's default seccomp profile blocks io_uring, which the disk PLE loader needs.
set -euo pipefail
docker rm -f vllm_node >/dev/null 2>&1 || true
mkdir -p $HOME/kk-cache
exec docker run -d --name vllm_node --gpus all --network host --ipc host --init \
  --ulimit memlock=-1 --ulimit stack=67108864 --shm-size 16g \
  --security-opt seccomp=unconfined \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface -v $HOME/kk-cache:/cache -v "$(cd "$(dirname "$0")" && pwd)":/serving:ro \
  -e B12X_COMPILE_CACHE_DIR=/cache/jit/spark-solo/qwen38-flash-next-4990d7e4c4262aeb/b12x/compile \
  -e B12X_COMPILE_WORKERS=2 \
  -e B12X_CUTE_COMPILE_CACHE_DIR=/cache/jit/spark-solo/qwen38-flash-next-4990d7e4c4262aeb/b12x/cute \
  -e B12X_DENSE_SPLITK_TURBO=1 \
  -e B12X_DYNAMIC_DETERMINISTIC_OUTPUT=0 \
  -e B12X_DYNAMIC_DIRECT_EXPERT_SCALES=1 \
  -e B12X_DYNAMIC_SKIP_SPLIT_BARRIER_RESET=1 \
  -e B12X_DYNAMIC_SPLIT_COMPUTE_MAC=224 \
  -e B12X_DYNAMIC_SPLIT_FAST_PREPARE=1 \
  -e B12X_DYNAMIC_SPLIT_LOW_SMEM=1 \
  -e B12X_DYNAMIC_SPLIT_ROUTE_COMPUTE=1 \
  -e B12X_DYNAMIC_WORK_SOURCE=persistent_grid \
  -e B12X_MHC_PDL=1 \
  -e B12X_PCIE_ONESHOT_BLOCK_LIMIT=4 \
  -e B12X_PCIE_ONESHOT_PDL=1 \
  -e B12X_PCIE_ONESHOT_THREADS=512 \
  -e CUDA_ARCH_LIST='8.0 8.6 9.0 10.0 11.0 12.0' \
  -e CUDA_BINARY_LOADER_THREAD_COUNT=8 \
  -e CUDA_CACHE_PATH=/cache/jit/spark-solo/qwen38-flash-next-4990d7e4c4262aeb/cuda \
  -e CUDA_COMPONENT_LIST='crt nvrtc driver-dev culibos-dev cudart cudart-dev nvcc tileiras cupti' \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_DRIVER_VERSION=615.65.02 \
  -e CUDA_HOME=/usr/local/cuda \
  -e CUDA_MODULE_LOADING=LAZY \
  -e CUDA_VERSION=13.4.1.012 \
  -e CUPY_CACHE_DIR=/cache/jit/spark-solo/qwen38-flash-next-4990d7e4c4262aeb/cupy \
  -e CUTE_DSL_CACHE_DIR=/cache/jit/spark-solo/qwen38-flash-next-4990d7e4c4262aeb/cute-dsl \
  -e FLASHINFER_WORKSPACE_BASE=/cache/jit/spark-solo/qwen38-flash-next-4990d7e4c4262aeb/flashinfer \
  -e FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/cache/jit/spark-solo/qwen38-flash-next-4990d7e4c4262aeb/flash-attention-cute-dsl \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 \
  -e INSTANTTENSOR_BACKEND=BUFFERED \
  -e NCCL_NET_PLUGIN=spcx \
  -e NCCL_VERSION=2.30.7+cuda13.3 \
  -e NUMBA_CACHE_DIR=/cache/jit/spark-solo/qwen38-flash-next-4990d7e4c4262aeb/numba \
  -e OMP_NUM_THREADS=2 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SAFETENSORS_FAST_GPU=1 \
  -e SPARKINFER_COMPILE_CACHE_DIR=/cache/jit/spark-solo/qwen38-flash-next-4990d7e4c4262aeb/b12x/compile \
  -e TILELANG_CACHE_DIR=/cache/jit/spark-solo/qwen38-flash-next-4990d7e4c4262aeb/tilelang \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/jit/spark-solo/qwen38-flash-next-4990d7e4c4262aeb/torchinductor \
  -e TORCHINDUCTOR_CUTLASS_DIR=/opt/pytorch/pytorch/third_party/cutlass \
  -e TORCHINDUCTOR_LOOP_ORDERING_AFTER_FUSION=0 \
  -e TORCH_CUDA_ARCH_LIST='8.0 8.6 9.0 10.0 11.0 12.0+PTX' \
  -e TORCH_EXTENSIONS_DIR=/cache/jit/spark-solo/qwen38-flash-next-4990d7e4c4262aeb/torch-extensions \
  -e TRITON_CACHE_DIR=/cache/jit/spark-solo/qwen38-flash-next-4990d7e4c4262aeb/triton \
  -e TRITON_CUDACRT_PATH=/usr/local/cuda/include \
  -e TRITON_CUDART_PATH=/usr/local/cuda/include \
  -e TRITON_CUOBJDUMP_PATH=/usr/local/cuda/bin/cuobjdump \
  -e TRITON_CUPTI_INCLUDE_PATH=/usr/local/cuda/include \
  -e TRITON_CUPTI_LIB_PATH=/usr/local/cuda/lib64 \
  -e TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm \
  -e TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
  -e TVM_CACHE_DIR=/cache/jit/spark-solo/qwen38-flash-next-4990d7e4c4262aeb/tvm \
  -e TVM_FFI_CACHE_DIR=/cache/jit/spark-solo/qwen38-flash-next-4990d7e4c4262aeb/tvm-ffi \
  -e VLLM_B12X_DENSE_ACTIVATION_MODE=auto \
  -e VLLM_CACHE_DIR=/cache/jit/spark-solo/qwen38-flash-next-4990d7e4c4262aeb/vllm \
  -e VLLM_CACHE_ROOT=/cache/jit/spark-solo/qwen38-flash-next-4990d7e4c4262aeb/vllm \
  -e VLLM_CAUSAL_CONV1D_UPDATE_HOIST=1 \
  -e VLLM_COMPUTE_NANS_IN_LOGITS=0 \
  -e VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel \
  -e VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH=1 \
  -e VLLM_LM_HEAD_A16=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -e VLLM_MTP_NVFP4_LM_HEAD=1 \
  -e VLLM_MXFP8_LM_HEAD=0 \
  -e VLLM_PLE_CPU_OFFLOAD=1 \
  -e VLLM_PLE_TABLE_MEMORY=disk \
  -e VLLM_QWEN3_8_FLASH_NEXT_MTP_COMPACT=1 \
  -e VLLM_QWEN3_8_FLASH_NEXT_OVERLAP=1 \
  -e VLLM_SSM_CONV_STATE_LAYOUT=DS \
  -e VLLM_USE_FASTOKENS=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e XDG_CACHE_HOME=/cache/jit/spark-solo/qwen38-flash-next-4990d7e4c4262aeb \
  lil-kk-qwen38:arm64-sm121-beta20261003-uring \
  /opt/venv/bin/python -m vllm.entrypoints.cli.main serve local-inference-lab/Qwen3.8-Flash-Next-NVFP4 --async-scheduling --block-size 64 --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' --decode-context-parallel-size 1 --dtype bfloat16 --enable-auto-tool-choice --enable-chunked-prefill --no-enable-flashinfer-autotune --enable-prefix-caching --enable-prompt-tokens-details --gdn-decode-kernel b12x --gpu-memory-utilization 0.80 --host 0.0.0.0 --kv-cache-dtype fp8 --no-language-model-only --linear-backend b12x --load-format safetensors --mamba-cache-mode align --mamba-ssm-cache-dtype auto --max-cudagraph-capture-size 64 --max-model-len 262144 --max-num-batched-tokens 6019 --max-num-seqs 4 --mm-encoder-tp-mode data --mm-processor-cache-gb 0.0 --moe-backend b12x --pipeline-parallel-size 1 --port 8000 --quantization modelopt_mixed --reasoning-parser qwen3 --served-model-name qwen3.8-flash-next --speculative-config '{"method":"mtp","draft_sample_method":"probabilistic","rejection_sample_method":"standard","moe_backend":"b12x","num_speculative_tokens":3}' --tensor-parallel-size 1 --tool-call-parser qwen3_xml --kv-cache-memory-bytes 8589934592 --chat-template /serving/qwen38fn-chat-template.jinja
