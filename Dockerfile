# syntax=docker/dockerfile:1.7
# LIL "Karmic Kraken" BETA (2026-10-03) Qwen3.8 Flash-Next runtime rebuilt for GB10 (aarch64, sm_121a).
# vLLM integration/karmic-kraken-beta @1286ae9, b12x beta @78ee52c, FlashInfer beta @dbd6238 (reads qad-step5500-ple1000).
# Same components as the x86 KK image: NGC PyTorch 26.08 foundation, LIL vLLM fork,
# b12x (SparkInfer), LIL FlashInfer, InstantTensor, qwen38-runtime pins, KK python patches.
ARG BASE=nvcr.io/nvidia/pytorch@sha256:237ecf9ac7373daf91b31bb4f86651ce1ce57b676366ed435aa1aba61dad81d5

FROM ${BASE} AS build
ARG BUILD_JOBS=4
RUN apt-get update && apt-get install -y --no-install-recommends libdw-dev git && rm -rf /var/lib/apt/lists/*
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.95.0 --profile minimal
ENV PATH=/root/.cargo/bin:/build-venv/bin:${PATH} \
    TORCH_CUDA_ARCH_LIST=12.1a \
    CMAKE_CUDA_ARCHITECTURES=121a \
    MAX_JOBS=${BUILD_JOBS} \
    CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} \
    NVCC_THREADS=2
RUN pip install --no-cache-dir uv==0.11.30 \
 && uv venv --python /usr/bin/python --system-site-packages /build-venv \
 && uv pip install --python /build-venv/bin/python cmake==3.31.6 jinja2==3.1.6 ninja==1.13.0 \
      packaging==26.2 semantic-version==2.10.0 setuptools==80.10.2 setuptools-rust==1.13.0 \
      setuptools-scm==10.2.1 vcs-versioning==2.3.1 wheel==0.47.0
COPY vllm /src/vllm
RUN git config --global --add safe.directory '*' \
 && cd /src/vllm && env -u PYTHONPATH VLLM_TARGET_DEVICE=cuda \
    CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=121a -DVLLM_BUILD_CUTLASS_SCALED_MM_C2X=OFF" \
    uv build --wheel --no-build-isolation --python /build-venv/bin/python --out-dir /wheelhouse /src/vllm
COPY InstantTensor /src/InstantTensor
RUN cd /src/InstantTensor && (test -x checkout_submodules.sh && ./checkout_submodules.sh || true) \
 && /build-venv/bin/python -m pip wheel --no-build-isolation --no-deps -w /wheelhouse . \
 || echo "INSTANTTENSOR_BUILD_FAILED" > /wheelhouse/instanttensor.failed

FROM ${BASE} AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends liburing2 libnuma1 libaio1t64 patch && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.11.30 && uv venv --python /usr/bin/python --system-site-packages /opt/venv
ENV PATH=/opt/venv/bin:${PATH} VIRTUAL_ENV=/opt/venv
COPY blackwell-llm-docker/tools/jovian_wheel_runtime/qwen38-runtime.in /tmp/qwen38-runtime.in
COPY --from=build /wheelhouse /wheelhouse
RUN uv pip install --python /opt/venv/bin/python --no-deps -r /tmp/qwen38-runtime.in \
 && uv pip install --python /opt/venv/bin/python --no-deps quack-kernels==0.6.4 \
 && (uv pip install --python /opt/venv/bin/python --no-deps torch-c-dlpack-ext==0.1.5 || echo "no aarch64 torch-c-dlpack-ext") \
 && uv pip install --python /opt/venv/bin/python --no-deps /wheelhouse/*.whl \
      https://github.com/local-inference-lab/b12x/releases/download/b12x-cu134-beta-78ee52c302abcef019d7ce17634c80b76f0975c9/b12x-1.3.0-py3-none-any.whl \
      "https://github.com/local-inference-lab/flashinfer/releases/download/flashinfer-cu134-sm120-beta-dbd6238c6655b98195fdf77f04bba6facf5a38a4/flashinfer_python-0.6.18%2Blil.cu134.sm120.gdbd6238c6655-py3-none-any.whl"
COPY blackwell-llm-docker/recipes/glm53 /opt/kk-patches
RUN cd / && patch -p1 --forward -r - usr/local/lib/python3.12/dist-packages/torch/_library/utils.py < /opt/kk-patches/torch-schema-enumeration.patch ; \
    for f in /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/dsl_packages/cutlass/base_dsl/jit_executor.py \
             /opt/venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/dsl_packages/cutlass/base_dsl/jit_executor.py; do \
      test -f "$f" && patch --forward -r - "$f" < /opt/kk-patches/cutlass-sentinel-identity.patch ; done ; \
    python -c "import vllm, b12x, flashinfer; print('vllm', vllm.__version__, 'b12x ok, flashinfer', flashinfer.__version__)"
ENTRYPOINT []
CMD ["python", "-m", "vllm.entrypoints.cli.main", "--help"]
