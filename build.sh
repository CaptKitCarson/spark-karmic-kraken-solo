#!/bin/bash
# Build LIL Karmic Kraken BETA for GB10 (arm64, sm_121a). Run from this directory. Log: ./build.log
# Takes about 2 h 15 min with BUILD_JOBS=4 (raise it if nothing else is using the box's memory).
set -u
T=karmic-kraken-beta-d4499365dba23083d2003d0e77ce487dccdce5d2e15ab00cfe72119d60803e3d
cd "$(dirname "$0")"
log() { echo "[$(date +%T)] $*" | tee -a build.log; }
log "sources"
fetch() { # repo dir ref
  rm -rf "$2"; mkdir -p "$2"; git -C "$2" init -q; git -C "$2" remote add origin "https://github.com/local-inference-lab/$1.git"
  git -C "$2" fetch -q --depth 1 origin "$3" && git -C "$2" checkout -q FETCH_HEAD
}
fetch vllm vllm 1286ae9c9a3fd9376b82da39f7bc87f6a99037d9 || { log "vllm fetch FAILED"; exit 1; }
# setuptools-scm needs a version tag in a shallow clone
git -C vllm tag -f v0.1.0 >/dev/null
fetch InstantTensor InstantTensor 95d4729b6d6a991bb8de61877147a9d9d9100b23 || { log "InstantTensor fetch FAILED"; exit 1; }
fetch blackwell-llm-docker blackwell-llm-docker "refs/tags/$T" || { log "blackwell-llm-docker fetch FAILED"; exit 1; }
log "vllm $(git -C vllm rev-parse --short HEAD), InstantTensor $(git -C InstantTensor rev-parse --short HEAD), bld $(git -C blackwell-llm-docker rev-parse --short HEAD)"
ls blackwell-llm-docker/recipes/glm53/*.patch blackwell-llm-docker/tools/jovian_wheel_runtime/qwen38-runtime.in >> build.log 2>&1 || log "WARN patch/runtime files moved"
log "docker build start"
DOCKER_BUILDKIT=1 nice -n 10 docker build --build-arg BUILD_JOBS=${BUILD_JOBS:-4} -t lil-kk-qwen38:arm64-sm121-beta20261003 . >> build.log 2>&1
rc=$?
log "BUILD_EXIT=$rc"
[ $rc -eq 0 ] && docker build -f Dockerfile.uring -t lil-kk-qwen38:arm64-sm121-beta20261003-uring . >> build.log 2>&1 && log "uring layer OK"
