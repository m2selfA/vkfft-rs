#!/bin/sh
set -eu

expected_commit=066a17c17068c0f11c9298d848c2976c71fad1c1
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
upstream_dir=${VKFFT_UPSTREAM_DIR:-"$repo_root/target/upstream-vkfft"}

if [ ! -d "$upstream_dir/.git" ]; then
    echo "missing pinned upstream VkFFT clone: $upstream_dir" >&2
    exit 2
fi
actual_commit=$(git -C "$upstream_dir" rev-parse HEAD)
if [ "$actual_commit" != "$expected_commit" ]; then
    echo "upstream VkFFT commit mismatch: expected $expected_commit, got $actual_commit" >&2
    exit 2
fi

structs="$upstream_dir/vkFFT/vkFFT/vkFFT_Structs/vkFFT_Structs.h"
initialize="$upstream_dir/vkFFT/vkFFT/vkFFT_AppManagement/vkFFT_InitializeApp.h"
api_params="$upstream_dir/vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_InitAPIParameters.h"
convolution="$upstream_dir/vkFFT/vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel1/PrePostProcessing/vkFFT_Convolution.h"

# Pinned C2C application/mixed-storage composition contract:
# 1. performConvolution is a general application flag and does not name a precision restriction.
# 2. halfPrecisionMemoryOnly keeps compute/kernel memory in FP32 while only the true caller
#    input/output boundaries become FP16.
# 3. doublePrecisionFloatMemory selects FP64 arithmetic with FP32 memory storage.
# 4. convolutionStep consumes the specialization memory types; there is no joint precision x
#    performConvolution/convolutionStep branch in the pinned implementation.
grep -F 'pfUINT performConvolution; //perform convolution in this application (0 - off, 1 - on). Disables reorderFourStep parameter' "$structs" >/dev/null
grep -F 'pfUINT halfPrecisionMemoryOnly; //use half precision only as input/output buffer.' "$structs" >/dev/null
grep -F 'pfUINT doublePrecisionFloatMemory; //use FP64 precision for all calculations, while all memory storage is done in FP32.' "$structs" >/dev/null
grep -F 'if (app->configuration.halfPrecisionMemoryOnly) {' "$api_params" >/dev/null
grep -F 'sc->floatTypeKernelMemoryCode = 12;' "$api_params" >/dev/null
grep -F 'sc->floatTypeInputMemoryCode = 02;' "$api_params" >/dev/null
grep -F 'sc->floatTypeOutputMemoryCode = 02;' "$api_params" >/dev/null
grep -F 'if (app->configuration.doublePrecisionFloatMemory) {' "$api_params" >/dev/null
grep -F 'sc->floatTypeCode = 22;' "$api_params" >/dev/null
grep -F 'sc->floatTypeInputMemoryCode = 12;' "$api_params" >/dev/null
grep -F 'sc->floatTypeOutputMemoryCode = 12;' "$api_params" >/dev/null
grep -F 'axis->specializationConstants.convolutionStep = 1;' "$upstream_dir/vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_Plans/vkFFT_Plan_FFT.h" >/dev/null
if grep -R -E '(halfPrecisionMemoryOnly|doublePrecisionFloatMemory).*(performConvolution|convolutionStep)|(performConvolution|convolutionStep).*(halfPrecisionMemoryOnly|doublePrecisionFloatMemory)' "$upstream_dir/vkFFT/vkFFT" >/dev/null; then
    echo 'performConvolution unexpectedly has a mixed-storage joint branch' >&2
    exit 3
fi
if grep -E 'halfPrecisionMemoryOnly|doublePrecisionFloatMemory' "$convolution" >/dev/null; then
    echo 'convolutionStep body unexpectedly branches on mixed-storage configuration' >&2
    exit 3
fi

printf 'upstream C2C performConvolution mixed-storage: 14/14 source contracts matched at %s\n' "$expected_commit"
