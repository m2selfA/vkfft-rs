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
run_app="$upstream_dir/vkFFT/vkFFT/vkFFT_AppManagement/vkFFT_RunApp.h"
sample50="$upstream_dir/benchmark_scripts/vkFFT_scripts/src/sample_50_convolution_VkFFT_single_1d_matrix.cpp"
sample51="$upstream_dir/benchmark_scripts/vkFFT_scripts/src/sample_51_convolution_VkFFT_single_3d_matrix_zeropadding_r2c.cpp"
sample52="$upstream_dir/benchmark_scripts/vkFFT_scripts/src/sample_52_convolution_VkFFT_single_2d_batched_r2c.cpp"

# Kernel preparation is a forward FFT mode with application-convolution capacity knobs,
# not an application midpoint. Official Real convolution samples configure R2C and
# kernelConvolution together, even though their demo FFT invocation is commented out.
grep -F 'pfUINT kernelConvolution;// specify if this application is used to create kernel for convolution, so it has the same properties. performConvolution has to be set to 0 for kernel creation' "$structs" >/dev/null
grep -F 'pfUINT performR2C; //perform R2C/C2R decomposition (0 - off, 1 - on)' "$structs" >/dev/null
grep -F 'if (inputLaunchConfiguration.performR2C != 0) {' "$initialize" >/dev/null
grep -F 'app->configuration.performR2C = inputLaunchConfiguration.performR2C;' "$initialize" >/dev/null
grep -F 'if (inputLaunchConfiguration.kernelConvolution != 0) {' "$initialize" >/dev/null
grep -F 'app->configuration.kernelConvolution = inputLaunchConfiguration.kernelConvolution;' "$initialize" >/dev/null
grep -F 'app->configuration.reorderFourStep = 0;' "$initialize" >/dev/null
grep -F 'app->configuration.registerBoost = 1;' "$initialize" >/dev/null
grep -F 'configuration.kernelConvolution = true;' "$sample50" >/dev/null
grep -F 'configuration.coordinateFeatures = 9;' "$sample50" >/dev/null
grep -F 'uint64_t kernelSize = ((uint64_t)configuration.coordinateFeatures) * sizeof(float) * 2 * (configuration.size[0])' "$sample50" >/dev/null
grep -F 'configuration.kernelConvolution = true;' "$sample51" >/dev/null
grep -F 'configuration.performR2C = true;' "$sample51" >/dev/null
grep -F 'configuration.coordinateFeatures = 9;' "$sample51" >/dev/null
grep -F 'configuration.performZeropadding[0] = true;' "$sample51" >/dev/null
grep -F 'configuration.performZeropadding[1] = true;' "$sample51" >/dev/null
grep -F 'configuration.performZeropadding[2] = true;' "$sample51" >/dev/null
grep -F 'uint64_t kernelSize = ((uint64_t)configuration.coordinateFeatures) * sizeof(float) * 2 * (configuration.size[0] / 2 + 1) * configuration.size[1] * configuration.size[2];' "$sample51" >/dev/null
grep -F 'for (uint64_t v = 0; v < configuration.coordinateFeatures; v++) {' "$sample51" >/dev/null
grep -F 'v * (configuration.size[0] + 2) * configuration.size[1] * configuration.size[2]' "$sample51" >/dev/null
grep -F 'configuration.kernelConvolution = true;' "$sample52" >/dev/null
grep -F 'configuration.performR2C = true;' "$sample52" >/dev/null
grep -F 'configuration.FFTdim = 2;' "$sample52" >/dev/null
grep -F 'configuration.size[0] = 32;' "$sample52" >/dev/null
grep -F 'configuration.size[1] = 32;' "$sample52" >/dev/null
grep -F 'configuration.coordinateFeatures = 2;' "$sample52" >/dev/null
grep -F 'configuration.numberBatches = 2;' "$sample52" >/dev/null
grep -F 'uint64_t kernelSize = ((uint64_t)configuration.numberBatches) * configuration.coordinateFeatures * sizeof(float) * 2 * (configuration.size[0] / 2 + 1)' "$sample52" >/dev/null
grep -F 'for (uint64_t f = 0; f < configuration.numberBatches; f++) {' "$sample52" >/dev/null
grep -F 'for (uint64_t v = 0; v < configuration.coordinateFeatures; v++) {' "$sample52" >/dev/null
grep -F 'f * configuration.coordinateFeatures * (configuration.size[0] + 2)' "$sample52" >/dev/null
grep -F 'dispatchBlock[2] = app->configuration.coordinateFeatures * app->configuration.numberBatches;' "$run_app" >/dev/null

printf 'upstream kernelConvolution source: 32/32 contracts matched at %s\n' "$expected_commit"
