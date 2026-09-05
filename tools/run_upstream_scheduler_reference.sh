#!/bin/sh
set -eu

expected_commit=066a17c17068c0f11c9298d848c2976c71fad1c1
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
upstream_dir=${VKFFT_UPSTREAM_DIR:-"$repo_root/target/upstream-vkfft"}
cc_bin=${CC:-cc}
cxx_bin=${CXX:-c++}

if [ ! -d "$upstream_dir/.git" ]; then
    echo "missing pinned upstream VkFFT clone: $upstream_dir" >&2
    exit 2
fi
actual_commit=$(git -C "$upstream_dir" rev-parse HEAD)
if [ "$actual_commit" != "$expected_commit" ]; then
    echo "upstream VkFFT commit mismatch: expected $expected_commit, got $actual_commit" >&2
    exit 2
fi
if [ ! -f /usr/include/vulkan/vulkan.h ] && [ ! -f /usr/local/include/vulkan/vulkan.h ]; then
    echo "Vulkan headers are required to compile the pinned scheduler structs" >&2
    exit 2
fi

build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT HUP INT TERM
mkdir -p "$build_dir/include"
# vkFFT_Structs.h includes glslang's C interface for backend 0, but the scheduler-only
# structs/functions used here do not reference any glslang declarations.
: > "$build_dir/include/glslang_c_interface.h"
# Compile the pinned upstream auto-padding function itself instead of copying its
# vendor tables into this repository. The commit guard above makes this extraction stable.
sed -n '/^static inline VkFFTResult initializeBluesteinAutoPadding/,/^}$/p' \
    "$upstream_dir/vkFFT/vkFFT/vkFFT_AppManagement/vkFFT_InitializeApp.h" \
    > "$build_dir/include/upstream_bluestein_auto_padding.h"
if [ ! -s "$build_dir/include/upstream_bluestein_auto_padding.h" ]; then
    echo "failed to extract initializeBluesteinAutoPadding from pinned upstream" >&2
    exit 2
fi

# Execute scheduler-policy expressions from the pinned initialization source rather
# than copying their constants into the reference helper. Numeric ranges are safe here
# because the commit guard above makes this exact source layout part of the oracle.
initialize_app="$upstream_dir/vkFFT/vkFFT/vkFFT_AppManagement/vkFFT_InitializeApp.h"
sed -n '496,533p' "$initialize_app" > "$build_dir/include/upstream_policy_vulkan.inc"
sed -n '644,650p' "$initialize_app" > "$build_dir/include/upstream_policy_cuda.inc"
sed -n '748,755p' "$initialize_app" > "$build_dir/include/upstream_policy_hip.inc"
sed -n '818,856p' "$initialize_app" > "$build_dir/include/upstream_policy_opencl.inc"
sed -n '893,901p' "$initialize_app" > "$build_dir/include/upstream_policy_level_zero.inc"
sed -n '949,959p' "$initialize_app" > "$build_dir/include/upstream_policy_metal.inc"
sed -n '1235,1247p' "$initialize_app" > "$build_dir/include/upstream_policy_lut_finalize.inc"
sed -n '1312,1316p' "$initialize_app" > "$build_dir/include/upstream_policy_reorder.inc"
for include_file in \
    upstream_policy_vulkan.inc \
    upstream_policy_cuda.inc \
    upstream_policy_hip.inc \
    upstream_policy_opencl.inc \
    upstream_policy_level_zero.inc \
    upstream_policy_metal.inc \
    upstream_policy_lut_finalize.inc \
    upstream_policy_reorder.inc
do
    if [ ! -s "$build_dir/include/$include_file" ]; then
        echo "failed to extract pinned upstream policy source: $include_file" >&2
        exit 2
    fi
done

"$cxx_bin" -std=c++17 -O2 \
    -I "$build_dir/include" \
    "$script_dir/upstream_scheduler_policy_reference.cpp" \
    -o "$build_dir/upstream_scheduler_policy_reference"

"$cc_bin" -std=c11 -O2 -DVKFFT_BACKEND=0 \
    -I "$build_dir/include" \
    -I "$upstream_dir/vkFFT" \
    "$script_dir/upstream_scheduler_reference.c" \
    -lm \
    -o "$build_dir/upstream_scheduler_reference"

if [ "$#" -eq 0 ]; then
    "$build_dir/upstream_scheduler_policy_reference"
    exec "$build_dir/upstream_scheduler_reference"
fi
if [ "$#" -eq 1 ] && [ "$1" = "--list-cases" ]; then
    "$build_dir/upstream_scheduler_policy_reference" --list-cases
    exec "$build_dir/upstream_scheduler_reference" --list-cases
fi
if [ "$#" -eq 2 ] && [ "$1" = "--case" ]; then
    case "$2" in
        policy-*) exec "$build_dir/upstream_scheduler_policy_reference" "$@" ;;
        *) exec "$build_dir/upstream_scheduler_reference" "$@" ;;
    esac
fi
exec "$build_dir/upstream_scheduler_reference" "$@"
