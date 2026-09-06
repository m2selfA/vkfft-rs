# vkfft-rs

[![CI](https://github.com/m2selfA/vkfft-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/m2selfA/vkfft-rs/actions/workflows/ci.yml)

`vkfft-rs` is a from-source, pure-Rust reimplementation of Dmitrii Tolmachev's [VkFFT](https://github.com/DTolm/VkFFT). The goal is not to wrap `vkFFT.h`; the goal is to reproduce VkFFT's planner, runtime GPU kernel generation, transforms, memory-layout optimizations, and multi-backend execution in Rust.

The initial porting baseline is VkFFT **1.3.4**, upstream commit `066a17c17068c0f11c9298d848c2976c71fad1c1`.

The current release-candidate validation is **1,598/1,598 target tests passed** (1,440 library tests plus 158 integration tests), with eight example harnesses also completing successfully. Hardware claims remain backend-specific: Vulkan, CUDA, OpenCL, HIP, and Level Zero have real-device coverage on the configurations documented below, while real Apple-GPU Metal execution remains an open validation item.

## Status

`vkfft-rs` 0.1.0 is a correctness-first Rust reimplementation of VkFFT with executable GPU runtimes, not an FFI wrapper.

- **Transforms:** 1D and multidimensional C2C, R2C/C2R, DCT/DST I-IV, convolution, zero padding, strides, batching, and grouped batching.
- **Planning and kernels:** Stockham, direct/FFT-convolution Rader, Bluestein, recursive mixed-radix planning, typed kernel/program IR, and backend-specific code generation.
- **Precision:** F16 storage, F32/F64 compute, mixed storage, and double-double paths where supported.
- **Backends:** Vulkan, CUDA, HIP, OpenCL, Level Zero, and Metal. Vulkan/CUDA/OpenCL/HIP/Level Zero have real-device validation; real Apple-GPU Metal validation and FP64-capable Level Zero validation remain open hardware items.
- **Validation:** the 0.1.0 release candidate completed **1,598/1,598** target tests (1,440 library + 158 integration), with additional backend-specific real-device gates.

The crate is still pre-1.0, so APIs may evolve. See [`docs/PORTING_PLAN.md`](docs/PORTING_PLAN.md) for detailed feature mapping, hardware evidence, and remaining work.

## Design direction

VkFFT's paper describes a hierarchical `Application -> Plan -> Code` architecture. The Rust port keeps the same separation while replacing C structs and stringly code generation with typed Rust state:

1. **Application/configuration** — validate user intent and own backend resources.
2. **Plan** — select Stockham/Rader/Bluestein, split large axes, choose register/shared-memory strategy, and allocate LUT/scratch requirements.
3. **Code generation** — lower a typed kernel IR into backend source/binaries and cache the result.
4. **Backend dispatch** — Vulkan has the full cached Rust runtime adapter with pooled multi-in-flight tickets; CUDA Driver+NVRTC and OpenCL have validated correctness runtimes plus asynchronous ticket submission; HIP has real AMD execution coverage through HIPRTC and the Windows AMD COMGR fallback on wave64 and wave32 hardware; Level Zero has a correctness runtime with real Intel execution and a Windows Intel OpenCL native compiler fallback; Metal now has a correctness-first F32/F16 runtime with async tickets, cached pipelines, pooled transient buffers, immutable LUT reuse, and compiled-pipeline resource reports, while real Apple-GPU execution remains the outstanding hardware gate.

See [`docs/PORTING_PLAN.md`](docs/PORTING_PLAN.md) for the detailed mapping and roadmap.

## Example: inspect a plan

```rust
use vkfft_rs::{AlgorithmKind, FftConfig, FftPlan};

let plan = FftPlan::build(FftConfig::new(vec![1024, 17, 65537]))?;
assert_eq!(plan.axes[0].algorithm.kind(), AlgorithmKind::Stockham);
assert_eq!(plan.axes[1].algorithm.kind(), AlgorithmKind::Rader);
assert_eq!(plan.axes[2].algorithm.kind(), AlgorithmKind::Bluestein);
# Ok::<(), vkfft_rs::VkFftError>(())
```

## Example: generate the first Vulkan Stockham shader

```rust
use vkfft_rs::backend::{KernelBackend, vulkan::VulkanGlslBackend};
use vkfft_rs::{Backend, DeviceProfile, Direction, FftConfig, FftPlan, GpuVendor, KernelIr};

let plan = FftPlan::build(FftConfig::new(vec![256]).with_batch_count(8))?;
let device = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);
let ir = KernelIr::stockham_1d(&plan, Direction::Forward, device)?;
let shader = VulkanGlslBackend.lower(&ir)?;
let spirv = shader.compile_spirv()?;

assert_eq!(shader.dispatch.x, 8);
assert!(shader.glsl.starts_with("#version 450"));
assert_eq!(spirv.words[0], 0x0723_0203);
# Ok::<(), vkfft_rs::VkFftError>(())
```

The selected shader layout is scheduler-driven: portable profiles retain explicit shared ping-pong, while recognized GPU policies can select one-shared/registerBoost/register-resident layouts and proven Vulkan subgroup exchange. Unknown vendor profiles fail soft to the portable layout.

## Native source backends

`cargo run --example native_source -- cuda 256 f32` prints a CUDA kernel for the same typed Stockham IR. Replace `cuda` with `hip`, `opencl`, `level-zero`, or `metal`; the optional third argument is `f32`/`f64` (Metal currently rejects F64). Passing `direct-rader` as the fourth argument emits a direct-Rader kernel, for example `cargo run --example native_source -- opencl 47 f32 direct-rader`.

## Optional native runtimes

`examples/native_execute.rs` drives the same F32 `NativeRuntime` program through all five native runtime backends. For example:

```text
cargo run --example native_execute --features cuda-runtime -- cuda 64
cargo run --example native_execute --features hip-runtime -- hip 64
cargo run --example native_execute --features opencl-runtime -- opencl 64
cargo run --example native_execute --features level-zero-runtime -- level-zero 64
cargo run --example native_execute --features metal-runtime -- metal 64
```

HIP has real-device coverage on AMD `gfx803` wave64 and `gfx1103` wave32 hardware, including the Windows AMD COMGR compiler fallback. Level Zero has real Intel execution coverage, including its Windows native-binary compiler fallback. Metal dynamically enumerates `MTLDevice`, compiles MSL with `newLibraryWithSource`, supports synchronous and asynchronous F32/F16 execution, and reuses pipelines/transient buffers/immutable LUTs; Metal F64/DD and SIMD-group lowering remain fail-closed, and real Apple-GPU execution is still open because the available macOS validation host exposes no `MTLDevice`. Runtime features depend only on optional `libloading`; SDK/runtime libraries are discovered dynamically. `examples/native_benchmark.rs` accepts `cuda`, `opencl`, or `metal` and reports compiler/driver resource metrics when the runtime exposes them.

## Precision reports

The sample-11-style four-metric precision definitions are available as `PrecisionMetrics`: average/max absolute difference plus average/max per-bin relative epsilon. `PrecisionCaseReport::to_json_line()` emits dependency-free schema-v1 JSONL records containing the fixed upstream VkFFT commit, vkfft-rs version, backend/device, transform family, shape, precision, and all four metrics.

`examples/precision_report.rs` runs a deterministic matrix across ND C2C `[17,34]`, odd 1D R2C `65`, ND R2C `[17,34]`, 1D DCT-II `257`, and ND DCT-II `[17,34]`. Vulkan, CUDA Driver+NVRTC, and OpenCL execute paired F32/F64 GPU cases and have been run successfully on the development RTX 2080 Ti, producing ten JSONL records per backend. Metal preserves its actual precision contract instead of fabricating F64 support: it executes one F32 GPU case per shape and compares it against an independently built F64 CPU reference, producing five F32 JSONL records when a real Metal device is available. For example:

```text
cargo run --release --example precision_report --features vulkan-runtime -- --backend vulkan --device 1 --output precision-vulkan.jsonl
cargo run --release --example precision_report --features cuda-runtime -- --backend cuda --output precision-cuda.jsonl
cargo run --release --example precision_report --features opencl-runtime -- --backend opencl --output precision-opencl.jsonl
cargo run --release --example precision_report --features metal-runtime -- --backend metal --output precision-metal.jsonl
```

The Metal command is implemented and reaches `MetalExecutionContext`; on the current VMware macOS validation host it remains hardware-gated because `MTLCopyAllDevices` reports zero devices. The stored F32/F16 FFTW execution corpora, F16 real R2C/C2R round trip, and formatted ND/stride/padding/async gates are already wired to become strict when `VKFFT_REQUIRE_METAL_RUNTIME=1` is run on an actual Apple GPU.

`--paper` explicitly adds a `2^27`-sample 1D C2C case and prints a multi-GiB memory warning to stderr. It is intentionally outside the ordinary test path and was not run during the normal validation sweep; paper-scale performance reproduction and A100/MI250-class measurements remain separate open work.

The expanded precision matrix also exposed a previously masked CUDA F32 even-real N=34 failure: the old max-error helper could swallow NaNs, while the new metrics require finite values. The failing kernel was narrowed to a chained native `VkFFTComplex` temporary in the even-half real postprocess; the shared real shader now combines x/y components explicitly, and N=34 plus ND fastest-axis-34 regressions are finite on CUDA while Vulkan/OpenCL remain unchanged.

## Runtime features

The default build keeps GPU runtime loading disabled: planner/scheduler logic, CPU references, typed IR, LUT generation, native source generation, and Vulkan GLSL/SPIR-V generation remain available without a vendor runtime installed. Enable only the execution runtime(s) you need:

| Feature | Execution surface | Current hardware evidence |
| --- | --- | --- |
| `vulkan-runtime` | Vulkan 1.1 through `ash`; automatic or explicit device selection, async tickets, pools/caches, resident chaining | NVIDIA, AMD, and Intel Vulkan |
| `cuda-runtime` | CUDA Driver + NVRTC, dynamically loaded | NVIDIA |
| `hip-runtime` | HIP runtime plus HIPRTC/AMD compiler paths, dynamically loaded | AMD wave64 and wave32 |
| `opencl-runtime` | OpenCL runtime/compiler, dynamically loaded | NVIDIA plus Intel compiler/fallback use |
| `level-zero-runtime` | Intel Level Zero queue/module execution, dynamically loaded | Intel UHD 770 F32/F16; FP64-capable hardware remains open |
| `metal-runtime` | Metal F32/F16 runtime, async/cache/pool surface | ABI/no-device validation only; a real Apple `MTLDevice` remains open |

The native runtimes share the same `NativeRuntime`/`ProgramIr` contract. A minimal execution example accepts `BACKEND LENGTH DEVICE_INDEX`; for example:

```console
cargo run --example native_execute --features cuda-runtime -- cuda 256 0
cargo run --example native_execute --features hip-runtime -- hip 256 0
cargo run --example native_execute --features opencl-runtime -- opencl 256 0
cargo run --example native_execute --features level-zero-runtime -- level-zero 256 0
cargo run --example native_execute --features metal-runtime -- metal 256 0
```

On Intel systems without `ocloc`, enabling both `level-zero-runtime,opencl-runtime` allows the validated same-device Intel OpenCL native-binary compiler fallback. Runtime features discover vendor libraries at execution time; enabling a feature does not by itself claim that compatible hardware or drivers are present.

## Optional Vulkan runtime adapter

Enable `vulkan-runtime` to compile the `ash`-based runtime layer. `VulkanComputePipeline` keeps the low-level integration model: it accepts a caller-owned `ash::Device`, storage buffers, and command buffer so an application can append FFT work to its own Vulkan command stream. `VulkanExecutionContext` is the complementary convenience path for tests and simple tools: it dynamically loads Vulkan, creates a Vulkan 1.1 compute context, derives a `DeviceProfile` from physical-device limits/features, enables `shaderFloat64` when available, stages host data into device-local storage, records transfer/compute synchronization plus dispatches, submits once, waits on a fence, and stages F32/F64 results back.

The repository includes `cargo run --example vulkan_execute --features vulkan-runtime` as a minimal end-to-end sample. It keeps scored automatic device selection when no positional argument is supplied; append an explicit Vulkan compute-device ordinal such as `-- 1` to select a particular enumerated device. The self-contained path uses device-local compute storage with explicit host staging, pooled transient and staging buffers, persistent immutable LUTs, cached pipeline/descriptor objects, reusable command-buffer/fence slots, and a versioned/persistable driver pipeline-cache archive. The synchronous convenience calls still wait for completion, while the high-level ticket APIs can keep multiple submissions in flight with exclusive descriptor/pipeline, buffer, command-buffer, and fence ownership until each ticket completes.

The runtime golden test now sweeps representative Stockham radices and mixed-radix lengths in both forward and normalized-inverse directions. This also guards the Vulkan codegen's explicit workgroup-scope barrier normalization needed around Naga 29's GLSL-to-SPIR-V barrier representation.

A second conditional real-GPU golden test explicitly tunes the planner to exercise a 103-point Bluestein transform with batching, checks the forward result against the direct CPU DFT, and checks normalized inverse round-trip. For the covered single-Stockham inverse convolution it records preprocess -> forward padded Stockham FFT -> fused frequency-multiply/normalized inverse Stockham FFT -> postprocess, with the spectrum LUT bound directly to the inverse FFT; recursive inverse convolution roots keep the explicit multiply pass. The generic ProgramIr runtime inserts the required compute-to-compute barriers between remaining passes.

A third conditional real-GPU golden test covers a planner-selected 47-point direct-Rader transform with batching, checking forward DFT agreement and normalized inverse round-trip. Both Bluestein and direct Rader now exercise the generalized read-only storage-buffer LUT binding path.

A fourth conditional real-GPU golden test covers FFT-convolution Rader at 17, 19, 29, 31, 37, 41, 43, 107, and 257 points; 107 explicitly enables the recursive-Rader extension, while the other cases use the upstream-default planner. Single-Stockham convolution roots execute the two-dispatch generator-order path; 107 exercises the recursive envelope with `106 = 2 x 53`, including a two-container `[13,4]` p53 sub-Rader. The smooth non-power-of-two p19 container uses `[6,3]`; p29 uses `[7,4]` with a bounds-guarded partial final register group; the additional 31/37/41/43 cases exercise broader smooth mixed-radix schedules. Forward DFT agreement and normalized inverse round-trip are checked for the whole sweep.

A fifth conditional real-GPU golden test covers mixed Stockham/Rader axes at 34, 57, 68, 94, 136, 152, 174, 232, 514, 1028, and 2056 points. Eligible inner FFT-Rader containers use the same fused generator-order load/LUT/scatter path. The p17 cases group 2/4/8 one-stage containers; p257 uses 2/4 containers at 514/1028 and the exact subgroup-local proof can eliminate shared exchange in both one- and two-subgroup workgroups. The 2056 case combines the fused Rader IO with the 8-container p257 `raderTranspose` shared layout. A separate 4112-point F32 gate executes 16 p257 containers in a 256-thread transposed workgroup, checks sparse frequency bins against a direct oracle, and checks normalized inverse round-trip.

A sixth conditional real-GPU golden test covers the generic recursive tree at 289 points (`17^2`) and 323 points (`17 * 19`). The same depth-first tree order is independently used to build the `ProgramIr` pass/resource graph and the Vulkan shader list; both forward results are checked against direct DFTs and both normalized inverse paths round-trip the input.

An additional F64 real-GPU gate runs when `shaderFloat64` is supported. It covers Stockham kernels plus high-level FFT-Rader, multidimensional C2C, multidimensional R2C/C2R, and multidimensional DCT-II paths against the F64 CPU oracles. On NVIDIA/Vulkan these Stockham kernels consume the upstream-default immutable twiddle LUT; F32 continues to evaluate twiddles on the fly. Two-/three-upload Four-step kernels share a lazy full-period unit-root resource so `useLUT_4step` does not embed giant root vectors in `ProgramIr`. This gate caught and fixed an important compiler-boundary issue: unsuffixed GLSL decimal literals are F32, while Naga 29 rejects native 64-bit literal suffixes, so the F64 lowering now reconstructs floating constants from exact 32-bit integer-to-double arithmetic before SPIR-V emission.

## Licensing and provenance

VkFFT is MIT licensed. This rewrite preserves the upstream MIT notice in [`LICENSE`](LICENSE) and records the baseline/source mapping in [`NOTICE`](NOTICE). The implementation is being rewritten in Rust rather than mechanically translated or linked through FFI.
