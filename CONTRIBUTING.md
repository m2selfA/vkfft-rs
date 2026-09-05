# Contributing to vkfft-rs

Thanks for helping improve `vkfft-rs`.

## Development checks

Before opening a pull request, run the portable checks that do not require a GPU runtime:

```bash
cargo fmt --all -- --check
RUSTFLAGS="-D warnings" cargo check --all-targets --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo test --lib --no-default-features
```

For changes that affect planning, code generation, runtime execution, precision, or backend resource ownership, also run the smallest relevant focused tests. If compatible hardware is available, include the corresponding real-device runtime tests.

A test that skips because a runtime or device is unavailable is useful for portability, but it is not evidence that the backend was validated on hardware. Please keep hardware claims explicit about the device/runtime actually exercised.

## Design and compatibility

The implementation aims to reproduce VkFFT behavior while keeping the Rust API, scheduler state, kernel IR, and backend lowering typed and auditable. Preserve fail-closed behavior for unsupported precision or runtime capabilities, and prefer adding structural/CPU-oracle coverage before expanding hardware-specific execution paths.

The public roadmap and implementation mapping live in [`docs/PORTING_PLAN.md`](docs/PORTING_PLAN.md). `NOTICE` records the pinned upstream baseline and provenance.

## Pull requests

Keep changes focused, explain the scheduler/runtime behavior being changed, and include tests that would have failed before the fix when practical. Avoid committing generated build trees, local benchmark output, credentials, or machine-specific configuration.
