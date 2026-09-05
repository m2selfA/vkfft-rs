# Numerical golden fixtures

These fixtures are static numerical references generated offline with FFTW 3.3.9. They follow the transform conventions used by the pinned VkFFT 1.3.4 precision samples (sample 11 C2C, sample 15 R2C/C2R, and sample 16 DCT). Runtime tests do not link FFTW or VkFFT; they only parse committed fixture values.

Regenerate all three corpora with:

```sh
cc -O2 tools/generate_fftw_golden.c -lfftw3 -lm -o /tmp/vkfft-rs-generate-fftw-golden
/tmp/vkfft-rs-generate-fftw-golden > tests/fixtures/fftw_precision_v1.txt
/tmp/vkfft-rs-generate-fftw-golden medium > tests/fixtures/fftw_precision_medium_v1.txt
/tmp/vkfft-rs-generate-fftw-golden medium-f32 > tests/fixtures/fftw_precision_medium_f32_v1.txt
```

The first corpus stores both deterministic inputs and FFTW outputs for compact sanity cases. The `medium` corpus keeps the versioned `complex-v1`/`real-v1` input formulas in the generator/test and stores only FFTW outputs. `medium-f32` covers the same five broader algorithm/ND cases, but its `complex-f32-v1`/`real-f32-v1` formulas deliberately perform every input arithmetic operation in C `float` before promoting the values into FFTW's double-precision API; the Rust integration test reconstructs the same F32 values and compares real F32 CUDA/OpenCL/Vulkan execution against those external outputs. Review fixture diffs before committing regeneration, because FFTW is an external oracle rather than a Rust-generated candidate.
