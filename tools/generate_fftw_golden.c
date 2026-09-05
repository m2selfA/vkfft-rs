#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static double real_sample(size_t i) {
    double x = (double)i;
    return sin(0.173 * x) + 0.003 * x - 0.27 * cos(0.071 * x);
}

static void complex_sample(size_t i, double *re, double *im) {
    double x = (double)i;
    *re = sin(0.137 * x) + 0.0013 * x;
    *im = cos(0.071 * x) - 0.0009 * x;
}

static double real_sample_f32(size_t i) {
    float x = (float)i;
    float value = sinf(0.173f * x) + 0.003f * x - 0.27f * cosf(0.071f * x);
    return (double)value;
}

static void complex_sample_f32(size_t i, double *re, double *im) {
    float x = (float)i;
    float real = sinf(0.137f * x) + 0.0013f * x;
    float imag = cosf(0.071f * x) - 0.0009f * x;
    *re = (double)real;
    *im = (double)imag;
}

static uint16_t binary16_from_f32(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
    int exponent = (int)((bits >> 23) & 0xffu);
    uint32_t mantissa = bits & 0x007fffffu;

    if (exponent == 0xff) {
        if (mantissa == 0) return (uint16_t)(sign | 0x7c00u);
        uint16_t payload = (uint16_t)(mantissa >> 13);
        if (payload == 0) payload = 1;
        return (uint16_t)(sign | 0x7c00u | payload);
    }

    int half_exponent = exponent - 127 + 15;
    if (half_exponent >= 0x1f) return (uint16_t)(sign | 0x7c00u);

    if (half_exponent <= 0) {
        if (half_exponent < -10) return sign;
        mantissa |= 0x00800000u;
        unsigned shift = (unsigned)(14 - half_exponent);
        uint32_t half_mantissa = mantissa >> shift;
        uint32_t remainder_mask = (1u << shift) - 1u;
        uint32_t remainder = mantissa & remainder_mask;
        uint32_t halfway = 1u << (shift - 1u);
        if (remainder > halfway || (remainder == halfway && (half_mantissa & 1u) != 0)) {
            half_mantissa += 1u;
        }
        return (uint16_t)(sign | (uint16_t)half_mantissa);
    }

    uint32_t half_mantissa = mantissa >> 13;
    uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half_mantissa & 1u) != 0)) {
        half_mantissa += 1u;
        if (half_mantissa == 0x0400u) {
            half_mantissa = 0;
            half_exponent += 1;
            if (half_exponent >= 0x1f) return (uint16_t)(sign | 0x7c00u);
        }
    }
    return (uint16_t)(sign | ((uint16_t)half_exponent << 10) | (uint16_t)half_mantissa);
}

static float binary16_to_f32(uint16_t bits) {
    uint32_t sign = ((uint32_t)(bits & 0x8000u)) << 16;
    int exponent = (int)((bits >> 10) & 0x1fu);
    uint32_t mantissa = (uint32_t)(bits & 0x03ffu);
    uint32_t bits32;
    if (exponent == 0 && mantissa == 0) {
        bits32 = sign;
    } else if (exponent == 0) {
        uint32_t normalized = mantissa;
        int unbiased = -14;
        while ((normalized & 0x0400u) == 0) {
            normalized <<= 1;
            unbiased -= 1;
        }
        normalized &= 0x03ffu;
        bits32 = sign | ((uint32_t)(unbiased + 127) << 23) | (normalized << 13);
    } else if (exponent == 0x1f) {
        bits32 = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits32 = sign | ((uint32_t)(exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    float value;
    memcpy(&value, &bits32, sizeof(value));
    return value;
}

static float quantize_binary16_f32(float value) {
    return binary16_to_f32(binary16_from_f32(value));
}

static uint32_t binary32_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static void emit_header(const char *case_name, const char *kind, size_t n) {
    printf("case %s\nkind %s\nlength %zu\n", case_name, kind, n);
}

static void emit_c2c(size_t n) {
    fftw_complex *input = fftw_malloc(sizeof(fftw_complex) * n);
    fftw_complex *output = fftw_malloc(sizeof(fftw_complex) * n);
    if (!input || !output) exit(2);
    for (size_t i = 0; i < n; ++i) complex_sample(i, &input[i][0], &input[i][1]);
    fftw_plan plan = fftw_plan_dft_1d((int)n, input, output, FFTW_FORWARD, FFTW_ESTIMATE);
    if (!plan) exit(3);
    fftw_execute(plan);
    emit_header("c2c-n47", "c2c", n);
    for (size_t i = 0; i < n; ++i) {
        printf("%.17e %.17e %.17e %.17e\n", input[i][0], input[i][1], output[i][0], output[i][1]);
    }
    fftw_destroy_plan(plan);
    fftw_free(output);
    fftw_free(input);
}

static void emit_r2c(size_t n) {
    double *input = fftw_malloc(sizeof(double) * n);
    fftw_complex *output = fftw_malloc(sizeof(fftw_complex) * (n / 2 + 1));
    if (!input || !output) exit(2);
    for (size_t i = 0; i < n; ++i) input[i] = real_sample(i);
    fftw_plan plan = fftw_plan_dft_r2c_1d((int)n, input, output, FFTW_ESTIMATE);
    if (!plan) exit(3);
    fftw_execute(plan);
    emit_header("r2c-n45", "r2c", n);
    for (size_t i = 0; i < n; ++i) printf("%.17e\n", input[i]);
    printf("output %zu\n", n / 2 + 1);
    for (size_t i = 0; i < n / 2 + 1; ++i) printf("%.17e %.17e\n", output[i][0], output[i][1]);
    fftw_destroy_plan(plan);
    fftw_free(output);
    fftw_free(input);
}

static void emit_dct(size_t n) {
    double *input = fftw_malloc(sizeof(double) * n);
    double *output = fftw_malloc(sizeof(double) * n);
    if (!input || !output) exit(2);
    for (size_t i = 0; i < n; ++i) input[i] = real_sample(i);
    fftw_plan plan = fftw_plan_r2r_1d((int)n, input, output, FFTW_REDFT10, FFTW_ESTIMATE);
    if (!plan) exit(3);
    fftw_execute(plan);
    emit_header("dct2-n37", "dct2", n);
    for (size_t i = 0; i < n; ++i) printf("%.17e %.17e\n", input[i], output[i]);
    fftw_destroy_plan(plan);
    fftw_free(output);
    fftw_free(input);
}

static void emit_medium_header(
    const char *case_name,
    const char *kind,
    const size_t *shape,
    size_t dimensions,
    const char *input_formula,
    size_t output_count
) {
    printf("case %s\nkind %s\nshape", case_name, kind);
    for (size_t i = 0; i < dimensions; ++i) printf(" %zu", shape[i]);
    printf("\ninput %s\noutput %zu\n", input_formula, output_count);
}

static void emit_c2c_output_case(const char *case_name, size_t n) {
    fftw_complex *input = fftw_malloc(sizeof(fftw_complex) * n);
    fftw_complex *output = fftw_malloc(sizeof(fftw_complex) * n);
    if (!input || !output) exit(2);
    for (size_t i = 0; i < n; ++i) complex_sample(i, &input[i][0], &input[i][1]);
    fftw_plan plan = fftw_plan_dft_1d((int)n, input, output, FFTW_FORWARD, FFTW_ESTIMATE);
    if (!plan) exit(3);
    fftw_execute(plan);
    size_t shape[1] = {n};
    emit_medium_header(case_name, "c2c", shape, 1, "complex-v1", n);
    for (size_t i = 0; i < n; ++i) printf("%.17e %.17e\n", output[i][0], output[i][1]);
    fftw_destroy_plan(plan);
    fftw_free(output);
    fftw_free(input);
}

static void emit_r2c_2d_output_case(const char *case_name, size_t rows, size_t cols) {
    const size_t input_count = rows * cols;
    const size_t output_count = rows * (cols / 2 + 1);
    double *input = fftw_malloc(sizeof(double) * input_count);
    fftw_complex *output = fftw_malloc(sizeof(fftw_complex) * output_count);
    if (!input || !output) exit(2);
    for (size_t i = 0; i < input_count; ++i) input[i] = real_sample(i);
    fftw_plan plan = fftw_plan_dft_r2c_2d((int)rows, (int)cols, input, output, FFTW_ESTIMATE);
    if (!plan) exit(3);
    fftw_execute(plan);
    size_t shape[2] = {rows, cols};
    emit_medium_header(case_name, "r2c", shape, 2, "real-v1", output_count);
    for (size_t i = 0; i < output_count; ++i) printf("%.17e %.17e\n", output[i][0], output[i][1]);
    fftw_destroy_plan(plan);
    fftw_free(output);
    fftw_free(input);
}

static void emit_dct2_2d_output_case(const char *case_name, size_t rows, size_t cols) {
    const size_t count = rows * cols;
    double *input = fftw_malloc(sizeof(double) * count);
    double *output = fftw_malloc(sizeof(double) * count);
    if (!input || !output) exit(2);
    for (size_t i = 0; i < count; ++i) input[i] = real_sample(i);
    fftw_plan plan = fftw_plan_r2r_2d(
        (int)rows,
        (int)cols,
        input,
        output,
        FFTW_REDFT10,
        FFTW_REDFT10,
        FFTW_ESTIMATE
    );
    if (!plan) exit(3);
    fftw_execute(plan);
    size_t shape[2] = {rows, cols};
    emit_medium_header(case_name, "dct2", shape, 2, "real-v1", count);
    for (size_t i = 0; i < count; ++i) printf("%.17e\n", output[i]);
    fftw_destroy_plan(plan);
    fftw_free(output);
    fftw_free(input);
}

static void emit_c2c_f32_output_case(const char *case_name, size_t n) {
    fftw_complex *input = fftw_malloc(sizeof(fftw_complex) * n);
    fftw_complex *output = fftw_malloc(sizeof(fftw_complex) * n);
    if (!input || !output) exit(2);
    for (size_t i = 0; i < n; ++i) complex_sample_f32(i, &input[i][0], &input[i][1]);
    fftw_plan plan = fftw_plan_dft_1d((int)n, input, output, FFTW_FORWARD, FFTW_ESTIMATE);
    if (!plan) exit(3);
    fftw_execute(plan);
    size_t shape[1] = {n};
    emit_medium_header(case_name, "c2c", shape, 1, "complex-f32-v1", n);
    for (size_t i = 0; i < n; ++i) printf("%.17e %.17e\n", output[i][0], output[i][1]);
    fftw_destroy_plan(plan);
    fftw_free(output);
    fftw_free(input);
}

static void emit_r2c_2d_f32_output_case(const char *case_name, size_t rows, size_t cols) {
    const size_t input_count = rows * cols;
    const size_t output_count = rows * (cols / 2 + 1);
    double *input = fftw_malloc(sizeof(double) * input_count);
    fftw_complex *output = fftw_malloc(sizeof(fftw_complex) * output_count);
    if (!input || !output) exit(2);
    for (size_t i = 0; i < input_count; ++i) input[i] = real_sample_f32(i);
    fftw_plan plan = fftw_plan_dft_r2c_2d((int)rows, (int)cols, input, output, FFTW_ESTIMATE);
    if (!plan) exit(3);
    fftw_execute(plan);
    size_t shape[2] = {rows, cols};
    emit_medium_header(case_name, "r2c", shape, 2, "real-f32-v1", output_count);
    for (size_t i = 0; i < output_count; ++i) printf("%.17e %.17e\n", output[i][0], output[i][1]);
    fftw_destroy_plan(plan);
    fftw_free(output);
    fftw_free(input);
}

static void emit_dct2_2d_f32_output_case(const char *case_name, size_t rows, size_t cols) {
    const size_t count = rows * cols;
    double *input = fftw_malloc(sizeof(double) * count);
    double *output = fftw_malloc(sizeof(double) * count);
    if (!input || !output) exit(2);
    for (size_t i = 0; i < count; ++i) input[i] = real_sample_f32(i);
    fftw_plan plan = fftw_plan_r2r_2d(
        (int)rows,
        (int)cols,
        input,
        output,
        FFTW_REDFT10,
        FFTW_REDFT10,
        FFTW_ESTIMATE
    );
    if (!plan) exit(3);
    fftw_execute(plan);
    size_t shape[2] = {rows, cols};
    emit_medium_header(case_name, "dct2", shape, 2, "real-f32-v1", count);
    for (size_t i = 0; i < count; ++i) printf("%.17e\n", output[i]);
    fftw_destroy_plan(plan);
    fftw_free(output);
    fftw_free(input);
}

static void emit_c2c_f16_storage_output_case(const char *case_name, size_t n) {
    fftw_complex *input = fftw_malloc(sizeof(fftw_complex) * n);
    fftw_complex *output = fftw_malloc(sizeof(fftw_complex) * n);
    if (!input || !output) exit(2);
    for (size_t i = 0; i < n; ++i) {
        double re, im;
        complex_sample_f32(i, &re, &im);
        input[i][0] = (double)quantize_binary16_f32((float)re);
        input[i][1] = (double)quantize_binary16_f32((float)im);
    }
    fftw_plan plan = fftw_plan_dft_1d((int)n, input, output, FFTW_FORWARD, FFTW_ESTIMATE);
    if (!plan) exit(3);
    fftw_execute(plan);
    size_t shape[1] = {n};
    emit_medium_header(case_name, "c2c", shape, 1, "complex-f16-storage-v1", n);
    for (size_t i = 0; i < n; ++i) {
        printf("%04x %04x\n", (unsigned)binary16_from_f32((float)output[i][0]),
               (unsigned)binary16_from_f32((float)output[i][1]));
    }
    fftw_destroy_plan(plan);
    fftw_free(output);
    fftw_free(input);
}

static void emit_r2c_2d_f16_storage_output_case(const char *case_name, size_t rows, size_t cols) {
    const size_t input_count = rows * cols;
    const size_t output_count = rows * (cols / 2 + 1);
    double *input = fftw_malloc(sizeof(double) * input_count);
    fftw_complex *output = fftw_malloc(sizeof(fftw_complex) * output_count);
    if (!input || !output) exit(2);
    for (size_t i = 0; i < input_count; ++i) {
        input[i] = (double)quantize_binary16_f32((float)real_sample_f32(i));
    }
    fftw_plan plan = fftw_plan_dft_r2c_2d((int)rows, (int)cols, input, output, FFTW_ESTIMATE);
    if (!plan) exit(3);
    fftw_execute(plan);
    size_t shape[2] = {rows, cols};
    emit_medium_header(case_name, "r2c", shape, 2, "real-f16-storage-v1", output_count);
    for (size_t i = 0; i < output_count; ++i) {
        printf("%04x %04x\n", (unsigned)binary16_from_f32((float)output[i][0]),
               (unsigned)binary16_from_f32((float)output[i][1]));
    }
    fftw_destroy_plan(plan);
    fftw_free(output);
    fftw_free(input);
}

static void emit_dct2_2d_f16_storage_output_case(const char *case_name, size_t rows, size_t cols) {
    const size_t count = rows * cols;
    double *input = fftw_malloc(sizeof(double) * count);
    double *output = fftw_malloc(sizeof(double) * count);
    if (!input || !output) exit(2);
    for (size_t i = 0; i < count; ++i) {
        input[i] = (double)quantize_binary16_f32((float)real_sample_f32(i));
    }
    fftw_plan plan = fftw_plan_r2r_2d(
        (int)rows,
        (int)cols,
        input,
        output,
        FFTW_REDFT10,
        FFTW_REDFT10,
        FFTW_ESTIMATE
    );
    if (!plan) exit(3);
    fftw_execute(plan);
    size_t shape[2] = {rows, cols};
    emit_medium_header(case_name, "dct2", shape, 2, "real-f16-storage-v1", count);
    for (size_t i = 0; i < count; ++i) {
        printf("%04x\n", (unsigned)binary16_from_f32((float)output[i]));
    }
    fftw_destroy_plan(plan);
    fftw_free(output);
    fftw_free(input);
}

static void emit_c2c_f64_f32_storage_output_case(const char *case_name, size_t n) {
    fftw_complex *input = fftw_malloc(sizeof(fftw_complex) * n);
    fftw_complex *output = fftw_malloc(sizeof(fftw_complex) * n);
    if (!input || !output) exit(2);
    for (size_t i = 0; i < n; ++i) complex_sample_f32(i, &input[i][0], &input[i][1]);
    fftw_plan plan = fftw_plan_dft_1d((int)n, input, output, FFTW_FORWARD, FFTW_ESTIMATE);
    if (!plan) exit(3);
    fftw_execute(plan);
    size_t shape[1] = {n};
    emit_medium_header(case_name, "c2c", shape, 1, "complex-f32-v1", n);
    for (size_t i = 0; i < n; ++i) {
        printf("%08x %08x\n", (unsigned)binary32_bits((float)output[i][0]),
               (unsigned)binary32_bits((float)output[i][1]));
    }
    fftw_destroy_plan(plan);
    fftw_free(output);
    fftw_free(input);
}

static void emit_r2c_2d_f64_f32_storage_output_case(const char *case_name, size_t rows, size_t cols) {
    const size_t input_count = rows * cols;
    const size_t output_count = rows * (cols / 2 + 1);
    double *input = fftw_malloc(sizeof(double) * input_count);
    fftw_complex *output = fftw_malloc(sizeof(fftw_complex) * output_count);
    if (!input || !output) exit(2);
    for (size_t i = 0; i < input_count; ++i) input[i] = real_sample_f32(i);
    fftw_plan plan = fftw_plan_dft_r2c_2d((int)rows, (int)cols, input, output, FFTW_ESTIMATE);
    if (!plan) exit(3);
    fftw_execute(plan);
    size_t shape[2] = {rows, cols};
    emit_medium_header(case_name, "r2c", shape, 2, "real-f32-v1", output_count);
    for (size_t i = 0; i < output_count; ++i) {
        printf("%08x %08x\n", (unsigned)binary32_bits((float)output[i][0]),
               (unsigned)binary32_bits((float)output[i][1]));
    }
    fftw_destroy_plan(plan);
    fftw_free(output);
    fftw_free(input);
}

static void emit_dct2_2d_f64_f32_storage_output_case(const char *case_name, size_t rows, size_t cols) {
    const size_t count = rows * cols;
    double *input = fftw_malloc(sizeof(double) * count);
    double *output = fftw_malloc(sizeof(double) * count);
    if (!input || !output) exit(2);
    for (size_t i = 0; i < count; ++i) input[i] = real_sample_f32(i);
    fftw_plan plan = fftw_plan_r2r_2d(
        (int)rows,
        (int)cols,
        input,
        output,
        FFTW_REDFT10,
        FFTW_REDFT10,
        FFTW_ESTIMATE
    );
    if (!plan) exit(3);
    fftw_execute(plan);
    size_t shape[2] = {rows, cols};
    emit_medium_header(case_name, "dct2", shape, 2, "real-f32-v1", count);
    for (size_t i = 0; i < count; ++i) {
        printf("%08x\n", (unsigned)binary32_bits((float)output[i]));
    }
    fftw_destroy_plan(plan);
    fftw_free(output);
    fftw_free(input);
}

static int emit_medium_f32_corpus(void) {
    puts("schema 1");
    puts("source fftw-3.3.9");
    puts("upstream-samples VkFFT-1.3.4-sample11-sample14-sample15-sample16");
    puts("input-formulas complex-f32-v1 real-f32-v1");
    emit_c2c_f32_output_case("c2c-bluestein-n103", 103);
    emit_c2c_f32_output_case("c2c-rader-n257", 257);
    emit_c2c_f32_output_case("c2c-smooth-n1536", 1536);
    emit_c2c_f32_output_case("c2c-large-smooth-n4096", 4096);
    emit_r2c_2d_f32_output_case("r2c-nd-17x34", 17, 34);
    emit_r2c_2d_f32_output_case("r2c-nd-64x64", 64, 64);
    emit_dct2_2d_f32_output_case("dct2-nd-17x34", 17, 34);
    return 0;
}

static int emit_medium_f16_storage_corpus(void) {
    puts("schema 1");
    puts("source fftw-3.3.9");
    puts("upstream-samples VkFFT-1.3.4-sample11-sample14-sample15-sample16");
    puts("storage binary16-rne-from-f32-v1");
    puts("input-formulas complex-f16-storage-v1 real-f16-storage-v1");
    emit_c2c_f16_storage_output_case("c2c-bluestein-n103", 103);
    emit_c2c_f16_storage_output_case("c2c-rader-n257", 257);
    emit_c2c_f16_storage_output_case("c2c-smooth-n1536", 1536);
    emit_c2c_f16_storage_output_case("c2c-large-smooth-n4096", 4096);
    emit_r2c_2d_f16_storage_output_case("r2c-nd-17x34", 17, 34);
    emit_r2c_2d_f16_storage_output_case("r2c-nd-64x64", 64, 64);
    emit_dct2_2d_f16_storage_output_case("dct2-nd-17x34", 17, 34);
    return 0;
}

static int emit_medium_f64_f32_storage_corpus(void) {
    puts("schema 1");
    puts("source fftw-3.3.9");
    puts("upstream-samples VkFFT-1.3.4-sample11-sample14-sample15-sample16");
    puts("storage binary32-rne-from-f64-v1");
    puts("input-formulas complex-f32-v1 real-f32-v1");
    emit_c2c_f64_f32_storage_output_case("c2c-bluestein-n103", 103);
    emit_c2c_f64_f32_storage_output_case("c2c-rader-n257", 257);
    emit_c2c_f64_f32_storage_output_case("c2c-smooth-n1536", 1536);
    emit_c2c_f64_f32_storage_output_case("c2c-large-smooth-n4096", 4096);
    emit_r2c_2d_f64_f32_storage_output_case("r2c-nd-17x34", 17, 34);
    emit_r2c_2d_f64_f32_storage_output_case("r2c-nd-64x64", 64, 64);
    emit_dct2_2d_f64_f32_storage_output_case("dct2-nd-17x34", 17, 34);
    return 0;
}

static int emit_medium_corpus(void) {
    puts("schema 1");
    puts("source fftw-3.3.9");
    puts("upstream-samples VkFFT-1.3.4-sample11-sample14-sample15-sample16");
    puts("input-formulas complex-v1 real-v1");
    emit_c2c_output_case("c2c-bluestein-n103", 103);
    emit_c2c_output_case("c2c-rader-n257", 257);
    emit_c2c_output_case("c2c-smooth-n1536", 1536);
    emit_c2c_output_case("c2c-large-smooth-n4096", 4096);
    emit_r2c_2d_output_case("r2c-nd-17x34", 17, 34);
    emit_r2c_2d_output_case("r2c-nd-64x64", 64, 64);
    emit_dct2_2d_output_case("dct2-nd-17x34", 17, 34);
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "medium") == 0) return emit_medium_corpus();
    if (argc == 2 && strcmp(argv[1], "medium-f32") == 0) return emit_medium_f32_corpus();
    if (argc == 2 && strcmp(argv[1], "medium-f16-storage") == 0) return emit_medium_f16_storage_corpus();
    if (argc == 2 && strcmp(argv[1], "medium-f64-f32-storage") == 0) return emit_medium_f64_f32_storage_corpus();
    if (argc != 1) {
        fprintf(stderr, "usage: %s [medium|medium-f32|medium-f16-storage|medium-f64-f32-storage]\n", argv[0]);
        return 2;
    }
    puts("schema 1");
    puts("source fftw-3.3.9");
    puts("upstream-samples VkFFT-1.3.4-sample11-sample15-sample16");
    emit_c2c(47);
    emit_r2c(45);
    emit_dct(37);
    return 0;
}
