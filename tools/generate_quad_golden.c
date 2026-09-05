#include <fenv.h>
#include <math.h>
#include <quadmath.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    __float128 re;
    __float128 im;
} qcomplex;

static uint64_t binary64_bits(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static double real_sample(size_t i) {
    double x = (double)i;
    return sin(0.173 * x) + 0.003 * x - 0.27 * cos(0.071 * x);
}

static void complex_sample(size_t i, double *re, double *im) {
    double x = (double)i;
    *re = sin(0.137 * x) + 0.0013 * x;
    *im = cos(0.071 * x) - 0.0009 * x;
}

static qcomplex qadd(qcomplex a, qcomplex b) {
    qcomplex out = {a.re + b.re, a.im + b.im};
    return out;
}

static qcomplex qmul(qcomplex a, qcomplex b) {
    qcomplex out = {
        a.re * b.re - a.im * b.im,
        a.re * b.im + a.im * b.re,
    };
    return out;
}

static qcomplex qscale(qcomplex value, __float128 scalar) {
    qcomplex out = {value.re * scalar, value.im * scalar};
    return out;
}

static qcomplex qphase(__float128 angle) {
    qcomplex out = {cosq(angle), sinq(angle)};
    return out;
}

static __float128 qpi(void) {
    return acosq(-1.0Q);
}

typedef struct {
    double hi;
    double lo;
} dd_pair;

static dd_pair q_to_dd(__float128 value) {
    double hi = (double)value;
    double lo = (double)(value - (__float128)hi);
    double sum = hi + lo;
    double bb = sum - hi;
    double error = (hi - (sum - bb)) + (lo - bb);
    dd_pair out = {sum, error};
    return out;
}

static __float128 dd_to_q(dd_pair value) {
    return (__float128)value.hi + (__float128)value.lo;
}

static __float128 q_real_dd_sample(size_t i) {
    return (__float128)real_sample(i) + (__float128)(i + 1) * 7.0e-32Q;
}

static qcomplex q_complex_dd_sample(size_t i) {
    double re, im;
    complex_sample(i, &re, &im);
    qcomplex out = {
        (__float128)re + (__float128)(i + 1) * 8.0e-32Q,
        (__float128)im - (__float128)(i + 1) * 4.0e-32Q,
    };
    return out;
}

static void qkahan_add(__float128 value, __float128 *sum, __float128 *correction) {
    __float128 y = value - *correction;
    __float128 next = *sum + y;
    *correction = (next - *sum) - y;
    *sum = next;
}

static qcomplex qkahan_sum_terms(const qcomplex *terms, size_t count) {
    __float128 re = 0.0Q, re_c = 0.0Q;
    __float128 im = 0.0Q, im_c = 0.0Q;
    for (size_t i = 0; i < count; ++i) {
        qkahan_add(terms[i].re, &re, &re_c);
        qkahan_add(terms[i].im, &im, &im_c);
    }
    qcomplex out = {re, im};
    return out;
}

static void print_dd(dd_pair value) {
    printf("%016llx %016llx", (unsigned long long)binary64_bits(value.hi),
           (unsigned long long)binary64_bits(value.lo));
}


static void emit_case_header(
    const char *name,
    const char *kind,
    const size_t *shape,
    size_t dimensions,
    const char *input_format,
    size_t input_count,
    const char *output_format,
    size_t output_count
) {
    printf("case %s\nkind %s\nshape", name, kind);
    for (size_t i = 0; i < dimensions; ++i) printf(" %zu", shape[i]);
    printf("\ninput-format %s\ninput-count %zu\n", input_format, input_count);
    printf("output-format %s\noutput-count %zu\n", output_format, output_count);
}

static void emit_c2c_case(const char *name, size_t n) {
    double *input_re = calloc(n, sizeof(*input_re));
    double *input_im = calloc(n, sizeof(*input_im));
    qcomplex *input = calloc(n, sizeof(*input));
    qcomplex *output = calloc(n, sizeof(*output));
    if (!input_re || !input_im || !input || !output) exit(2);

    for (size_t j = 0; j < n; ++j) {
        complex_sample(j, &input_re[j], &input_im[j]);
        input[j].re = (__float128)input_re[j];
        input[j].im = (__float128)input_im[j];
    }

    const __float128 two_pi = 2.0Q * qpi();
    for (size_t k = 0; k < n; ++k) {
        const __float128 angle = -two_pi * (__float128)k / (__float128)n;
        const qcomplex step = qphase(angle);
        qcomplex phase = {1.0Q, 0.0Q};
        qcomplex sum = {0.0Q, 0.0Q};
        for (size_t j = 0; j < n; ++j) {
            sum = qadd(sum, qmul(input[j], phase));
            phase = qmul(phase, step);
        }
        output[k] = sum;
    }

    const size_t shape[1] = {n};
    emit_case_header(name, "c2c", shape, 1, "complex-binary64", n, "complex-binary64", n);
    for (size_t j = 0; j < n; ++j) {
        printf("%016llx %016llx\n", (unsigned long long)binary64_bits(input_re[j]),
               (unsigned long long)binary64_bits(input_im[j]));
    }
    puts("output-data");
    for (size_t k = 0; k < n; ++k) {
        printf("%016llx %016llx\n", (unsigned long long)binary64_bits((double)output[k].re),
               (unsigned long long)binary64_bits((double)output[k].im));
    }

    free(output);
    free(input);
    free(input_im);
    free(input_re);
}

static void emit_r2c_2d_case(const char *name, size_t rows, size_t cols) {
    const size_t input_count = rows * cols;
    const size_t half_cols = cols / 2 + 1;
    const size_t output_count = rows * half_cols;
    double *input_f64 = calloc(input_count, sizeof(*input_f64));
    __float128 *input = calloc(input_count, sizeof(*input));
    qcomplex *row_spectrum = calloc(rows * half_cols, sizeof(*row_spectrum));
    qcomplex *output = calloc(output_count, sizeof(*output));
    if (!input_f64 || !input || !row_spectrum || !output) exit(2);

    for (size_t i = 0; i < input_count; ++i) {
        input_f64[i] = real_sample(i);
        input[i] = (__float128)input_f64[i];
    }

    const __float128 two_pi = 2.0Q * qpi();
    for (size_t row = 0; row < rows; ++row) {
        for (size_t kc = 0; kc < half_cols; ++kc) {
            const __float128 angle = -two_pi * (__float128)kc / (__float128)cols;
            const qcomplex step = qphase(angle);
            qcomplex phase = {1.0Q, 0.0Q};
            qcomplex sum = {0.0Q, 0.0Q};
            for (size_t col = 0; col < cols; ++col) {
                sum = qadd(sum, qscale(phase, input[row * cols + col]));
                phase = qmul(phase, step);
            }
            row_spectrum[row * half_cols + kc] = sum;
        }
    }

    for (size_t kr = 0; kr < rows; ++kr) {
        const __float128 angle = -two_pi * (__float128)kr / (__float128)rows;
        const qcomplex step = qphase(angle);
        for (size_t kc = 0; kc < half_cols; ++kc) {
            qcomplex phase = {1.0Q, 0.0Q};
            qcomplex sum = {0.0Q, 0.0Q};
            for (size_t row = 0; row < rows; ++row) {
                sum = qadd(sum, qmul(row_spectrum[row * half_cols + kc], phase));
                phase = qmul(phase, step);
            }
            output[kr * half_cols + kc] = sum;
        }
    }

    const size_t shape[2] = {rows, cols};
    emit_case_header(name, "r2c", shape, 2, "real-binary64", input_count,
                     "complex-binary64", output_count);
    for (size_t i = 0; i < input_count; ++i) {
        printf("%016llx\n", (unsigned long long)binary64_bits(input_f64[i]));
    }
    puts("output-data");
    for (size_t i = 0; i < output_count; ++i) {
        printf("%016llx %016llx\n", (unsigned long long)binary64_bits((double)output[i].re),
               (unsigned long long)binary64_bits((double)output[i].im));
    }

    free(output);
    free(row_spectrum);
    free(input);
    free(input_f64);
}

static void emit_dct2_2d_case(const char *name, size_t rows, size_t cols) {
    const size_t count = rows * cols;
    double *input_f64 = calloc(count, sizeof(*input_f64));
    __float128 *input = calloc(count, sizeof(*input));
    __float128 *temp = calloc(count, sizeof(*temp));
    __float128 *output = calloc(count, sizeof(*output));
    __float128 *col_cos = calloc(cols * cols, sizeof(*col_cos));
    __float128 *row_cos = calloc(rows * rows, sizeof(*row_cos));
    if (!input_f64 || !input || !temp || !output || !col_cos || !row_cos) exit(2);

    for (size_t i = 0; i < count; ++i) {
        input_f64[i] = real_sample(i);
        input[i] = (__float128)input_f64[i];
    }

    const __float128 pi = qpi();
    for (size_t kc = 0; kc < cols; ++kc) {
        for (size_t col = 0; col < cols; ++col) {
            col_cos[kc * cols + col] =
                cosq(pi * (__float128)kc * (__float128)(2 * col + 1) / (2.0Q * (__float128)cols));
        }
    }
    for (size_t kr = 0; kr < rows; ++kr) {
        for (size_t row = 0; row < rows; ++row) {
            row_cos[kr * rows + row] =
                cosq(pi * (__float128)kr * (__float128)(2 * row + 1) / (2.0Q * (__float128)rows));
        }
    }

    for (size_t row = 0; row < rows; ++row) {
        for (size_t kc = 0; kc < cols; ++kc) {
            __float128 sum = 0.0Q;
            for (size_t col = 0; col < cols; ++col) {
                sum += input[row * cols + col] * col_cos[kc * cols + col];
            }
            temp[row * cols + kc] = 2.0Q * sum;
        }
    }
    for (size_t kr = 0; kr < rows; ++kr) {
        for (size_t kc = 0; kc < cols; ++kc) {
            __float128 sum = 0.0Q;
            for (size_t row = 0; row < rows; ++row) {
                sum += temp[row * cols + kc] * row_cos[kr * rows + row];
            }
            output[kr * cols + kc] = 2.0Q * sum;
        }
    }

    const size_t shape[2] = {rows, cols};
    emit_case_header(name, "dct2", shape, 2, "real-binary64", count, "real-binary64", count);
    for (size_t i = 0; i < count; ++i) {
        printf("%016llx\n", (unsigned long long)binary64_bits(input_f64[i]));
    }
    puts("output-data");
    for (size_t i = 0; i < count; ++i) {
        printf("%016llx\n", (unsigned long long)binary64_bits((double)output[i]));
    }

    free(row_cos);
    free(col_cos);
    free(output);
    free(temp);
    free(input);
    free(input_f64);
}

static void emit_full_dd_c2c_case(const char *name, size_t n) {
    dd_pair *input_re = calloc(n, sizeof(*input_re));
    dd_pair *input_im = calloc(n, sizeof(*input_im));
    qcomplex *input = calloc(n, sizeof(*input));
    qcomplex *phase = calloc(n, sizeof(*phase));
    qcomplex *terms = calloc(n, sizeof(*terms));
    if (!input_re || !input_im || !input || !phase || !terms) exit(2);

    for (size_t j = 0; j < n; ++j) {
        qcomplex source = q_complex_dd_sample(j);
        input_re[j] = q_to_dd(source.re);
        input_im[j] = q_to_dd(source.im);
        input[j].re = dd_to_q(input_re[j]);
        input[j].im = dd_to_q(input_im[j]);
    }

    const __float128 two_pi = 2.0Q * qpi();
    for (size_t j = 0; j < n; ++j) {
        phase[j] = qphase(-two_pi * (__float128)j / (__float128)n);
    }

    const size_t shape[1] = {n};
    emit_case_header(name, "c2c", shape, 1, "complex-double-double-binary64", n,
                     "complex-double-double-binary64", n);
    for (size_t j = 0; j < n; ++j) {
        print_dd(input_re[j]);
        putchar(' ');
        print_dd(input_im[j]);
        putchar('\n');
    }
    puts("output-data");
    for (size_t k = 0; k < n; ++k) {
        for (size_t j = 0; j < n; ++j) {
            terms[j] = qmul(input[j], phase[(k * j) % n]);
        }
        qcomplex sum = qkahan_sum_terms(terms, n);
        print_dd(q_to_dd(sum.re));
        putchar(' ');
        print_dd(q_to_dd(sum.im));
        putchar('\n');
    }

    free(terms);
    free(phase);
    free(input);
    free(input_im);
    free(input_re);
}

static void emit_full_dd_r2c_2d_case(const char *name, size_t rows, size_t cols) {
    const size_t input_count = rows * cols;
    const size_t half_cols = cols / 2 + 1;
    const size_t output_count = rows * half_cols;
    dd_pair *input_dd = calloc(input_count, sizeof(*input_dd));
    __float128 *input = calloc(input_count, sizeof(*input));
    qcomplex *row_phase = calloc(rows * rows, sizeof(*row_phase));
    qcomplex *col_phase = calloc(half_cols * cols, sizeof(*col_phase));
    qcomplex *terms = calloc(input_count, sizeof(*terms));
    if (!input_dd || !input || !row_phase || !col_phase || !terms) exit(2);

    for (size_t i = 0; i < input_count; ++i) {
        input_dd[i] = q_to_dd(q_real_dd_sample(i));
        input[i] = dd_to_q(input_dd[i]);
    }
    const __float128 two_pi = 2.0Q * qpi();
    for (size_t kr = 0; kr < rows; ++kr) {
        for (size_t row = 0; row < rows; ++row) {
            row_phase[kr * rows + row] =
                qphase(-two_pi * (__float128)(kr * row) / (__float128)rows);
        }
    }
    for (size_t kc = 0; kc < half_cols; ++kc) {
        for (size_t col = 0; col < cols; ++col) {
            col_phase[kc * cols + col] =
                qphase(-two_pi * (__float128)(kc * col) / (__float128)cols);
        }
    }

    const size_t shape[2] = {rows, cols};
    emit_case_header(name, "r2c", shape, 2, "real-double-double-binary64", input_count,
                     "complex-double-double-binary64", output_count);
    for (size_t i = 0; i < input_count; ++i) {
        print_dd(input_dd[i]);
        putchar('\n');
    }
    puts("output-data");
    for (size_t kr = 0; kr < rows; ++kr) {
        for (size_t kc = 0; kc < half_cols; ++kc) {
            size_t term = 0;
            for (size_t row = 0; row < rows; ++row) {
                for (size_t col = 0; col < cols; ++col) {
                    qcomplex phase_value =
                        qmul(row_phase[kr * rows + row], col_phase[kc * cols + col]);
                    terms[term++] = qscale(phase_value, input[row * cols + col]);
                }
            }
            qcomplex sum = qkahan_sum_terms(terms, input_count);
            print_dd(q_to_dd(sum.re));
            putchar(' ');
            print_dd(q_to_dd(sum.im));
            putchar('\n');
        }
    }

    free(terms);
    free(col_phase);
    free(row_phase);
    free(input);
    free(input_dd);
}

static void emit_full_dd_dct2_2d_case(const char *name, size_t rows, size_t cols) {
    const size_t count = rows * cols;
    dd_pair *input_dd = calloc(count, sizeof(*input_dd));
    __float128 *input = calloc(count, sizeof(*input));
    __float128 *row_cos = calloc(rows * rows, sizeof(*row_cos));
    __float128 *col_cos = calloc(cols * cols, sizeof(*col_cos));
    if (!input_dd || !input || !row_cos || !col_cos) exit(2);

    for (size_t i = 0; i < count; ++i) {
        input_dd[i] = q_to_dd(q_real_dd_sample(i));
        input[i] = dd_to_q(input_dd[i]);
    }
    const __float128 pi = qpi();
    for (size_t kr = 0; kr < rows; ++kr) {
        for (size_t row = 0; row < rows; ++row) {
            row_cos[kr * rows + row] =
                cosq(pi * (__float128)kr * (__float128)(2 * row + 1) /
                     (2.0Q * (__float128)rows));
        }
    }
    for (size_t kc = 0; kc < cols; ++kc) {
        for (size_t col = 0; col < cols; ++col) {
            col_cos[kc * cols + col] =
                cosq(pi * (__float128)kc * (__float128)(2 * col + 1) /
                     (2.0Q * (__float128)cols));
        }
    }

    const size_t shape[2] = {rows, cols};
    emit_case_header(name, "dct2", shape, 2, "real-double-double-binary64", count,
                     "real-double-double-binary64", count);
    for (size_t i = 0; i < count; ++i) {
        print_dd(input_dd[i]);
        putchar('\n');
    }
    puts("output-data");
    for (size_t kr = 0; kr < rows; ++kr) {
        for (size_t kc = 0; kc < cols; ++kc) {
            __float128 sum = 0.0Q, correction = 0.0Q;
            for (size_t row = 0; row < rows; ++row) {
                for (size_t col = 0; col < cols; ++col) {
                    __float128 term = input[row * cols + col] * row_cos[kr * rows + row] *
                                      col_cos[kc * cols + col];
                    qkahan_add(term, &sum, &correction);
                }
            }
            print_dd(q_to_dd(4.0Q * sum));
            putchar('\n');
        }
    }

    free(col_cos);
    free(row_cos);
    free(input);
    free(input_dd);
}

static int emit_full_dd_corpus(void) {
    puts("schema 1");
    puts("source libquadmath-direct-113bit-kahan-v1");
    puts("rounding double-double-hi-lo-rne-from-f128-v1");
    puts("input-storage double-double-binary64-v1");
    emit_full_dd_c2c_case("c2c-bluestein-n103", 103);
    emit_full_dd_c2c_case("c2c-rader-n257", 257);
    emit_full_dd_c2c_case("c2c-smooth-n1536", 1536);
    emit_full_dd_c2c_case("c2c-large-smooth-n4096", 4096);
    emit_full_dd_r2c_2d_case("r2c-nd-17x34", 17, 34);
    emit_full_dd_r2c_2d_case("r2c-nd-64x64", 64, 64);
    emit_full_dd_dct2_2d_case("dct2-nd-17x34", 17, 34);
    return 0;
}

int main(int argc, char **argv) {
    if (fesetround(FE_TONEAREST) != 0) return 4;
    if (argc == 2 && strcmp(argv[1], "full-dd") == 0) return emit_full_dd_corpus();
    if (argc != 1) {
        fprintf(stderr, "usage: %s [full-dd]\n", argv[0]);
        return 2;
    }
    puts("schema 1");
    puts("source libquadmath-direct-113bit-v1");
    puts("rounding binary64-rne-from-f128-v1");
    puts("input-storage binary64-exact-v1");
    emit_c2c_case("c2c-bluestein-n103", 103);
    emit_c2c_case("c2c-rader-n257", 257);
    emit_c2c_case("c2c-smooth-n1536", 1536);
    emit_c2c_case("c2c-large-smooth-n4096", 4096);
    emit_r2c_2d_case("r2c-nd-17x34", 17, 34);
    emit_r2c_2d_case("r2c-nd-64x64", 64, 64);
    emit_dct2_2d_case("dct2-nd-17x34", 17, 34);
    return 0;
}
