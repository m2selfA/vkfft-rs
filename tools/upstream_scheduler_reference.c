// Scheduler-only reference extractor for the pinned upstream VkFFT headers.
//
// This intentionally does not initialize a GPU runtime or link VkFFT into vkfft-rs.
// It is compiled as a standalone validation helper by run_upstream_scheduler_reference.sh.
#ifndef VKFFT_BACKEND
#define VKFFT_BACKEND 0
#endif
#define VKFFT_MAX_FFT_DIMENSIONS 4

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define pfLD long double
#define pfINT int64_t
#define pfUINT uint64_t
#define pfceil ceil

#include "vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_Scheduler.h"
#include "upstream_bluestein_auto_padding.h"
#include "vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_AxisBlockSplitter.h"

#define SNAPSHOT_SCHEMA_VERSION 1
#define UPSTREAM_COMMIT "066a17c17068c0f11c9298d848c2976c71fad1c1"

typedef struct {
    const char* case_name;
    const char* vendor_name;
    pfUINT vendor_id;
    pfUINT sequence_len;
    pfUINT outer_batches;
    int prime;
    pfUINT shared_memory_bytes;
    pfUINT shared_memory_pow2_bytes;
    pfUINT coalesced_memory_bytes;
    pfUINT register_boost;
    pfUINT swap_to_three_stage;
    int perform_convolution;
    pfUINT number_kernels;
} RaderCase;

static const RaderCase RADER_CASES[] = {
    {
        "rader-nvidia-p257-batch32",
        "nvidia",
        0x10DE,
        257,
        32,
        257,
        48 * 1024,
        32 * 1024,
        32,
        4,
        4194305,
    },
    {
        .case_name = "rader-nvidia-p257-batch1-convolution",
        .vendor_name = "nvidia",
        .vendor_id = 0x10DE,
        .sequence_len = 257,
        .outer_batches = 1,
        .prime = 257,
        .shared_memory_bytes = 48 * 1024,
        .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32,
        .register_boost = 4,
        .swap_to_three_stage = 4194305,
        .perform_convolution = 1,
    },
    {
        .case_name = "rader-nvidia-p257-batch1-convolution-k3",
        .vendor_name = "nvidia",
        .vendor_id = 0x10DE,
        .sequence_len = 257,
        .outer_batches = 1,
        .prime = 257,
        .shared_memory_bytes = 48 * 1024,
        .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32,
        .register_boost = 4,
        .swap_to_three_stage = 4194305,
        .perform_convolution = 1,
        .number_kernels = 3,
    },
    {
        "rader-nvidia-p769-batch1",
        "nvidia",
        0x10DE,
        769,
        1,
        769,
        64 * 1024,
        64 * 1024,
        32,
        4,
        4194305,
    },
    {
        "rader-nvidia-p2801-batch1",
        "nvidia",
        0x10DE,
        2801,
        1,
        2801,
        64 * 1024,
        64 * 1024,
        32,
        4,
        4194305,
    },
    {
        "rader-nvidia-p257-containers8",
        "nvidia",
        0x10DE,
        2056,
        1,
        257,
        48 * 1024,
        32 * 1024,
        32,
        4,
        4194305,
    },
    {
        "rader-nvidia-p29-containers8",
        "nvidia",
        0x10DE,
        232,
        2,
        29,
        48 * 1024,
        32 * 1024,
        32,
        4,
        4194305,
    },
    {
        "rader-amd-p19-containers8",
        "amd",
        0x1002,
        152,
        1,
        19,
        64 * 1024,
        64 * 1024,
        32,
        2,
        524288,
    },

};

typedef struct {
    const char* case_name;
    pfUINT sequence_len;
    pfUINT shared_memory_bytes;
    pfUINT shared_memory_pow2_bytes;
    pfUINT max_threads_num;
    int fft_rader_prime;
    int double_double;
    int min_direct_prime;
    int max_direct_prime;
    int min_fft_prime;
    int max_fft_prime;
    int profile_kind;
    pfUINT max_workgroup_x;
    pfUINT max_workgroup_y;
    int strided_axis;
    pfUINT fastest_axis_len;
    int bandwidth_boost;
    int half_precision;
    pfUINT batch_count;
    pfUINT grouped_batch_override;
    int perform_zero_padding;
    int double_precision_float_memory;
    int perform_r2c;
    int perform_dct;
    int perform_dst;
    pfUINT axis_id_override;
    pfUINT middle_axis_len;
    pfUINT axis1_grouped_batch_override;
} RaderUploadCase;

static const RaderUploadCase RADER_UPLOAD_CASES[] = {
    {"rader-upload-nvidia-n4089-capacity", 4089, 32 * 1024, 32 * 1024, 1024, 0, 0, 17, 89, 17, 16384},
    {"rader-upload-nvidia-n8704-capacity", 8704, 32 * 1024, 32 * 1024, 1024, 17, 0, 17, 89, 17, 16384},
    {"rader-upload-nvidia-n5100-forced", 5100, 48 * 1024, 32 * 1024, 128, 17, 0, 17, 89, 17, 16384},
    {"rader-upload-nvidia-n1922-three", 1922, 1024, 1024, 32, 31, 0, 17, 89, 17, 16384},
    {"rader-upload-nvidia-n33728-three", 33728, 2 * 1024, 2 * 1024, 64, 31, 0, 17, 89, 17, 16384},
    {"rader-upload-nvidia-n4352-two", 4352, 48 * 1024, 32 * 1024, 128, 17, 0, 17, 89, 17, 16384},
    {"rader-upload-nvidia-n6592-p103-cross-bluestein", 64 * 103, 32 * 1024, 32 * 1024, 1024, 103, 0, 29, 89, 29, 16384},
    {"rader-upload-nvidia-dd-n9367-capacity", 17 * 19 * 29, 256 * 1024, 256 * 1024, 1024, 29, 1, 17, 89, 19, 16384},
    {"rader-upload-nvidia-dd-n3553-capacity-256t", 11 * 17 * 19, 48 * 1024, 32 * 1024, 256, 19, 1, 11, 89, 17, 16384},
    {"rader-upload-nvidia-dd-n15810-capacity", 30 * 17 * 31, 256 * 1024, 256 * 1024, 1024, 31, 1, 17, 89, 19, 16384},
    {"rader-upload-amd-vulkan-n33728-cross-vendor", 33728, 48 * 1024, 32 * 1024, 1024, 31, 0, 17, 89, 17, 16384, 1},
    {"rader-upload-intel-opencl-n33728-cross-vendor", 33728, 32 * 1024, 32 * 1024, 256, 31, 0, 17, 89, 17, 16384, 2},
    {"rader-upload-amd-vulkan-n4089-cross-vendor", 4089, 48 * 1024, 32 * 1024, 1024, 0, 0, 17, 89, 17, 16384, 1},
    {"rader-upload-intel-opencl-n4089-cross-vendor", 4089, 32 * 1024, 32 * 1024, 256, 0, 0, 17, 89, 17, 16384, 2},
    {"rader-upload-amd-vulkan-n6592-p103-cross-bluestein", 64 * 103, 48 * 1024, 32 * 1024, 1024, 103, 0, 29, 89, 29, 16384, 1},
    {"rader-upload-intel-opencl-n6592-p103-cross-bluestein", 64 * 103, 32 * 1024, 32 * 1024, 256, 103, 0, 29, 89, 29, 16384, 2},
    {"rader-upload-intel-opencl-dd-n323-mixed-48t", 17 * 19, 48 * 1024, 32 * 1024, 48, 19, 1, 17, 31, 19, 16384, 2, 48},
    {"rader-upload-intel-opencl-dd-n104329-mixed-48t", 17 * 19 * 17 * 19, 48 * 1024, 32 * 1024, 48, 19, 1, 17, 31, 19, 16384, 2, 48},
    {"rader-upload-nvidia-n69632-k0-floor", 17 * 4096, 32 * 1024, 32 * 1024, 64, 17, 0, 17, 89, 17, 16384, 0, 64},
    {"rader-upload-nvidia-n139264-k1-floor", 17 * 8192, 32 * 1024, 32 * 1024, 64, 17, 0, 17, 89, 17, 16384, 0, 64},
    {"rader-upload-nvidia-n33728-direct-u1-64t", 33728, 2 * 1024, 2 * 1024, 64, 31, 0, 17, 89, 19, 16384, 0, 64, 1024},
    {
        "rader-upload-intel-vulkan-n278528-strided-b0",
        17 * 16384, 32 * 1024, 32 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        3, 1024, 1024, 1, 8, 0,
    },
    {
        "rader-upload-intel-vulkan-n278528-strided-b2",
        17 * 16384, 32 * 1024, 32 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        3, 1024, 1024, 1, 8, 2,
    },
    {
        "rader-upload-nvidia-f16-n278528-32k",
        17 * 16384, 32 * 1024, 32 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        0, 1024, 1024, 0, 0, 0, 1,
    },
    {
        "rader-upload-nvidia-f16-n8789-8k",
        11 * 17 * 47, 8 * 1024, 8 * 1024, 1024, 0, 0, 17, 89, 17, 16384,
        0, 1024, 1024, 0, 0, 0, 1,
    },
    {
        "rader-upload-nvidia-f16-n1922-2k",
        1922, 2 * 1024, 2 * 1024, 32, 31, 0, 17, 89, 17, 16384,
        0, 32, 32, 0, 0, 0, 1,
    },
    {
        "rader-upload-nvidia-f16-n33728-4k",
        17 * 31 * 64, 4 * 1024, 4 * 1024, 1024, 31, 0, 17, 89, 17, 16384,
        0, 1024, 1024, 0, 0, 0, 1,
    },
    {"rader-upload-nvidia-dd-n544-16t", 17 * 32, 48 * 1024, 32 * 1024, 16, 17, 1, 11, 29, 17, 16384},
    {"rader-upload-amd-vulkan-dd-n544-16t", 17 * 32, 48 * 1024, 32 * 1024, 16, 0, 1, 11, 29, 19, 16384, 1},
    {"rader-upload-amd-vulkan-n33728-32k", 33728, 32 * 1024, 32 * 1024, 1024, 31, 0, 17, 89, 17, 16384, 1},
    {
        "rader-upload-amd-vulkan-f16-n33728-4k",
        17 * 31 * 64, 4 * 1024, 4 * 1024, 1024, 31, 0, 17, 89, 17, 16384,
        1, 1024, 1024, 0, 0, 0, 1,
    },
    {
        "rader-upload-amd-vulkan-f32-n9367-32k",
        17 * 19 * 29, 32 * 1024, 32 * 1024, 1024, 29, 0, 17, 89, 19, 16384,
        1, 1024, 1024,
    },
    {
        "rader-upload-amd-vulkan-f32-n102272-4k",
        17 * 47 * 128, 4 * 1024, 4 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        1, 1024, 1024,
    },
    {
        "rader-upload-amd-vulkan-f32-n8789-8k",
        11 * 17 * 47, 8 * 1024, 8 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        1, 1024, 1024,
    },
    {
        "rader-upload-amd-vulkan-f16-n8789-8k",
        11 * 17 * 47, 8 * 1024, 8 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        1, 1024, 1024, 0, 0, 0, 1,
    },
    {
        "rader-upload-amd-vulkan-f32-n1922-1k-32t",
        2 * 31 * 31, 1024, 1024, 32, 31, 0, 17, 89, 17, 16384,
        1, 32, 1024,
    },
    {
        "rader-upload-amd-vulkan-f32-n33728-direct-u1-2k-64t",
        17 * 31 * 64, 2 * 1024, 2 * 1024, 64, 31, 0, 17, 89, 19, 16384,
        1, 64, 1024,
    },
    {
        "rader-upload-amd-vulkan-f32-n4112-p257-32k",
        16 * 257, 32 * 1024, 32 * 1024, 1024, 257, 0, 17, 89, 17, 16384,
        1, 1024, 1024,
    },
    {"rader-upload-amd-vulkan-n139264-32k-64t", 17 * 8192, 32 * 1024, 32 * 1024, 64, 17, 0, 17, 89, 17, 16384, 1, 64},
    {
        "rader-upload-amd-vulkan-f16-n139264-32k",
        17 * 8192, 32 * 1024, 32 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        1, 1024, 1024, 0, 0, 0, 1,
    },
    {
        "rader-upload-amd-vulkan-f32-n139264-b5-g3-32k",
        17 * 8192, 32 * 1024, 32 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        1, 1024, 1024, 0, 0, 0, 0, 5, 3,
    },
    {
        "rader-upload-amd-vulkan-f16-n139264-b5-g3-zp-32k",
        17 * 8192, 32 * 1024, 32 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        1, 1024, 1024, 0, 0, 0, 1, 5, 3, 1,
    },
    {
        "rader-upload-amd-vulkan-f32-n1114112-b2-g3-32k",
        17 * 65536, 32 * 1024, 32 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        1, 1024, 1024, 0, 0, 0, 0, 2, 3,
    },
    {
        "rader-upload-nvidia-vulkan-f32-n278528-strided-b0-32k",
        17 * 16384, 32 * 1024, 32 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        0, 1024, 1024, 1, 8, 0, 0, 0, 0, 0,
    },
    {
        "rader-upload-nvidia-vulkan-f32-n278528-strided-b2-32k",
        17 * 16384, 32 * 1024, 32 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        0, 1024, 1024, 1, 8, 2, 0, 0, 0, 0,
    },
    {
        "rader-upload-amd-vulkan-f32-n278528-strided-b0-32k",
        17 * 16384, 32 * 1024, 32 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        1, 1024, 1024, 1, 8, 0, 0, 0, 0, 0,
    },
    {
        "rader-upload-amd-vulkan-f32-n278528-strided-b2-32k",
        17 * 16384, 32 * 1024, 32 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        1, 1024, 1024, 1, 8, 2, 0, 0, 0, 0,
    },
    {
        "rader-upload-nvidia-vulkan-f16-n278528-strided-b0-32k",
        17 * 16384, 32 * 1024, 32 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        0, 1024, 1024, 1, 8, 0, 1, 0, 0, 0,
    },
    {
        "rader-upload-nvidia-vulkan-f16-n278528-strided-b2-32k",
        17 * 16384, 32 * 1024, 32 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        0, 1024, 1024, 1, 8, 2, 1, 0, 0, 0,
    },
    {
        "rader-upload-amd-vulkan-f16-n278528-strided-b0-32k",
        17 * 16384, 32 * 1024, 32 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        1, 1024, 1024, 1, 8, 0, 1, 0, 0, 0,
    },
    {
        "rader-upload-amd-vulkan-f16-n278528-strided-b2-32k",
        17 * 16384, 32 * 1024, 32 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        1, 1024, 1024, 1, 8, 2, 1, 0, 0, 0,
    },
    {
        "rader-upload-nvidia-vulkan-f16-n278528-strided-b0-b5-g3-32k",
        17 * 16384, 32 * 1024, 32 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        0, 1024, 1024, 1, 8, 0, 1, 5, 3, 0,
    },
    {
        "rader-upload-nvidia-vulkan-f16-n278528-strided-b2-b5-g3-32k",
        17 * 16384, 32 * 1024, 32 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        0, 1024, 1024, 1, 8, 2, 1, 5, 3, 0,
    },
    {
        "rader-upload-amd-vulkan-f16-n278528-strided-b0-b5-g3-32k",
        17 * 16384, 32 * 1024, 32 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        1, 1024, 1024, 1, 8, 0, 1, 5, 3, 0,
    },
    {
        "rader-upload-amd-vulkan-f16-n278528-strided-b2-b5-g3-32k",
        17 * 16384, 32 * 1024, 32 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        1, 1024, 1024, 1, 8, 2, 1, 5, 3, 0,
    },
    {
        "rader-upload-nvidia-vulkan-f16-n278528-strided-b0-b5-g3-zp-32k",
        17 * 16384, 32 * 1024, 32 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        0, 1024, 1024, 1, 8, 0, 1, 5, 3, 1,
    },
    {
        "rader-upload-nvidia-vulkan-f16-n278528-strided-b2-b5-g3-zp-32k",
        17 * 16384, 32 * 1024, 32 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        0, 1024, 1024, 1, 8, 2, 1, 5, 3, 1,
    },
    {
        "rader-upload-amd-vulkan-f16-n278528-strided-b0-b5-g3-zp-32k",
        17 * 16384, 32 * 1024, 32 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        1, 1024, 1024, 1, 8, 0, 1, 5, 3, 1,
    },
    {
        "rader-upload-amd-vulkan-f16-n278528-strided-b2-b5-g3-zp-32k",
        17 * 16384, 32 * 1024, 32 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        1, 1024, 1024, 1, 8, 2, 1, 5, 3, 1,
    },
    {
        "rader-upload-nvidia-vulkan-f64-f32-storage-n278528-strided-b0-b5-g3-zp-32k",
        17 * 16384, 32 * 1024, 32 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        0, 1024, 1024, 1, 8, 0, 0, 5, 3, 1, 1,
    },
    {
        "rader-upload-nvidia-vulkan-f64-f32-storage-n278528-strided-b2-b5-g3-zp-32k",
        17 * 16384, 32 * 1024, 32 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        0, 1024, 1024, 1, 8, 2, 0, 5, 3, 1, 1,
    },
    {
        "rader-upload-amd-vulkan-f64-f32-storage-n278528-strided-b0-b5-g3-zp-32k",
        17 * 16384, 32 * 1024, 32 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        1, 1024, 1024, 1, 8, 0, 0, 5, 3, 1, 1,
    },
    {
        "rader-upload-amd-vulkan-f64-f32-storage-n278528-strided-b2-b5-g3-zp-32k",
        17 * 16384, 32 * 1024, 32 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        1, 1024, 1024, 1, 8, 2, 0, 5, 3, 1, 1,
    },
    {
        "rader-upload-nvidia-vulkan-f16-nd-r2c-n278528x14-strided-b0-b5-g3-zp-32k",
        17 * 16384, 32 * 1024, 32 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        0, 1024, 1024, 1, 14, 0, 1, 5, 3, 1, 0, 1,
    },
    {
        "rader-upload-nvidia-vulkan-f16-nd-r2c-n278528x14-strided-b2-b5-g3-zp-32k",
        17 * 16384, 32 * 1024, 32 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        0, 1024, 1024, 1, 14, 2, 1, 5, 3, 1, 0, 1,
    },
    {
        "rader-upload-amd-vulkan-f16-nd-r2c-n278528x14-strided-b0-b5-g3-zp-32k",
        17 * 16384, 32 * 1024, 32 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        1, 1024, 1024, 1, 14, 0, 1, 5, 3, 1, 0, 1,
    },
    {
        "rader-upload-amd-vulkan-f16-nd-r2c-n278528x14-strided-b2-b5-g3-zp-32k",
        17 * 16384, 32 * 1024, 32 * 1024, 1024, 17, 0, 17, 89, 17, 16384,
        1, 1024, 1024, 1, 14, 2, 1, 5, 3, 1, 0, 1,
    },
    {
        "rader-upload-nvidia-vulkan-dd-n391-nested-p23-strided-b5-g3-zp",
        17 * 23, 32 * 1024, 32 * 1024, 1024, 23, 1, 17, 89, 17, 16384,
        0, 1024, 1024, 1, 8, 0, 0, 5, 3, 1,
    },
    {
        "rader-upload-amd-vulkan-dd-n391-nested-p23-strided-b5-g3-zp",
        17 * 23, 32 * 1024, 32 * 1024, 1024, 23, 1, 17, 89, 17, 16384,
        1, 1024, 1024, 1, 8, 0, 0, 5, 3, 1,
    },
    {
        "rader-upload-nvidia-vulkan-dd-n102272-three-strided-b5-g3-zp-8k",
        17 * 47 * 128, 8 * 1024, 8 * 1024, 1024, 17, 1, 17, 89, 17, 16384,
        0, 1024, 1024, 1, 8, 0, 0, 5, 3, 1,
    },
    {
        "rader-upload-amd-vulkan-dd-n102272-three-strided-b5-g3-zp-8k",
        17 * 47 * 128, 8 * 1024, 8 * 1024, 1024, 17, 1, 17, 89, 17, 16384,
        1, 1024, 1024, 1, 8, 0, 0, 5, 3, 1,
    },
    {
        "rader-upload-nvidia-vulkan-dd-nd-r2c-n391x14-nested-p23-strided-b5-g3-zp-32k",
        17 * 23, 32 * 1024, 32 * 1024, 1024, 23, 1, 17, 89, 17, 16384,
        0, 1024, 1024, 1, 14, 0, 0, 5, 3, 1, 0, 1,
    },
    {
        "rader-upload-amd-vulkan-dd-nd-r2c-n391x14-nested-p23-strided-b5-g3-zp-32k",
        17 * 23, 32 * 1024, 32 * 1024, 1024, 23, 1, 17, 89, 17, 16384,
        1, 1024, 1024, 1, 14, 0, 0, 5, 3, 1, 0, 1,
    },
    {
        "rader-upload-nvidia-vulkan-dd-nd-r2c-n391x126-nested-p23-strided-b5-g3-zp-32k",
        17 * 23, 32 * 1024, 32 * 1024, 1024, 23, 1, 17, 89, 17, 16384,
        0, 1024, 1024, 1, 126, 0, 0, 5, 3, 1, 0, 1,
    },
    {
        "rader-upload-amd-vulkan-dd-nd-r2c-n391x126-nested-p23-strided-b5-g3-zp-32k",
        17 * 23, 32 * 1024, 32 * 1024, 1024, 23, 1, 17, 89, 17, 16384,
        1, 1024, 1024, 1, 126, 0, 0, 5, 3, 1, 0, 1,
    },
    {
        "rader-upload-nvidia-vulkan-dd-nd-dct4-n782x8-child391-nested-p23-strided-b5-g3-zp-32k",
        782, 32 * 1024, 32 * 1024, 1024, 23, 1, 17, 89, 17, 16384,
        0, 1024, 1024, 1, 8, 0, 0, 5, 3, 1, 0, 0, 4,
    },
    {
        "rader-upload-amd-vulkan-dd-nd-dct4-n782x8-child391-nested-p23-strided-b5-g3-zp-32k",
        782, 32 * 1024, 32 * 1024, 1024, 23, 1, 17, 89, 17, 16384,
        1, 1024, 1024, 1, 8, 0, 0, 5, 3, 1, 0, 0, 4,
    },
    {
        "rader-upload-nvidia-vulkan-dd-nd-dct1-n392x8-child782-nested-p23-strided-b5-g3-zp-32k",
        392, 32 * 1024, 32 * 1024, 1024, 23, 1, 17, 89, 17, 16384,
        0, 1024, 1024, 1, 8, 0, 0, 5, 3, 1, 0, 0, 1,
    },
    {
        "rader-upload-amd-vulkan-dd-nd-dct1-n392x8-child782-nested-p23-strided-b5-g3-zp-32k",
        392, 32 * 1024, 32 * 1024, 1024, 23, 1, 17, 89, 17, 16384,
        1, 1024, 1024, 1, 8, 0, 0, 5, 3, 1, 0, 0, 1,
    },
    {
        "rader-upload-intel-opencl-dd-nd-dct1-n392x8-child782-nested-p23-strided-b5-g3-zp-32k",
        392, 32 * 1024, 32 * 1024, 512, 23, 1, 17, 89, 17, 16384,
        2, 512, 512, 1, 8, 0, 0, 5, 3, 1, 0, 0, 1,
    },
    {
        "rader-upload-intel-level-zero-dd-nd-dct1-n392x8-child782-nested-p23-strided-b5-g3-zp-32k",
        392, 32 * 1024, 32 * 1024, 512, 23, 1, 17, 89, 17, 16384,
        4, 512, 512, 1, 8, 0, 0, 5, 3, 1, 0, 0, 1,
    },
    {
        "rader-upload-nvidia-vulkan-dd-nd-dct4-n204544x8-child102272-three-strided-b5-g3-zp-8k",
        204544, 8 * 1024, 8 * 1024, 1024, 17, 1, 17, 89, 17, 16384,
        0, 1024, 1024, 1, 8, 0, 0, 5, 3, 1, 0, 0, 4,
    },
    {
        "rader-upload-amd-vulkan-dd-nd-dct4-n204544x8-child102272-three-strided-b5-g3-zp-8k",
        204544, 8 * 1024, 8 * 1024, 1024, 17, 1, 17, 89, 17, 16384,
        1, 1024, 1024, 1, 8, 0, 0, 5, 3, 1, 0, 0, 4,
    },
    {
        "rader-upload-nvidia-vulkan-dd-nd-dct2-n391x8-npoint-nested-p23-strided-b5-g3-zp-32k",
        391, 32 * 1024, 32 * 1024, 1024, 23, 1, 17, 89, 17, 16384,
        0, 1024, 1024, 1, 8, 0, 0, 5, 3, 1, 0, 0, 2,
    },
    {
        "rader-upload-amd-vulkan-dd-nd-dct2-n391x8-npoint-nested-p23-strided-b5-g3-zp-32k",
        391, 32 * 1024, 32 * 1024, 1024, 23, 1, 17, 89, 17, 16384,
        1, 1024, 1024, 1, 8, 0, 0, 5, 3, 1, 0, 0, 2,
    },
    {
        "rader-upload-nvidia-vulkan-dd-nd-dst1-n390x8-child782-nested-p23-strided-b5-g3-zp-32k",
        390, 32 * 1024, 32 * 1024, 1024, 23, 1, 17, 89, 17, 16384,
        0, 1024, 1024, 1, 8, 0, 0, 5, 3, 1, 0, 0, 0, 1,
    },
    {
        "rader-upload-amd-vulkan-dd-nd-dst1-n390x8-child782-nested-p23-strided-b5-g3-zp-32k",
        390, 32 * 1024, 32 * 1024, 1024, 23, 1, 17, 89, 17, 16384,
        1, 1024, 1024, 1, 8, 0, 0, 5, 3, 1, 0, 0, 0, 1,
    },
    {
        "rader-upload-nvidia-vulkan-dd-nd-dct2-n102272x8-npoint-three-strided-b5-g3-zp-8k",
        102272, 8 * 1024, 8 * 1024, 1024, 17, 1, 17, 89, 17, 16384,
        0, 1024, 1024, 1, 8, 0, 0, 5, 3, 1, 0, 0, 2,
    },
    {
        "rader-upload-amd-vulkan-dd-nd-dct2-n102272x8-npoint-three-strided-b5-g3-zp-8k",
        102272, 8 * 1024, 8 * 1024, 1024, 17, 1, 17, 89, 17, 16384,
        1, 1024, 1024, 1, 8, 0, 0, 5, 3, 1, 0, 0, 2,
    }
};

typedef struct {
    const char* case_name;
    size_t rader_upload_case_index;
    pfUINT upload_id;
} RaderUploadAxisBlockReference;

static const RaderUploadAxisBlockReference RADER_UPLOAD_AXIS_BLOCK_REFERENCES[] = {
    {"axis-block-nvidia-f32-n8704-rader-u0-stockham", 1, 0},
    {"axis-block-nvidia-f32-n5100-rader-u1-stockham", 2, 1},
    {"axis-block-nvidia-f32-n1922-rader-u0-stockham", 3, 0},
    {"axis-block-nvidia-f32-n1922-rader-u1-p31", 3, 1},
    {"axis-block-nvidia-f32-n1922-rader-u2-p31", 3, 2},
    {"axis-block-nvidia-f32-n33728-rader-u0-stockham", 4, 0},
    {"axis-block-nvidia-f32-n33728-rader-u2-p31", 4, 2},
    {"axis-block-nvidia-f32-n4352-rader-u0-stockham", 5, 0},
    {"axis-block-nvidia-f32-n6592-rader-u1-p103-cross-bluestein", 6, 1},
    {"axis-block-amd-vulkan-f32-n33728-rader-u0", 10, 0},
    {"axis-block-amd-vulkan-f32-n33728-rader-u1", 10, 1},
    {"axis-block-intel-opencl-f32-n33728-rader-u0", 11, 0},
    {"axis-block-intel-opencl-f32-n33728-rader-u1", 11, 1},
    {"axis-block-amd-vulkan-f32-n33728-32k-u0", 29, 0},
    {"axis-block-amd-vulkan-f32-n33728-32k-u1", 29, 1},
    {"axis-block-amd-vulkan-f16-n33728-rader-u0", 30, 0},
    {"axis-block-amd-vulkan-f16-n33728-rader-u1", 30, 1},
    {"axis-block-amd-vulkan-f16-n33728-rader-u2", 30, 2},
    {"axis-block-amd-vulkan-f32-n9367-32k-u0", 31, 0},
    {"axis-block-amd-vulkan-f32-n9367-32k-u1", 31, 1},
    {"axis-block-amd-vulkan-f32-n102272-4k-u0", 32, 0},
    {"axis-block-amd-vulkan-f32-n102272-4k-u1", 32, 1},
    {"axis-block-amd-vulkan-f32-n102272-4k-u2", 32, 2},
    {"axis-block-amd-vulkan-f32-n8789-8k-u0", 33, 0},
    {"axis-block-amd-vulkan-f32-n8789-8k-u1", 33, 1},
    {"axis-block-amd-vulkan-f16-n8789-8k-u0", 34, 0},
    {"axis-block-amd-vulkan-f16-n8789-8k-u1", 34, 1},
    {"axis-block-amd-vulkan-f16-n8789-8k-u2", 34, 2},
    {"axis-block-amd-vulkan-f32-n1922-1k-32t-u0", 35, 0},
    {"axis-block-amd-vulkan-f32-n1922-1k-32t-u1", 35, 1},
    {"axis-block-amd-vulkan-f32-n1922-1k-32t-u2", 35, 2},
    {"axis-block-amd-vulkan-f32-n33728-direct-u1-2k-64t-u0", 36, 0},
    {"axis-block-amd-vulkan-f32-n33728-direct-u1-2k-64t-u1", 36, 1},
    {"axis-block-amd-vulkan-f32-n33728-direct-u1-2k-64t-u2", 36, 2},
    {"axis-block-amd-vulkan-f32-n4112-p257-32k-u0", 37, 0},
    {"axis-block-amd-vulkan-f32-n4112-p257-32k-u1", 37, 1},
    {"axis-block-amd-vulkan-f32-n4089-rader-u0-single", 12, 0},
    {"axis-block-intel-opencl-f32-n4089-rader-u0", 13, 0},
    {"axis-block-intel-opencl-f32-n4089-rader-u1-p47", 13, 1},
    {"axis-block-amd-vulkan-f32-n6592-rader-u0-stockham", 14, 0},
    {"axis-block-amd-vulkan-f32-n6592-rader-u1-p103-cross-bluestein", 14, 1},
    {"axis-block-intel-opencl-f32-n6592-rader-u0-stockham", 15, 0},
    {"axis-block-intel-opencl-f32-n6592-rader-u1-p103-cross-bluestein", 15, 1},
    {"axis-block-intel-opencl-dd-n323-mixed-48t-u0", 16, 0},
    {"axis-block-intel-opencl-dd-n104329-mixed-48t-u0", 17, 0},
    {"axis-block-intel-opencl-dd-n104329-mixed-48t-u1", 17, 1},
    {"axis-block-nvidia-f32-n69632-rader-u0-p17", 18, 0},
    {"axis-block-nvidia-f32-n139264-rader-u1-p17", 19, 1},
    {"axis-block-nvidia-f32-n33728-rader-u1-direct-64t", 20, 1},
    {"axis-block-nvidia-f16-n278528-rader-u0", 23, 0},
    {"axis-block-nvidia-f16-n278528-rader-u1", 23, 1},
    {"axis-block-nvidia-f16-n278528-rader-u2", 23, 2},
    {"axis-block-nvidia-f16-n8789-rader-u0", 24, 0},
    {"axis-block-nvidia-f16-n8789-rader-u1", 24, 1},
    {"axis-block-nvidia-f16-n8789-rader-u2", 24, 2},
    {"axis-block-nvidia-f16-n1922-rader-u0", 25, 0},
    {"axis-block-nvidia-f16-n1922-rader-u1", 25, 1},
    {"axis-block-nvidia-f16-n1922-rader-u2", 25, 2},
    {"axis-block-nvidia-f16-n33728-rader-u0", 26, 0},
    {"axis-block-nvidia-f16-n33728-rader-u1", 26, 1},
    {"axis-block-nvidia-f16-n33728-rader-u2", 26, 2},
    {"axis-block-amd-vulkan-f32-n139264-32k-64t-u0", 38, 0},
    {"axis-block-amd-vulkan-f32-n139264-32k-64t-u1", 38, 1},
    {"axis-block-amd-vulkan-f16-n139264-32k-u0", 39, 0},
    {"axis-block-amd-vulkan-f16-n139264-32k-u1", 39, 1},
    {"axis-block-amd-vulkan-f32-n139264-b5-g3-32k-u0", 40, 0},
    {"axis-block-amd-vulkan-f32-n139264-b5-g3-32k-u1", 40, 1},
    {"axis-block-amd-vulkan-f16-n139264-b5-g3-zp-32k-u0", 41, 0},
    {"axis-block-amd-vulkan-f16-n139264-b5-g3-zp-32k-u1", 41, 1},
    {"axis-block-amd-vulkan-f32-n1114112-b2-g3-32k-u0", 42, 0},
    {"axis-block-amd-vulkan-f32-n1114112-b2-g3-32k-u1", 42, 1},
    {"axis-block-amd-vulkan-f32-n1114112-b2-g3-32k-u2", 42, 2},
    {"axis-block-nvidia-vulkan-f32-n278528-strided-b0-u0", 43, 0},
    {"axis-block-nvidia-vulkan-f32-n278528-strided-b0-u1", 43, 1},
    {"axis-block-nvidia-vulkan-f32-n278528-strided-b2-u0", 44, 0},
    {"axis-block-nvidia-vulkan-f32-n278528-strided-b2-u1", 44, 1},
    {"axis-block-amd-vulkan-f32-n278528-strided-b0-u0", 45, 0},
    {"axis-block-amd-vulkan-f32-n278528-strided-b0-u1", 45, 1},
    {"axis-block-amd-vulkan-f32-n278528-strided-b2-u0", 46, 0},
    {"axis-block-amd-vulkan-f32-n278528-strided-b2-u1", 46, 1},
    {"axis-block-nvidia-vulkan-f16-n278528-strided-b0-u0", 47, 0},
    {"axis-block-nvidia-vulkan-f16-n278528-strided-b0-u1", 47, 1},
    {"axis-block-nvidia-vulkan-f16-n278528-strided-b0-u2", 47, 2},
    {"axis-block-nvidia-vulkan-f16-n278528-strided-b2-u0", 48, 0},
    {"axis-block-nvidia-vulkan-f16-n278528-strided-b2-u1", 48, 1},
    {"axis-block-amd-vulkan-f16-n278528-strided-b0-u0", 49, 0},
    {"axis-block-amd-vulkan-f16-n278528-strided-b0-u1", 49, 1},
    {"axis-block-amd-vulkan-f16-n278528-strided-b0-u2", 49, 2},
    {"axis-block-amd-vulkan-f16-n278528-strided-b2-u0", 50, 0},
    {"axis-block-amd-vulkan-f16-n278528-strided-b2-u1", 50, 1},
    {"axis-block-nvidia-vulkan-f16-n278528-strided-b0-b5-g3-u0", 51, 0},
    {"axis-block-nvidia-vulkan-f16-n278528-strided-b0-b5-g3-u1", 51, 1},
    {"axis-block-nvidia-vulkan-f16-n278528-strided-b0-b5-g3-u2", 51, 2},
    {"axis-block-nvidia-vulkan-f16-n278528-strided-b2-b5-g3-u0", 52, 0},
    {"axis-block-nvidia-vulkan-f16-n278528-strided-b2-b5-g3-u1", 52, 1},
    {"axis-block-amd-vulkan-f16-n278528-strided-b0-b5-g3-u0", 53, 0},
    {"axis-block-amd-vulkan-f16-n278528-strided-b0-b5-g3-u1", 53, 1},
    {"axis-block-amd-vulkan-f16-n278528-strided-b0-b5-g3-u2", 53, 2},
    {"axis-block-amd-vulkan-f16-n278528-strided-b2-b5-g3-u0", 54, 0},
    {"axis-block-amd-vulkan-f16-n278528-strided-b2-b5-g3-u1", 54, 1},
    {"axis-block-nvidia-vulkan-f16-n278528-strided-b0-b5-g3-zp-u0", 55, 0},
    {"axis-block-nvidia-vulkan-f16-n278528-strided-b0-b5-g3-zp-u1", 55, 1},
    {"axis-block-nvidia-vulkan-f16-n278528-strided-b0-b5-g3-zp-u2", 55, 2},
    {"axis-block-nvidia-vulkan-f16-n278528-strided-b2-b5-g3-zp-u0", 56, 0},
    {"axis-block-nvidia-vulkan-f16-n278528-strided-b2-b5-g3-zp-u1", 56, 1},
    {"axis-block-amd-vulkan-f16-n278528-strided-b0-b5-g3-zp-u0", 57, 0},
    {"axis-block-amd-vulkan-f16-n278528-strided-b0-b5-g3-zp-u1", 57, 1},
    {"axis-block-amd-vulkan-f16-n278528-strided-b0-b5-g3-zp-u2", 57, 2},
    {"axis-block-amd-vulkan-f16-n278528-strided-b2-b5-g3-zp-u0", 58, 0},
    {"axis-block-amd-vulkan-f16-n278528-strided-b2-b5-g3-zp-u1", 58, 1},
    {"axis-block-nvidia-vulkan-f64-f32-storage-n278528-strided-b0-b5-g3-zp-u0", 59, 0},
    {"axis-block-nvidia-vulkan-f64-f32-storage-n278528-strided-b0-b5-g3-zp-u1", 59, 1},
    {"axis-block-nvidia-vulkan-f64-f32-storage-n278528-strided-b2-b5-g3-zp-u0", 60, 0},
    {"axis-block-nvidia-vulkan-f64-f32-storage-n278528-strided-b2-b5-g3-zp-u1", 60, 1},
    {"axis-block-amd-vulkan-f64-f32-storage-n278528-strided-b0-b5-g3-zp-u0", 61, 0},
    {"axis-block-amd-vulkan-f64-f32-storage-n278528-strided-b0-b5-g3-zp-u1", 61, 1},
    {"axis-block-amd-vulkan-f64-f32-storage-n278528-strided-b2-b5-g3-zp-u0", 62, 0},
    {"axis-block-amd-vulkan-f64-f32-storage-n278528-strided-b2-b5-g3-zp-u1", 62, 1},
    {"axis-block-nvidia-vulkan-f16-nd-r2c-n278528x14-strided-b0-b5-g3-zp-u0", 63, 0},
    {"axis-block-nvidia-vulkan-f16-nd-r2c-n278528x14-strided-b0-b5-g3-zp-u1", 63, 1},
    {"axis-block-nvidia-vulkan-f16-nd-r2c-n278528x14-strided-b0-b5-g3-zp-u2", 63, 2},
    {"axis-block-nvidia-vulkan-f16-nd-r2c-n278528x14-strided-b2-b5-g3-zp-u0", 64, 0},
    {"axis-block-nvidia-vulkan-f16-nd-r2c-n278528x14-strided-b2-b5-g3-zp-u1", 64, 1},
    {"axis-block-amd-vulkan-f16-nd-r2c-n278528x14-strided-b0-b5-g3-zp-u0", 65, 0},
    {"axis-block-amd-vulkan-f16-nd-r2c-n278528x14-strided-b0-b5-g3-zp-u1", 65, 1},
    {"axis-block-amd-vulkan-f16-nd-r2c-n278528x14-strided-b0-b5-g3-zp-u2", 65, 2},
    {"axis-block-amd-vulkan-f16-nd-r2c-n278528x14-strided-b2-b5-g3-zp-u0", 66, 0},
    {"axis-block-amd-vulkan-f16-nd-r2c-n278528x14-strided-b2-b5-g3-zp-u1", 66, 1},
    {"axis-block-nvidia-vulkan-dd-n391-nested-p23-strided-b5-g3-zp-u0", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 22, 0},
    {"axis-block-amd-vulkan-dd-n391-nested-p23-strided-b5-g3-zp-u0", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 21, 0},
    {"axis-block-nvidia-vulkan-dd-n102272-three-strided-b5-g3-zp-8k-u0", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 20, 0},
    {"axis-block-nvidia-vulkan-dd-n102272-three-strided-b5-g3-zp-8k-u1", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 20, 1},
    {"axis-block-nvidia-vulkan-dd-n102272-three-strided-b5-g3-zp-8k-u2", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 20, 2},
    {"axis-block-amd-vulkan-dd-n102272-three-strided-b5-g3-zp-8k-u0", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 19, 0},
    {"axis-block-amd-vulkan-dd-n102272-three-strided-b5-g3-zp-8k-u1", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 19, 1},
    {"axis-block-amd-vulkan-dd-n102272-three-strided-b5-g3-zp-8k-u2", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 19, 2},
    {"axis-block-nvidia-vulkan-dd-nd-r2c-n391x14-nested-p23-strided-b5-g3-zp-32k-u0", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 18, 0},
    {"axis-block-amd-vulkan-dd-nd-r2c-n391x14-nested-p23-strided-b5-g3-zp-32k-u0", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 17, 0},
    {"axis-block-nvidia-vulkan-dd-nd-r2c-n391x126-nested-p23-strided-b5-g3-zp-32k-u0", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 16, 0},
    {"axis-block-amd-vulkan-dd-nd-r2c-n391x126-nested-p23-strided-b5-g3-zp-32k-u0", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 15, 0},
    {"axis-block-nvidia-vulkan-dd-nd-dct4-n782x8-child391-nested-p23-strided-b5-g3-zp-32k-u0", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 14, 0},
    {"axis-block-amd-vulkan-dd-nd-dct4-n782x8-child391-nested-p23-strided-b5-g3-zp-32k-u0", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 13, 0},
    {"axis-block-nvidia-vulkan-dd-nd-dct1-n392x8-child782-nested-p23-strided-b5-g3-zp-32k-u0", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 12, 0},
    {"axis-block-amd-vulkan-dd-nd-dct1-n392x8-child782-nested-p23-strided-b5-g3-zp-32k-u0", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 11, 0},
    {"axis-block-intel-opencl-dd-nd-dct1-n392x8-child782-nested-p23-strided-b5-g3-zp-32k-u0", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 10, 0},
    {"axis-block-intel-opencl-dd-nd-dct1-n392x8-child782-nested-p23-strided-b5-g3-zp-32k-u1", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 10, 1},
    {"axis-block-intel-level-zero-dd-nd-dct1-n392x8-child782-nested-p23-strided-b5-g3-zp-32k-u0", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 9, 0},
    {"axis-block-intel-level-zero-dd-nd-dct1-n392x8-child782-nested-p23-strided-b5-g3-zp-32k-u1", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 9, 1},
    {"axis-block-nvidia-vulkan-dd-nd-dct4-n204544x8-child102272-three-strided-b5-g3-zp-8k-u0", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 8, 0},
    {"axis-block-nvidia-vulkan-dd-nd-dct4-n204544x8-child102272-three-strided-b5-g3-zp-8k-u1", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 8, 1},
    {"axis-block-nvidia-vulkan-dd-nd-dct4-n204544x8-child102272-three-strided-b5-g3-zp-8k-u2", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 8, 2},
    {"axis-block-amd-vulkan-dd-nd-dct4-n204544x8-child102272-three-strided-b5-g3-zp-8k-u0", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 7, 0},
    {"axis-block-amd-vulkan-dd-nd-dct4-n204544x8-child102272-three-strided-b5-g3-zp-8k-u1", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 7, 1},
    {"axis-block-amd-vulkan-dd-nd-dct4-n204544x8-child102272-three-strided-b5-g3-zp-8k-u2", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 7, 2},
    {"axis-block-nvidia-vulkan-dd-nd-dct2-n391x8-npoint-nested-p23-strided-b5-g3-zp-32k-u0", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 6, 0},
    {"axis-block-amd-vulkan-dd-nd-dct2-n391x8-npoint-nested-p23-strided-b5-g3-zp-32k-u0", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 5, 0},
    {"axis-block-nvidia-vulkan-dd-nd-dst1-n390x8-child782-nested-p23-strided-b5-g3-zp-32k-u0", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 4, 0},
    {"axis-block-amd-vulkan-dd-nd-dst1-n390x8-child782-nested-p23-strided-b5-g3-zp-32k-u0", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 3, 0},
    {"axis-block-nvidia-vulkan-dd-nd-dct2-n102272x8-npoint-three-strided-b5-g3-zp-8k-u0", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 2, 0},
    {"axis-block-nvidia-vulkan-dd-nd-dct2-n102272x8-npoint-three-strided-b5-g3-zp-8k-u1", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 2, 1},
    {"axis-block-nvidia-vulkan-dd-nd-dct2-n102272x8-npoint-three-strided-b5-g3-zp-8k-u2", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 2, 2},
    {"axis-block-amd-vulkan-dd-nd-dct2-n102272x8-npoint-three-strided-b5-g3-zp-8k-u0", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 1, 0},
    {"axis-block-amd-vulkan-dd-nd-dct2-n102272x8-npoint-three-strided-b5-g3-zp-8k-u1", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 1, 1},
    {"axis-block-amd-vulkan-dd-nd-dct2-n102272x8-npoint-three-strided-b5-g3-zp-8k-u2", sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]) - 1, 2},
};

typedef struct {
    const char* case_name;
    const char* backend_name;
    const char* vendor_name;
    pfUINT vendor_id;
    pfUINT sequence_len;
    pfUINT batch_count;
    pfUINT shared_memory_bytes;
    pfUINT shared_memory_pow2_bytes;
    pfUINT max_threads_num;
    pfUINT max_workgroup_x;
    pfUINT coalesced_memory_bytes;
    pfUINT warp_size;
    int register_boost;
    pfUINT swap_threshold;
    int min_direct_prime;
    int max_direct_prime;
    int min_fft_prime;
    int max_fft_prime;
} ForcedAxisBlockCase;

static const ForcedAxisBlockCase FORCED_AXIS_BLOCK_CASES[] = {
    {
        "forced-axis-block-nvidia-dd-n9367-256t",
        "vulkan",
        "nvidia",
        0x10DE,
        17 * 19 * 29,
        1,
        48 * 1024,
        32 * 1024,
        256,
        256,
        32,
        32,
        4,
        4194305,
        17,
        89,
        19,
        16384,
    },
    {
        "forced-axis-block-intel-opencl-dd-n1054-32t",
        "opencl",
        "intel",
        0x8086,
        2 * 17 * 31,
        1,
        48 * 1024,
        32 * 1024,
        32,
        32,
        64,
        32,
        2,
        262144,
        17,
        31,
        19,
        16384,
    },
    {
        "forced-axis-block-nvidia-dd-n3196-three",
        "vulkan",
        "nvidia",
        0x10DE,
        4 * 17 * 47,
        1,
        3 * 1024,
        3 * 1024,
        128,
        1024,
        32,
        32,
        4,
        4194305,
        17,
        89,
        17,
        16384,
    },
    {
        "forced-axis-block-nvidia-dd-n1922-three",
        "vulkan",
        "nvidia",
        0x10DE,
        2 * 31 * 31,
        1,
        1024,
        1024,
        32,
        1024,
        32,
        32,
        4,
        4194305,
        17,
        89,
        17,
        16384,
    },
    {
        "forced-axis-block-nvidia-dd-n102272-three",
        "vulkan",
        "nvidia",
        0x10DE,
        17 * 47 * 128,
        1,
        8 * 1024,
        8 * 1024,
        1024,
        1024,
        32,
        32,
        4,
        4194305,
        17,
        89,
        17,
        16384,
    },
    {
        "forced-axis-block-amd-vulkan-dd-n9367-48k",
        "vulkan",
        "amd",
        0x1002,
        17 * 19 * 29,
        1,
        48 * 1024,
        32 * 1024,
        1024,
        1024,
        32,
        64,
        4,
        524288,
        11,
        29,
        19,
        16384,
    },
    {
        "forced-axis-block-amd-vulkan-dd-n9367-64k",
        "vulkan",
        "amd",
        0x1002,
        17 * 19 * 29,
        1,
        64 * 1024,
        64 * 1024,
        1024,
        1024,
        32,
        64,
        2,
        524288,
        11,
        29,
        19,
        16384,
    },
    {
        "forced-axis-block-intel-opencl-dd-n9367",
        "opencl",
        "intel",
        0x8086,
        17 * 19 * 29,
        1,
        32 * 1024,
        32 * 1024,
        256,
        256,
        64,
        32,
        2,
        524288,
        11,
        29,
        17,
        16384,
    },
    {
        "forced-axis-block-amd-vulkan-dd-n9367-32k-portable",
        "vulkan",
        "amd",
        0x1002,
        17 * 19 * 29,
        1,
        32 * 1024,
        32 * 1024,
        1024,
        1024,
        32,
        64,
        4,
        524288,
        17,
        89,
        19,
        16384,
    },
    {
        "forced-axis-block-amd-vulkan-dd-n9367-32k-256t",
        "vulkan",
        "amd",
        0x1002,
        17 * 19 * 29,
        1,
        32 * 1024,
        32 * 1024,
        256,
        256,
        32,
        64,
        4,
        524288,
        17,
        89,
        19,
        16384,
    },
    {
        "forced-axis-block-amd-vulkan-dd-n4352-32k-128t",
        "vulkan",
        "amd",
        0x1002,
        17 * 256,
        1,
        32 * 1024,
        32 * 1024,
        128,
        1024,
        32,
        64,
        4,
        524288,
        11,
        29,
        19,
        16384,
    },
    {
        "forced-axis-block-amd-vulkan-dd-n5100-32k-128t",
        "vulkan",
        "amd",
        0x1002,
        17 * 300,
        1,
        32 * 1024,
        32 * 1024,
        128,
        1024,
        32,
        64,
        4,
        524288,
        11,
        29,
        19,
        16384,
    },
    {
        "forced-axis-block-nvidia-vulkan-dd-n4352-48k-128t",
        "vulkan",
        "nvidia",
        0x10DE,
        17 * 256,
        1,
        48 * 1024,
        32 * 1024,
        128,
        1024,
        32,
        32,
        4,
        4194305,
        11,
        29,
        17,
        16384,
    },
    {
        "forced-axis-block-amd-vulkan-dd-n102272-three-8k",
        "vulkan",
        "amd",
        0x1002,
        17 * 47 * 128,
        1,
        8 * 1024,
        8 * 1024,
        1024,
        1024,
        32,
        64,
        4,
        524288,
        17,
        89,
        17,
        16384,
    },
    {
        "forced-axis-block-nvidia-vulkan-dd-n5100-48k-128t",
        "vulkan",
        "nvidia",
        0x10DE,
        17 * 300,
        1,
        48 * 1024,
        32 * 1024,
        128,
        1024,
        32,
        32,
        4,
        4194305,
        11,
        29,
        17,
        16384,
    },
    {
        "forced-axis-block-amd-vulkan-dd-n5797-24k-128t",
        "vulkan",
        "amd",
        0x1002,
        11 * 17 * 31,
        1,
        24 * 1024,
        24 * 1024,
        128,
        1024,
        32,
        64,
        4,
        524288,
        11,
        29,
        19,
        16384,
    },
    {
        "forced-axis-block-amd-vulkan-dd-n8789-8k",
        "vulkan",
        "amd",
        0x1002,
        11 * 17 * 47,
        1,
        8 * 1024,
        8 * 1024,
        1024,
        1024,
        32,
        64,
        4,
        524288,
        11,
        89,
        17,
        16384,
    },
    {
        "forced-axis-block-nvidia-vulkan-dd-n5797-24k-128t",
        "vulkan",
        "nvidia",
        0x10DE,
        11 * 17 * 31,
        1,
        24 * 1024,
        24 * 1024,
        128,
        1024,
        32,
        32,
        4,
        4194305,
        11,
        29,
        17,
        16384,
    },
    {
        "forced-axis-block-nvidia-vulkan-dd-n8789-8k",
        "vulkan",
        "nvidia",
        0x10DE,
        11 * 17 * 47,
        1,
        8 * 1024,
        8 * 1024,
        1024,
        1024,
        32,
        32,
        4,
        4194305,
        11,
        89,
        17,
        16384,
    },
};

typedef struct {
    const char* case_name;
    size_t forced_case_index;
    pfUINT upload_id;
} ForcedAxisBlockUploadReference;

static const ForcedAxisBlockUploadReference FORCED_AXIS_BLOCK_UPLOAD_REFERENCES[] = {
    {"axis-block-nvidia-dd-n9367-forced-u0", 0, 0},
    {"axis-block-nvidia-dd-n9367-forced-u1", 0, 1},
    {"axis-block-intel-opencl-dd-n1054-forced-u0", 1, 0},
    {"axis-block-intel-opencl-dd-n1054-forced-u1", 1, 1},
    {"axis-block-nvidia-dd-n3196-forced-u0", 2, 0},
    {"axis-block-nvidia-dd-n3196-forced-u1", 2, 1},
    {"axis-block-nvidia-dd-n3196-forced-u2", 2, 2},
    {"axis-block-nvidia-dd-n1922-forced-u0", 3, 0},
    {"axis-block-nvidia-dd-n1922-forced-u1", 3, 1},
    {"axis-block-nvidia-dd-n1922-forced-u2", 3, 2},
    {"axis-block-nvidia-dd-n102272-forced-u0", 4, 0},
    {"axis-block-nvidia-dd-n102272-forced-u1", 4, 1},
    {"axis-block-nvidia-dd-n102272-forced-u2", 4, 2},
    {"axis-block-amd-vulkan-dd-n9367-48k-u0", 5, 0},
    {"axis-block-amd-vulkan-dd-n9367-48k-u1", 5, 1},
    {"axis-block-amd-vulkan-dd-n9367-64k-u0", 6, 0},
    {"axis-block-amd-vulkan-dd-n9367-64k-u1", 6, 1},
    {"axis-block-intel-opencl-dd-n9367-u0", 7, 0},
    {"axis-block-intel-opencl-dd-n9367-u1", 7, 1},
    {"axis-block-amd-vulkan-dd-n9367-32k-portable-u0", 8, 0},
    {"axis-block-amd-vulkan-dd-n9367-32k-portable-u1", 8, 1},
    {"axis-block-amd-vulkan-dd-n9367-32k-forced256-u0", 9, 0},
    {"axis-block-amd-vulkan-dd-n9367-32k-forced256-u1", 9, 1},
    {"axis-block-amd-vulkan-dd-n4352-32k-128t-u0", 10, 0},
    {"axis-block-amd-vulkan-dd-n4352-32k-128t-u1", 10, 1},
    {"axis-block-amd-vulkan-dd-n5100-32k-128t-u0", 11, 0},
    {"axis-block-amd-vulkan-dd-n5100-32k-128t-u1", 11, 1},
    {"axis-block-nvidia-vulkan-dd-n4352-48k-128t-u0", 12, 0},
    {"axis-block-nvidia-vulkan-dd-n4352-48k-128t-u1", 12, 1},
    {"axis-block-nvidia-vulkan-dd-n5100-48k-128t-u0", 14, 0},
    {"axis-block-nvidia-vulkan-dd-n5100-48k-128t-u1", 14, 1},
    {"axis-block-amd-vulkan-dd-n102272-forced-u0", 13, 0},
    {"axis-block-amd-vulkan-dd-n102272-forced-u1", 13, 1},
    {"axis-block-amd-vulkan-dd-n102272-forced-u2", 13, 2},
    {"axis-block-amd-vulkan-dd-n5797-forced-u0", 15, 0},
    {"axis-block-amd-vulkan-dd-n5797-forced-u1", 15, 1},
    {"axis-block-amd-vulkan-dd-n8789-forced-u0", 16, 0},
    {"axis-block-amd-vulkan-dd-n8789-forced-u1", 16, 1},
    {"axis-block-nvidia-vulkan-dd-n5797-forced-u0", 17, 0},
    {"axis-block-nvidia-vulkan-dd-n5797-forced-u1", 17, 1},
    {"axis-block-nvidia-vulkan-dd-n8789-forced-u0", 18, 0},
    {"axis-block-nvidia-vulkan-dd-n8789-forced-u1", 18, 1},
};

typedef struct {
    const char* case_name;
    pfUINT sequence_len;
    pfUINT batch_count;
    pfUINT shared_memory_bytes;
    pfUINT shared_memory_pow2_bytes;
    pfUINT max_threads_num;
    const char* vendor_name;
    pfUINT vendor_id;
    pfUINT coalesced_memory_bytes;
    pfUINT register_boost;
    pfUINT swap_to_three_stage;
    int warp_size;
} RaderParentCase;

static const RaderParentCase RADER_PARENT_CASES[] = {
    {"rader-parent-nvidia-n361-p19x2", 19 * 19, 2, 48 * 1024, 32 * 1024, 1024},
    {"rader-parent-nvidia-n437-p19-p23", 19 * 23, 2, 48 * 1024, 32 * 1024, 1024},
    {"rader-parent-nvidia-n551-p19-p29", 19 * 29, 2, 48 * 1024, 32 * 1024, 1024},
    {"rader-parent-nvidia-n589-p19-p31", 19 * 31, 2, 48 * 1024, 32 * 1024, 1024},
    {"rader-parent-nvidia-n667-p23-p29", 23 * 29, 2, 48 * 1024, 32 * 1024, 1024},
    {"rader-parent-nvidia-n289-p17x2", 17 * 17, 2, 48 * 1024, 32 * 1024, 1024},
    {"rader-parent-nvidia-n323-p17-p19", 17 * 19, 2, 48 * 1024, 32 * 1024, 1024},
    {"rader-parent-nvidia-n391-p17-p23", 17 * 23, 2, 48 * 1024, 32 * 1024, 1024},
    {"rader-parent-intel-vulkan-n391-p17-p23", 17 * 23, 2, 32 * 1024, 32 * 1024, 1024, "intel", 0x8086, 64, 2, 524288, 32},
    {"rader-parent-amd-vulkan-n391-p17-p23", 17 * 23, 2, 48 * 1024, 32 * 1024, 1024, "amd", 0x1002, 32, 4, 524288, 64},
    {"rader-parent-amd-vulkan-n391-p17-p23-64k", 17 * 23, 2, 64 * 1024, 64 * 1024, 1024, "amd", 0x1002, 32, 2, 524288, 64},
    {"rader-parent-intel-vulkan-n391-p17-p23-64k", 17 * 23, 2, 64 * 1024, 64 * 1024, 1024, "intel", 0x8086, 64, 1, 524288, 32},
    {"rader-parent-nvidia-n493-p17-p29", 17 * 29, 2, 48 * 1024, 32 * 1024, 1024},
    {"rader-parent-nvidia-n527-p17-p31", 17 * 31, 2, 48 * 1024, 32 * 1024, 1024},
    {"rader-parent-nvidia-n529-p23x2", 23 * 23, 2, 48 * 1024, 32 * 1024, 1024},
    {"rader-parent-nvidia-n713-p23-p31", 23 * 31, 2, 48 * 1024, 32 * 1024, 1024},
    {"rader-parent-nvidia-n899-p29-p31", 29 * 31, 2, 48 * 1024, 32 * 1024, 1024},
    {"rader-parent-nvidia-n961-p31x2", 31 * 31, 2, 48 * 1024, 32 * 1024, 1024},
    {"rader-parent-nvidia-n8303-p19x2-p23-256k", 19 * 19 * 23, 1, 256 * 1024, 256 * 1024, 1024},
    {"rader-parent-nvidia-n7429-p17-p19-p23-256k", 17 * 19 * 23, 1, 256 * 1024, 256 * 1024, 1024},
};

#define MIXED_RADER_MAX_PRIMES 3

typedef struct {
    const char* case_name;
    pfUINT sequence_len;
    pfUINT batch_count;
    pfUINT shared_memory_bytes;
    pfUINT shared_memory_pow2_bytes;
    pfUINT max_threads_num;
    int double_double;
    int min_direct_prime;
    int max_direct_prime;
    int min_fft_prime;
    int max_fft_prime;
    int direct_count;
    int direct_primes[MIXED_RADER_MAX_PRIMES];
    int fft_count;
    int fft_primes[MIXED_RADER_MAX_PRIMES];
} MixedRaderParentCase;

static const MixedRaderParentCase MIXED_RADER_PARENT_CASES[] = {
    {"rader-mixed-parent-nvidia-n527-p17d-p31f", 17 * 31, 2, 48 * 1024, 32 * 1024, 1024, 0, 17, 89, 19, 16384, 1, {17, 0, 0}, 1, {31, 0, 0}},
    {"rader-mixed-parent-nvidia-n7905-r15-p17d-p31f", 15 * 17 * 31, 1, 256 * 1024, 256 * 1024, 1024, 0, 17, 89, 19, 16384, 1, {17, 0, 0}, 1, {31, 0, 0}},
    {"rader-mixed-parent-nvidia-n7429-p17d-p19f-p23f", 17 * 19 * 23, 1, 256 * 1024, 256 * 1024, 1024, 0, 17, 89, 19, 16384, 1, {17, 0, 0}, 2, {19, 23, 0}},
    {"rader-mixed-parent-nvidia-n9367-p17d-p19f-p29f", 17 * 19 * 29, 1, 256 * 1024, 256 * 1024, 1024, 0, 17, 89, 19, 16384, 1, {17, 0, 0}, 2, {19, 29, 0}},
    {"rader-mixed-parent-nvidia-n3553-p11d-p17f-p19f-256t", 11 * 17 * 19, 2, 48 * 1024, 32 * 1024, 256, 0, 11, 89, 17, 16384, 1, {11, 0, 0}, 2, {17, 19, 0}},
    {"rader-mixed-parent-nvidia-dd-n527-p17d-p31f", 17 * 31, 2, 48 * 1024, 32 * 1024, 1024, 1, 17, 89, 19, 16384, 1, {17, 0, 0}, 1, {31, 0, 0}},
    {"rader-mixed-parent-nvidia-dd-n323-p17d-p19f-b29", 17 * 19, 29, 48 * 1024, 32 * 1024, 1024, 1, 17, 89, 19, 16384, 1, {17, 0, 0}, 1, {19, 0, 0}},
    {"rader-mixed-parent-nvidia-dd-n7905-r15-p17d-p31f", 15 * 17 * 31, 1, 256 * 1024, 256 * 1024, 1024, 1, 17, 89, 19, 16384, 1, {17, 0, 0}, 1, {31, 0, 0}},
    {"rader-mixed-parent-nvidia-dd-n7429-p17d-p19f-p23f", 17 * 19 * 23, 1, 256 * 1024, 256 * 1024, 1024, 1, 17, 89, 19, 16384, 1, {17, 0, 0}, 2, {19, 23, 0}},
    {"rader-mixed-parent-nvidia-dd-n1054-r2-p17d-p31f", 2 * 17 * 31, 1, 48 * 1024, 32 * 1024, 1024, 1, 17, 89, 19, 16384, 1, {17, 0, 0}, 1, {31, 0, 0}},
    {"rader-mixed-parent-nvidia-n206-p103f-cross-bluestein", 2 * 103, 1, 48 * 1024, 32 * 1024, 1024, 0, 29, 89, 29, 16384, 0, {0, 0, 0}, 1, {103, 0, 0}},
    {"rader-mixed-parent-nvidia-n3193-p31f-p103f-cross-bluestein", 31 * 103, 1, 256 * 1024, 256 * 1024, 1024, 0, 29, 89, 29, 16384, 0, {0, 0, 0}, 2, {31, 103, 0}},
};

typedef struct {
    const char* case_name;
    const char* vendor_name;
    pfUINT vendor_id;
    pfUINT sequence_len;
    pfUINT batch_count;
    pfUINT shared_memory_bytes;
    pfUINT shared_memory_pow2_bytes;
    pfUINT coalesced_memory_bytes;
    pfUINT register_boost;
    pfUINT swap_to_three_stage;
    int warp_size;
    int double_double;
    pfUINT axis_id;
    pfUINT fastest_axis_len;
    int perform_bandwidth_boost;
    pfUINT bluestein_logical_len;
    int double_precision;
    const char* backend_name;
    int half_precision;
    int double_precision_float_memory;
    pfUINT grouped_batch_override;
    int perform_zero_padding;
    int min_direct_prime_override;
    int max_direct_prime_override;
    int min_fft_prime_override;
    int max_fft_prime_override;
    int perform_convolution;
    pfUINT number_kernels;
    int kernel_convolution;
} StockhamCase;

static const StockhamCase STOCKHAM_CASES[] = {
    {
        "stockham-nvidia-1048576-f32",
        "nvidia",
        0x10DE,
        1048576,
        1,
        48 * 1024,
        32 * 1024,
        32,
        4,
        4194305,
        32,
    },
    {
        "stockham-amd-vulkan-1048576-f32-32k",
        "amd",
        0x1002,
        1048576,
        1,
        32 * 1024,
        32 * 1024,
        32,
        4,
        524288,
        64,
    },
    {
        "stockham-nvidia-vulkan-3840-f32-32k",
        "nvidia",
        0x10DE,
        3840,
        1,
        32 * 1024,
        32 * 1024,
        32,
        4,
        4194305,
        32,
    },
    {
        "stockham-amd-vulkan-3840-f32-32k",
        "amd",
        0x1002,
        3840,
        1,
        32 * 1024,
        32 * 1024,
        32,
        4,
        524288,
        64,
    },
    {
        "stockham-nvidia-vulkan-8192-f32-32k",
        "nvidia", 0x10DE, 8192, 1, 32 * 1024, 32 * 1024, 32, 4, 4194305, 32,
    },
    {
        .case_name = "stockham-nvidia-vulkan-8192-f32-32k-convolution",
        .vendor_name = "nvidia", .vendor_id = 0x10DE,
        .sequence_len = 8192, .batch_count = 1,
        .shared_memory_bytes = 32 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32, .register_boost = 4,
        .swap_to_three_stage = 4194305, .warp_size = 32,
        .perform_convolution = 1,
    },
    {
        .case_name = "stockham-nvidia-vulkan-8192-f32-32k-kernel-convolution",
        .vendor_name = "nvidia", .vendor_id = 0x10DE,
        .sequence_len = 8192, .batch_count = 1,
        .shared_memory_bytes = 32 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32, .register_boost = 4,
        .swap_to_three_stage = 4194305, .warp_size = 32,
        .kernel_convolution = 1,
    },
    {
        .case_name = "stockham-nvidia-vulkan-8192-f32-32k-kernel-convolution-batch4",
        .vendor_name = "nvidia", .vendor_id = 0x10DE,
        .sequence_len = 8192, .batch_count = 4,
        .shared_memory_bytes = 32 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32, .register_boost = 4,
        .swap_to_three_stage = 4194305, .warp_size = 32,
        .kernel_convolution = 1,
    },
    {
        .case_name = "stockham-nvidia-vulkan-8192-f32-32k-convolution-k3",
        .vendor_name = "nvidia", .vendor_id = 0x10DE,
        .sequence_len = 8192, .batch_count = 1,
        .shared_memory_bytes = 32 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32, .register_boost = 4,
        .swap_to_three_stage = 4194305, .warp_size = 32,
        .perform_convolution = 1, .number_kernels = 3,
    },
    {
        .case_name = "stockham-nvidia-vulkan-4096-f64-32k",
        .vendor_name = "nvidia", .vendor_id = 0x10DE,
        .sequence_len = 4096, .batch_count = 1,
        .shared_memory_bytes = 32 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32, .register_boost = 4,
        .swap_to_three_stage = 4194305, .warp_size = 32,
        .double_precision = 1,
    },
    {
        .case_name = "stockham-nvidia-vulkan-4096-f64-32k-convolution",
        .vendor_name = "nvidia", .vendor_id = 0x10DE,
        .sequence_len = 4096, .batch_count = 1,
        .shared_memory_bytes = 32 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32, .register_boost = 4,
        .swap_to_three_stage = 4194305, .warp_size = 32,
        .double_precision = 1, .perform_convolution = 1,
    },
    {
        .case_name = "stockham-nvidia-vulkan-4096-f64-32k-convolution-k3",
        .vendor_name = "nvidia", .vendor_id = 0x10DE,
        .sequence_len = 4096, .batch_count = 1,
        .shared_memory_bytes = 32 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32, .register_boost = 4,
        .swap_to_three_stage = 4194305, .warp_size = 32,
        .double_precision = 1, .perform_convolution = 1, .number_kernels = 3,
    },
    {"stockham-nvidia-vulkan-1920-f32-32k", "nvidia", 0x10DE, 1920, 1, 32 * 1024, 32 * 1024, 32, 4, 4194305, 32},
    {"stockham-amd-vulkan-1920-f32-32k", "amd", 0x1002, 1920, 1, 32 * 1024, 32 * 1024, 32, 4, 524288, 64},
    {"stockham-nvidia-vulkan-2880-f32-32k", "nvidia", 0x10DE, 2880, 1, 32 * 1024, 32 * 1024, 32, 4, 4194305, 32},
    {"stockham-amd-vulkan-2880-f32-32k", "amd", 0x1002, 2880, 1, 32 * 1024, 32 * 1024, 32, 4, 524288, 64},
    {"stockham-nvidia-vulkan-5760-f32-32k", "nvidia", 0x10DE, 5760, 1, 32 * 1024, 32 * 1024, 32, 4, 4194305, 32},
    {
        .case_name = "stockham-nvidia-vulkan-5760-f32-32k-convolution",
        .vendor_name = "nvidia", .vendor_id = 0x10DE,
        .sequence_len = 5760, .batch_count = 1,
        .shared_memory_bytes = 32 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32, .register_boost = 4,
        .swap_to_three_stage = 4194305, .warp_size = 32,
        .perform_convolution = 1,
    },
    {"stockham-amd-vulkan-5760-f32-32k", "amd", 0x1002, 5760, 1, 32 * 1024, 32 * 1024, 32, 4, 524288, 64},
    {"stockham-nvidia-vulkan-7680-f32-32k", "nvidia", 0x10DE, 7680, 1, 32 * 1024, 32 * 1024, 32, 4, 4194305, 32},
    {"stockham-amd-vulkan-7680-f32-32k", "amd", 0x1002, 7680, 1, 32 * 1024, 32 * 1024, 32, 4, 524288, 64},
    {
        "stockham-nvidia-8388608-f32",
        "nvidia",
        0x10DE,
        8388608,
        1,
        48 * 1024,
        32 * 1024,
        32,
        4,
        4194305,
        32,
    },
    {
        .case_name = "stockham-nvidia-8388608-f32-convolution",
        .vendor_name = "nvidia", .vendor_id = 0x10DE,
        .sequence_len = 8388608, .batch_count = 1,
        .shared_memory_bytes = 48 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32, .register_boost = 4,
        .swap_to_three_stage = 4194305, .warp_size = 32,
        .perform_convolution = 1,
    },
    {
        .case_name = "stockham-nvidia-8388608-f32-convolution-k3",
        .vendor_name = "nvidia", .vendor_id = 0x10DE,
        .sequence_len = 8388608, .batch_count = 1,
        .shared_memory_bytes = 48 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32, .register_boost = 4,
        .swap_to_three_stage = 4194305, .warp_size = 32,
        .perform_convolution = 1, .number_kernels = 3,
    },
    {
        "stockham-amd-vulkan-8388608-f32-32k",
        "amd",
        0x1002,
        8388608,
        1,
        32 * 1024,
        32 * 1024,
        32,
        4,
        524288,
        64,
    },
    {
        "stockham-amd-vulkan-49152-f32-32k",
        "amd",
        0x1002,
        49152,
        1,
        32 * 1024,
        32 * 1024,
        32,
        4,
        524288,
        64,
    },
    {
        "stockham-amd-vulkan-4718592-f32-32k",
        "amd",
        0x1002,
        4718592,
        1,
        32 * 1024,
        32 * 1024,
        32,
        4,
        524288,
        64,
    },
    {
        "stockham-nvidia-6144-f32-32k",
        "nvidia",
        0x10DE,
        6144,
        40,
        32 * 1024,
        32 * 1024,
        32,
        4,
        4194305,
        32,
    },
    {
        "stockham-amd-vulkan-4096-f32-32k",
        "amd",
        0x1002,
        4096,
        1,
        32 * 1024,
        32 * 1024,
        32,
        4,
        524288,
        64,
    },
    {
        "stockham-amd-vulkan-8192-f32-32k",
        "amd",
        0x1002,
        8192,
        1,
        32 * 1024,
        32 * 1024,
        32,
        4,
        524288,
        64,
    },
    {
        "stockham-amd-vulkan-16384-f32-32k",
        "amd",
        0x1002,
        16384,
        1,
        32 * 1024,
        32 * 1024,
        32,
        4,
        524288,
        64,
    },
    {
        "stockham-amd-16384-f32-48k",
        "amd",
        0x1002,
        16384,
        1,
        48 * 1024,
        32 * 1024,
        32,
        4,
        524288,
        64,
    },
    {
        "stockham-amd-16384-f32-64k",
        "amd",
        0x1002,
        16384,
        1,
        64 * 1024,
        64 * 1024,
        32,
        2,
        524288,
        64,
    },
    {
        "stockham-nvidia-vulkan-dd-n32768-32k",
        "nvidia", 0x10DE, 32768, 1, 32 * 1024, 32 * 1024, 32, 4, 4194305, 32,
        1,
    },
    {
        "stockham-amd-vulkan-dd-n32768-32k",
        "amd", 0x1002, 32768, 1, 32 * 1024, 32 * 1024, 32, 4, 524288, 64,
        1,
    },
    {
        "stockham-nvidia-vulkan-dd-n1679616-48k",
        "nvidia", 0x10DE, 1679616, 1, 48 * 1024, 32 * 1024, 32, 4, 4194305, 32,
        1,
    },
    {
        "stockham-amd-vulkan-dd-n1679616-48k",
        "amd", 0x1002, 1679616, 1, 48 * 1024, 32 * 1024, 32, 4, 524288, 64,
        1,
    },
    {
        "stockham-amd-vulkan-dd-n1679616-32k",
        "amd", 0x1002, 1679616, 1, 32 * 1024, 32 * 1024, 32, 4, 524288, 64,
        1,
    },
    {
        "stockham-nvidia-vulkan-dd-n823543-16k",
        "nvidia", 0x10DE, 823543, 1, 16 * 1024, 16 * 1024, 32, 4, 4194305, 32,
        1,
    },
    {
        "stockham-amd-vulkan-dd-n823543-16k",
        "amd", 0x1002, 823543, 1, 16 * 1024, 16 * 1024, 32, 4, 524288, 64,
        1,
    },
    {
        "stockham-intel-vulkan-16384-f32-32k",
        "intel",
        0x8086,
        16384,
        1,
        32 * 1024,
        32 * 1024,
        64,
        2,
        524288,
        32,
    },
    {
        "stockham-intel-vulkan-16384-f32-64k",
        "intel",
        0x8086,
        16384,
        1,
        64 * 1024,
        64 * 1024,
        64,
        1,
        524288,
        32,
    },
    {
        "stockham-nvidia-524288-f32-32k",
        "nvidia", 0x10DE, 524288, 1, 48 * 1024, 32 * 1024, 32, 4, 4194305, 32,
    },
    {
        "stockham-amd-524288-f32-32k",
        "amd", 0x1002, 524288, 1, 48 * 1024, 32 * 1024, 32, 4, 524288, 64,
    },
    {
        "stockham-intel-vulkan-524288-f32-32k",
        "intel", 0x8086, 524288, 1, 48 * 1024, 32 * 1024, 64, 2, 524288, 32,
    },
    {
        "stockham-nvidia-524288-f16-32k",
        "nvidia", 0x10DE, 524288, 1, 48 * 1024, 32 * 1024, 64, 4, 4194305, 32,
        0, 0, 0, 0, 0, 0, NULL, 1,
    },
    {
        "stockham-amd-524288-f16-32k",
        "amd", 0x1002, 524288, 1, 48 * 1024, 32 * 1024, 64, 4, 524288, 64,
        0, 0, 0, 0, 0, 0, NULL, 1,
    },
    {
        "stockham-intel-vulkan-524288-f16-32k",
        "intel", 0x8086, 524288, 1, 48 * 1024, 32 * 1024, 128, 2, 524288, 32,
        0, 0, 0, 0, 0, 0, NULL, 1,
    },
    {
        "stockham-amd-hip-2097152-f32-32k",
        "amd", 0x1002, 2097152, 1, 48 * 1024, 32 * 1024, 32, 1, 2097152, 64,
        0, 0, 0, 0, 0, 0, "hip",
    },
    {
        "stockham-amd-hip-2097152-f16-32k",
        "amd", 0x1002, 2097152, 1, 48 * 1024, 32 * 1024, 64, 1, 2097152, 64,
        0, 0, 0, 0, 0, 0, "hip", 1, 0,
    },
    {
        "stockham-intel-vulkan-dd-n1024-strided-b0",
        "intel", 0x8086, 1024, 1, 32 * 1024, 32 * 1024, 64, 2, 524288, 32,
        1, 1, 1, 0,
    },
    {
        "stockham-intel-vulkan-dd-n1024-strided-b1",
        "intel", 0x8086, 1024, 1, 32 * 1024, 32 * 1024, 64, 2, 524288, 32,
        1, 1, 1, 1,
    },
    {
        "stockham-intel-vulkan-dd-n1024-strided-b2",
        "intel", 0x8086, 1024, 1, 32 * 1024, 32 * 1024, 64, 2, 524288, 32,
        1, 1, 1, 2,
    },
    {
        "stockham-intel-vulkan-f32-p263-m625-bluestein-strided-b0",
        "intel", 0x8086, 625, 2, 32 * 1024, 32 * 1024, 64, 2, 524288, 32,
        0, 1, 2, 0, 263,
    },
    {
        "stockham-intel-vulkan-f32-p263-m625-bluestein-strided-b2",
        "intel", 0x8086, 625, 2, 32 * 1024, 32 * 1024, 64, 2, 524288, 32,
        0, 1, 2, 2, 263,
    },
    {
        "stockham-intel-vulkan-f32-p389-m1024-bluestein-strided-b0",
        "intel", 0x8086, 1024, 2, 32 * 1024, 32 * 1024, 64, 2, 524288, 32,
        0, 1, 2, 0, 389,
    },
    {
        "stockham-intel-vulkan-f32-p389-m1024-bluestein-strided-b2",
        "intel", 0x8086, 1024, 2, 32 * 1024, 32 * 1024, 64, 2, 524288, 32,
        0, 1, 2, 2, 389,
    },
    {
        "stockham-amd-vulkan-f32-p263-m625-bluestein-strided-b0",
        "amd", 0x1002, 625, 2, 32 * 1024, 32 * 1024, 32, 4, 524288, 64,
        0, 1, 2, 0, 263,
    },
    {
        "stockham-amd-vulkan-f32-p389-m1024-bluestein-strided-b0",
        "amd", 0x1002, 1024, 2, 32 * 1024, 32 * 1024, 32, 4, 524288, 64,
        0, 1, 2, 0, 389,
    },
    {
        "stockham-amd-vulkan-4096-f64-32k",
        "amd", 0x1002, 4096, 1, 32 * 1024, 32 * 1024, 32, 4, 524288, 64,
        0, 0, 0, 0, 0, 1,
    },
    {
        "stockham-nvidia-262144-f64-64k",
        "nvidia", 0x10DE, 262144, 1, 64 * 1024, 64 * 1024, 32, 4, 4194305, 32,
        0, 0, 0, 0, 0, 1,
    },
    {
        "stockham-amd-262144-f64-64k",
        "amd", 0x1002, 262144, 1, 64 * 1024, 64 * 1024, 32, 2, 262144, 64,
        0, 0, 0, 0, 0, 1,
    },
    {
        "stockham-intel-vulkan-262144-f64-64k",
        "intel", 0x8086, 262144, 1, 64 * 1024, 64 * 1024, 64, 1, 262144, 32,
        0, 0, 0, 0, 0, 1,
    },
    {
        "stockham-nvidia-262144-f64-f32-storage-64k",
        "nvidia", 0x10DE, 262144, 1, 64 * 1024, 64 * 1024, 32, 4, 4194305, 32,
        0, 0, 0, 0, 0, 0, NULL, 0, 1,
    },
    {
        "stockham-amd-262144-f64-f32-storage-64k",
        "amd", 0x1002, 262144, 1, 64 * 1024, 64 * 1024, 32, 2, 524288, 64,
        0, 0, 0, 0, 0, 0, NULL, 0, 1,
    },
    {
        "stockham-intel-vulkan-262144-f64-f32-storage-64k",
        "intel", 0x8086, 262144, 1, 64 * 1024, 64 * 1024, 64, 1, 524288, 32,
        0, 0, 0, 0, 0, 0, NULL, 0, 1,
    },
    {
        "stockham-amd-hip-262144-f64-64k",
        "amd", 0x1002, 262144, 1, 64 * 1024, 64 * 1024, 32, 1, 1048576, 64,
        0, 0, 0, 0, 0, 1, "hip",
    },
    {
        "stockham-amd-hip-1048576-f64-64k",
        "amd", 0x1002, 1048576, 1, 64 * 1024, 64 * 1024, 32, 1, 1048576, 64,
        0, 0, 0, 0, 0, 1, "hip",
    },
    {
        "stockham-amd-hip-1048576-f64-f32-storage-64k",
        "amd", 0x1002, 1048576, 1, 64 * 1024, 64 * 1024, 32, 1, 2097152, 64,
        0, 0, 0, 0, 0, 0, "hip", 0, 1,
    },
    {
        "stockham-apple-metal-8192-f32-32k",
        "apple", 0x1027f00, 8192, 1, 32 * 1024, 32 * 1024, 64, 1, 524288, 1,
        0, 0, 0, 0, 0, 0, "metal",
    },
    {
        "stockham-intel-level-zero-8192-f32-32k",
        "intel", 0x8086, 8192, 1, 32 * 1024, 32 * 1024, 64, 2, 524288, 1,
        0, 0, 0, 0, 0, 0, "level-zero",
    },
    {
        "stockham-intel-level-zero-262144-f64-64k",
        "intel", 0x8086, 262144, 1, 64 * 1024, 64 * 1024, 64, 1, 262144, 1,
        0, 0, 0, 0, 0, 1, "level-zero",
    },
    {
        "stockham-intel-level-zero-262144-f64-f32-storage-64k",
        "intel", 0x8086, 262144, 1, 64 * 1024, 64 * 1024, 64, 1, 524288, 1,
        0, 0, 0, 0, 0, 0, "level-zero", 0, 1,
    },
    {
        "stockham-amd-opencl-262144-f64-64k",
        "amd", 0x1002, 262144, 1, 64 * 1024, 64 * 1024, 32, 2, 262144, 64,
        0, 0, 0, 0, 0, 1, "opencl",
    },
    {
        "stockham-amd-opencl-262144-f64-f32-storage-64k",
        "amd", 0x1002, 262144, 1, 64 * 1024, 64 * 1024, 32, 2, 524288, 64,
        0, 0, 0, 0, 0, 0, "opencl", 0, 1,
    },
    {"stockham-nvidia-vulkan-3840-f32-b7-32k", "nvidia", 0x10DE, 3840, 7, 32 * 1024, 32 * 1024, 32, 4, 4194305, 32},
    {"stockham-amd-vulkan-3840-f32-b7-32k", "amd", 0x1002, 3840, 7, 32 * 1024, 32 * 1024, 32, 4, 524288, 64},
    {"stockham-nvidia-vulkan-5760-f32-b7-32k", "nvidia", 0x10DE, 5760, 7, 32 * 1024, 32 * 1024, 32, 4, 4194305, 32},
    {"stockham-amd-vulkan-5760-f32-b7-32k", "amd", 0x1002, 5760, 7, 32 * 1024, 32 * 1024, 32, 4, 524288, 64},
    {"stockham-nvidia-vulkan-3840-f16-32k", "nvidia", 0x10DE, 3840, 1, 32 * 1024, 32 * 1024, 64, 4, 4194305, 32, 0, 0, 0, 0, 0, 0, NULL, 1, 0},
    {"stockham-amd-vulkan-3840-f16-32k", "amd", 0x1002, 3840, 1, 32 * 1024, 32 * 1024, 64, 4, 524288, 64, 0, 0, 0, 0, 0, 0, NULL, 1, 0},
    {"stockham-nvidia-vulkan-5760-f16-32k", "nvidia", 0x10DE, 5760, 1, 32 * 1024, 32 * 1024, 64, 4, 4194305, 32, 0, 0, 0, 0, 0, 0, NULL, 1, 0},
    {"stockham-amd-vulkan-5760-f16-32k", "amd", 0x1002, 5760, 1, 32 * 1024, 32 * 1024, 64, 4, 524288, 64, 0, 0, 0, 0, 0, 0, NULL, 1, 0},
    {"stockham-nvidia-vulkan-3840-f64-32k", "nvidia", 0x10DE, 3840, 1, 32 * 1024, 32 * 1024, 32, 4, 4194305, 32, 0, 0, 0, 0, 0, 1, NULL, 0, 0},
    {"stockham-amd-vulkan-3840-f64-32k", "amd", 0x1002, 3840, 1, 32 * 1024, 32 * 1024, 32, 4, 524288, 64, 0, 0, 0, 0, 0, 1, NULL, 0, 0},
    {"stockham-nvidia-vulkan-3840-f64-f32-storage-32k", "nvidia", 0x10DE, 3840, 1, 32 * 1024, 32 * 1024, 32, 4, 4194305, 32, 0, 0, 0, 0, 0, 0, NULL, 0, 1},
    {"stockham-amd-vulkan-3840-f64-f32-storage-32k", "amd", 0x1002, 3840, 1, 32 * 1024, 32 * 1024, 32, 4, 524288, 64, 0, 0, 0, 0, 0, 0, NULL, 0, 1},
    {"stockham-nvidia-vulkan-3840-f32-higher-b0-32k", "nvidia", 0x10DE, 3840, 1, 32 * 1024, 32 * 1024, 32, 4, 4194305, 32, 0, 1, 8, 0, 0, 0, NULL, 0, 0},
    {"stockham-amd-vulkan-3840-f32-higher-b0-32k", "amd", 0x1002, 3840, 1, 32 * 1024, 32 * 1024, 32, 4, 524288, 64, 0, 1, 8, 0, 0, 0, NULL, 0, 0},
    {"stockham-nvidia-vulkan-3840-f32-higher-b2-32k", "nvidia", 0x10DE, 3840, 1, 32 * 1024, 32 * 1024, 32, 4, 4194305, 32, 0, 1, 8, 2, 0, 0, NULL, 0, 0},
    {"stockham-amd-vulkan-3840-f32-higher-b2-32k", "amd", 0x1002, 3840, 1, 32 * 1024, 32 * 1024, 32, 4, 524288, 64, 0, 1, 8, 2, 0, 0, NULL, 0, 0},
    {"stockham-intel-vulkan-3840-f32-32k", "intel", 0x8086, 3840, 1, 32 * 1024, 32 * 1024, 64, 2, 524288, 32},
    {"stockham-intel-vulkan-3840-f16-32k", "intel", 0x8086, 3840, 1, 32 * 1024, 32 * 1024, 128, 2, 524288, 32, 0, 0, 0, 0, 0, 0, NULL, 1, 0},
    {"stockham-apple-metal-3840-f32-32k", "apple", 0x1027f00, 3840, 1, 32 * 1024, 32 * 1024, 64, 1, 524288, 1, 0, 0, 0, 0, 0, 0, "metal", 0, 0},
    {"stockham-intel-level-zero-3840-f32-32k", "intel", 0x8086, 3840, 1, 32 * 1024, 32 * 1024, 64, 2, 524288, 1, 0, 0, 0, 0, 0, 0, "level-zero", 0, 0},
    {"stockham-nvidia-vulkan-3145728-f32-higher-b0-32k", "nvidia", 0x10DE, 3145728, 1, 32 * 1024, 32 * 1024, 32, 4, 4194305, 32, 0, 1, 8, 0, 0, 0, NULL, 0, 0},
    {"stockham-amd-vulkan-3145728-f32-higher-b0-32k", "amd", 0x1002, 3145728, 1, 32 * 1024, 32 * 1024, 32, 4, 524288, 64, 0, 1, 8, 0, 0, 0, NULL, 0, 0},
    {"stockham-nvidia-vulkan-3145728-f32-higher-b2-32k", "nvidia", 0x10DE, 3145728, 1, 32 * 1024, 32 * 1024, 32, 4, 4194305, 32, 0, 1, 8, 2, 0, 0, NULL, 0, 0},
    {"stockham-amd-vulkan-3145728-f32-higher-b2-32k", "amd", 0x1002, 3145728, 1, 32 * 1024, 32 * 1024, 32, 4, 524288, 64, 0, 1, 8, 2, 0, 0, NULL, 0, 0},
    {"stockham-nvidia-vulkan-2097152-f32-higher-b0-32k", "nvidia", 0x10DE, 2097152, 1, 32 * 1024, 32 * 1024, 32, 4, 4194305, 32, 0, 1, 8, 0, 0, 0, NULL, 0, 0},
    {"stockham-amd-vulkan-2097152-f32-higher-b0-32k", "amd", 0x1002, 2097152, 1, 32 * 1024, 32 * 1024, 32, 4, 524288, 64, 0, 1, 8, 0, 0, 0, NULL, 0, 0},
    {"stockham-nvidia-vulkan-2097152-f32-higher-b2-32k", "nvidia", 0x10DE, 2097152, 1, 32 * 1024, 32 * 1024, 32, 4, 4194305, 32, 0, 1, 8, 2, 0, 0, NULL, 0, 0},
    {"stockham-amd-vulkan-2097152-f32-higher-b2-32k", "amd", 0x1002, 2097152, 1, 32 * 1024, 32 * 1024, 32, 4, 524288, 64, 0, 1, 8, 2, 0, 0, NULL, 0, 0},
    {"stockham-nvidia-vulkan-f32-p263-m567-bluestein-strided-b0", "nvidia", 0x10DE, 567, 2, 32 * 1024, 32 * 1024, 32, 4, 4194305, 32, 0, 1, 2, 0, 263},
    {"stockham-nvidia-vulkan-f32-p263-m567-bluestein-strided-b2", "nvidia", 0x10DE, 567, 2, 32 * 1024, 32 * 1024, 32, 4, 4194305, 32, 0, 1, 2, 2, 263},
    {"stockham-amd-vulkan-f32-p263-m625-bluestein-strided-b2", "amd", 0x1002, 625, 2, 32 * 1024, 32 * 1024, 32, 4, 524288, 64, 0, 1, 2, 2, 263},
    {"stockham-nvidia-vulkan-f32-p389-m832-bluestein-strided-b0", "nvidia", 0x10DE, 832, 2, 32 * 1024, 32 * 1024, 32, 4, 4194305, 32, 0, 1, 2, 0, 389},
    {"stockham-nvidia-vulkan-f32-p389-m832-bluestein-strided-b2", "nvidia", 0x10DE, 832, 2, 32 * 1024, 32 * 1024, 32, 4, 4194305, 32, 0, 1, 2, 2, 389},
    {"stockham-amd-vulkan-f32-p389-m1024-bluestein-strided-b2", "amd", 0x1002, 1024, 2, 32 * 1024, 32 * 1024, 32, 4, 524288, 64, 0, 1, 2, 2, 389},
    {
        .case_name = "stockham-nvidia-vulkan-f32-p263-m567-bluestein-strided-b0-b5-g3-zp",
        .vendor_name = "nvidia", .vendor_id = 0x10DE,
        .sequence_len = 567, .batch_count = 5,
        .shared_memory_bytes = 32 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32, .register_boost = 4,
        .swap_to_three_stage = 4194305, .warp_size = 32,
        .axis_id = 1, .fastest_axis_len = 8, .perform_bandwidth_boost = 0,
        .bluestein_logical_len = 263, .grouped_batch_override = 3,
        .perform_zero_padding = 1,
    },
    {
        .case_name = "stockham-nvidia-vulkan-f32-p263-m567-bluestein-strided-b2-b5-g3-zp",
        .vendor_name = "nvidia", .vendor_id = 0x10DE,
        .sequence_len = 567, .batch_count = 5,
        .shared_memory_bytes = 32 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32, .register_boost = 4,
        .swap_to_three_stage = 4194305, .warp_size = 32,
        .axis_id = 1, .fastest_axis_len = 8, .perform_bandwidth_boost = 2,
        .bluestein_logical_len = 263, .grouped_batch_override = 3,
        .perform_zero_padding = 1,
    },
    {
        .case_name = "stockham-amd-vulkan-f32-p263-m625-bluestein-strided-b0-b5-g3-zp",
        .vendor_name = "amd", .vendor_id = 0x1002,
        .sequence_len = 625, .batch_count = 5,
        .shared_memory_bytes = 32 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32, .register_boost = 4,
        .swap_to_three_stage = 524288, .warp_size = 64,
        .axis_id = 1, .fastest_axis_len = 8, .perform_bandwidth_boost = 0,
        .bluestein_logical_len = 263, .grouped_batch_override = 3,
        .perform_zero_padding = 1,
    },
    {
        .case_name = "stockham-amd-vulkan-f32-p263-m625-bluestein-strided-b2-b5-g3-zp",
        .vendor_name = "amd", .vendor_id = 0x1002,
        .sequence_len = 625, .batch_count = 5,
        .shared_memory_bytes = 32 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32, .register_boost = 4,
        .swap_to_three_stage = 524288, .warp_size = 64,
        .axis_id = 1, .fastest_axis_len = 8, .perform_bandwidth_boost = 2,
        .bluestein_logical_len = 263, .grouped_batch_override = 3,
        .perform_zero_padding = 1,
    },
    {
        .case_name = "stockham-nvidia-vulkan-f32-p2053-m4368-bluestein-strided-b0-b5-g3-zp",
        .vendor_name = "nvidia", .vendor_id = 0x10DE,
        .sequence_len = 4368, .batch_count = 5,
        .shared_memory_bytes = 32 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32, .register_boost = 4,
        .swap_to_three_stage = 4194305, .warp_size = 32,
        .axis_id = 1, .fastest_axis_len = 8, .perform_bandwidth_boost = 0,
        .bluestein_logical_len = 2053, .grouped_batch_override = 3,
        .perform_zero_padding = 1,
    },
    {
        .case_name = "stockham-nvidia-vulkan-f32-p2053-m4368-bluestein-strided-b2-b5-g3-zp",
        .vendor_name = "nvidia", .vendor_id = 0x10DE,
        .sequence_len = 4368, .batch_count = 5,
        .shared_memory_bytes = 32 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32, .register_boost = 4,
        .swap_to_three_stage = 4194305, .warp_size = 32,
        .axis_id = 1, .fastest_axis_len = 8, .perform_bandwidth_boost = 2,
        .bluestein_logical_len = 2053, .grouped_batch_override = 3,
        .perform_zero_padding = 1,
    },
    {
        .case_name = "stockham-amd-vulkan-f32-p2053-m4375-bluestein-strided-b0-b5-g3-zp",
        .vendor_name = "amd", .vendor_id = 0x1002,
        .sequence_len = 4375, .batch_count = 5,
        .shared_memory_bytes = 32 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32, .register_boost = 4,
        .swap_to_three_stage = 524288, .warp_size = 64,
        .axis_id = 1, .fastest_axis_len = 8, .perform_bandwidth_boost = 0,
        .bluestein_logical_len = 2053, .grouped_batch_override = 3,
        .perform_zero_padding = 1,
    },
    {
        .case_name = "stockham-amd-vulkan-f32-p2053-m4375-bluestein-strided-b2-b5-g3-zp",
        .vendor_name = "amd", .vendor_id = 0x1002,
        .sequence_len = 4375, .batch_count = 5,
        .shared_memory_bytes = 32 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32, .register_boost = 4,
        .swap_to_three_stage = 524288, .warp_size = 64,
        .axis_id = 1, .fastest_axis_len = 8, .perform_bandwidth_boost = 2,
        .bluestein_logical_len = 2053, .grouped_batch_override = 3,
        .perform_zero_padding = 1,
    },
    {
        .case_name = "stockham-nvidia-vulkan-dd-p2053-m4368-bluestein-strided-b0-b5-g3-zp",
        .vendor_name = "nvidia", .vendor_id = 0x10DE,
        .sequence_len = 4368, .batch_count = 5,
        .shared_memory_bytes = 32 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32, .register_boost = 1,
        .swap_to_three_stage = 4194305, .warp_size = 32,
        .double_double = 1, .axis_id = 1, .fastest_axis_len = 8,
        .perform_bandwidth_boost = 0, .bluestein_logical_len = 2053,
        .grouped_batch_override = 3, .perform_zero_padding = 1,
    },
    {
        .case_name = "stockham-nvidia-vulkan-dd-p2053-m4368-bluestein-strided-b2-b5-g3-zp",
        .vendor_name = "nvidia", .vendor_id = 0x10DE,
        .sequence_len = 4368, .batch_count = 5,
        .shared_memory_bytes = 32 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32, .register_boost = 1,
        .swap_to_three_stage = 4194305, .warp_size = 32,
        .double_double = 1, .axis_id = 1, .fastest_axis_len = 8,
        .perform_bandwidth_boost = 2, .bluestein_logical_len = 2053,
        .grouped_batch_override = 3, .perform_zero_padding = 1,
    },
    {
        .case_name = "stockham-amd-vulkan-dd-p2053-m4224-bluestein-strided-b0-b5-g3-zp",
        .vendor_name = "amd", .vendor_id = 0x1002,
        .sequence_len = 4224, .batch_count = 5,
        .shared_memory_bytes = 32 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32, .register_boost = 1,
        .swap_to_three_stage = 524288, .warp_size = 64,
        .double_double = 1, .axis_id = 1, .fastest_axis_len = 8,
        .perform_bandwidth_boost = 0, .bluestein_logical_len = 2053,
        .grouped_batch_override = 3, .perform_zero_padding = 1,
    },
    {
        .case_name = "stockham-amd-vulkan-dd-p2053-m4224-bluestein-strided-b2-b5-g3-zp",
        .vendor_name = "amd", .vendor_id = 0x1002,
        .sequence_len = 4224, .batch_count = 5,
        .shared_memory_bytes = 32 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32, .register_boost = 1,
        .swap_to_three_stage = 524288, .warp_size = 64,
        .double_double = 1, .axis_id = 1, .fastest_axis_len = 8,
        .perform_bandwidth_boost = 2, .bluestein_logical_len = 2053,
        .grouped_batch_override = 3, .perform_zero_padding = 1,
    },
    {
        .case_name = "stockham-nvidia-vulkan-dd-p2503-m5184-bluestein-strided-b0-b5-g3-zp",
        .vendor_name = "nvidia", .vendor_id = 0x10DE,
        .sequence_len = 5184, .batch_count = 5,
        .shared_memory_bytes = 32 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32, .register_boost = 1,
        .swap_to_three_stage = 4194305, .warp_size = 32,
        .double_double = 1, .axis_id = 1, .fastest_axis_len = 8,
        .perform_bandwidth_boost = 0, .bluestein_logical_len = 2503,
        .grouped_batch_override = 3, .perform_zero_padding = 1,
    },
    {
        .case_name = "stockham-nvidia-vulkan-dd-p2503-m5184-bluestein-strided-b2-b5-g3-zp",
        .vendor_name = "nvidia", .vendor_id = 0x10DE,
        .sequence_len = 5184, .batch_count = 5,
        .shared_memory_bytes = 32 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32, .register_boost = 1,
        .swap_to_three_stage = 4194305, .warp_size = 32,
        .double_double = 1, .axis_id = 1, .fastest_axis_len = 8,
        .perform_bandwidth_boost = 2, .bluestein_logical_len = 2503,
        .grouped_batch_override = 3, .perform_zero_padding = 1,
    },
    {
        .case_name = "stockham-amd-vulkan-dd-p2503-m5005-bluestein-strided-b0-b5-g3-zp",
        .vendor_name = "amd", .vendor_id = 0x1002,
        .sequence_len = 5005, .batch_count = 5,
        .shared_memory_bytes = 32 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32, .register_boost = 1,
        .swap_to_three_stage = 524288, .warp_size = 64,
        .double_double = 1, .axis_id = 1, .fastest_axis_len = 8,
        .perform_bandwidth_boost = 0, .bluestein_logical_len = 2503,
        .grouped_batch_override = 3, .perform_zero_padding = 1,
    },
    {
        .case_name = "stockham-amd-vulkan-dd-p2503-m5005-bluestein-strided-b2-b5-g3-zp",
        .vendor_name = "amd", .vendor_id = 0x1002,
        .sequence_len = 5005, .batch_count = 5,
        .shared_memory_bytes = 32 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32, .register_boost = 1,
        .swap_to_three_stage = 524288, .warp_size = 64,
        .double_double = 1, .axis_id = 1, .fastest_axis_len = 8,
        .perform_bandwidth_boost = 2, .bluestein_logical_len = 2503,
        .grouped_batch_override = 3, .perform_zero_padding = 1,
    },
    {
        .case_name = "stockham-nvidia-vulkan-dd-p659-m1331-bluestein-strided-b0-b5-g3-zp",
        .vendor_name = "nvidia", .vendor_id = 0x10DE,
        .sequence_len = 1331, .batch_count = 5,
        .shared_memory_bytes = 32 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32, .register_boost = 1,
        .swap_to_three_stage = 4194305, .warp_size = 32,
        .double_double = 1, .axis_id = 1, .fastest_axis_len = 8,
        .perform_bandwidth_boost = 0, .bluestein_logical_len = 659,
        .grouped_batch_override = 3, .perform_zero_padding = 1,
    },
    {
        .case_name = "stockham-nvidia-vulkan-dd-p659-m1331-bluestein-strided-b2-b5-g3-zp",
        .vendor_name = "nvidia", .vendor_id = 0x10DE,
        .sequence_len = 1331, .batch_count = 5,
        .shared_memory_bytes = 32 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32, .register_boost = 1,
        .swap_to_three_stage = 4194305, .warp_size = 32,
        .double_double = 1, .axis_id = 1, .fastest_axis_len = 8,
        .perform_bandwidth_boost = 2, .bluestein_logical_len = 659,
        .grouped_batch_override = 3, .perform_zero_padding = 1,
    },
    {
        .case_name = "stockham-amd-vulkan-dd-p659-m1323-bluestein-strided-b0-b5-g3-zp",
        .vendor_name = "amd", .vendor_id = 0x1002,
        .sequence_len = 1323, .batch_count = 5,
        .shared_memory_bytes = 32 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32, .register_boost = 1,
        .swap_to_three_stage = 524288, .warp_size = 64,
        .double_double = 1, .axis_id = 1, .fastest_axis_len = 8,
        .perform_bandwidth_boost = 0, .bluestein_logical_len = 659,
        .grouped_batch_override = 3, .perform_zero_padding = 1,
    },
    {
        .case_name = "stockham-amd-vulkan-dd-p659-m1323-bluestein-strided-b2-b5-g3-zp",
        .vendor_name = "amd", .vendor_id = 0x1002,
        .sequence_len = 1323, .batch_count = 5,
        .shared_memory_bytes = 32 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32, .register_boost = 1,
        .swap_to_three_stage = 524288, .warp_size = 64,
        .double_double = 1, .axis_id = 1, .fastest_axis_len = 8,
        .perform_bandwidth_boost = 2, .bluestein_logical_len = 659,
        .grouped_batch_override = 3, .perform_zero_padding = 1,
    },
    {
        .case_name = "stockham-nvidia-vulkan-dd-n2431-multi-rader-strided-b5-g3-zp",
        .vendor_name = "nvidia", .vendor_id = 0x10DE,
        .sequence_len = 2431, .batch_count = 5,
        .shared_memory_bytes = 32 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32, .register_boost = 1,
        .swap_to_three_stage = 4194305, .warp_size = 32,
        .double_double = 1, .axis_id = 1, .fastest_axis_len = 8,
        .grouped_batch_override = 3, .perform_zero_padding = 1,
    },
    {
        .case_name = "stockham-amd-vulkan-dd-n2431-multi-rader-strided-b5-g3-zp",
        .vendor_name = "amd", .vendor_id = 0x1002,
        .sequence_len = 2431, .batch_count = 5,
        .shared_memory_bytes = 32 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32, .register_boost = 1,
        .swap_to_three_stage = 524288, .warp_size = 64,
        .double_double = 1, .axis_id = 1, .fastest_axis_len = 8,
        .grouped_batch_override = 3, .perform_zero_padding = 1,
    },
    {
        .case_name = "stockham-nvidia-vulkan-dd-n94-higher-portable-48k",
        .vendor_name = "nvidia", .vendor_id = 0x10DE,
        .sequence_len = 94, .batch_count = 1,
        .shared_memory_bytes = 48 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .coalesced_memory_bytes = 32, .register_boost = 4,
        .swap_to_three_stage = 4194305, .warp_size = 32,
        .double_double = 1, .axis_id = 1, .fastest_axis_len = 8,
        .min_direct_prime_override = 17, .max_direct_prime_override = 89,
        .min_fft_prime_override = 17, .max_fft_prime_override = 16384,
    }

};

typedef struct {
    const char* case_name;
    const char* stockham_case_name;
    pfUINT upload_id;
} StockhamAxisBlockReference;

static const StockhamAxisBlockReference STOCKHAM_AXIS_BLOCK_REFERENCES[] = {
    {"axis-block-nvidia-f32-n8388608-convolution-u0", "stockham-nvidia-8388608-f32-convolution", 0},
    {"axis-block-nvidia-f32-n8388608-convolution-u1", "stockham-nvidia-8388608-f32-convolution", 1},
    {"axis-block-nvidia-f32-n8388608-convolution-u2", "stockham-nvidia-8388608-f32-convolution", 2},
    {"axis-block-nvidia-f32-n8388608-convolution-k3-u0", "stockham-nvidia-8388608-f32-convolution-k3", 0},
    {"axis-block-nvidia-f32-n8388608-convolution-k3-u1", "stockham-nvidia-8388608-f32-convolution-k3", 1},
    {"axis-block-nvidia-f32-n8388608-convolution-k3-u2", "stockham-nvidia-8388608-f32-convolution-k3", 2},
    {"axis-block-nvidia-vulkan-f32-n5760-convolution-u0", "stockham-nvidia-vulkan-5760-f32-32k-convolution", 0},
    {"axis-block-nvidia-vulkan-f32-n5760-convolution-u1", "stockham-nvidia-vulkan-5760-f32-32k-convolution", 1},
    {"axis-block-nvidia-vulkan-f64-n4096-convolution-u0", "stockham-nvidia-vulkan-4096-f64-32k-convolution", 0},
    {"axis-block-nvidia-vulkan-f64-n4096-convolution-u1", "stockham-nvidia-vulkan-4096-f64-32k-convolution", 1},
    {"axis-block-nvidia-vulkan-f64-n4096-convolution-k3-u0", "stockham-nvidia-vulkan-4096-f64-32k-convolution-k3", 0},
    {"axis-block-nvidia-vulkan-f64-n4096-convolution-k3-u1", "stockham-nvidia-vulkan-4096-f64-32k-convolution-k3", 1},
    {"axis-block-nvidia-vulkan-f32-n8192-convolution-u0", "stockham-nvidia-vulkan-8192-f32-32k-convolution", 0},
    {"axis-block-nvidia-vulkan-f32-n8192-convolution-u1", "stockham-nvidia-vulkan-8192-f32-32k-convolution", 1},
    {"axis-block-nvidia-vulkan-f32-n8192-kernel-convolution-u0", "stockham-nvidia-vulkan-8192-f32-32k-kernel-convolution", 0},
    {"axis-block-nvidia-vulkan-f32-n8192-kernel-convolution-u1", "stockham-nvidia-vulkan-8192-f32-32k-kernel-convolution", 1},
    {"axis-block-nvidia-vulkan-f32-n8192-kernel-convolution-batch4-u0", "stockham-nvidia-vulkan-8192-f32-32k-kernel-convolution-batch4", 0},
    {"axis-block-nvidia-vulkan-f32-n8192-kernel-convolution-batch4-u1", "stockham-nvidia-vulkan-8192-f32-32k-kernel-convolution-batch4", 1},
    {"axis-block-nvidia-vulkan-f32-n8192-convolution-k3-u0", "stockham-nvidia-vulkan-8192-f32-32k-convolution-k3", 0},
    {"axis-block-nvidia-vulkan-f32-n8192-convolution-k3-u1", "stockham-nvidia-vulkan-8192-f32-32k-convolution-k3", 1},
    {"axis-block-nvidia-vulkan-dd-n32768-stockham-u0", "stockham-nvidia-vulkan-dd-n32768-32k", 0},
    {"axis-block-nvidia-vulkan-dd-n32768-stockham-u1", "stockham-nvidia-vulkan-dd-n32768-32k", 1},
    {"axis-block-amd-vulkan-dd-n32768-stockham-u0", "stockham-amd-vulkan-dd-n32768-32k", 0},
    {"axis-block-amd-vulkan-dd-n32768-stockham-u1", "stockham-amd-vulkan-dd-n32768-32k", 1},
    {"axis-block-nvidia-vulkan-dd-n1679616-stockham-u0", "stockham-nvidia-vulkan-dd-n1679616-48k", 0},
    {"axis-block-nvidia-vulkan-dd-n1679616-stockham-u1", "stockham-nvidia-vulkan-dd-n1679616-48k", 1},
    {"axis-block-amd-vulkan-dd-n1679616-stockham-u0", "stockham-amd-vulkan-dd-n1679616-48k", 0},
    {"axis-block-amd-vulkan-dd-n1679616-stockham-u1", "stockham-amd-vulkan-dd-n1679616-48k", 1},
    {"axis-block-amd-vulkan-dd-n1679616-stockham-u2", "stockham-amd-vulkan-dd-n1679616-48k", 2},
    {"axis-block-amd-vulkan-dd-n1679616-32k-stockham-u0", "stockham-amd-vulkan-dd-n1679616-32k", 0},
    {"axis-block-amd-vulkan-dd-n1679616-32k-stockham-u1", "stockham-amd-vulkan-dd-n1679616-32k", 1},
    {"axis-block-amd-vulkan-dd-n1679616-32k-stockham-u2", "stockham-amd-vulkan-dd-n1679616-32k", 2},
    {"axis-block-nvidia-vulkan-dd-n823543-stockham-u0", "stockham-nvidia-vulkan-dd-n823543-16k", 0},
    {"axis-block-nvidia-vulkan-dd-n823543-stockham-u1", "stockham-nvidia-vulkan-dd-n823543-16k", 1},
    {"axis-block-nvidia-vulkan-dd-n823543-stockham-u2", "stockham-nvidia-vulkan-dd-n823543-16k", 2},
    {"axis-block-amd-vulkan-dd-n823543-stockham-u0", "stockham-amd-vulkan-dd-n823543-16k", 0},
    {"axis-block-amd-vulkan-dd-n823543-stockham-u1", "stockham-amd-vulkan-dd-n823543-16k", 1},
    {"axis-block-amd-vulkan-dd-n823543-stockham-u2", "stockham-amd-vulkan-dd-n823543-16k", 2},
    {"axis-block-amd-vulkan-f32-n524288-stockham-u0", "stockham-amd-524288-f32-32k", 0},
    {"axis-block-amd-vulkan-f32-n524288-stockham-u1", "stockham-amd-524288-f32-32k", 1},
    {"axis-block-amd-vulkan-f32-n524288-stockham-u2", "stockham-amd-524288-f32-32k", 2},
    {"axis-block-amd-vulkan-f32-n8388608-stockham-u0", "stockham-amd-vulkan-8388608-f32-32k", 0},
    {"axis-block-amd-vulkan-f32-n8388608-stockham-u1", "stockham-amd-vulkan-8388608-f32-32k", 1},
    {"axis-block-amd-vulkan-f32-n8388608-stockham-u2", "stockham-amd-vulkan-8388608-f32-32k", 2},
    {"axis-block-amd-vulkan-f32-n1048576-stockham-u0", "stockham-amd-vulkan-1048576-f32-32k", 0},
    {"axis-block-amd-vulkan-f32-n1048576-stockham-u1", "stockham-amd-vulkan-1048576-f32-32k", 1},
    {"axis-block-amd-vulkan-f32-n1048576-stockham-u2", "stockham-amd-vulkan-1048576-f32-32k", 2},
    {"axis-block-nvidia-vulkan-f32-n3840-stockham-u0", "stockham-nvidia-vulkan-3840-f32-32k", 0},
    {"axis-block-amd-vulkan-f32-n3840-stockham-u0", "stockham-amd-vulkan-3840-f32-32k", 0},
    {"axis-block-amd-vulkan-f32-n49152-stockham-u0", "stockham-amd-vulkan-49152-f32-32k", 0},
    {"axis-block-amd-vulkan-f32-n49152-stockham-u1", "stockham-amd-vulkan-49152-f32-32k", 1},
    {"axis-block-amd-vulkan-f32-n4718592-stockham-u0", "stockham-amd-vulkan-4718592-f32-32k", 0},
    {"axis-block-amd-vulkan-f32-n4718592-stockham-u1", "stockham-amd-vulkan-4718592-f32-32k", 1},
    {"axis-block-amd-vulkan-f32-n4718592-stockham-u2", "stockham-amd-vulkan-4718592-f32-32k", 2},
    {"axis-block-intel-vulkan-f32-n524288-stockham-u0", "stockham-intel-vulkan-524288-f32-32k", 0},
    {"axis-block-intel-vulkan-f32-n524288-stockham-u1", "stockham-intel-vulkan-524288-f32-32k", 1},
    {"axis-block-intel-vulkan-f32-n524288-stockham-u2", "stockham-intel-vulkan-524288-f32-32k", 2},
    {"axis-block-nvidia-vulkan-f16-n524288-stockham-u0", "stockham-nvidia-524288-f16-32k", 0},
    {"axis-block-nvidia-vulkan-f16-n524288-stockham-u1", "stockham-nvidia-524288-f16-32k", 1},
    {"axis-block-nvidia-vulkan-f16-n524288-stockham-u2", "stockham-nvidia-524288-f16-32k", 2},
    {"axis-block-amd-vulkan-f16-n524288-stockham-u0", "stockham-amd-524288-f16-32k", 0},
    {"axis-block-amd-vulkan-f16-n524288-stockham-u1", "stockham-amd-524288-f16-32k", 1},
    {"axis-block-amd-vulkan-f16-n524288-stockham-u2", "stockham-amd-524288-f16-32k", 2},
    {"axis-block-intel-vulkan-f16-n524288-stockham-u0", "stockham-intel-vulkan-524288-f16-32k", 0},
    {"axis-block-intel-vulkan-f16-n524288-stockham-u1", "stockham-intel-vulkan-524288-f16-32k", 1},
    {"axis-block-intel-vulkan-f16-n524288-stockham-u2", "stockham-intel-vulkan-524288-f16-32k", 2},
    {"axis-block-amd-hip-f32-n2097152-stockham-u0", "stockham-amd-hip-2097152-f32-32k", 0},
    {"axis-block-amd-hip-f32-n2097152-stockham-u1", "stockham-amd-hip-2097152-f32-32k", 1},
    {"axis-block-amd-hip-f32-n2097152-stockham-u2", "stockham-amd-hip-2097152-f32-32k", 2},
    {"axis-block-amd-hip-f16-n2097152-stockham-u0", "stockham-amd-hip-2097152-f16-32k", 0},
    {"axis-block-amd-hip-f16-n2097152-stockham-u1", "stockham-amd-hip-2097152-f16-32k", 1},
    {"axis-block-amd-hip-f16-n2097152-stockham-u2", "stockham-amd-hip-2097152-f16-32k", 2},
    {"axis-block-nvidia-vulkan-f64-f32-storage-n262144-stockham-u0", "stockham-nvidia-262144-f64-f32-storage-64k", 0},
    {"axis-block-nvidia-vulkan-f64-f32-storage-n262144-stockham-u1", "stockham-nvidia-262144-f64-f32-storage-64k", 1},
    {"axis-block-amd-vulkan-f64-f32-storage-n262144-stockham-u0", "stockham-amd-262144-f64-f32-storage-64k", 0},
    {"axis-block-amd-vulkan-f64-f32-storage-n262144-stockham-u1", "stockham-amd-262144-f64-f32-storage-64k", 1},
    {"axis-block-intel-vulkan-f64-f32-storage-n262144-stockham-u0", "stockham-intel-vulkan-262144-f64-f32-storage-64k", 0},
    {"axis-block-intel-vulkan-f64-f32-storage-n262144-stockham-u1", "stockham-intel-vulkan-262144-f64-f32-storage-64k", 1},
    {"axis-block-amd-hip-f64-n262144-stockham-u0", "stockham-amd-hip-262144-f64-64k", 0},
    {"axis-block-amd-hip-f64-n262144-stockham-u1", "stockham-amd-hip-262144-f64-64k", 1},
    {"axis-block-amd-hip-f64-n1048576-stockham-u0", "stockham-amd-hip-1048576-f64-64k", 0},
    {"axis-block-amd-hip-f64-n1048576-stockham-u1", "stockham-amd-hip-1048576-f64-64k", 1},
    {"axis-block-amd-hip-f64-n1048576-stockham-u2", "stockham-amd-hip-1048576-f64-64k", 2},
    {"axis-block-amd-hip-f64-f32-storage-n1048576-stockham-u0", "stockham-amd-hip-1048576-f64-f32-storage-64k", 0},
    {"axis-block-amd-hip-f64-f32-storage-n1048576-stockham-u1", "stockham-amd-hip-1048576-f64-f32-storage-64k", 1},
    {"axis-block-nvidia-vulkan-f32-n3840-b7-u0", "stockham-nvidia-vulkan-3840-f32-b7-32k", 0},
    {"axis-block-amd-vulkan-f32-n3840-b7-u0", "stockham-amd-vulkan-3840-f32-b7-32k", 0},
    {"axis-block-nvidia-vulkan-f16-n3840-u0", "stockham-nvidia-vulkan-3840-f16-32k", 0},
    {"axis-block-amd-vulkan-f16-n3840-u0", "stockham-amd-vulkan-3840-f16-32k", 0},
    {"axis-block-nvidia-vulkan-f16-n5760-u0", "stockham-nvidia-vulkan-5760-f16-32k", 0},
    {"axis-block-nvidia-vulkan-f16-n5760-u1", "stockham-nvidia-vulkan-5760-f16-32k", 1},
    {"axis-block-amd-vulkan-f16-n5760-u0", "stockham-amd-vulkan-5760-f16-32k", 0},
    {"axis-block-amd-vulkan-f16-n5760-u1", "stockham-amd-vulkan-5760-f16-32k", 1},
    {"axis-block-nvidia-vulkan-f64-n3840-u0", "stockham-nvidia-vulkan-3840-f64-32k", 0},
    {"axis-block-nvidia-vulkan-f64-n3840-u1", "stockham-nvidia-vulkan-3840-f64-32k", 1},
    {"axis-block-amd-vulkan-f64-n3840-u0", "stockham-amd-vulkan-3840-f64-32k", 0},
    {"axis-block-amd-vulkan-f64-n3840-u1", "stockham-amd-vulkan-3840-f64-32k", 1},
    {"axis-block-nvidia-vulkan-f64-f32-storage-n3840-u0", "stockham-nvidia-vulkan-3840-f64-f32-storage-32k", 0},
    {"axis-block-nvidia-vulkan-f64-f32-storage-n3840-u1", "stockham-nvidia-vulkan-3840-f64-f32-storage-32k", 1},
    {"axis-block-amd-vulkan-f64-f32-storage-n3840-u0", "stockham-amd-vulkan-3840-f64-f32-storage-32k", 0},
    {"axis-block-amd-vulkan-f64-f32-storage-n3840-u1", "stockham-amd-vulkan-3840-f64-f32-storage-32k", 1},
    {"axis-block-nvidia-vulkan-f32-n3840-higher-b0-u0", "stockham-nvidia-vulkan-3840-f32-higher-b0-32k", 0},
    {"axis-block-nvidia-vulkan-f32-n3840-higher-b0-u1", "stockham-nvidia-vulkan-3840-f32-higher-b0-32k", 1},
    {"axis-block-amd-vulkan-f32-n3840-higher-b0-u0", "stockham-amd-vulkan-3840-f32-higher-b0-32k", 0},
    {"axis-block-amd-vulkan-f32-n3840-higher-b0-u1", "stockham-amd-vulkan-3840-f32-higher-b0-32k", 1},
    {"axis-block-nvidia-vulkan-f32-n3840-higher-b2-u0", "stockham-nvidia-vulkan-3840-f32-higher-b2-32k", 0},
    {"axis-block-nvidia-vulkan-f32-n3840-higher-b2-u1", "stockham-nvidia-vulkan-3840-f32-higher-b2-32k", 1},
    {"axis-block-amd-vulkan-f32-n3840-higher-b2-u0", "stockham-amd-vulkan-3840-f32-higher-b2-32k", 0},
    {"axis-block-amd-vulkan-f32-n3840-higher-b2-u1", "stockham-amd-vulkan-3840-f32-higher-b2-32k", 1},
    {"axis-block-intel-vulkan-f32-n3840-u0", "stockham-intel-vulkan-3840-f32-32k", 0},
    {"axis-block-intel-vulkan-f16-n3840-u0", "stockham-intel-vulkan-3840-f16-32k", 0},
    {"axis-block-apple-metal-f32-n3840-u0", "stockham-apple-metal-3840-f32-32k", 0},
    {"axis-block-intel-level-zero-f32-n3840-u0", "stockham-intel-level-zero-3840-f32-32k", 0},
    {"axis-block-intel-level-zero-f64-n262144-stockham-u0", "stockham-intel-level-zero-262144-f64-64k", 0},
    {"axis-block-intel-level-zero-f64-n262144-stockham-u1", "stockham-intel-level-zero-262144-f64-64k", 1},
    {"axis-block-intel-level-zero-f64-n262144-stockham-u2", "stockham-intel-level-zero-262144-f64-64k", 2},
    {"axis-block-intel-level-zero-f64-f32-storage-n262144-stockham-u0", "stockham-intel-level-zero-262144-f64-f32-storage-64k", 0},
    {"axis-block-intel-level-zero-f64-f32-storage-n262144-stockham-u1", "stockham-intel-level-zero-262144-f64-f32-storage-64k", 1},
    {"axis-block-amd-opencl-f64-n262144-stockham-u0", "stockham-amd-opencl-262144-f64-64k", 0},
    {"axis-block-amd-opencl-f64-n262144-stockham-u1", "stockham-amd-opencl-262144-f64-64k", 1},
    {"axis-block-amd-opencl-f64-n262144-stockham-u2", "stockham-amd-opencl-262144-f64-64k", 2},
    {"axis-block-amd-opencl-f64-f32-storage-n262144-stockham-u0", "stockham-amd-opencl-262144-f64-f32-storage-64k", 0},
    {"axis-block-amd-opencl-f64-f32-storage-n262144-stockham-u1", "stockham-amd-opencl-262144-f64-f32-storage-64k", 1},
    {"axis-block-nvidia-vulkan-f32-n3145728-higher-b0-u0", "stockham-nvidia-vulkan-3145728-f32-higher-b0-32k", 0},
    {"axis-block-nvidia-vulkan-f32-n3145728-higher-b0-u1", "stockham-nvidia-vulkan-3145728-f32-higher-b0-32k", 1},
    {"axis-block-nvidia-vulkan-f32-n3145728-higher-b0-u2", "stockham-nvidia-vulkan-3145728-f32-higher-b0-32k", 2},
    {"axis-block-amd-vulkan-f32-n3145728-higher-b0-u0", "stockham-amd-vulkan-3145728-f32-higher-b0-32k", 0},
    {"axis-block-amd-vulkan-f32-n3145728-higher-b0-u1", "stockham-amd-vulkan-3145728-f32-higher-b0-32k", 1},
    {"axis-block-amd-vulkan-f32-n3145728-higher-b0-u2", "stockham-amd-vulkan-3145728-f32-higher-b0-32k", 2},
    {"axis-block-nvidia-vulkan-f32-n3145728-higher-b2-u0", "stockham-nvidia-vulkan-3145728-f32-higher-b2-32k", 0},
    {"axis-block-nvidia-vulkan-f32-n3145728-higher-b2-u1", "stockham-nvidia-vulkan-3145728-f32-higher-b2-32k", 1},
    {"axis-block-nvidia-vulkan-f32-n3145728-higher-b2-u2", "stockham-nvidia-vulkan-3145728-f32-higher-b2-32k", 2},
    {"axis-block-amd-vulkan-f32-n3145728-higher-b2-u0", "stockham-amd-vulkan-3145728-f32-higher-b2-32k", 0},
    {"axis-block-amd-vulkan-f32-n3145728-higher-b2-u1", "stockham-amd-vulkan-3145728-f32-higher-b2-32k", 1},
    {"axis-block-amd-vulkan-f32-n3145728-higher-b2-u2", "stockham-amd-vulkan-3145728-f32-higher-b2-32k", 2},
    {"axis-block-nvidia-vulkan-f32-n2097152-higher-b0-u0", "stockham-nvidia-vulkan-2097152-f32-higher-b0-32k", 0},
    {"axis-block-nvidia-vulkan-f32-n2097152-higher-b0-u1", "stockham-nvidia-vulkan-2097152-f32-higher-b0-32k", 1},
    {"axis-block-nvidia-vulkan-f32-n2097152-higher-b0-u2", "stockham-nvidia-vulkan-2097152-f32-higher-b0-32k", 2},
    {"axis-block-amd-vulkan-f32-n2097152-higher-b0-u0", "stockham-amd-vulkan-2097152-f32-higher-b0-32k", 0},
    {"axis-block-amd-vulkan-f32-n2097152-higher-b0-u1", "stockham-amd-vulkan-2097152-f32-higher-b0-32k", 1},
    {"axis-block-amd-vulkan-f32-n2097152-higher-b0-u2", "stockham-amd-vulkan-2097152-f32-higher-b0-32k", 2},
    {"axis-block-nvidia-vulkan-f32-n2097152-higher-b2-u0", "stockham-nvidia-vulkan-2097152-f32-higher-b2-32k", 0},
    {"axis-block-nvidia-vulkan-f32-n2097152-higher-b2-u1", "stockham-nvidia-vulkan-2097152-f32-higher-b2-32k", 1},
    {"axis-block-amd-vulkan-f32-n2097152-higher-b2-u0", "stockham-amd-vulkan-2097152-f32-higher-b2-32k", 0},
    {"axis-block-amd-vulkan-f32-n2097152-higher-b2-u1", "stockham-amd-vulkan-2097152-f32-higher-b2-32k", 1},
    {"axis-block-amd-vulkan-f32-n2097152-higher-b2-u2", "stockham-amd-vulkan-2097152-f32-higher-b2-32k", 2},
    {"axis-block-nvidia-vulkan-f32-p263-m567-bluestein-strided-b0-b5-g3-zp-u0", "stockham-nvidia-vulkan-f32-p263-m567-bluestein-strided-b0-b5-g3-zp", 0},
    {"axis-block-nvidia-vulkan-f32-p263-m567-bluestein-strided-b2-b5-g3-zp-u0", "stockham-nvidia-vulkan-f32-p263-m567-bluestein-strided-b2-b5-g3-zp", 0},
    {"axis-block-amd-vulkan-f32-p263-m625-bluestein-strided-b0-b5-g3-zp-u0", "stockham-amd-vulkan-f32-p263-m625-bluestein-strided-b0-b5-g3-zp", 0},
    {"axis-block-amd-vulkan-f32-p263-m625-bluestein-strided-b2-b5-g3-zp-u0", "stockham-amd-vulkan-f32-p263-m625-bluestein-strided-b2-b5-g3-zp", 0},
    {"axis-block-nvidia-vulkan-f32-p2053-m4368-bluestein-strided-b0-b5-g3-zp-u0", "stockham-nvidia-vulkan-f32-p2053-m4368-bluestein-strided-b0-b5-g3-zp", 0},
    {"axis-block-nvidia-vulkan-f32-p2053-m4368-bluestein-strided-b0-b5-g3-zp-u1", "stockham-nvidia-vulkan-f32-p2053-m4368-bluestein-strided-b0-b5-g3-zp", 1},
    {"axis-block-nvidia-vulkan-f32-p2053-m4368-bluestein-strided-b2-b5-g3-zp-u0", "stockham-nvidia-vulkan-f32-p2053-m4368-bluestein-strided-b2-b5-g3-zp", 0},
    {"axis-block-nvidia-vulkan-f32-p2053-m4368-bluestein-strided-b2-b5-g3-zp-u1", "stockham-nvidia-vulkan-f32-p2053-m4368-bluestein-strided-b2-b5-g3-zp", 1},
    {"axis-block-amd-vulkan-f32-p2053-m4375-bluestein-strided-b0-b5-g3-zp-u0", "stockham-amd-vulkan-f32-p2053-m4375-bluestein-strided-b0-b5-g3-zp", 0},
    {"axis-block-amd-vulkan-f32-p2053-m4375-bluestein-strided-b0-b5-g3-zp-u1", "stockham-amd-vulkan-f32-p2053-m4375-bluestein-strided-b0-b5-g3-zp", 1},
    {"axis-block-amd-vulkan-f32-p2053-m4375-bluestein-strided-b2-b5-g3-zp-u0", "stockham-amd-vulkan-f32-p2053-m4375-bluestein-strided-b2-b5-g3-zp", 0},
    {"axis-block-amd-vulkan-f32-p2053-m4375-bluestein-strided-b2-b5-g3-zp-u1", "stockham-amd-vulkan-f32-p2053-m4375-bluestein-strided-b2-b5-g3-zp", 1},
    {"axis-block-nvidia-vulkan-dd-p2053-m4368-bluestein-strided-b0-b5-g3-zp-u0", "stockham-nvidia-vulkan-dd-p2053-m4368-bluestein-strided-b0-b5-g3-zp", 0},
    {"axis-block-nvidia-vulkan-dd-p2053-m4368-bluestein-strided-b0-b5-g3-zp-u1", "stockham-nvidia-vulkan-dd-p2053-m4368-bluestein-strided-b0-b5-g3-zp", 1},
    {"axis-block-nvidia-vulkan-dd-p2053-m4368-bluestein-strided-b2-b5-g3-zp-u0", "stockham-nvidia-vulkan-dd-p2053-m4368-bluestein-strided-b2-b5-g3-zp", 0},
    {"axis-block-nvidia-vulkan-dd-p2053-m4368-bluestein-strided-b2-b5-g3-zp-u1", "stockham-nvidia-vulkan-dd-p2053-m4368-bluestein-strided-b2-b5-g3-zp", 1},
    {"axis-block-amd-vulkan-dd-p2053-m4224-bluestein-strided-b0-b5-g3-zp-u0", "stockham-amd-vulkan-dd-p2053-m4224-bluestein-strided-b0-b5-g3-zp", 0},
    {"axis-block-amd-vulkan-dd-p2053-m4224-bluestein-strided-b0-b5-g3-zp-u1", "stockham-amd-vulkan-dd-p2053-m4224-bluestein-strided-b0-b5-g3-zp", 1},
    {"axis-block-amd-vulkan-dd-p2053-m4224-bluestein-strided-b2-b5-g3-zp-u0", "stockham-amd-vulkan-dd-p2053-m4224-bluestein-strided-b2-b5-g3-zp", 0},
    {"axis-block-amd-vulkan-dd-p2053-m4224-bluestein-strided-b2-b5-g3-zp-u1", "stockham-amd-vulkan-dd-p2053-m4224-bluestein-strided-b2-b5-g3-zp", 1},
    {"axis-block-nvidia-vulkan-dd-p2503-m5184-bluestein-strided-b0-b5-g3-zp-u0", "stockham-nvidia-vulkan-dd-p2503-m5184-bluestein-strided-b0-b5-g3-zp", 0},
    {"axis-block-nvidia-vulkan-dd-p2503-m5184-bluestein-strided-b0-b5-g3-zp-u1", "stockham-nvidia-vulkan-dd-p2503-m5184-bluestein-strided-b0-b5-g3-zp", 1},
    {"axis-block-nvidia-vulkan-dd-p2503-m5184-bluestein-strided-b2-b5-g3-zp-u0", "stockham-nvidia-vulkan-dd-p2503-m5184-bluestein-strided-b2-b5-g3-zp", 0},
    {"axis-block-nvidia-vulkan-dd-p2503-m5184-bluestein-strided-b2-b5-g3-zp-u1", "stockham-nvidia-vulkan-dd-p2503-m5184-bluestein-strided-b2-b5-g3-zp", 1},
    {"axis-block-amd-vulkan-dd-p2503-m5005-bluestein-strided-b0-b5-g3-zp-u0", "stockham-amd-vulkan-dd-p2503-m5005-bluestein-strided-b0-b5-g3-zp", 0},
    {"axis-block-amd-vulkan-dd-p2503-m5005-bluestein-strided-b0-b5-g3-zp-u1", "stockham-amd-vulkan-dd-p2503-m5005-bluestein-strided-b0-b5-g3-zp", 1},
    {"axis-block-amd-vulkan-dd-p2503-m5005-bluestein-strided-b2-b5-g3-zp-u0", "stockham-amd-vulkan-dd-p2503-m5005-bluestein-strided-b2-b5-g3-zp", 0},
    {"axis-block-amd-vulkan-dd-p2503-m5005-bluestein-strided-b2-b5-g3-zp-u1", "stockham-amd-vulkan-dd-p2503-m5005-bluestein-strided-b2-b5-g3-zp", 1},
    {"axis-block-nvidia-vulkan-dd-p659-m1331-bluestein-strided-b0-b5-g3-zp-u0", "stockham-nvidia-vulkan-dd-p659-m1331-bluestein-strided-b0-b5-g3-zp", 0},
    {"axis-block-nvidia-vulkan-dd-p659-m1331-bluestein-strided-b0-b5-g3-zp-u1", "stockham-nvidia-vulkan-dd-p659-m1331-bluestein-strided-b0-b5-g3-zp", 1},
    {"axis-block-nvidia-vulkan-dd-p659-m1331-bluestein-strided-b2-b5-g3-zp-u0", "stockham-nvidia-vulkan-dd-p659-m1331-bluestein-strided-b2-b5-g3-zp", 0},
    {"axis-block-nvidia-vulkan-dd-p659-m1331-bluestein-strided-b2-b5-g3-zp-u1", "stockham-nvidia-vulkan-dd-p659-m1331-bluestein-strided-b2-b5-g3-zp", 1},
    {"axis-block-amd-vulkan-dd-p659-m1323-bluestein-strided-b0-b5-g3-zp-u0", "stockham-amd-vulkan-dd-p659-m1323-bluestein-strided-b0-b5-g3-zp", 0},
    {"axis-block-amd-vulkan-dd-p659-m1323-bluestein-strided-b0-b5-g3-zp-u1", "stockham-amd-vulkan-dd-p659-m1323-bluestein-strided-b0-b5-g3-zp", 1},
    {"axis-block-amd-vulkan-dd-p659-m1323-bluestein-strided-b2-b5-g3-zp-u0", "stockham-amd-vulkan-dd-p659-m1323-bluestein-strided-b2-b5-g3-zp", 0},
    {"axis-block-amd-vulkan-dd-p659-m1323-bluestein-strided-b2-b5-g3-zp-u1", "stockham-amd-vulkan-dd-p659-m1323-bluestein-strided-b2-b5-g3-zp", 1},
    {"axis-block-nvidia-vulkan-dd-n2431-multi-rader-strided-b5-g3-zp-u0", "stockham-nvidia-vulkan-dd-n2431-multi-rader-strided-b5-g3-zp", 0},
    {"axis-block-nvidia-vulkan-dd-n2431-multi-rader-strided-b5-g3-zp-u1", "stockham-nvidia-vulkan-dd-n2431-multi-rader-strided-b5-g3-zp", 1},
    {"axis-block-amd-vulkan-dd-n2431-multi-rader-strided-b5-g3-zp-u0", "stockham-amd-vulkan-dd-n2431-multi-rader-strided-b5-g3-zp", 0},
    {"axis-block-amd-vulkan-dd-n2431-multi-rader-strided-b5-g3-zp-u1", "stockham-amd-vulkan-dd-n2431-multi-rader-strided-b5-g3-zp", 1},
    {"axis-block-nvidia-vulkan-dd-n94-higher-portable-48k-u0", "stockham-nvidia-vulkan-dd-n94-higher-portable-48k", 0},
};

typedef struct {
    const char* case_name;
    const char* backend_name;
    const char* vendor_name;
    pfUINT vendor_id;
    pfUINT sequence_len;
    pfUINT shared_memory_bytes;
    pfUINT shared_memory_pow2_bytes;
    pfUINT max_threads_num;
    pfUINT max_workgroup_x;
    pfUINT max_workgroup_y;
    pfUINT max_workgroup_z;
    pfUINT device_coalesced_memory_bytes;
    pfUINT scheduler_coalesced_memory_bytes;
    pfUINT warp_size;
    int register_boost;
    pfUINT swap_threshold;
    int supports_f64;
    const char* precision_name;
    int quad_double_double_precision;
    int quad_double_double_precision_double_memory;
    int half_precision;
    int double_precision;
    int double_precision_float_memory;
} RealShapeCase;

static const RealShapeCase REAL_SHAPE_CASES[] = {
    {
        "real-shape-apple-metal-r2c-n8192", "metal", "apple", 0x1027f00,
        8192, 32 * 1024, 32 * 1024, 256, 256, 256, 256,
        32, 64, 1, 1, 524288, 0, "f32", 0, 0,
    },
    {
        "real-shape-intel-level-zero-r2c-n8192", "level-zero", "intel", 0x8086,
        8192, 32 * 1024, 32 * 1024, 256, 256, 256, 256,
        32, 64, 1, 1, 524288, 1, "f32", 0, 0,
    },
    {
        "real-shape-nvidia-vulkan-dd-r2c-n64-32k", "vulkan", "nvidia", 0x10DE,
        64, 32 * 1024, 32 * 1024, 1024, 1024, 1024, 64,
        32, 32, 32, 4, 4194305, 1, "double-double", 1, 0,
    },
    {
        "real-shape-nvidia-vulkan-dd-r2c-n4096-32k", "vulkan", "nvidia", 0x10DE,
        4096, 32 * 1024, 32 * 1024, 1024, 1024, 1024, 64,
        32, 32, 32, 4, 4194305, 1, "double-double", 1, 0,
    },
    {
        "real-shape-nvidia-vulkan-dd-f64-storage-r2c-n64-32k", "vulkan", "nvidia", 0x10DE,
        64, 32 * 1024, 32 * 1024, 1024, 1024, 1024, 64,
        32, 32, 32, 4, 4194305, 1, "double-double-f64-storage", 0, 1,
    },
    {
        "real-shape-nvidia-vulkan-dd-f64-storage-r2c-n4096-32k", "vulkan", "nvidia", 0x10DE,
        4096, 32 * 1024, 32 * 1024, 1024, 1024, 1024, 64,
        32, 32, 32, 4, 4194305, 1, "double-double-f64-storage", 0, 1,
    },
    {
        "real-shape-nvidia-vulkan-dd-r2c-n26-1k-rader-reserve", "vulkan", "nvidia", 0x10DE,
        26, 1024, 1024, 1024, 1024, 1024, 64,
        32, 32, 32, 4, 4194305, 1, "double-double", 1, 0,
    },
    {
        "real-shape-nvidia-vulkan-dd-r2c-n26-2k-rader-reserve", "vulkan", "nvidia", 0x10DE,
        26, 2 * 1024, 2 * 1024, 1024, 1024, 1024, 64,
        32, 32, 32, 4, 4194305, 1, "double-double", 1, 0,
    },
    {
        "real-shape-nvidia-vulkan-dd-r2c-n94-4k-bluestein-padding", "vulkan", "nvidia", 0x10DE,
        94, 4 * 1024, 4 * 1024, 1024, 1024, 1024, 64,
        32, 32, 32, 4, 4194305, 1, "double-double", 1, 0,
    },
    {
        "real-shape-nvidia-vulkan-dd-r2c-n94-8k-bluestein-padding", "vulkan", "nvidia", 0x10DE,
        94, 8 * 1024, 8 * 1024, 1024, 1024, 1024, 64,
        32, 32, 32, 4, 4194305, 1, "double-double", 1, 0,
    },
    {
        "real-shape-nvidia-vulkan-dd-r2c-n34-1536-fft-rader", "vulkan", "nvidia", 0x10DE,
        34, 1536, 1536, 1024, 1024, 1024, 64,
        32, 32, 32, 4, 4194305, 1, "double-double", 1, 0,
    },
    {
        "real-shape-amd-vulkan-dd-r2c-n34-1536-direct-rader", "vulkan", "amd", 0x1002,
        34, 1536, 1536, 1024, 1024, 1024, 64,
        32, 32, 64, 4, 262144, 1, "double-double", 1, 0,
    },
    {
        "real-shape-nvidia-vulkan-dd-r2c-n1088-48k-threads32-force-rader-two-upload", "vulkan", "nvidia", 0x10DE,
        1088, 48 * 1024, 32 * 1024, 32, 32, 32, 64,
        32, 32, 32, 4, 4194305, 1, "double-double", 1, 0,
    },
    {
        "real-shape-nvidia-vulkan-dd-r2c-n1088-48k-threads128-force-rader-two-upload", "vulkan", "nvidia", 0x10DE,
        1088, 48 * 1024, 32 * 1024, 128, 128, 128, 64,
        32, 32, 32, 4, 4194305, 1, "double-double", 1, 0,
    },
    {
        "real-shape-nvidia-vulkan-f16-r2c-n94-4k-128t", "vulkan", "nvidia", 0x10DE,
        94, 4 * 1024, 4 * 1024, 128, 128, 128, 64,
        32, 64, 32, 4, 4194305, 1, "f16-storage-f32-compute", 0, 0, 1, 0, 0,
    },
    {
        "real-shape-nvidia-vulkan-f16-r2c-n94-8k-128t", "vulkan", "nvidia", 0x10DE,
        94, 8 * 1024, 8 * 1024, 128, 128, 128, 64,
        32, 64, 32, 4, 4194305, 1, "f16-storage-f32-compute", 0, 0, 1, 0, 0,
    },
};

typedef struct {
    const char* case_name;
    pfUINT sequence_len;
    int snapshot_upload_id;
    pfUINT shared_memory_bytes;
    pfUINT shared_memory_pow2_bytes;
    int double_double;
    const char* vendor_name;
    pfUINT vendor_id;
    pfUINT max_threads_num;
    pfUINT max_workgroup_x;
    pfUINT max_workgroup_y;
    pfUINT coalesced_memory_bytes;
    int warp_size;
    pfUINT register_boost;
    pfUINT swap_threshold;
    int double_precision;
    int automatic_grouping;
    int min_direct_prime;
    int max_direct_prime;
    int min_fft_prime;
    int max_fft_prime;
    pfUINT fastest_axis_len;
    pfUINT outer_batches;
    int half_precision;
} AxisBlockCase;

static const AxisBlockCase AXIS_BLOCK_CASES[] = {
    {"axis-block-nvidia-6144-upload1-higher-g3", 6144, 1, 32 * 1024, 32 * 1024, 0},
    {"axis-block-nvidia-6144-upload0-higher-g3", 6144, 0, 32 * 1024, 32 * 1024, 0},
    {
        "axis-block-nvidia-6144-upload1-higher-auto", 6144, 1, 32 * 1024, 32 * 1024, 0,
        "nvidia", 0x10DE, 1024, 1024, 1024, 32, 32, 4, 4194305, 0, 1,
        17, 89, 17, 16384, 64, 1,
    },
    {
        "axis-block-nvidia-6144-upload0-higher-auto", 6144, 0, 32 * 1024, 32 * 1024, 0,
        "nvidia", 0x10DE, 1024, 1024, 1024, 32, 32, 4, 4194305, 0, 1,
        17, 89, 17, 16384, 64, 1,
    },
    {
        "axis-block-amd-6144-upload1-higher-auto", 6144, 1, 32 * 1024, 32 * 1024, 0,
        "amd", 0x1002, 1024, 1024, 1024, 32, 64, 4, 524288, 0, 1,
        17, 89, 17, 16384, 64, 1,
    },
    {
        "axis-block-amd-6144-upload0-higher-auto", 6144, 0, 32 * 1024, 32 * 1024, 0,
        "amd", 0x1002, 1024, 1024, 1024, 32, 64, 4, 524288, 0, 1,
        17, 89, 17, 16384, 64, 1,
    },
    {"axis-block-nvidia-p47-higher-g3", 47, -1, 48 * 1024, 32 * 1024, 0},
    {"axis-block-nvidia-p257-higher-g3", 257, -1, 48 * 1024, 32 * 1024, 0},
    {"axis-block-nvidia-p31-higher-g3", 31, -1, 48 * 1024, 32 * 1024, 0},
    {"axis-block-nvidia-p281-higher-g3", 281, -1, 48 * 1024, 32 * 1024, 0},
    {"axis-block-nvidia-dd-p13-higher-g3", 13, -1, 48 * 1024, 32 * 1024, 1},
    {"axis-block-nvidia-dd-p17-higher-g3", 17, -1, 48 * 1024, 32 * 1024, 1},
    {"axis-block-nvidia-dd-p257-higher-g3", 257, -1, 48 * 1024, 32 * 1024, 1},
    {"axis-block-nvidia-dd-n4800-higher-164k", 4800, -1, 164 * 1024, 164 * 1024, 1},
    {"axis-block-nvidia-dd-n5000-higher-164k", 5000, -1, 164 * 1024, 164 * 1024, 1},
    {"axis-block-nvidia-dd-n5120-higher-164k", 5120, -1, 164 * 1024, 164 * 1024, 1},
    {
        "axis-block-nvidia-dd-n192-higher-auto-48k", 192, -1, 48 * 1024, 32 * 1024, 1,
        "nvidia", 0x10DE, 1024, 1024, 1024, 32, 32, 4, 4194305, 0, 1,
        11, 29, 17, 16384, 8, 5,
    },
    {
        "axis-block-amd-dd-n192-higher-auto-48k", 192, -1, 48 * 1024, 32 * 1024, 1,
        "amd", 0x1002, 1024, 1024, 1024, 32, 64, 4, 524288, 0, 1,
        11, 29, 17, 16384, 8, 5,
    },
    {
        "axis-block-intel-dd-n192-higher-auto-48k", 192, -1, 48 * 1024, 32 * 1024, 1,
        "intel", 0x8086, 1024, 1024, 1024, 64, 32, 2, 524288, 0, 1,
        11, 29, 17, 16384, 8, 5,
    },
    {
        "axis-block-nvidia-f64-p47-higher-auto-128t", 47, -1, 48 * 1024, 32 * 1024, 0,
        "nvidia", 0x10DE, 128, 1024, 1024, 32, 32, 4, 4194305, 1, 1,
        17, 89, 53, 16384,
    },
    {
        "axis-block-intel-f64-p47-higher-auto-128t", 47, -1, 48 * 1024, 32 * 1024, 0,
        "intel", 0x8086, 128, 1024, 1024, 64, 32, 2, 524288, 1, 1,
        17, 89, 53, 16384,
    },
    {
        "axis-block-nvidia-dd-p47-higher-auto-128t", 47, -1, 48 * 1024, 32 * 1024, 1,
        "nvidia", 0x10DE, 128, 1024, 1024, 32, 32, 4, 4194305, 0, 1,
        17, 89, 53, 16384,
    },
    {
        "axis-block-intel-dd-p47-higher-auto-128t", 47, -1, 48 * 1024, 32 * 1024, 1,
        "intel", 0x8086, 128, 1024, 1024, 64, 32, 2, 524288, 0, 1,
        17, 89, 53, 16384,
    },
    {
        "axis-block-nvidia-f16-6144-upload1-higher-auto", 6144, 1, 32 * 1024, 32 * 1024, 0,
        "nvidia", 0x10DE, 1024, 1024, 1024, 64, 32, 4, 4194305, 0, 1,
        17, 89, 17, 16384, 64, 1, 1,
    },
    {
        "axis-block-nvidia-f16-6144-upload0-higher-auto", 6144, 0, 32 * 1024, 32 * 1024, 0,
        "nvidia", 0x10DE, 1024, 1024, 1024, 64, 32, 4, 4194305, 0, 1,
        17, 89, 17, 16384, 64, 1, 1,
    },
    {
        "axis-block-amd-f16-6144-upload1-higher-auto", 6144, 1, 32 * 1024, 32 * 1024, 0,
        "amd", 0x1002, 1024, 1024, 1024, 64, 64, 4, 524288, 0, 1,
        17, 89, 17, 16384, 64, 1, 1,
    },
    {
        "axis-block-amd-f16-6144-upload0-higher-auto", 6144, 0, 32 * 1024, 32 * 1024, 0,
        "amd", 0x1002, 1024, 1024, 1024, 64, 64, 4, 524288, 0, 1,
        17, 89, 17, 16384, 64, 1, 1,
    },
    {
        "axis-block-intel-f16-6144-upload1-higher-auto", 6144, 1, 32 * 1024, 32 * 1024, 0,
        "intel", 0x8086, 256, 256, 256, 128, 32, 2, 524288, 0, 1,
        17, 89, 17, 16384, 64, 1, 1,
    },
    {
        "axis-block-intel-f16-6144-upload0-higher-auto", 6144, 0, 32 * 1024, 32 * 1024, 0,
        "intel", 0x8086, 256, 256, 256, 128, 32, 2, 524288, 0, 1,
        17, 89, 17, 16384, 64, 1, 1,
    },
    {
        "axis-block-nvidia-f16-n524288-higher-u0", 524288, 0, 48 * 1024, 32 * 1024, 0,
        "nvidia", 0x10DE, 1024, 1024, 1024, 64, 32, 4, 4194305, 0, 1,
        17, 89, 17, 16384, 64, 1, 1,
    },
    {
        "axis-block-nvidia-f16-n524288-higher-u1", 524288, 1, 48 * 1024, 32 * 1024, 0,
        "nvidia", 0x10DE, 1024, 1024, 1024, 64, 32, 4, 4194305, 0, 1,
        17, 89, 17, 16384, 64, 1, 1,
    },
    {
        "axis-block-nvidia-f16-n524288-higher-u2", 524288, 2, 48 * 1024, 32 * 1024, 0,
        "nvidia", 0x10DE, 1024, 1024, 1024, 64, 32, 4, 4194305, 0, 1,
        17, 89, 17, 16384, 64, 1, 1,
    },
    {
        "axis-block-amd-f16-n524288-higher-u0", 524288, 0, 48 * 1024, 32 * 1024, 0,
        "amd", 0x1002, 1024, 1024, 1024, 64, 64, 4, 524288, 0, 1,
        17, 89, 17, 16384, 64, 1, 1,
    },
    {
        "axis-block-amd-f16-n524288-higher-u1", 524288, 1, 48 * 1024, 32 * 1024, 0,
        "amd", 0x1002, 1024, 1024, 1024, 64, 64, 4, 524288, 0, 1,
        17, 89, 17, 16384, 64, 1, 1,
    },
    {
        "axis-block-amd-f16-n524288-higher-u2", 524288, 2, 48 * 1024, 32 * 1024, 0,
        "amd", 0x1002, 1024, 1024, 1024, 64, 64, 4, 524288, 0, 1,
        17, 89, 17, 16384, 64, 1, 1,
    },
    {
        "axis-block-intel-f16-n524288-higher-u0", 524288, 0, 48 * 1024, 32 * 1024, 0,
        "intel", 0x8086, 1024, 1024, 1024, 128, 32, 2, 524288, 0, 1,
        17, 89, 17, 16384, 64, 1, 1,
    },
    {
        "axis-block-intel-f16-n524288-higher-u1", 524288, 1, 48 * 1024, 32 * 1024, 0,
        "intel", 0x8086, 1024, 1024, 1024, 128, 32, 2, 524288, 0, 1,
        17, 89, 17, 16384, 64, 1, 1,
    },
    {
        "axis-block-intel-f16-n524288-higher-u2", 524288, 2, 48 * 1024, 32 * 1024, 0,
        "intel", 0x8086, 1024, 1024, 1024, 128, 32, 2, 524288, 0, 1,
        17, 89, 17, 16384, 64, 1, 1,
    },
    {
        "axis-block-nvidia-f16-n512-higher-auto", 512, -1, 32 * 1024, 32 * 1024, 0,
        "nvidia", 0x10DE, 1024, 1024, 1024, 64, 32, 4, 4194305, 0, 1,
        17, 89, 17, 16384, 64, 1, 1,
    },
    {
        "axis-block-amd-f16-n512-higher-auto", 512, -1, 32 * 1024, 32 * 1024, 0,
        "amd", 0x1002, 1024, 1024, 1024, 64, 64, 4, 524288, 0, 1,
        17, 89, 17, 16384, 64, 1, 1,
    },
    {
        "axis-block-intel-f16-n512-higher-auto", 512, -1, 32 * 1024, 32 * 1024, 0,
        "intel", 0x8086, 1024, 1024, 1024, 128, 32, 2, 524288, 0, 1,
        17, 89, 17, 16384, 64, 1, 1,
    },
    {
        "axis-block-nvidia-f16-p47-higher-auto", 47, -1, 32 * 1024, 32 * 1024, 0,
        "nvidia", 0x10DE, 1024, 1024, 1024, 64, 32, 4, 4194305, 0, 1,
        17, 89, 53, 16384, 64, 1, 1,
    },
    {
        "axis-block-amd-f16-p47-higher-auto", 47, -1, 32 * 1024, 32 * 1024, 0,
        "amd", 0x1002, 1024, 1024, 1024, 64, 64, 4, 524288, 0, 1,
        17, 89, 53, 16384, 64, 1, 1,
    },
    {
        "axis-block-intel-f16-p47-higher-auto", 47, -1, 32 * 1024, 32 * 1024, 0,
        "intel", 0x8086, 1024, 1024, 1024, 128, 32, 2, 524288, 0, 1,
        17, 89, 53, 16384, 64, 1, 1,
    },
    {
        "axis-block-nvidia-dd-n16384-higher-u1-auto", 16384, 1, 48 * 1024, 32 * 1024, 1,
        "nvidia", 0x10DE, 1024, 1024, 1024, 32, 32, 4, 4194305, 0, 1,
        11, 29, 17, 16384, 64, 1,
    },
    {
        "axis-block-amd-dd-n16384-higher-u1-auto", 16384, 1, 48 * 1024, 32 * 1024, 1,
        "amd", 0x1002, 1024, 1024, 1024, 32, 64, 4, 524288, 0, 1,
        11, 29, 17, 16384, 64, 1,
    },
    {
        "axis-block-intel-dd-n16384-higher-u1-auto", 16384, 1, 48 * 1024, 32 * 1024, 1,
        "intel", 0x8086, 1024, 1024, 1024, 64, 32, 2, 524288, 0, 1,
        11, 29, 17, 16384, 64, 1,
    }
};

typedef struct {
    const char* case_name;
    pfUINT sequence_len;
    pfUINT batch_count;
    pfUINT shared_memory_bytes;
    pfUINT shared_memory_pow2_bytes;
    const char* backend_name;
    const char* vendor_name;
    pfUINT vendor_id;
    pfUINT max_threads_num;
    pfUINT max_workgroup_x;
    pfUINT coalesced_memory_bytes;
    pfUINT warp_size;
    int register_boost;
    pfUINT swap_threshold;
    int min_direct_prime;
    int max_direct_prime;
    int min_fft_prime;
    int max_fft_prime;
    int half_precision;
    int expect_direct_rader;
    pfUINT grouped_batch_override;
    int perform_convolution;
    pfUINT number_kernels;
} Axis0RaderBlockCase;

static const Axis0RaderBlockCase AXIS0_RADER_BLOCK_CASES[] = {
    {"axis-block-nvidia-p17-b1", 17, 1, 48 * 1024, 32 * 1024},
    {"axis-block-nvidia-p17-b32", 17, 32, 48 * 1024, 32 * 1024},
    {"axis-block-nvidia-p31-b1", 31, 1, 48 * 1024, 32 * 1024},
    {"axis-block-nvidia-p31-b32", 31, 32, 48 * 1024, 32 * 1024},
    {"axis-block-nvidia-p257-b1", 257, 1, 48 * 1024, 32 * 1024},
    {"axis-block-nvidia-p7681-b1", 7681, 1, 64 * 1024, 64 * 1024},
    {
        "axis-block-nvidia-p67-direct-b32", 67, 32, 32 * 1024, 32 * 1024,
        "vulkan", "nvidia", 0x10DE, 512, 512, 32, 32, 4, 4194305,
        17, 89, 89, 16384,
    },
    {
        "axis-block-amd-p67-direct-b32", 67, 32, 32 * 1024, 32 * 1024,
        "vulkan", "amd", 0x1002, 512, 512, 32, 64, 4, 524288,
        17, 89, 89, 16384,
    },
    {
        "axis-block-amd-hip-p67-direct-b32-wave32", 67, 32, 32 * 1024, 32 * 1024,
        "hip", "amd", 0x1002, 512, 512, 32, 32, 1, 2097152,
        17, 89, 89, 16384, 0, 1,
    },
    {
        "axis-block-amd-hip-p67-direct-b32-wave64", 67, 32, 32 * 1024, 32 * 1024,
        "hip", "amd", 0x1002, 512, 512, 32, 64, 1, 2097152,
        17, 89, 89, 16384, 0, 1,
    },
    {
        "axis-block-intel-opencl-p67-direct-b32", 67, 32, 32 * 1024, 32 * 1024,
        "opencl", "intel", 0x8086, 512, 512, 64, 32, 2, 524288,
        17, 89, 89, 16384,
    },
    {
        "axis-block-intel-level-zero-p67-direct-b32", 67, 32, 32 * 1024, 32 * 1024,
        "level-zero", "intel", 0x8086, 512, 512, 64, 1, 2, 524288,
        17, 89, 89, 16384,
    },
    {
        "axis-block-nvidia-f16-p83-direct-b32-512t", 83, 32, 32 * 1024, 32 * 1024,
        "vulkan", "nvidia", 0x10DE, 512, 512, 64, 32, 4, 4194305,
        17, 89, 17, 16384, 1, 1,
    },
    {
        "axis-block-amd-f16-p83-direct-b32-512t", 83, 32, 32 * 1024, 32 * 1024,
        "vulkan", "amd", 0x1002, 512, 512, 64, 64, 4, 524288,
        17, 89, 17, 16384, 1, 1,
    },
    {
        "axis-block-nvidia-f16-p47-direct-b1-256t", 47, 1, 48 * 1024, 32 * 1024,
        "vulkan", "nvidia", 0x10DE, 256, 256, 64, 32, 4, 4194305,
        17, 89, 17, 16384, 1, 1,
    },
    {
        "axis-block-nvidia-p47-direct-b1-convolution-k3", 47, 1, 48 * 1024, 32 * 1024,
        "vulkan", "nvidia", 0x10DE, 1024, 1024, 32, 32, 4, 4194305,
        17, 89, 17, 16384, 0, 1, 0, 1, 3,
    },
    {
        "axis-block-nvidia-f16-p769-fft-b32-g8", 769, 32, 32 * 1024, 32 * 1024,
        "vulkan", "nvidia", 0x10DE, 1024, 1024, 64, 32, 4, 4194305,
        17, 89, 17, 16384, 1, 0, 8,
    },
    {
        "axis-block-nvidia-f16-n323-composite-b32-g16", 323, 32, 32 * 1024, 32 * 1024,
        "vulkan", "nvidia", 0x10DE, 1024, 1024, 64, 32, 4, 4194305,
        17, 89, 17, 16384, 1, 0, 16,
    },
};

typedef struct {
    const char* case_name;
    pfUINT sequence_len;
    pfUINT batch_count;
    pfUINT shared_memory_bytes;
    pfUINT shared_memory_pow2_bytes;
} Axis0StockhamBlockCase;

static const Axis0StockhamBlockCase AXIS0_STOCKHAM_BLOCK_CASES[] = {
    {"axis-block-nvidia-dd-n4800-b1-164k", 4800, 1, 164 * 1024, 164 * 1024},
    {"axis-block-nvidia-dd-n5000-b1-164k", 5000, 1, 164 * 1024, 164 * 1024},
    {"axis-block-nvidia-dd-n5120-b1-164k", 5120, 1, 164 * 1024, 164 * 1024},
    {"axis-block-nvidia-dd-n4800-b32-164k", 4800, 32, 164 * 1024, 164 * 1024},
    {"axis-block-nvidia-dd-n5000-b32-164k", 5000, 32, 164 * 1024, 164 * 1024},
    {"axis-block-nvidia-dd-n5120-b32-164k", 5120, 32, 164 * 1024, 164 * 1024},
};

typedef struct {
    const char* case_name;
    const char* vendor_name;
    pfUINT vendor_id;
    pfUINT sequence_len;
    pfUINT shared_memory_bytes;
    pfUINT shared_memory_pow2_bytes;
    pfUINT register_boost;
    pfUINT swap_to_three_stage;
    int warp_size;
    int wide_precision;
    pfUINT max_threads_num;
    pfUINT max_workgroup_x;
    pfUINT coalesced_memory_bytes;
    int half_precision;
    int keep_default_rader_fft_max;
    const char* backend_name;
    pfUINT batch_count;
    int padding_only;
} BluesteinCase;

static const BluesteinCase BLUESTEIN_CASES[] = {
    {
        "bluestein-nvidia-n206-f32", "nvidia", 0x10DE, 206,
        48 * 1024, 32 * 1024, 4, 4194305, 32, 0,
    },
    {
        "bluestein-nvidia-p257-f32-autopad", "nvidia", 0x10DE, 257,
        48 * 1024, 32 * 1024, 4, 4194305, 32, 0,
    },
    {
        "bluestein-amd-p257-f32-autopad", "amd", 0x1002, 257,
        64 * 1024, 64 * 1024, 2, 524288, 64, 0,
    },
    {
        .case_name = "bluestein-nvidia-p2053-f32-autopad",
        .vendor_name = "nvidia", .vendor_id = 0x10DE, .sequence_len = 2053,
        .shared_memory_bytes = 48 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .register_boost = 4, .swap_to_three_stage = 4194305, .warp_size = 32,
        .padding_only = 1,
    },
    {
        .case_name = "bluestein-amd-p2053-f32-autopad",
        .vendor_name = "amd", .vendor_id = 0x1002, .sequence_len = 2053,
        .shared_memory_bytes = 64 * 1024, .shared_memory_pow2_bytes = 64 * 1024,
        .register_boost = 2, .swap_to_three_stage = 524288, .warp_size = 64,
        .padding_only = 1,
    },
    {
        "bluestein-nvidia-p103-f64-autopad", "nvidia", 0x10DE, 103,
        48 * 1024, 32 * 1024, 4, 4194305, 32, 1,
    },
    {
        "bluestein-amd-p103-f64-autopad", "amd", 0x1002, 103,
        64 * 1024, 64 * 1024, 2, 524288, 64, 1,
    },
    {
        .case_name = "bluestein-nvidia-n611-f64-autopad",
        .vendor_name = "nvidia", .vendor_id = 0x10DE, .sequence_len = 611,
        .shared_memory_bytes = 48 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .register_boost = 4, .swap_to_three_stage = 4194305, .warp_size = 32,
        .wide_precision = 1, .padding_only = 1,
    },
    {
        .case_name = "bluestein-amd-n611-f64-autopad",
        .vendor_name = "amd", .vendor_id = 0x1002, .sequence_len = 611,
        .shared_memory_bytes = 64 * 1024, .shared_memory_pow2_bytes = 64 * 1024,
        .register_boost = 2, .swap_to_three_stage = 524288, .warp_size = 64,
        .wide_precision = 1, .padding_only = 1,
    },
    {
        .case_name = "bluestein-nvidia-p2053-f64-autopad",
        .vendor_name = "nvidia", .vendor_id = 0x10DE, .sequence_len = 2053,
        .shared_memory_bytes = 48 * 1024, .shared_memory_pow2_bytes = 32 * 1024,
        .register_boost = 4, .swap_to_three_stage = 4194305, .warp_size = 32,
        .wide_precision = 1, .padding_only = 1,
    },
    {
        .case_name = "bluestein-amd-p2053-f64-autopad",
        .vendor_name = "amd", .vendor_id = 0x1002, .sequence_len = 2053,
        .shared_memory_bytes = 64 * 1024, .shared_memory_pow2_bytes = 64 * 1024,
        .register_boost = 2, .swap_to_three_stage = 524288, .warp_size = 64,
        .wide_precision = 1, .padding_only = 1,
    },
    {
        "bluestein-nvidia-p317-f64-autopad", "nvidia", 0x10DE, 317,
        48 * 1024, 32 * 1024, 4, 4194305, 32, 1,
    },
    {
        "bluestein-amd-p317-f64-autopad", "amd", 0x1002, 317,
        64 * 1024, 64 * 1024, 2, 262144, 64, 1,
    },
    {
        "bluestein-nvidia-n4106-f32", "nvidia", 0x10DE, 4106,
        48 * 1024, 32 * 1024, 4, 4194305, 32, 0,
    },
    {
        "bluestein-amd-n4106-f32", "amd", 0x1002, 4106,
        64 * 1024, 64 * 1024, 2, 524288, 64, 0,
    },
    {
        "bluestein-nvidia-p47-f16-thread-cap", "nvidia", 0x10DE, 47,
        48 * 1024, 32 * 1024, 4, 4194305, 32, 0,
        128, 128, 64, 1, 1,
    },
    {
        "bluestein-intel-opencl-p83-f16-thread-cap", "intel", 0x8086, 83,
        32 * 1024, 32 * 1024, 2, 524288, 32, 0,
        512, 512, 128, 1, 1, "opencl", 32,
    },
    {
        "bluestein-intel-opencl-p47-f16-thread-cap", "intel", 0x8086, 47,
        32 * 1024, 32 * 1024, 2, 524288, 32, 0,
        256, 256, 128, 1, 1, "opencl",
    },
};

typedef struct {
    const char* case_name;
    const char* backend_name;
    const char* vendor_name;
    pfUINT vendor_id;
    pfUINT sequence_len;
    pfUINT shared_memory_bytes;
    pfUINT shared_memory_pow2_bytes;
    pfUINT max_threads_num;
    pfUINT max_workgroup_x;
    pfUINT coalesced_memory_bytes;
    pfUINT warp_size;
    int register_boost;
    pfUINT swap_threshold;
    int min_direct_prime;
    int max_direct_prime;
    int min_fft_prime;
    int max_fft_prime;
} DeviceDoubleDoubleBluesteinCase;

static const DeviceDoubleDoubleBluesteinCase DEVICE_DD_BLUESTEIN_CASES[] = {
    {
        "bluestein-amd-vulkan-dd-n3196-48k", "vulkan", "amd", 0x1002, 3196,
        48 * 1024, 32 * 1024, 1024, 1024, 32, 64, 4, 524288,
        11, 29, 19, 16384,
    },
    {
        "bluestein-amd-vulkan-dd-n3196-64k", "vulkan", "amd", 0x1002, 3196,
        64 * 1024, 64 * 1024, 1024, 1024, 32, 64, 2, 524288,
        11, 29, 19, 16384,
    },
    {
        "bluestein-intel-opencl-dd-n3196", "opencl", "intel", 0x8086, 3196,
        32 * 1024, 32 * 1024, 256, 256, 64, 32, 2, 524288,
        11, 29, 17, 16384,
    },
    {
        "bluestein-nvidia-vulkan-dd-p2203-164k", "vulkan", "nvidia", 0x10DE, 2203,
        164 * 1024, 164 * 1024, 1024, 1024, 32, 32, 4, 4194305,
        11, 29, 17, 100,
    },
    {
        "bluestein-intel-vulkan-dd-p263-32k", "vulkan", "intel", 0x8086, 263,
        32 * 1024, 32 * 1024, 1024, 1024, 64, 32, 2, 262144,
        11, 29, 19, 16384,
    },
};

typedef struct {
    const char* case_name;
    size_t bluestein_case_index;
    pfUINT axis_id;
    pfUINT fastest_axis_len;
    pfUINT upload_id;
} DeviceDoubleDoubleBluesteinAxisBlockReference;

static const DeviceDoubleDoubleBluesteinAxisBlockReference DEVICE_DD_BLUESTEIN_AXIS_BLOCK_REFERENCES[] = {
    {"axis-block-amd-vulkan-dd-n3196-bluestein-u0", 0, 0, 1, 0},
    {"axis-block-amd-vulkan-dd-n3196-bluestein-u1", 0, 0, 1, 1},
    {"axis-block-amd-vulkan-dd-n3196-64k-bluestein-u0", 1, 0, 1, 0},
    {"axis-block-amd-vulkan-dd-n3196-64k-bluestein-u1", 1, 0, 1, 1},
    {"axis-block-intel-opencl-dd-n3196-bluestein-u0", 2, 0, 1, 0},
    {"axis-block-intel-opencl-dd-n3196-bluestein-u1", 2, 0, 1, 1},
    {"axis-block-nvidia-vulkan-dd-p2203-164k-bluestein-u0", 3, 0, 1, 0},
    {"axis-block-nvidia-vulkan-dd-p2203-164k-bluestein-higher-u0", 3, 1, 2, 0},
};

typedef struct {
    const char* case_name;
    size_t bluestein_case_index;
    pfUINT fastest_axis_len;
    int perform_bandwidth_boost;
} DeviceDoubleDoubleBluesteinStockhamReference;

static const DeviceDoubleDoubleBluesteinStockhamReference DEVICE_DD_BLUESTEIN_STOCKHAM_REFERENCES[] = {
    {"stockham-intel-vulkan-dd-p263-m625-bluestein-strided-b0", 4, 2, 0},
    {"stockham-intel-vulkan-dd-p263-m625-bluestein-strided-b2", 4, 2, 2},
};

static void configure_app(
    VkFFTApplication* app,
    const RaderCase* test_case,
    pfUINT* temp_buffer_size
) {
    memset(app, 0, sizeof(*app));
    app->configuration.FFTdim = 1;
    app->configuration.size[0] = test_case->sequence_len;
    app->configuration.vendorID = test_case->vendor_id;
    app->configuration.sharedMemorySize = test_case->shared_memory_bytes;
    app->configuration.sharedMemorySizePow2 = test_case->shared_memory_pow2_bytes;
    app->configuration.maxThreadsNum = 1024;
    app->configuration.maxComputeWorkGroupSize[0] = 1024;
    app->configuration.maxComputeWorkGroupSize[1] = 1024;
    app->configuration.maxComputeWorkGroupSize[2] = 64;
    app->configuration.coalescedMemory = test_case->coalesced_memory_bytes;
    app->configuration.aimThreads = 128;
    app->configuration.numSharedBanks = 32;
    app->configuration.warpSize = test_case->vendor_id == 0x1002 ? 64 : 32;
    app->configuration.registerBoost = test_case->register_boost;
    app->configuration.registerBoost4Step = 1;
    app->configuration.registerBoostNonPow2 = 0;
    app->configuration.swapTo3Stage4Step = test_case->swap_to_three_stage;
    app->configuration.swapTo2Stage4Step = test_case->swap_to_three_stage;
    app->configuration.reorderFourStep = 1;
    app->configuration.fixMinRaderPrimeMult = 17;
    app->configuration.fixMaxRaderPrimeMult = 89;
    app->configuration.fixMinRaderPrimeFFT = 17;
    app->configuration.fixMaxRaderPrimeFFT = 16384;
    app->configuration.coordinateFeatures = 1;
    app->configuration.numberBatches = 1;
    app->configuration.numberKernels = test_case->number_kernels != 0 ? test_case->number_kernels : 1;
    app->configuration.tempBufferSize = temp_buffer_size;
    app->actualNumBatches = test_case->outer_batches;
    if (test_case->perform_convolution) {
        // Mirror pinned initializeVkFFT defaults for performConvolution exactly as the
        // Stockham reference path does; this lets the same scheduler/container code
        // expose whether application mode changes FFT-Rader physical ownership.
        app->configuration.performConvolution = 1;
        app->configuration.reorderFourStep = 0;
        app->configuration.registerBoost = 1;
        app->configuration.registerBoostNonPow2 = 0;
        app->configuration.registerBoost4Step = 1;
    }
}

static const char* rader_upload_backend_name(const RaderUploadCase* test_case) {
    if (test_case->profile_kind == 2) return "opencl";
    if (test_case->profile_kind == 4) return "level-zero";
    return "vulkan";
}

static const char* rader_upload_vendor_name(const RaderUploadCase* test_case) {
    if (test_case->profile_kind == 1) return "amd";
    if ((test_case->profile_kind == 2) || (test_case->profile_kind == 3) || (test_case->profile_kind == 4)) return "intel";
    return "nvidia";
}

static pfUINT rader_upload_vendor_id(const RaderUploadCase* test_case) {
    if (test_case->profile_kind == 1) return 0x1002;
    if ((test_case->profile_kind == 2) || (test_case->profile_kind == 3) || (test_case->profile_kind == 4)) return 0x8086;
    return 0x10DE;
}

static pfUINT rader_upload_max_workgroup_x(const RaderUploadCase* test_case) {
    if (test_case->max_workgroup_x != 0) return test_case->max_workgroup_x;
    return test_case->profile_kind == 2 ? 256 : 1024;
}

static pfUINT rader_upload_max_workgroup_y(const RaderUploadCase* test_case) {
    if (test_case->max_workgroup_y != 0) return test_case->max_workgroup_y;
    return rader_upload_max_workgroup_x(test_case);
}

static pfUINT rader_upload_coalesced_memory(const RaderUploadCase* test_case) {
    const pfUINT base = ((test_case->profile_kind == 2) || (test_case->profile_kind == 3) || (test_case->profile_kind == 4)) ? 64 : 32;
    return test_case->half_precision ? 2 * base : base;
}

static const char* rader_upload_precision_name(const RaderUploadCase* test_case) {
    if (test_case->double_double) return "double-double";
    if (test_case->double_precision_float_memory) return "f64-compute-f32-storage";
    if (test_case->half_precision) return "f16-storage-f32-compute";
    return "f32";
}

static pfUINT rader_upload_complex_size(const RaderUploadCase* test_case) {
    if (test_case->double_double) return 4 * sizeof(double);
    if (test_case->double_precision_float_memory) return 2 * sizeof(double);
    return 2 * sizeof(float);
}

static pfUINT rader_upload_warp_size(const RaderUploadCase* test_case) {
    if (test_case->profile_kind == 1) return 64;
    if (test_case->profile_kind == 4) return 1;
    return 32;
}

static pfUINT rader_upload_batch_count(const RaderUploadCase* test_case) {
    return test_case->batch_count != 0 ? test_case->batch_count : 1;
}

static pfUINT rader_upload_axis_id(const RaderUploadCase* test_case) {
    if (test_case->axis_id_override != 0) return test_case->axis_id_override;
    return test_case->strided_axis ? 1 : 0;
}

static pfUINT rader_upload_middle_axis_len(const RaderUploadCase* test_case) {
    return test_case->middle_axis_len != 0 ? test_case->middle_axis_len : 1;
}

static int rader_upload_register_boost(const RaderUploadCase* test_case) {
    if (test_case->double_double) return 1;
    if (test_case->profile_kind == 0) return 4;
    if (test_case->profile_kind == 1) {
        return test_case->shared_memory_bytes >= 65536 ? 2 : 4;
    }
    if ((test_case->profile_kind == 2) || (test_case->profile_kind == 3) || (test_case->profile_kind == 4)) {
        return test_case->shared_memory_bytes >= 65536 ? 1 : 2;
    }
    return 1;
}

static pfUINT rader_upload_swap_threshold(const RaderUploadCase* test_case) {
    if (test_case->double_precision_float_memory) {
        return test_case->profile_kind == 0 ? 4194305 : 524288;
    }
    return test_case->profile_kind == 0 ? 4194305 : 524288;
}

static pfUINT parameterized_rader_upload_swap_threshold(const RaderUploadCase* test_case) {
    // Execute the pinned backend initialization policy for the precision classes used by
    // the parameterized sweep. NVIDIA/Vulkan keeps its 4194305 constant. AMD/Intel
    // Vulkan/OpenCL/Level Zero use the reduced 262144 threshold for native DD, while
    // doublePrecisionFloatMemory stays in the normal 524288 class.
    if (test_case->profile_kind == 0) return 4194305;
    if (test_case->double_double) return 262144;
    return 524288;
}

static void configure_rader_upload_app(
    VkFFTApplication* app,
    const RaderUploadCase* test_case,
    pfUINT* temp_buffer_size
) {
    memset(app, 0, sizeof(*app));
    const pfUINT configured_axis_id = rader_upload_axis_id(test_case);
    if (configured_axis_id == 0) {
        app->configuration.FFTdim = 1;
        app->configuration.size[0] = test_case->sequence_len;
    } else if (configured_axis_id == 1) {
        app->configuration.FFTdim = 2;
        app->configuration.size[0] = test_case->fastest_axis_len != 0 ? test_case->fastest_axis_len : 8;
        app->configuration.size[1] = test_case->sequence_len;
    } else {
        app->configuration.FFTdim = configured_axis_id + 1;
        app->configuration.size[0] = test_case->fastest_axis_len != 0 ? test_case->fastest_axis_len : 8;
        app->configuration.size[1] = rader_upload_middle_axis_len(test_case);
        app->configuration.size[configured_axis_id] = test_case->sequence_len;
    }
    app->configuration.vendorID = rader_upload_vendor_id(test_case);
    app->configuration.sharedMemorySize = test_case->shared_memory_bytes;
    app->configuration.sharedMemorySizePow2 = test_case->shared_memory_pow2_bytes;
    app->configuration.maxThreadsNum = test_case->max_threads_num;
    app->configuration.maxComputeWorkGroupSize[0] = rader_upload_max_workgroup_x(test_case);
    app->configuration.maxComputeWorkGroupSize[1] = rader_upload_max_workgroup_y(test_case);
    app->configuration.maxComputeWorkGroupSize[2] = 64;
    app->configuration.coalescedMemory = rader_upload_coalesced_memory(test_case);
    app->configuration.aimThreads = 128;
    app->configuration.numSharedBanks = 32;
    app->configuration.warpSize = rader_upload_warp_size(test_case);
    app->configuration.registerBoost = rader_upload_register_boost(test_case);
    app->configuration.registerBoost4Step = 1;
    app->configuration.registerBoostNonPow2 = 0;
    app->configuration.swapTo3Stage4Step = rader_upload_swap_threshold(test_case);
    app->configuration.swapTo2Stage4Step = rader_upload_swap_threshold(test_case);
    app->configuration.reorderFourStep = 1;
    app->configuration.performBandwidthBoost = test_case->bandwidth_boost;
    if (test_case->half_precision) {
        app->configuration.halfPrecision = 1;
    }
    if (test_case->double_double) {
        app->configuration.quadDoubleDoublePrecision = 1;
        app->configuration.useLUT = 1;
    }
    if (test_case->double_precision_float_memory) {
        app->configuration.doublePrecisionFloatMemory = 1;
        app->configuration.useLUT = 1;
    }
    if (test_case->perform_r2c) {
        app->configuration.performR2C = 1;
    }
    if (test_case->perform_dct) {
        app->configuration.performDCT = (pfUINT)test_case->perform_dct;
    }
    if (test_case->perform_dst) {
        app->configuration.performDST = (pfUINT)test_case->perform_dst;
    }
    app->configuration.fixMinRaderPrimeMult = test_case->min_direct_prime;
    app->configuration.fixMaxRaderPrimeMult = test_case->max_direct_prime;
    app->configuration.fixMinRaderPrimeFFT = test_case->min_fft_prime;
    app->configuration.fixMaxRaderPrimeFFT = test_case->max_fft_prime;
    app->configuration.coordinateFeatures = 1;
    app->configuration.numberBatches = 1;
    app->configuration.numberKernels = 1;
    app->configuration.tempBufferSize = temp_buffer_size;
    app->actualNumBatches = rader_upload_batch_count(test_case);
    if (test_case->grouped_batch_override != 0) {
        app->configuration.groupedBatch[configured_axis_id] = test_case->grouped_batch_override;
    }
    if (test_case->axis1_grouped_batch_override != 0) {
        app->configuration.groupedBatch[1] = test_case->axis1_grouped_batch_override;
    }
    if (test_case->perform_zero_padding) {
        app->configuration.performZeropadding[configured_axis_id] = 1;
    }
}

static void configure_forced_axis_block_app(
    VkFFTApplication* app,
    const ForcedAxisBlockCase* test_case,
    pfUINT* temp_buffer_size
) {
    memset(app, 0, sizeof(*app));
    app->configuration.FFTdim = 1;
    app->configuration.size[0] = test_case->sequence_len;
    app->configuration.vendorID = test_case->vendor_id;
    app->configuration.sharedMemorySize = test_case->shared_memory_bytes;
    app->configuration.sharedMemorySizePow2 = test_case->shared_memory_pow2_bytes;
    app->configuration.maxThreadsNum = test_case->max_threads_num;
    app->configuration.maxComputeWorkGroupSize[0] = test_case->max_workgroup_x;
    app->configuration.maxComputeWorkGroupSize[1] = test_case->max_workgroup_x;
    app->configuration.maxComputeWorkGroupSize[2] = 64;
    app->configuration.coalescedMemory = test_case->coalesced_memory_bytes;
    app->configuration.aimThreads = 128;
    app->configuration.numSharedBanks = 32;
    app->configuration.warpSize = test_case->warp_size;
    app->configuration.registerBoost = test_case->register_boost;
    app->configuration.registerBoost4Step = 1;
    app->configuration.registerBoostNonPow2 = 0;
    app->configuration.swapTo3Stage4Step = test_case->swap_threshold;
    app->configuration.swapTo2Stage4Step = test_case->swap_threshold;
    app->configuration.reorderFourStep = 1;
    app->configuration.quadDoubleDoublePrecision = 1;
    app->configuration.useLUT = 1;
    app->configuration.fixMinRaderPrimeMult = test_case->min_direct_prime;
    app->configuration.fixMaxRaderPrimeMult = test_case->max_direct_prime;
    app->configuration.fixMinRaderPrimeFFT = test_case->min_fft_prime;
    app->configuration.fixMaxRaderPrimeFFT = test_case->max_fft_prime;
    app->configuration.coordinateFeatures = 1;
    app->configuration.numberBatches = 1;
    app->configuration.numberKernels = 1;
    app->configuration.tempBufferSize = temp_buffer_size;
    app->actualNumBatches = test_case->batch_count;
}

static void configure_rader_parent_app(
    VkFFTApplication* app,
    const RaderParentCase* test_case,
    pfUINT* temp_buffer_size
) {
    memset(app, 0, sizeof(*app));
    app->configuration.FFTdim = 1;
    app->configuration.size[0] = test_case->sequence_len;
    app->configuration.vendorID = test_case->vendor_id != 0 ? test_case->vendor_id : 0x10DE;
    app->configuration.sharedMemorySize = test_case->shared_memory_bytes;
    app->configuration.sharedMemorySizePow2 = test_case->shared_memory_pow2_bytes;
    app->configuration.maxThreadsNum = test_case->max_threads_num;
    app->configuration.maxComputeWorkGroupSize[0] = 1024;
    app->configuration.maxComputeWorkGroupSize[1] = 1024;
    app->configuration.maxComputeWorkGroupSize[2] = 64;
    app->configuration.coalescedMemory = test_case->coalesced_memory_bytes != 0
        ? test_case->coalesced_memory_bytes
        : 32;
    app->configuration.aimThreads = 128;
    app->configuration.numSharedBanks = 32;
    app->configuration.warpSize = test_case->warp_size != 0 ? test_case->warp_size : 32;
    app->configuration.registerBoost = test_case->register_boost != 0 ? test_case->register_boost : 4;
    app->configuration.registerBoost4Step = 1;
    app->configuration.registerBoostNonPow2 = 0;
    const pfUINT swap_to_three_stage = test_case->swap_to_three_stage != 0
        ? test_case->swap_to_three_stage
        : 4194305;
    app->configuration.swapTo3Stage4Step = swap_to_three_stage;
    app->configuration.swapTo2Stage4Step = swap_to_three_stage;
    app->configuration.reorderFourStep = 1;
    app->configuration.fixMinRaderPrimeMult = 17;
    app->configuration.fixMaxRaderPrimeMult = 89;
    app->configuration.fixMinRaderPrimeFFT = 17;
    app->configuration.fixMaxRaderPrimeFFT = 16384;
    app->configuration.coordinateFeatures = 1;
    app->configuration.numberBatches = 1;
    app->configuration.numberKernels = 1;
    app->configuration.tempBufferSize = temp_buffer_size;
    app->actualNumBatches = test_case->batch_count;
}

static void configure_mixed_rader_parent_app(
    VkFFTApplication* app,
    const MixedRaderParentCase* test_case,
    pfUINT* temp_buffer_size
) {
    memset(app, 0, sizeof(*app));
    app->configuration.FFTdim = 1;
    app->configuration.size[0] = test_case->sequence_len;
    app->configuration.vendorID = 0x10DE;
    app->configuration.sharedMemorySize = test_case->shared_memory_bytes;
    app->configuration.sharedMemorySizePow2 = test_case->shared_memory_pow2_bytes;
    app->configuration.maxThreadsNum = test_case->max_threads_num;
    app->configuration.maxComputeWorkGroupSize[0] = 1024;
    app->configuration.maxComputeWorkGroupSize[1] = 1024;
    app->configuration.maxComputeWorkGroupSize[2] = 64;
    app->configuration.coalescedMemory = 32;
    app->configuration.aimThreads = 128;
    app->configuration.numSharedBanks = 32;
    app->configuration.warpSize = 32;
    app->configuration.registerBoost = 4;
    app->configuration.registerBoost4Step = 1;
    app->configuration.registerBoostNonPow2 = 0;
    app->configuration.swapTo3Stage4Step = 4194305;
    app->configuration.swapTo2Stage4Step = 4194305;
    app->configuration.reorderFourStep = 1;
    if (test_case->double_double) {
        app->configuration.quadDoubleDoublePrecision = 1;
        app->configuration.useLUT = 1;
    }
    app->configuration.fixMinRaderPrimeMult = test_case->min_direct_prime;
    app->configuration.fixMaxRaderPrimeMult = test_case->max_direct_prime;
    app->configuration.fixMinRaderPrimeFFT = test_case->min_fft_prime;
    app->configuration.fixMaxRaderPrimeFFT = test_case->max_fft_prime;
    app->configuration.coordinateFeatures = 1;
    app->configuration.numberBatches = 1;
    app->configuration.numberKernels = 1;
    app->configuration.tempBufferSize = temp_buffer_size;
    app->actualNumBatches = test_case->batch_count;
}

static void configure_stockham_app(
    VkFFTApplication* app,
    const StockhamCase* test_case,
    pfUINT* temp_buffer_size
) {
    memset(app, 0, sizeof(*app));
    const pfUINT axis_id = test_case->axis_id;
    app->configuration.FFTdim = axis_id > 0 ? 2 : 1;
    const pfUINT scheduler_sequence_len = test_case->bluestein_logical_len != 0
        ? test_case->bluestein_logical_len
        : test_case->sequence_len;
    if (axis_id > 0) {
        app->configuration.size[0] = test_case->fastest_axis_len != 0 ? test_case->fastest_axis_len : 1;
        app->configuration.size[axis_id] = scheduler_sequence_len;
    } else {
        app->configuration.size[0] = scheduler_sequence_len;
    }
    app->configuration.vendorID = test_case->vendor_id;
    app->configuration.sharedMemorySize = test_case->shared_memory_bytes;
    app->configuration.sharedMemorySizePow2 = test_case->shared_memory_pow2_bytes;
    app->configuration.maxThreadsNum = 1024;
    app->configuration.maxComputeWorkGroupSize[0] = 1024;
    app->configuration.maxComputeWorkGroupSize[1] = 1024;
    app->configuration.maxComputeWorkGroupSize[2] = 64;
    app->configuration.coalescedMemory = test_case->coalesced_memory_bytes;
    app->configuration.aimThreads = 128;
    app->configuration.numSharedBanks = 32;
    app->configuration.warpSize = test_case->warp_size;
    app->configuration.registerBoost = test_case->register_boost;
    app->configuration.registerBoost4Step = 1;
    app->configuration.registerBoostNonPow2 = 0;
    app->configuration.swapTo3Stage4Step = test_case->swap_to_three_stage;
    app->configuration.swapTo2Stage4Step = test_case->swap_to_three_stage;
    app->configuration.reorderFourStep = 1;
    app->configuration.performBandwidthBoost = test_case->perform_bandwidth_boost;
    if (test_case->half_precision) {
        app->configuration.halfPrecision = 1;
    }
    if (test_case->double_double) {
        app->configuration.quadDoubleDoublePrecision = 1;
        app->configuration.useLUT = 1;
    } else if (test_case->double_precision_float_memory) {
        app->configuration.doublePrecisionFloatMemory = 1;
        app->configuration.useLUT = 1;
    } else if (test_case->double_precision) {
        app->configuration.doublePrecision = 1;
        app->configuration.useLUT = 1;
    }
    if (test_case->double_double) {
        app->configuration.fixMinRaderPrimeMult = 11;
        app->configuration.fixMaxRaderPrimeMult = 29;
        app->configuration.fixMinRaderPrimeFFT = test_case->vendor_id == 0x1002 ? 19 : 17;
        app->configuration.fixMaxRaderPrimeFFT = 16384;
    } else {
        app->configuration.fixMinRaderPrimeMult = 17;
        app->configuration.fixMaxRaderPrimeMult = 89;
        app->configuration.fixMinRaderPrimeFFT = 17;
        app->configuration.fixMaxRaderPrimeFFT = 16384;
    }
    if (test_case->min_direct_prime_override != 0) {
        app->configuration.fixMinRaderPrimeMult = test_case->min_direct_prime_override;
    }
    if (test_case->max_direct_prime_override != 0) {
        app->configuration.fixMaxRaderPrimeMult = test_case->max_direct_prime_override;
    }
    if (test_case->min_fft_prime_override != 0) {
        app->configuration.fixMinRaderPrimeFFT = test_case->min_fft_prime_override;
    }
    if (test_case->max_fft_prime_override != 0) {
        app->configuration.fixMaxRaderPrimeFFT = test_case->max_fft_prime_override;
    }
    app->configuration.coordinateFeatures = 1;
    app->configuration.numberBatches = 1;
    app->configuration.numberKernels = test_case->number_kernels != 0 ? test_case->number_kernels : 1;
    app->configuration.tempBufferSize = temp_buffer_size;
    app->actualNumBatches = test_case->batch_count;
    if (test_case->grouped_batch_override != 0) {
        app->configuration.groupedBatch[axis_id] = test_case->grouped_batch_override;
    }
    if (test_case->perform_zero_padding) {
        app->configuration.performZeropadding[axis_id] = 1;
    }
    if (test_case->perform_convolution) {
        // Mirror pinned initializeVkFFT defaults for performConvolution rather than
        // toggling the scheduler flag in isolation.
        app->configuration.performConvolution = 1;
        app->configuration.reorderFourStep = 0;
        app->configuration.registerBoost = 1;
        app->configuration.registerBoostNonPow2 = 0;
        app->configuration.registerBoost4Step = 1;
    }
    if (test_case->kernel_convolution) {
        // Pinned initializeVkFFT applies the same capacity/reorder knobs when this
        // application prepares a convolution kernel, but leaves performConvolution=0.
        app->configuration.kernelConvolution = 1;
        app->configuration.reorderFourStep = 0;
        app->configuration.registerBoost = 1;
        app->configuration.registerBoostNonPow2 = 0;
        app->configuration.registerBoost4Step = 1;
    }
}

static VkFFTRaderContainer* find_rader_container(
    VkFFTPlan* plan,
    int prime,
    pfUINT* upload_id,
    VkFFTAxis** axis_out
) {
    for (pfUINT upload = 0; upload < plan->numAxisUploads[0]; ++upload) {
        VkFFTAxis* axis = &plan->axes[0][upload];
        for (int index = 0; index < axis->specializationConstants.numRaderPrimes; ++index) {
            VkFFTRaderContainer* container = &axis->specializationConstants.raderContainer[index];
            if (container->prime == prime && container->type == 0) {
                *upload_id = upload;
                *axis_out = axis;
                return container;
            }
        }
    }
    return NULL;
}

static VkFFTRaderContainer* find_axis_rader_container_of_type(
    VkFFTAxis* axis,
    int prime,
    int type
) {
    for (int index = 0; index < axis->specializationConstants.numRaderPrimes; ++index) {
        VkFFTRaderContainer* container = &axis->specializationConstants.raderContainer[index];
        if (container->prime == prime && container->type == type) {
            return container;
        }
    }
    return NULL;
}

static int required_local_registers(const VkFFTRaderContainer* container) {
    int registers[33];
    int multipliers[33] = {0};
    memcpy(registers, container->registers_per_thread_per_radix, sizeof(registers));
    for (int stage = 0; stage < container->numStages; ++stage) {
        const int radix = container->stageRadix[stage];
        if (radix < 2 || radix >= 33) {
            return -1;
        }
        multipliers[radix]++;
    }
    int max_non_power_of_two_radix = 1;
    int required = 1;
    VkFFTResult result = VkFFTOptimizeRadixKernels(
        registers,
        multipliers,
        1,
        &max_non_power_of_two_radix,
        &required,
        NULL,
        0
    );
    if (result != VKFFT_SUCCESS) {
        return -1;
    }
    return required;
}

static int transpose_workgroup_threads(const VkFFTRaderContainer* container) {
    if (container->containerFFTNum < 8 || container->numStages <= 1) {
        return 0;
    }
    int workgroup_threads = 0;
    for (int stage = 0; stage < container->numStages; ++stage) {
        const int radix = container->stageRadix[stage];
        const int storage = container->registers_per_thread_per_radix[radix];
        if (storage <= 0) {
            return -1;
        }
        const int local_threads = (int)ceil(container->containerFFTDim / (double)storage);
        const int active_threads = container->containerFFTNum * local_threads;
        if (active_threads > workgroup_threads) {
            workgroup_threads = active_threads;
        }
    }
    return workgroup_threads;
}

static int effective_stage_register_extrema(
    const VkFFTRaderContainer* container,
    int* max_registers,
    int* min_registers
) {
    *max_registers = 0;
    *min_registers = 0;
    for (int stage = 0; stage < container->numStages; ++stage) {
        const int radix = container->stageRadix[stage];
        if (radix < 2 || radix >= 33) {
            return -1;
        }
        const int value = container->registers_per_thread_per_radix[radix];
        if (value <= 0) {
            return -1;
        }
        if (value > *max_registers) {
            *max_registers = value;
        }
        if (*min_registers == 0 || value < *min_registers) {
            *min_registers = value;
        }
    }
    return (*max_registers > 0 && *min_registers > 0) ? 0 : -1;
}

static int emit_rader_case(const RaderCase* test_case) {
    VkFFTApplication* app = (VkFFTApplication*)calloc(1, sizeof(*app));
    VkFFTPlan* plan = (VkFFTPlan*)calloc(1, sizeof(*plan));
    pfUINT temp_buffer_size = 0;
    if (app == NULL || plan == NULL) {
        fprintf(stderr, "failed to allocate upstream scheduler state\n");
        free(plan);
        free(app);
        return 1;
    }
    configure_app(app, test_case, &temp_buffer_size);
    VkFFTResult result = VkFFTScheduler(app, plan, 0);
    if (result != VKFFT_SUCCESS) {
        fprintf(
            stderr,
            "VkFFTScheduler failed for %s with result %d\n",
            test_case->case_name,
            (int)result
        );
        free(plan);
        free(app);
        return 1;
    }

    pfUINT upload_id = 0;
    VkFFTAxis* axis = NULL;
    VkFFTRaderContainer* container =
        find_rader_container(plan, test_case->prime, &upload_id, &axis);
    if (container == NULL || axis == NULL) {
        fprintf(stderr, "upstream scheduler did not produce p%d FFT-Rader for %s\n", test_case->prime, test_case->case_name);
        free(plan);
        free(app);
        return 1;
    }
    const pfUINT expected_outer = (pfUINT)test_case->prime * (pfUINT)container->containerFFTNum;
    if (plan->axisSplit[0][upload_id] != expected_outer) {
        fprintf(
            stderr,
            "upstream %s Rader container belongs to upload length %llu, expected %llu\n",
            test_case->case_name,
            (unsigned long long)plan->axisSplit[0][upload_id],
            (unsigned long long)expected_outer
        );
        free(plan);
        free(app);
        return 1;
    }

    const int required_local = required_local_registers(container);
    const int transpose_threads = transpose_workgroup_threads(container);
    int effective_registers = 0;
    int effective_min_registers = 0;
    const int extrema_result = effective_stage_register_extrema(
        container,
        &effective_registers,
        &effective_min_registers
    );
    if (required_local <= 0 || transpose_threads < 0 || extrema_result != 0) {
        fprintf(stderr, "failed to derive upstream Rader execution metadata for %s\n", test_case->case_name);
        free(plan);
        free(app);
        return 1;
    }
    const int has_transpose = container->containerFFTNum >= 8 && container->numStages > 1;
    const int min_threads = axis->specializationConstants.minRaderFFTThreadNum;
    const int execution_containers = container->containerFFTNum;
    const pfUINT rader_batch_count = test_case->outer_batches * (pfUINT)container->containerFFTNum;
    const pfUINT workgroup_count = rader_batch_count / (pfUINT)execution_containers;

    printf(
        "{\"schema_version\":%d,\"upstream_commit\":\"%s\",\"case\":\"%s\",\"kind\":\"rader\","
        "\"backend\":\"vulkan\",\"vendor\":\"%s\",\"precision\":\"f32\","
        "\"device\":{\"shared_memory_bytes\":%llu,\"shared_memory_pow2_bytes\":%llu,"
        "\"max_threads_per_block\":1024,\"max_workgroup_size\":[1024,1024,64],"
        "\"coalesced_memory_bytes\":%llu,\"shared_banks\":32,\"supports_f64\":true},"
        "\"payload\":{\"prime\":%d,\"convolution_len\":%d,\"outer_fft_len\":%llu,"
        "\"container_fft_num\":%d,\"min_rader_fft_thread_num\":%d,"
        "\"execution_container_fft_num\":%d,\"execution_workgroup_count\":%llu,"
        "\"execution_threads_per_workgroup\":%d,\"rader_transpose\":%s,"
        "\"rader_transpose_workgroup_threads\":",
        SNAPSHOT_SCHEMA_VERSION,
        UPSTREAM_COMMIT,
        test_case->case_name,
        test_case->vendor_name,
        (unsigned long long)test_case->shared_memory_bytes,
        (unsigned long long)test_case->shared_memory_pow2_bytes,
        (unsigned long long)test_case->coalesced_memory_bytes,
        container->prime,
        container->containerFFTDim,
        (unsigned long long)expected_outer,
        container->containerFFTNum,
        min_threads,
        execution_containers,
        (unsigned long long)workgroup_count,
        min_threads,
        has_transpose ? "true" : "false"
    );
    if (has_transpose) {
        printf("%d", transpose_threads);
    } else {
        fputs("null", stdout);
    }
    fputs(",\"stage_radices\":[", stdout);
    for (int stage = 0; stage < container->numStages; ++stage) {
        if (stage != 0) {
            fputc(',', stdout);
        }
        printf("%d", container->stageRadix[stage]);
    }
    printf(
        "],\"registers_per_thread\":%d,\"min_registers_per_thread\":%d,"
        "\"required_local_registers\":%d}}\n",
        effective_registers,
        effective_min_registers,
        required_local
    );

    free(plan);
    free(app);
    return 0;
}

static int emit_rader_upload_case_impl(const RaderUploadCase* test_case, int infer_force_from_plan) {
    VkFFTApplication* app = (VkFFTApplication*)calloc(1, sizeof(*app));
    VkFFTPlan* plan = (VkFFTPlan*)calloc(1, sizeof(*plan));
    pfUINT temp_buffer_size = 0;
    if (app == NULL || plan == NULL) {
        fprintf(stderr, "failed to allocate upstream Rader upload state\n");
        free(plan);
        free(app);
        return 1;
    }
    configure_rader_upload_app(app, test_case, &temp_buffer_size);
    if (infer_force_from_plan) {
        const pfUINT swap_threshold = parameterized_rader_upload_swap_threshold(test_case);
        app->configuration.swapTo3Stage4Step = swap_threshold;
        app->configuration.swapTo2Stage4Step = swap_threshold;
    }
    const pfUINT axis_id = rader_upload_axis_id(test_case);
    VkFFTResult result = VkFFTScheduler(app, plan, axis_id);
    if (result != VKFFT_SUCCESS) {
        if (infer_force_from_plan && (result == VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH)) {
            free(plan);
            free(app);
            return 3;
        }
        fprintf(stderr, "VkFFTScheduler failed for %s with result %d\n", test_case->case_name, (int)result);
        free(plan);
        free(app);
        return 1;
    }
    const pfUINT scheduled_sequence_len = plan->actualFFTSizePerAxis[axis_id][axis_id];
    // The parameterized Rader-upload sweep intentionally covers axes that remain at
    // their requested C2C length. A changed scheduler length means upstream promoted
    // the whole axis to Bluestein; that belongs to the Bluestein/classification family,
    // not to this logical Rader-upload differential.
    if (infer_force_from_plan && (scheduled_sequence_len != test_case->sequence_len)) {
        free(plan);
        free(app);
        return 3;
    }
    if (test_case->fft_rader_prime > 0) {
        int found_fft_rader = 0;
        for (pfUINT upload = 0; upload < plan->numAxisUploads[axis_id]; ++upload) {
            VkFFTAxis* axis = &plan->axes[axis_id][upload];
            if (find_axis_rader_container_of_type(axis, test_case->fft_rader_prime, 0) != NULL) {
                found_fft_rader = 1;
                break;
            }
        }
        if (!found_fft_rader) {
            fprintf(
                stderr,
                "upstream %s did not classify p%d as FFT/type-0 Rader\n",
                test_case->case_name,
                test_case->fft_rader_prime
            );
            free(plan);
            free(app);
            return 1;
        }
    }
    int force_rader_two_upload = 0;
    if (infer_force_from_plan) {
        pfUINT remaining = scheduled_sequence_len;
        for (pfUINT prime = 2; prime <= remaining / prime; ++prime) {
            if ((remaining % prime) != 0) continue;
            int found_fft_rader = 0;
            for (pfUINT upload = 0; upload < plan->numAxisUploads[axis_id]; ++upload) {
                if (find_axis_rader_container_of_type(&plan->axes[axis_id][upload], (int)prime, 0) != NULL) {
                    found_fft_rader = 1;
                    break;
                }
            }
            if (found_fft_rader
                && ((scheduled_sequence_len / prime > 512)
                    || (scheduled_sequence_len / prime > test_case->max_threads_num))) {
                force_rader_two_upload = 1;
            }
            while ((remaining % prime) == 0) remaining /= prime;
        }
        if (remaining > 1) {
            int found_fft_rader = 0;
            for (pfUINT upload = 0; upload < plan->numAxisUploads[axis_id]; ++upload) {
                if (find_axis_rader_container_of_type(&plan->axes[axis_id][upload], (int)remaining, 0) != NULL) {
                    found_fft_rader = 1;
                    break;
                }
            }
            if (found_fft_rader
                && ((scheduled_sequence_len / remaining > 512)
                    || (scheduled_sequence_len / remaining > test_case->max_threads_num))) {
                force_rader_two_upload = 1;
            }
        }
    } else {
        force_rader_two_upload = test_case->fft_rader_prime > 0
            && ((scheduled_sequence_len / (pfUINT)test_case->fft_rader_prime > 512)
                || (scheduled_sequence_len / (pfUINT)test_case->fft_rader_prime > test_case->max_threads_num));
    }
    printf(
        "{\"schema_version\":%d,\"upstream_commit\":\"%s\",\"case\":\"%s\",\"kind\":\"rader-upload\","
        "\"backend\":\"%s\",\"vendor\":\"%s\",\"precision\":\"%s\","
        "\"device\":{\"shared_memory_bytes\":%llu,\"shared_memory_pow2_bytes\":%llu,"
        "\"max_threads_per_block\":%llu,\"max_workgroup_size\":[%llu,%llu,64],"
        "\"coalesced_memory_bytes\":%llu,\"shared_banks\":32,\"supports_f64\":true},"
        "\"payload\":{\"sequence_len\":%llu,\"force_rader_two_upload\":%s,\"upload_count\":%llu,\"axis_split\":[",
        SNAPSHOT_SCHEMA_VERSION,
        UPSTREAM_COMMIT,
        test_case->case_name,
        rader_upload_backend_name(test_case),
        rader_upload_vendor_name(test_case),
        rader_upload_precision_name(test_case),
        (unsigned long long)test_case->shared_memory_bytes,
        (unsigned long long)test_case->shared_memory_pow2_bytes,
        (unsigned long long)test_case->max_threads_num,
        (unsigned long long)rader_upload_max_workgroup_x(test_case),
        (unsigned long long)rader_upload_max_workgroup_y(test_case),
        (unsigned long long)rader_upload_coalesced_memory(test_case),
        (unsigned long long)scheduled_sequence_len,
        force_rader_two_upload ? "true" : "false",
        (unsigned long long)plan->numAxisUploads[axis_id]
    );
    for (pfUINT upload = 0; upload < plan->numAxisUploads[axis_id]; ++upload) {
        if (upload != 0) {
            fputc(',', stdout);
        }
        printf("%llu", (unsigned long long)plan->axisSplit[axis_id][upload]);
    }
    fputs("]}}\n", stdout);
    free(plan);
    free(app);
    return 0;
}

static int emit_rader_upload_case(const RaderUploadCase* test_case) {
    return emit_rader_upload_case_impl(test_case, 0);
}

static int emit_parameterized_rader_upload_case(const RaderUploadCase* test_case) {
    return emit_rader_upload_case_impl(test_case, 1);
}

static int emit_parameterized_axis_classification_case(const RaderUploadCase* test_case) {
    VkFFTApplication* app = (VkFFTApplication*)calloc(1, sizeof(*app));
    VkFFTPlan* plan = (VkFFTPlan*)calloc(1, sizeof(*plan));
    pfUINT temp_buffer_size = 0;
    if ((app == NULL) || (plan == NULL)) {
        fprintf(stderr, "failed to allocate upstream parameterized classification state\n");
        free(plan);
        free(app);
        return 1;
    }
    configure_rader_upload_app(app, test_case, &temp_buffer_size);
    const pfUINT swap_threshold = parameterized_rader_upload_swap_threshold(test_case);
    app->configuration.swapTo3Stage4Step = swap_threshold;
    app->configuration.swapTo2Stage4Step = swap_threshold;
    const pfUINT axis_id = rader_upload_axis_id(test_case);
    const VkFFTResult result = VkFFTScheduler(app, plan, axis_id);
    if (result != VKFFT_SUCCESS) {
        if (result == VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH) {
            free(plan);
            free(app);
            return 3;
        }
        fprintf(stderr, "VkFFTScheduler failed for %s with result %d\n", test_case->case_name, (int)result);
        free(plan);
        free(app);
        return 1;
    }

    const pfUINT scheduled_sequence_len = plan->actualFFTSizePerAxis[axis_id][axis_id];
    if (scheduled_sequence_len != test_case->sequence_len) {
        printf(
            "%s\tbluestein\t%llu\t-\t-\n",
            test_case->case_name,
            (unsigned long long)scheduled_sequence_len
        );
        free(plan);
        free(app);
        return 0;
    }

    pfUINT direct_primes[32] = {0};
    pfUINT fft_primes[32] = {0};
    size_t direct_count = 0;
    size_t fft_count = 0;
    pfUINT remaining = scheduled_sequence_len;
    for (pfUINT prime = 2; prime <= remaining / prime; ++prime) {
        if ((remaining % prime) != 0) continue;
        int found_direct = 0;
        int found_fft = 0;
        for (pfUINT upload = 0; upload < plan->numAxisUploads[axis_id]; ++upload) {
            VkFFTAxis* axis = &plan->axes[axis_id][upload];
            if (find_axis_rader_container_of_type(axis, (int)prime, 1) != NULL) found_direct = 1;
            if (find_axis_rader_container_of_type(axis, (int)prime, 0) != NULL) found_fft = 1;
        }
        if (found_direct && (direct_count < 32)) direct_primes[direct_count++] = prime;
        if (found_fft && (fft_count < 32)) fft_primes[fft_count++] = prime;
        while ((remaining % prime) == 0) remaining /= prime;
    }
    if (remaining > 1) {
        int found_direct = 0;
        int found_fft = 0;
        for (pfUINT upload = 0; upload < plan->numAxisUploads[axis_id]; ++upload) {
            VkFFTAxis* axis = &plan->axes[axis_id][upload];
            if (find_axis_rader_container_of_type(axis, (int)remaining, 1) != NULL) found_direct = 1;
            if (find_axis_rader_container_of_type(axis, (int)remaining, 0) != NULL) found_fft = 1;
        }
        if (found_direct && (direct_count < 32)) direct_primes[direct_count++] = remaining;
        if (found_fft && (fft_count < 32)) fft_primes[fft_count++] = remaining;
    }

    printf(
        "%s\t%s\t%llu\t",
        test_case->case_name,
        ((direct_count + fft_count) > 0) ? "rader" : "stockham",
        (unsigned long long)scheduled_sequence_len
    );
    if (direct_count == 0) {
        fputc('-', stdout);
    } else {
        for (size_t index = 0; index < direct_count; ++index) {
            if (index != 0) fputc(',', stdout);
            printf("%llu", (unsigned long long)direct_primes[index]);
        }
    }
    fputc('\t', stdout);
    if (fft_count == 0) {
        fputc('-', stdout);
    } else {
        for (size_t index = 0; index < fft_count; ++index) {
            if (index != 0) fputc(',', stdout);
            printf("%llu", (unsigned long long)fft_primes[index]);
        }
    }
    fputc('\n', stdout);
    free(plan);
    free(app);
    return 0;
}

static int emit_parameterized_axis_block_case(const RaderUploadCase* test_case) {
    VkFFTApplication* app = (VkFFTApplication*)calloc(1, sizeof(*app));
    VkFFTPlan* plan = (VkFFTPlan*)calloc(1, sizeof(*plan));
    pfUINT temp_buffer_size = 0;
    if ((app == NULL) || (plan == NULL)) {
        fprintf(stderr, "failed to allocate upstream parameterized AxisBlock state\n");
        free(plan);
        free(app);
        return 1;
    }
    configure_rader_upload_app(app, test_case, &temp_buffer_size);
    const pfUINT swap_threshold = parameterized_rader_upload_swap_threshold(test_case);
    app->configuration.swapTo3Stage4Step = swap_threshold;
    app->configuration.swapTo2Stage4Step = swap_threshold;
    const pfUINT axis_id = rader_upload_axis_id(test_case);
    VkFFTResult result = VkFFTScheduler(app, plan, axis_id);
    if (result != VKFFT_SUCCESS) {
        if (result == VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH) {
            free(plan);
            free(app);
            return 3;
        }
        fprintf(stderr, "VkFFTScheduler failed for %s with result %d\n", test_case->case_name, (int)result);
        free(plan);
        free(app);
        return 1;
    }
    const pfUINT scheduled_sequence_len = plan->actualFFTSizePerAxis[axis_id][axis_id];
    if ((scheduled_sequence_len != test_case->sequence_len) || (plan->numAxisUploads[axis_id] < 2)) {
        free(plan);
        free(app);
        return 3;
    }

    int has_rader = 0;
    pfUINT remaining = scheduled_sequence_len;
    for (pfUINT prime = 2; prime <= remaining / prime; ++prime) {
        if ((remaining % prime) != 0) continue;
        for (pfUINT upload = 0; upload < plan->numAxisUploads[axis_id]; ++upload) {
            VkFFTAxis* axis = &plan->axes[axis_id][upload];
            if ((find_axis_rader_container_of_type(axis, (int)prime, 0) != NULL)
                || (find_axis_rader_container_of_type(axis, (int)prime, 1) != NULL)) {
                has_rader = 1;
            }
        }
        while ((remaining % prime) == 0) remaining /= prime;
    }
    if (remaining > 1) {
        for (pfUINT upload = 0; upload < plan->numAxisUploads[axis_id]; ++upload) {
            VkFFTAxis* axis = &plan->axes[axis_id][upload];
            if ((find_axis_rader_container_of_type(axis, (int)remaining, 0) != NULL)
                || (find_axis_rader_container_of_type(axis, (int)remaining, 1) != NULL)) {
                has_rader = 1;
            }
        }
    }
    if (!has_rader) {
        free(plan);
        free(app);
        return 3;
    }

    const pfUINT complex_size = rader_upload_complex_size(test_case);
    for (pfUINT upload = 0; upload < plan->numAxisUploads[axis_id]; ++upload) {
        VkFFTAxis* axis = &plan->axes[axis_id][upload];
        axis->specializationConstants.complexSize = complex_size;
        axis->specializationConstants.numAxisUploads = (int)plan->numAxisUploads[axis_id];
        axis->specializationConstants.reorderFourStep = 1;
        axis->specializationConstants.stageStartSize.type = 31;
        axis->specializationConstants.stageStartSize.data.i = 1;
        for (pfUINT prior_upload = 0; prior_upload < upload; ++prior_upload) {
            axis->specializationConstants.stageStartSize.data.i *= plan->axisSplit[axis_id][prior_upload];
        }
        pfUINT allowed_shared = test_case->shared_memory_bytes;
        pfUINT allowed_shared_pow2 = test_case->shared_memory_pow2_bytes;
        if (axis->specializationConstants.useRaderMult > 0) {
            const pfUINT reserve =
                (pfUINT)(axis->specializationConstants.useRaderMult - 1) * complex_size;
            if ((reserve >= allowed_shared) || (reserve >= allowed_shared_pow2)) {
                free(plan);
                free(app);
                return 3;
            }
            allowed_shared -= reserve;
            allowed_shared_pow2 -= reserve;
        }
        result = VkFFTSplitAxisBlock(
            app,
            plan,
            axis,
            axis_id,
            upload,
            allowed_shared,
            allowed_shared_pow2
        );
        if (result != VKFFT_SUCCESS) {
            fprintf(
                stderr,
                "VkFFTSplitAxisBlock failed for %s upload %llu with result %d\n",
                test_case->case_name,
                (unsigned long long)upload,
                (int)result
            );
            free(plan);
            free(app);
            return 1;
        }
        const pfUINT fft_dim = (pfUINT)axis->specializationConstants.fftDim.data.i;
        if ((fft_dim == 0) || ((scheduled_sequence_len % fft_dim) != 0)) {
            fprintf(stderr, "invalid parameterized AxisBlock geometry for %s\n", test_case->case_name);
            free(plan);
            free(app);
            return 1;
        }
        const pfUINT higher_axis_extent = test_case->fastest_axis_len != 0
            ? test_case->fastest_axis_len
            : 1;
        pfUINT independent_lines = rader_upload_batch_count(test_case);
        if (axis_id > 0) independent_lines *= higher_axis_extent;
        if (axis_id > 1) independent_lines *= rader_upload_middle_axis_len(test_case);
        const pfUINT transform_count = (scheduled_sequence_len / fft_dim) * independent_lines;
        const int axis_swapped = axis->specializationConstants.axisSwapped != 0;
        const int transforms_on_x = (axis_id > 0) || (upload > 0) || axis_swapped;
        const pfUINT threads_per_transform =
            transforms_on_x ? axis->axisBlock[1] : axis->axisBlock[0];
        printf(
            "{\"schema_version\":%d,\"upstream_commit\":\"%s\",\"case\":\"%s-u%llu\",\"kind\":\"axis-block\","
            "\"backend\":\"%s\",\"vendor\":\"%s\",\"precision\":\"%s\","
            "\"device\":{\"shared_memory_bytes\":%llu,\"shared_memory_pow2_bytes\":%llu,"
            "\"max_threads_per_block\":%llu,\"max_workgroup_size\":[%llu,%llu,64],"
            "\"coalesced_memory_bytes\":%llu,\"shared_banks\":32,\"supports_f64\":true},"
            "\"payload\":{\"sequence_len\":%llu,\"batch_count\":%llu,\"axis_upload_id\":%llu,"
            "\"transform_count\":%llu,\"threads_per_transform\":%llu,\"grouped_batch\":%llu,"
            "\"transforms_on_x\":%s,\"axis_swapped\":%s,\"local_size_x\":%llu,\"local_size_y\":%llu}}\n",
            SNAPSHOT_SCHEMA_VERSION,
            UPSTREAM_COMMIT,
            test_case->case_name,
            (unsigned long long)upload,
            rader_upload_backend_name(test_case),
            rader_upload_vendor_name(test_case),
            rader_upload_precision_name(test_case),
            (unsigned long long)test_case->shared_memory_bytes,
            (unsigned long long)test_case->shared_memory_pow2_bytes,
            (unsigned long long)test_case->max_threads_num,
            (unsigned long long)rader_upload_max_workgroup_x(test_case),
            (unsigned long long)rader_upload_max_workgroup_y(test_case),
            (unsigned long long)rader_upload_coalesced_memory(test_case),
            (unsigned long long)fft_dim,
            (unsigned long long)transform_count,
            (unsigned long long)upload,
            (unsigned long long)transform_count,
            (unsigned long long)threads_per_transform,
            (unsigned long long)axis->groupedBatch,
            transforms_on_x ? "true" : "false",
            axis_swapped ? "true" : "false",
            (unsigned long long)axis->axisBlock[0],
            (unsigned long long)axis->axisBlock[1]
        );
    }
    free(plan);
    free(app);
    return 0;
}

static int emit_rader_upload_axis_block_reference(
    const RaderUploadAxisBlockReference* reference
) {
    const size_t case_count = sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]);
    if (reference->rader_upload_case_index >= case_count) {
        fprintf(stderr, "invalid Rader-upload axis-block case index for %s\n", reference->case_name);
        return 1;
    }
    const RaderUploadCase* test_case = &RADER_UPLOAD_CASES[reference->rader_upload_case_index];
    VkFFTApplication* app = (VkFFTApplication*)calloc(1, sizeof(*app));
    VkFFTPlan* plan = (VkFFTPlan*)calloc(1, sizeof(*plan));
    pfUINT temp_buffer_size = 0;
    if (app == NULL || plan == NULL) {
        fprintf(stderr, "failed to allocate upstream Rader-upload axis-block state\n");
        free(plan);
        free(app);
        return 1;
    }
    configure_rader_upload_app(app, test_case, &temp_buffer_size);
    const pfUINT axis_id = rader_upload_axis_id(test_case);
    VkFFTResult result = VkFFTScheduler(app, plan, axis_id);
    if (result != VKFFT_SUCCESS) {
        fprintf(stderr, "VkFFTScheduler failed for %s with result %d\n", reference->case_name, (int)result);
        free(plan);
        free(app);
        return 1;
    }
    const pfUINT upload = reference->upload_id;
    if (upload >= plan->numAxisUploads[axis_id]) {
        fprintf(stderr, "upstream %s is missing requested upload %llu\n", reference->case_name, (unsigned long long)upload);
        free(plan);
        free(app);
        return 1;
    }
    VkFFTAxis* axis = &plan->axes[axis_id][upload];
    const pfUINT complex_size = rader_upload_complex_size(test_case);
    axis->specializationConstants.complexSize = complex_size;
    axis->specializationConstants.numAxisUploads = (int)plan->numAxisUploads[axis_id];
    axis->specializationConstants.reorderFourStep = 1;
    axis->specializationConstants.stageStartSize.type = 31;
    axis->specializationConstants.stageStartSize.data.i = 1;
    for (pfUINT prior_upload = 0; prior_upload < upload; ++prior_upload) {
        axis->specializationConstants.stageStartSize.data.i *= plan->axisSplit[axis_id][prior_upload];
    }
    pfUINT allowed_shared = test_case->shared_memory_bytes;
    pfUINT allowed_shared_pow2 = test_case->shared_memory_pow2_bytes;
    if (axis->specializationConstants.useRaderMult > 0) {
        const pfUINT reserve =
            (pfUINT)(axis->specializationConstants.useRaderMult - 1) * complex_size;
        allowed_shared -= reserve;
        allowed_shared_pow2 -= reserve;
    }
    result = VkFFTSplitAxisBlock(
        app,
        plan,
        axis,
        axis_id,
        upload,
        allowed_shared,
        allowed_shared_pow2
    );
    if (result != VKFFT_SUCCESS) {
        fprintf(stderr, "VkFFTSplitAxisBlock failed for %s with result %d\n", reference->case_name, (int)result);
        free(plan);
        free(app);
        return 1;
    }
    const pfUINT fft_dim = (pfUINT)axis->specializationConstants.fftDim.data.i;
    const pfUINT scheduled_sequence_len = plan->actualFFTSizePerAxis[axis_id][axis_id];
    if (fft_dim == 0 || scheduled_sequence_len % fft_dim != 0) {
        fprintf(stderr, "invalid Rader-upload component geometry for %s\n", reference->case_name);
        free(plan);
        free(app);
        return 1;
    }
    const pfUINT higher_axis_extent = test_case->perform_r2c
        ? plan->actualFFTSizePerAxis[axis_id][0]
        : (test_case->fastest_axis_len != 0 ? test_case->fastest_axis_len : 1);
    const pfUINT independent_lines = axis_id > 0
        ? higher_axis_extent * rader_upload_batch_count(test_case)
        : rader_upload_batch_count(test_case);
    const pfUINT transform_count =
        (scheduled_sequence_len / fft_dim) * independent_lines;
    const int axis_swapped = axis->specializationConstants.axisSwapped != 0;
    const int transforms_on_x = axis_id > 0 || upload > 0 || axis_swapped;
    const pfUINT threads_per_transform =
        transforms_on_x ? axis->axisBlock[1] : axis->axisBlock[0];
    printf(
        "{\"schema_version\":%d,\"upstream_commit\":\"%s\",\"case\":\"%s\",\"kind\":\"axis-block\","
        "\"backend\":\"%s\",\"vendor\":\"%s\",\"precision\":\"%s\","
        "\"device\":{\"shared_memory_bytes\":%llu,\"shared_memory_pow2_bytes\":%llu,"
        "\"max_threads_per_block\":%llu,\"max_workgroup_size\":[%llu,%llu,64],"
        "\"coalesced_memory_bytes\":%llu,\"shared_banks\":32,\"supports_f64\":true},"
        "\"payload\":{\"sequence_len\":%llu,\"batch_count\":%llu,\"axis_upload_id\":%llu,"
        "\"transform_count\":%llu,\"threads_per_transform\":%llu,\"grouped_batch\":%llu,"
        "\"transforms_on_x\":%s,\"axis_swapped\":%s,\"local_size_x\":%llu,\"local_size_y\":%llu}}\n",
        SNAPSHOT_SCHEMA_VERSION,
        UPSTREAM_COMMIT,
        reference->case_name,
        rader_upload_backend_name(test_case),
        rader_upload_vendor_name(test_case),
        rader_upload_precision_name(test_case),
        (unsigned long long)test_case->shared_memory_bytes,
        (unsigned long long)test_case->shared_memory_pow2_bytes,
        (unsigned long long)test_case->max_threads_num,
        (unsigned long long)rader_upload_max_workgroup_x(test_case),
        (unsigned long long)rader_upload_max_workgroup_y(test_case),
        (unsigned long long)rader_upload_coalesced_memory(test_case),
        (unsigned long long)fft_dim,
        (unsigned long long)transform_count,
        (unsigned long long)upload,
        (unsigned long long)transform_count,
        (unsigned long long)threads_per_transform,
        (unsigned long long)axis->groupedBatch,
        transforms_on_x ? "true" : "false",
        axis_swapped ? "true" : "false",
        (unsigned long long)axis->axisBlock[0],
        (unsigned long long)axis->axisBlock[1]
    );
    free(plan);
    free(app);
    return 0;
}

static int emit_forced_axis_block_case(const ForcedAxisBlockCase* test_case) {
    VkFFTApplication* app = (VkFFTApplication*)calloc(1, sizeof(*app));
    VkFFTPlan* plan = (VkFFTPlan*)calloc(1, sizeof(*plan));
    pfUINT temp_buffer_size = 0;
    if (app == NULL || plan == NULL) {
        fprintf(stderr, "failed to allocate upstream forced-axis state\n");
        free(plan);
        free(app);
        return 1;
    }
    configure_forced_axis_block_app(app, test_case, &temp_buffer_size);
    VkFFTResult result = VkFFTScheduler(app, plan, 0);
    if (result != VKFFT_SUCCESS) {
        fprintf(stderr, "VkFFTScheduler failed for %s with result %d\n", test_case->case_name, (int)result);
        free(plan);
        free(app);
        return 1;
    }
    if (plan->numAxisUploads[0] < 2) {
        fprintf(stderr, "upstream %s expected forced multi-upload, got %llu upload(s)\n", test_case->case_name, (unsigned long long)plan->numAxisUploads[0]);
        free(plan);
        free(app);
        return 1;
    }

    const pfUINT complex_size = 4 * sizeof(double);
    printf(
        "{\"schema_version\":%d,\"upstream_commit\":\"%s\",\"case\":\"%s\",\"kind\":\"forced-axis-block-probe\","
        "\"backend\":\"%s\",\"vendor\":\"%s\",\"precision\":\"double-double\","
        "\"payload\":{\"sequence_len\":%llu,\"upload_count\":%llu,\"axis_split\":[",
        SNAPSHOT_SCHEMA_VERSION,
        UPSTREAM_COMMIT,
        test_case->case_name,
        test_case->backend_name,
        test_case->vendor_name,
        (unsigned long long)test_case->sequence_len,
        (unsigned long long)plan->numAxisUploads[0]
    );
    for (pfUINT upload = 0; upload < plan->numAxisUploads[0]; ++upload) {
        if (upload != 0) {
            fputc(',', stdout);
        }
        printf("%llu", (unsigned long long)plan->axisSplit[0][upload]);
    }
    fputs("],\"uploads\":[", stdout);
    for (pfUINT upload = 0; upload < plan->numAxisUploads[0]; ++upload) {
        VkFFTAxis* axis = &plan->axes[0][upload];
        axis->specializationConstants.complexSize = complex_size;
        axis->specializationConstants.numAxisUploads = (int)plan->numAxisUploads[0];
        axis->specializationConstants.reorderFourStep = 1;
        axis->specializationConstants.stageStartSize.type = 31;
        axis->specializationConstants.stageStartSize.data.i = 1;
        for (pfUINT prior_upload = 0; prior_upload < upload; ++prior_upload) {
            axis->specializationConstants.stageStartSize.data.i *= plan->axisSplit[0][prior_upload];
        }
        pfUINT allowed_shared = test_case->shared_memory_bytes;
        pfUINT allowed_shared_pow2 = test_case->shared_memory_pow2_bytes;
        if (axis->specializationConstants.useRaderMult > 0) {
            const pfUINT reserve =
                (pfUINT)(axis->specializationConstants.useRaderMult - 1) * complex_size;
            allowed_shared -= reserve;
            allowed_shared_pow2 -= reserve;
        }
        result = VkFFTSplitAxisBlock(
            app,
            plan,
            axis,
            0,
            upload,
            allowed_shared,
            allowed_shared_pow2
        );
        if (result != VKFFT_SUCCESS) {
            fprintf(stderr, "VkFFTSplitAxisBlock failed for %s upload %llu with result %d\n", test_case->case_name, (unsigned long long)upload, (int)result);
            free(plan);
            free(app);
            return 1;
        }
        if (upload != 0) {
            fputc(',', stdout);
        }
        printf(
            "{\"fft_dim\":%lld,\"stage_start_size\":%lld,\"min_registers\":%d,\"register_boost\":%d,"
            "\"rader_min_registers\":%d,\"min_rader_fft_threads\":%d,\"use_rader_mult\":%d,"
            "\"use_rader_fft\":%d,\"num_rader_primes\":%d,\"grouped_batch\":%llu,"
            "\"axis_swapped\":%d,\"axis_block\":[%.0f,%.0f],\"rader_containers\":[",
            (long long)axis->specializationConstants.fftDim.data.i,
            (long long)axis->specializationConstants.stageStartSize.data.i,
            axis->specializationConstants.min_registers_per_thread,
            axis->specializationConstants.registerBoost,
            axis->specializationConstants.rader_min_registers,
            axis->specializationConstants.minRaderFFTThreadNum,
            axis->specializationConstants.useRaderMult,
            axis->specializationConstants.useRaderFFT,
            axis->specializationConstants.numRaderPrimes,
            (unsigned long long)axis->groupedBatch,
            axis->specializationConstants.axisSwapped,
            (double)axis->axisBlock[0],
            (double)axis->axisBlock[1]
        );
        for (int rader_index = 0; rader_index < axis->specializationConstants.numRaderPrimes; ++rader_index) {
            if (rader_index != 0) {
                fputc(',', stdout);
            }
            VkFFTRaderContainer* container = &axis->specializationConstants.raderContainer[rader_index];
            printf("[%d,%d]", container->prime, container->type);
        }
        fputs("]}", stdout);
    }
    fputs("]}}\n", stdout);
    free(plan);
    free(app);
    return 0;
}

static int emit_forced_axis_block_upload_reference(
    const ForcedAxisBlockUploadReference* reference
) {
    const size_t forced_count = sizeof(FORCED_AXIS_BLOCK_CASES) / sizeof(FORCED_AXIS_BLOCK_CASES[0]);
    if (reference->forced_case_index >= forced_count) {
        fprintf(stderr, "invalid forced-axis upload reference index for %s\n", reference->case_name);
        return 1;
    }
    const ForcedAxisBlockCase* test_case = &FORCED_AXIS_BLOCK_CASES[reference->forced_case_index];
    VkFFTApplication* app = (VkFFTApplication*)calloc(1, sizeof(*app));
    VkFFTPlan* plan = (VkFFTPlan*)calloc(1, sizeof(*plan));
    pfUINT temp_buffer_size = 0;
    if (app == NULL || plan == NULL) {
        fprintf(stderr, "failed to allocate upstream forced-axis upload state\n");
        free(plan);
        free(app);
        return 1;
    }
    configure_forced_axis_block_app(app, test_case, &temp_buffer_size);
    VkFFTResult result = VkFFTScheduler(app, plan, 0);
    if (result != VKFFT_SUCCESS) {
        fprintf(stderr, "VkFFTScheduler failed for %s with result %d\n", reference->case_name, (int)result);
        free(plan);
        free(app);
        return 1;
    }
    const pfUINT upload = reference->upload_id;
    if (upload >= plan->numAxisUploads[0]) {
        fprintf(stderr, "upstream %s is missing requested upload %llu\n", reference->case_name, (unsigned long long)upload);
        free(plan);
        free(app);
        return 1;
    }
    VkFFTAxis* axis = &plan->axes[0][upload];
    const pfUINT complex_size = 4 * sizeof(double);
    axis->specializationConstants.complexSize = complex_size;
    axis->specializationConstants.numAxisUploads = (int)plan->numAxisUploads[0];
    axis->specializationConstants.reorderFourStep = 1;
    axis->specializationConstants.stageStartSize.type = 31;
    axis->specializationConstants.stageStartSize.data.i = 1;
    for (pfUINT prior_upload = 0; prior_upload < upload; ++prior_upload) {
        axis->specializationConstants.stageStartSize.data.i *= plan->axisSplit[0][prior_upload];
    }
    pfUINT allowed_shared = test_case->shared_memory_bytes;
    pfUINT allowed_shared_pow2 = test_case->shared_memory_pow2_bytes;
    if (axis->specializationConstants.useRaderMult > 0) {
        const pfUINT reserve =
            (pfUINT)(axis->specializationConstants.useRaderMult - 1) * complex_size;
        allowed_shared -= reserve;
        allowed_shared_pow2 -= reserve;
    }
    result = VkFFTSplitAxisBlock(
        app,
        plan,
        axis,
        0,
        upload,
        allowed_shared,
        allowed_shared_pow2
    );
    if (result != VKFFT_SUCCESS) {
        fprintf(stderr, "VkFFTSplitAxisBlock failed for %s with result %d\n", reference->case_name, (int)result);
        free(plan);
        free(app);
        return 1;
    }
    const pfUINT fft_dim = (pfUINT)axis->specializationConstants.fftDim.data.i;
    if (fft_dim == 0 || !test_case->sequence_len || test_case->sequence_len % fft_dim != 0) {
        fprintf(stderr, "invalid forced-axis component geometry for %s\n", reference->case_name);
        free(plan);
        free(app);
        return 1;
    }
    const pfUINT transform_count =
        (test_case->sequence_len / fft_dim) * test_case->batch_count;
    const int axis_swapped = axis->specializationConstants.axisSwapped != 0;
    const int transforms_on_x = upload > 0 || axis_swapped;
    const pfUINT threads_per_transform =
        (upload == 0 && !axis_swapped) ? axis->axisBlock[0] : axis->axisBlock[1];
    printf(
        "{\"schema_version\":%d,\"upstream_commit\":\"%s\",\"case\":\"%s\",\"kind\":\"axis-block\","
        "\"backend\":\"%s\",\"vendor\":\"%s\",\"precision\":\"double-double\","
        "\"device\":{\"shared_memory_bytes\":%llu,\"shared_memory_pow2_bytes\":%llu,"
        "\"max_threads_per_block\":%llu,\"max_workgroup_size\":[%llu,%llu,64],"
        "\"coalesced_memory_bytes\":%llu,\"shared_banks\":32,\"supports_f64\":true},"
        "\"payload\":{\"sequence_len\":%llu,\"batch_count\":%llu,\"axis_upload_id\":%llu,"
        "\"transform_count\":%llu,\"threads_per_transform\":%llu,\"grouped_batch\":%llu,"
        "\"transforms_on_x\":%s,\"axis_swapped\":%s,\"local_size_x\":%llu,\"local_size_y\":%llu}}\n",
        SNAPSHOT_SCHEMA_VERSION,
        UPSTREAM_COMMIT,
        reference->case_name,
        test_case->backend_name,
        test_case->vendor_name,
        (unsigned long long)test_case->shared_memory_bytes,
        (unsigned long long)test_case->shared_memory_pow2_bytes,
        (unsigned long long)test_case->max_threads_num,
        (unsigned long long)test_case->max_workgroup_x,
        (unsigned long long)test_case->max_workgroup_x,
        (unsigned long long)test_case->coalesced_memory_bytes,
        (unsigned long long)fft_dim,
        (unsigned long long)transform_count,
        (unsigned long long)upload,
        (unsigned long long)transform_count,
        (unsigned long long)threads_per_transform,
        (unsigned long long)axis->groupedBatch,
        transforms_on_x ? "true" : "false",
        axis_swapped ? "true" : "false",
        (unsigned long long)axis->axisBlock[0],
        (unsigned long long)axis->axisBlock[1]
    );
    free(plan);
    free(app);
    return 0;
}

static int emit_rader_parent_case(const RaderParentCase* test_case) {
    VkFFTApplication* app = (VkFFTApplication*)calloc(1, sizeof(*app));
    VkFFTPlan* plan = (VkFFTPlan*)calloc(1, sizeof(*plan));
    pfUINT temp_buffer_size = 0;
    if (app == NULL || plan == NULL) {
        fprintf(stderr, "failed to allocate upstream Rader parent state\n");
        free(plan);
        free(app);
        return 1;
    }
    configure_rader_parent_app(app, test_case, &temp_buffer_size);
    VkFFTResult result = VkFFTScheduler(app, plan, 0);
    if (result != VKFFT_SUCCESS) {
        fprintf(stderr, "VkFFTScheduler failed for %s with result %d\n", test_case->case_name, (int)result);
        free(plan);
        free(app);
        return 1;
    }
    if (plan->numAxisUploads[0] != 1) {
        fprintf(stderr, "upstream %s expected one Rader parent upload, got %llu\n", test_case->case_name, (unsigned long long)plan->numAxisUploads[0]);
        free(plan);
        free(app);
        return 1;
    }
    VkFFTAxis* axis = &plan->axes[0][0];
    const int min_registers = axis->specializationConstants.min_registers_per_thread;
    const int min_rader_threads = axis->specializationConstants.minRaderFFTThreadNum;
    if (min_registers <= 0 || min_rader_threads <= 0) {
        fprintf(stderr, "upstream %s did not produce an executable joint Rader parent\n", test_case->case_name);
        free(plan);
        free(app);
        return 1;
    }
    const int ordinary_threads = (int)((test_case->sequence_len + (pfUINT)min_registers - 1) / (pfUINT)min_registers);
    const int threads_per_transform = ordinary_threads > min_rader_threads ? ordinary_threads : min_rader_threads;
    const pfUINT complex_size = 2 * sizeof(float);
    axis->specializationConstants.complexSize = complex_size;
    axis->specializationConstants.numAxisUploads = (int)plan->numAxisUploads[0];
    axis->specializationConstants.reorderFourStep = plan->numAxisUploads[0] > 1 ? 1 : 0;
    axis->specializationConstants.stageStartSize.type = 31;
    axis->specializationConstants.stageStartSize.data.i = 1;
    pfUINT allowed_shared = test_case->shared_memory_bytes;
    pfUINT allowed_shared_pow2 = test_case->shared_memory_pow2_bytes;
    if (axis->specializationConstants.useRaderMult > 0) {
        const pfUINT reserve =
            (pfUINT)(axis->specializationConstants.useRaderMult - 1) * complex_size;
        allowed_shared -= reserve;
        allowed_shared_pow2 -= reserve;
    }
    result = VkFFTSplitAxisBlock(
        app,
        plan,
        axis,
        0,
        0,
        allowed_shared,
        allowed_shared_pow2
    );
    if (result != VKFFT_SUCCESS) {
        fprintf(stderr, "VkFFTSplitAxisBlock failed for %s with result %d\n", test_case->case_name, (int)result);
        free(plan);
        free(app);
        return 1;
    }
    if ((int)axis->axisBlock[0] != threads_per_transform) {
        fprintf(
            stderr,
            "upstream %s split axisBlock[0]=%llu disagrees with reconstructed parent threads=%d\n",
            test_case->case_name,
            (unsigned long long)axis->axisBlock[0],
            threads_per_transform
        );
        free(plan);
        free(app);
        return 1;
    }
    const char* vendor_name = test_case->vendor_name != NULL ? test_case->vendor_name : "nvidia";
    const pfUINT coalesced_memory_bytes = test_case->coalesced_memory_bytes != 0
        ? test_case->coalesced_memory_bytes
        : 32;
    printf(
        "{\"schema_version\":%d,\"upstream_commit\":\"%s\",\"case\":\"%s\",\"kind\":\"rader-parent\","
        "\"backend\":\"vulkan\",\"vendor\":\"%s\",\"precision\":\"f32\","
        "\"device\":{\"shared_memory_bytes\":%llu,\"shared_memory_pow2_bytes\":%llu,"
        "\"max_threads_per_block\":%llu,\"max_workgroup_size\":[1024,1024,64],"
        "\"coalesced_memory_bytes\":%llu,\"shared_banks\":32,\"supports_f64\":true},"
        "\"payload\":{\"sequence_len\":%llu,\"batch_count\":%llu,\"upload_count\":1,\"axis_split\":[%llu],"
        "\"final_min_registers\":%d,\"min_rader_fft_thread_num\":%d,\"threads_per_transform\":%d}}\n",
        SNAPSHOT_SCHEMA_VERSION,
        UPSTREAM_COMMIT,
        test_case->case_name,
        vendor_name,
        (unsigned long long)test_case->shared_memory_bytes,
        (unsigned long long)test_case->shared_memory_pow2_bytes,
        (unsigned long long)test_case->max_threads_num,
        (unsigned long long)coalesced_memory_bytes,
        (unsigned long long)test_case->sequence_len,
        (unsigned long long)test_case->batch_count,
        (unsigned long long)plan->axisSplit[0][0],
        min_registers,
        min_rader_threads,
        threads_per_transform
    );
    free(plan);
    free(app);
    return 0;
}

static int emit_mixed_rader_parent_case(const MixedRaderParentCase* test_case) {
    VkFFTApplication* app = (VkFFTApplication*)calloc(1, sizeof(*app));
    VkFFTPlan* plan = (VkFFTPlan*)calloc(1, sizeof(*plan));
    pfUINT temp_buffer_size = 0;
    if (app == NULL || plan == NULL) {
        fprintf(stderr, "failed to allocate upstream mixed Rader parent state\n");
        free(plan);
        free(app);
        return 1;
    }
    configure_mixed_rader_parent_app(app, test_case, &temp_buffer_size);
    VkFFTResult result = VkFFTScheduler(app, plan, 0);
    if (result != VKFFT_SUCCESS) {
        fprintf(stderr, "VkFFTScheduler failed for %s with result %d\n", test_case->case_name, (int)result);
        free(plan);
        free(app);
        return 1;
    }
    if (plan->numAxisUploads[0] != 1) {
        fprintf(stderr, "upstream %s expected one mixed Rader parent upload, got %llu\n", test_case->case_name, (unsigned long long)plan->numAxisUploads[0]);
        free(plan);
        free(app);
        return 1;
    }
    VkFFTAxis* axis = &plan->axes[0][0];
    for (int index = 0; index < test_case->direct_count; ++index) {
        if (find_axis_rader_container_of_type(axis, test_case->direct_primes[index], 1) == NULL) {
            fprintf(stderr, "upstream %s did not classify p%d as Direct/type-1 Rader\n", test_case->case_name, test_case->direct_primes[index]);
            free(plan);
            free(app);
            return 1;
        }
    }
    for (int index = 0; index < test_case->fft_count; ++index) {
        if (find_axis_rader_container_of_type(axis, test_case->fft_primes[index], 0) == NULL) {
            fprintf(stderr, "upstream %s did not classify p%d as FFT/type-0 Rader\n", test_case->case_name, test_case->fft_primes[index]);
            free(plan);
            free(app);
            return 1;
        }
    }

    const pfUINT complex_size =
        test_case->double_double ? 4 * sizeof(double) : 2 * sizeof(float);
    axis->specializationConstants.complexSize = complex_size;
    axis->specializationConstants.numAxisUploads = (int)plan->numAxisUploads[0];
    axis->specializationConstants.reorderFourStep = 0;
    axis->specializationConstants.stageStartSize.type = 31;
    axis->specializationConstants.stageStartSize.data.i = 1;
    pfUINT allowed_shared = test_case->shared_memory_bytes;
    pfUINT allowed_shared_pow2 = test_case->shared_memory_pow2_bytes;
    if (axis->specializationConstants.useRaderMult > 0) {
        const pfUINT reserve =
            (pfUINT)(axis->specializationConstants.useRaderMult - 1) * complex_size;
        allowed_shared -= reserve;
        allowed_shared_pow2 -= reserve;
    }
    result = VkFFTSplitAxisBlock(app, plan, axis, 0, 0, allowed_shared, allowed_shared_pow2);
    if (result != VKFFT_SUCCESS || axis->axisBlock[0] == 0) {
        fprintf(stderr, "VkFFTSplitAxisBlock failed for %s with result %d / axisBlock0=%llu\n", test_case->case_name, (int)result, (unsigned long long)axis->axisBlock[0]);
        free(plan);
        free(app);
        return 1;
    }
    printf(
        "{\"schema_version\":%d,\"upstream_commit\":\"%s\",\"case\":\"%s\",\"kind\":\"rader-mixed-parent\","
        "\"backend\":\"vulkan\",\"vendor\":\"nvidia\",\"precision\":\"%s\","
        "\"device\":{\"shared_memory_bytes\":%llu,\"shared_memory_pow2_bytes\":%llu,"
        "\"max_threads_per_block\":%llu,\"max_workgroup_size\":[1024,1024,64],"
        "\"coalesced_memory_bytes\":32,\"shared_banks\":32,\"supports_f64\":true},"
        "\"payload\":{\"sequence_len\":%llu,\"batch_count\":%llu,\"direct_primes\":[",
        SNAPSHOT_SCHEMA_VERSION,
        UPSTREAM_COMMIT,
        test_case->case_name,
        test_case->double_double ? "double-double" : "f32",
        (unsigned long long)test_case->shared_memory_bytes,
        (unsigned long long)test_case->shared_memory_pow2_bytes,
        (unsigned long long)test_case->max_threads_num,
        (unsigned long long)test_case->sequence_len,
        (unsigned long long)test_case->batch_count
    );
    for (int index = 0; index < test_case->direct_count; ++index) {
        if (index != 0) {
            fputc(',', stdout);
        }
        printf("%d", test_case->direct_primes[index]);
    }
    fputs("],\"fft_primes\":[", stdout);
    for (int index = 0; index < test_case->fft_count; ++index) {
        if (index != 0) {
            fputc(',', stdout);
        }
        printf("%d", test_case->fft_primes[index]);
    }
    printf("],\"threads_per_transform\":%llu}}\n", (unsigned long long)axis->axisBlock[0]);
    free(plan);
    free(app);
    return 0;
}

static int emit_stockham_axis_block_reference(const StockhamAxisBlockReference* reference) {
    const StockhamCase* test_case = NULL;
    const size_t stockham_count = sizeof(STOCKHAM_CASES) / sizeof(STOCKHAM_CASES[0]);
    for (size_t index = 0; index < stockham_count; ++index) {
        if (strcmp(reference->stockham_case_name, STOCKHAM_CASES[index].case_name) == 0) {
            test_case = &STOCKHAM_CASES[index];
            break;
        }
    }
    if (test_case == NULL) {
        fprintf(stderr, "missing source Stockham case for %s\n", reference->case_name);
        return 1;
    }

    VkFFTApplication* app = (VkFFTApplication*)calloc(1, sizeof(*app));
    VkFFTPlan* plan = (VkFFTPlan*)calloc(1, sizeof(*plan));
    pfUINT temp_buffer_size = 0;
    if (app == NULL || plan == NULL) {
        fprintf(stderr, "failed to allocate upstream Stockham AxisBlock state\n");
        free(plan);
        free(app);
        return 1;
    }
    configure_stockham_app(app, test_case, &temp_buffer_size);
    const pfUINT axis_id = test_case->axis_id;
    VkFFTResult result = VKFFT_SUCCESS;
    if (test_case->bluestein_logical_len != 0) {
        result = initializeBluesteinAutoPadding(app);
    }
    if (result == VKFFT_SUCCESS) {
        result = VkFFTScheduler(app, plan, axis_id);
    }
    const pfUINT upload = reference->upload_id;
    if (result != VKFFT_SUCCESS || upload >= plan->numAxisUploads[axis_id]) {
        fprintf(
            stderr,
            "upstream Stockham AxisBlock scheduler failed for %s (result=%d, uploads=%llu)\n",
            reference->case_name,
            (int)result,
            (unsigned long long)plan->numAxisUploads[axis_id]
        );
        if (test_case->bluestein_logical_len != 0) {
            free(app->configuration.primeSizes);
            free(app->configuration.paddedSizes);
        }
        free(plan);
        free(app);
        return 1;
    }

    VkFFTAxis* axis = &plan->axes[axis_id][upload];
    const pfUINT complex_size = test_case->double_double
        ? 4 * sizeof(double)
        : ((test_case->double_precision || test_case->double_precision_float_memory)
            ? 2 * sizeof(double)
            : 2 * sizeof(float));
    axis->specializationConstants.complexSize = complex_size;
    axis->specializationConstants.numAxisUploads = (int)plan->numAxisUploads[axis_id];
    axis->specializationConstants.reorderFourStep = plan->numAxisUploads[axis_id] > 1 ? 1 : 0;
    axis->specializationConstants.stageStartSize.type = 31;
    axis->specializationConstants.stageStartSize.data.i = 1;
    for (pfUINT prior_upload = 0; prior_upload < upload; ++prior_upload) {
        axis->specializationConstants.stageStartSize.data.i *= plan->axisSplit[axis_id][prior_upload];
    }
    pfUINT allowed_shared = test_case->shared_memory_bytes;
    pfUINT allowed_shared_pow2 = test_case->shared_memory_pow2_bytes;
    if (axis->specializationConstants.useRaderMult > 0) {
        const pfUINT reserve =
            (pfUINT)(axis->specializationConstants.useRaderMult - 1) * complex_size;
        allowed_shared -= reserve;
        allowed_shared_pow2 -= reserve;
    }
    result = VkFFTSplitAxisBlock(
        app,
        plan,
        axis,
        axis_id,
        upload,
        allowed_shared,
        allowed_shared_pow2
    );
    if (result != VKFFT_SUCCESS) {
        fprintf(stderr, "VkFFTSplitAxisBlock failed for %s with result %d\n", reference->case_name, (int)result);
        if (test_case->bluestein_logical_len != 0) {
            free(app->configuration.primeSizes);
            free(app->configuration.paddedSizes);
        }
        free(plan);
        free(app);
        return 1;
    }

    const pfUINT fft_dim = (pfUINT)axis->specializationConstants.fftDim.data.i;
    const pfUINT independent_lines = axis_id > 0
        ? (test_case->fastest_axis_len != 0 ? test_case->fastest_axis_len : 1) * test_case->batch_count
        : test_case->batch_count;
    const pfUINT transform_count = test_case->sequence_len * independent_lines / fft_dim;
    const int axis_swapped = axis->specializationConstants.axisSwapped != 0;
    const int transforms_on_x = axis_id > 0 || upload > 0 || axis_swapped;
    const pfUINT threads_per_transform = transforms_on_x ? axis->axisBlock[1] : axis->axisBlock[0];
    const char* backend_name = test_case->backend_name != NULL ? test_case->backend_name : "vulkan";
    const char* precision_name = test_case->double_double
        ? "double-double"
        : (test_case->half_precision
            ? "f16-storage-f32-compute"
            : (test_case->double_precision_float_memory
                ? "f64-compute-f32-storage"
                : (test_case->double_precision ? "f64" : "f32")));
    printf(
        "{\"schema_version\":%d,\"upstream_commit\":\"%s\",\"case\":\"%s\",\"kind\":\"axis-block\","
        "\"backend\":\"%s\",\"vendor\":\"%s\",\"precision\":\"%s\","
        "\"device\":{\"shared_memory_bytes\":%llu,\"shared_memory_pow2_bytes\":%llu,"
        "\"max_threads_per_block\":1024,\"max_workgroup_size\":[1024,1024,64],"
        "\"coalesced_memory_bytes\":%llu,\"shared_banks\":32,\"supports_f64\":true},"
        "\"payload\":{\"sequence_len\":%llu,\"batch_count\":%llu,\"axis_upload_id\":%llu,"
        "\"transform_count\":%llu,\"threads_per_transform\":%llu,\"grouped_batch\":%llu,"
        "\"transforms_on_x\":%s,\"axis_swapped\":%s,\"local_size_x\":%llu,\"local_size_y\":%llu}}\n",
        SNAPSHOT_SCHEMA_VERSION,
        UPSTREAM_COMMIT,
        reference->case_name,
        backend_name,
        test_case->vendor_name,
        precision_name,
        (unsigned long long)test_case->shared_memory_bytes,
        (unsigned long long)test_case->shared_memory_pow2_bytes,
        (unsigned long long)test_case->coalesced_memory_bytes,
        (unsigned long long)fft_dim,
        (unsigned long long)transform_count,
        (unsigned long long)upload,
        (unsigned long long)transform_count,
        (unsigned long long)threads_per_transform,
        (unsigned long long)axis->groupedBatch,
        transforms_on_x ? "true" : "false",
        axis_swapped ? "true" : "false",
        (unsigned long long)axis->axisBlock[0],
        (unsigned long long)axis->axisBlock[1]
    );

    if (test_case->bluestein_logical_len != 0) {
        free(app->configuration.primeSizes);
        free(app->configuration.paddedSizes);
    }
    free(plan);
    free(app);
    return 0;
}

static int emit_stockham_case(const StockhamCase* test_case) {
    VkFFTApplication* app = (VkFFTApplication*)calloc(1, sizeof(*app));
    VkFFTPlan* plan = (VkFFTPlan*)calloc(1, sizeof(*plan));
    pfUINT temp_buffer_size = 0;
    if (app == NULL || plan == NULL) {
        fprintf(stderr, "failed to allocate upstream scheduler state\n");
        free(plan);
        free(app);
        return 1;
    }
    configure_stockham_app(app, test_case, &temp_buffer_size);
    const pfUINT axis_id = test_case->axis_id;
    VkFFTResult result = VKFFT_SUCCESS;
    if (test_case->bluestein_logical_len != 0) {
        result = initializeBluesteinAutoPadding(app);
    }
    if (result == VKFFT_SUCCESS) {
        result = VkFFTScheduler(app, plan, axis_id);
    }
    if (result != VKFFT_SUCCESS) {
        fprintf(
            stderr,
            "VkFFTScheduler failed for %s with result %d\n",
            test_case->case_name,
            (int)result
        );
        free(plan);
        free(app);
        return 1;
    }
    if (plan->numAxisUploads[axis_id] == 0) {
        fprintf(stderr, "upstream scheduler produced no uploads for %s\n", test_case->case_name);
        free(plan);
        free(app);
        return 1;
    }
    if (test_case->bluestein_logical_len != 0 && !app->useBluesteinFFT[axis_id]) {
        fprintf(stderr, "upstream scheduler did not select Bluestein for %s\n", test_case->case_name);
        free(app->configuration.primeSizes);
        free(app->configuration.paddedSizes);
        free(plan);
        free(app);
        return 1;
    }
    const pfUINT scheduled_sequence_len = plan->actualFFTSizePerAxis[axis_id][axis_id];
    if (scheduled_sequence_len != test_case->sequence_len) {
        fprintf(
            stderr,
            "upstream %s scheduled length %llu instead of expected %llu\n",
            test_case->case_name,
            (unsigned long long)scheduled_sequence_len,
            (unsigned long long)test_case->sequence_len
        );
        free(app->configuration.primeSizes);
        free(app->configuration.paddedSizes);
        free(plan);
        free(app);
        return 1;
    }

    const pfUINT used_shared =
        (scheduled_sequence_len & (scheduled_sequence_len - 1)) == 0
            ? test_case->shared_memory_pow2_bytes
            : test_case->shared_memory_bytes;
    const pfUINT complex_size = test_case->double_double
        ? 4 * sizeof(double)
        : ((test_case->double_precision || test_case->double_precision_float_memory)
            ? 2 * sizeof(double)
            : 2 * sizeof(float));
    const pfUINT max_shared = used_shared / complex_size;
    const pfUINT max_strided = test_case->coalesced_memory_bytes > complex_size
        ? used_shared / test_case->coalesced_memory_bytes
        : max_shared;
    const int schedule_boost = plan->axes[axis_id][0].specializationConstants.registerBoost;
    const char* backend_name = test_case->backend_name != NULL ? test_case->backend_name : "vulkan";

    if (test_case->double_double) {
        printf(
            "{\"schema_version\":%d,\"upstream_commit\":\"%s\",\"case\":\"%s\",\"kind\":\"dd-stockham\","
            "\"backend\":\"%s\",\"vendor\":\"%s\",\"precision\":\"double-double\","
            "\"device\":{\"shared_memory_bytes\":%llu,\"shared_memory_pow2_bytes\":%llu,"
            "\"max_threads_per_block\":1024,\"max_workgroup_size\":[1024,1024,64],"
            "\"coalesced_memory_bytes\":%llu,\"shared_banks\":32,\"supports_f64\":true},"
            "\"payload\":{\"sequence_len\":%llu,\"batch_count\":%llu,"
            "\"used_shared_memory_bytes\":%llu,\"max_sequence_len_shared\":%llu,"
            "\"max_sequence_len_strided\":%llu,\"register_boost\":%d,"
            "\"upload_count\":%llu,\"axis_split\":[",
            SNAPSHOT_SCHEMA_VERSION,
            UPSTREAM_COMMIT,
            test_case->case_name,
            backend_name,
            test_case->vendor_name,
            (unsigned long long)test_case->shared_memory_bytes,
            (unsigned long long)test_case->shared_memory_pow2_bytes,
            (unsigned long long)test_case->coalesced_memory_bytes,
            (unsigned long long)scheduled_sequence_len,
            (unsigned long long)test_case->batch_count,
            (unsigned long long)used_shared,
            (unsigned long long)max_shared,
            (unsigned long long)max_strided,
            schedule_boost,
            (unsigned long long)plan->numAxisUploads[axis_id]
        );
        for (pfUINT upload = 0; upload < plan->numAxisUploads[axis_id]; ++upload) {
            if (upload != 0) {
                fputc(',', stdout);
            }
            printf("%llu", (unsigned long long)plan->axisSplit[axis_id][upload]);
        }
        fputs("]}}\n", stdout);
        if (test_case->bluestein_logical_len != 0) {
            free(app->configuration.primeSizes);
            free(app->configuration.paddedSizes);
        }
        free(plan);
        free(app);
        return 0;
    }

    printf(
        "{\"schema_version\":%d,\"upstream_commit\":\"%s\",\"case\":\"%s\",\"kind\":\"stockham\","
        "\"backend\":\"%s\",\"vendor\":\"%s\",\"precision\":\"%s\","
        "\"device\":{\"shared_memory_bytes\":%llu,\"shared_memory_pow2_bytes\":%llu,"
        "\"max_threads_per_block\":1024,\"max_workgroup_size\":[1024,1024,64],"
        "\"coalesced_memory_bytes\":%llu,\"shared_banks\":32,\"supports_f64\":true},"
        "\"payload\":{\"sequence_len\":%llu,\"batch_count\":%llu,"
        "\"used_shared_memory_bytes\":%llu,\"max_sequence_len_shared\":%llu,"
        "\"max_sequence_len_strided\":%llu,\"register_boost\":%d,"
        "\"upload_count\":%llu,\"axis_split\":[",
        SNAPSHOT_SCHEMA_VERSION,
        UPSTREAM_COMMIT,
        test_case->case_name,
        backend_name,
        test_case->vendor_name,
        test_case->half_precision
            ? "f16-storage-f32-compute"
            : (test_case->double_precision_float_memory
                ? "f64-compute-f32-storage"
                : (test_case->double_precision ? "f64" : "f32")),
        (unsigned long long)test_case->shared_memory_bytes,
        (unsigned long long)test_case->shared_memory_pow2_bytes,
        (unsigned long long)test_case->coalesced_memory_bytes,
        (unsigned long long)scheduled_sequence_len,
        (unsigned long long)test_case->batch_count,
        (unsigned long long)used_shared,
        (unsigned long long)max_shared,
        (unsigned long long)max_strided,
        schedule_boost,
        (unsigned long long)plan->numAxisUploads[axis_id]
    );
    for (pfUINT upload = 0; upload < plan->numAxisUploads[axis_id]; ++upload) {
        if (upload != 0) {
            fputc(',', stdout);
        }
        printf("%llu", (unsigned long long)plan->axisSplit[axis_id][upload]);
    }
    fputs("],\"radix_schedules\":[", stdout);
    for (pfUINT upload = 0; upload < plan->numAxisUploads[axis_id]; ++upload) {
        VkFFTAxis* axis = &plan->axes[axis_id][upload];
        const int boost = axis->specializationConstants.registerBoost;
        if (boost != schedule_boost) {
            fprintf(
                stderr,
                "upstream %s uses inconsistent registerBoost across uploads\n",
                test_case->case_name
            );
            free(plan);
            free(app);
            return 1;
        }
        if (upload != 0) {
            fputc(',', stdout);
        }
        const pfUINT fft_len = plan->axisSplit[axis_id][upload];
        const pfUINT rhs = scheduled_sequence_len * test_case->batch_count / fft_len;
        const int registers = axis->specializationConstants.registers_per_thread;
        const int min_registers = axis->specializationConstants.min_registers_per_thread;
        const int good_sequence =
            !(registers > 16 || registers >= 2 * min_registers);
        printf(
            "{\"fft_len\":%llu,\"rhs_transform_count\":%llu,\"register_boost\":%d,"
            "\"stage_radices\":[",
            (unsigned long long)fft_len,
            (unsigned long long)rhs,
            boost
        );
        for (int stage = 0; stage < axis->specializationConstants.numStages; ++stage) {
            if (stage != 0) {
                fputc(',', stdout);
            }
            printf("%d", axis->specializationConstants.stageRadix[stage]);
        }
        fputs("],\"register_boost_stage_radix\":", stdout);
        if (boost > 1) {
            printf("%d", boost);
        } else {
            fputs("null", stdout);
        }
        printf(
            ",\"registers_per_thread\":%d,\"min_registers_per_thread\":%d,"
            "\"is_good_sequence\":%s,\"max_non_power_of_two_radix\":%d,"
            "\"required_local_registers\":%d}",
            registers,
            min_registers,
            good_sequence ? "true" : "false",
            axis->specializationConstants.maxNonPow2Radix,
            axis->specializationConstants.usedLocRegs
        );
    }
    fputs("]}}\n", stdout);

    if (test_case->bluestein_logical_len != 0) {
        free(app->configuration.primeSizes);
        free(app->configuration.paddedSizes);
    }
    free(plan);
    free(app);
    return 0;
}

static int emit_real_shape_case(const RealShapeCase* test_case) {
    VkFFTApplication* app = (VkFFTApplication*)calloc(1, sizeof(*app));
    VkFFTPlan* plan = (VkFFTPlan*)calloc(1, sizeof(*plan));
    pfUINT temp_buffer_size = 0;
    if (app == NULL || plan == NULL) {
        fprintf(stderr, "failed to allocate upstream real-shape scheduler state\n");
        free(plan);
        free(app);
        return 1;
    }

    app->configuration.FFTdim = 1;
    app->configuration.size[0] = test_case->sequence_len;
    app->configuration.performR2C = 1;
    app->configuration.quadDoubleDoublePrecision = test_case->quad_double_double_precision;
    app->configuration.quadDoubleDoublePrecisionDoubleMemory =
        test_case->quad_double_double_precision_double_memory;
    app->configuration.halfPrecision = test_case->half_precision;
    app->configuration.doublePrecision = test_case->double_precision;
    app->configuration.doublePrecisionFloatMemory = test_case->double_precision_float_memory;
    app->configuration.vendorID = test_case->vendor_id;
    app->configuration.sharedMemorySize = test_case->shared_memory_bytes;
    app->configuration.sharedMemorySizePow2 = test_case->shared_memory_pow2_bytes;
    app->configuration.maxThreadsNum = test_case->max_threads_num;
    app->configuration.maxComputeWorkGroupSize[0] = test_case->max_workgroup_x;
    app->configuration.maxComputeWorkGroupSize[1] = test_case->max_workgroup_y;
    app->configuration.maxComputeWorkGroupSize[2] = test_case->max_workgroup_z;
    app->configuration.coalescedMemory = test_case->scheduler_coalesced_memory_bytes;
    app->configuration.aimThreads = 128;
    app->configuration.numSharedBanks = 32;
    app->configuration.warpSize = test_case->warp_size;
    app->configuration.registerBoost = test_case->register_boost;
    app->configuration.registerBoost4Step = 1;
    app->configuration.registerBoostNonPow2 = 0;
    app->configuration.swapTo3Stage4Step = test_case->swap_threshold;
    app->configuration.swapTo2Stage4Step = test_case->swap_threshold;
    app->configuration.reorderFourStep = 1;
    const int is_quad = test_case->quad_double_double_precision
        || test_case->quad_double_double_precision_double_memory;
    app->configuration.fixMinRaderPrimeMult = is_quad ? 11 : 17;
    if (is_quad) {
        app->configuration.fixMaxRaderPrimeMult = 29;
    } else if (test_case->vendor_id == 0x10DE || test_case->vendor_id == 0x1002) {
        app->configuration.fixMaxRaderPrimeMult = 89;
    } else {
        app->configuration.fixMaxRaderPrimeMult = 17;
    }
    app->configuration.fixMinRaderPrimeFFT =
        (is_quad && test_case->vendor_id == 0x1002) ? 19 : 17;
    app->configuration.fixMaxRaderPrimeFFT = 16384;
    app->configuration.coordinateFeatures = 1;
    app->configuration.numberBatches = 1;
    app->configuration.numberKernels = 1;
    app->configuration.tempBufferSize = &temp_buffer_size;
    app->actualNumBatches = 1;

    VkFFTResult result = initializeBluesteinAutoPadding(app);
    if (result == VKFFT_SUCCESS) {
        result = VkFFTScheduler(app, plan, 0);
    }
    if (result != VKFFT_SUCCESS) {
        fprintf(
            stderr,
            "upstream real-shape scheduler failed for %s with result %d\n",
            test_case->case_name,
            (int)result
        );
        free(app->configuration.primeSizes);
        free(app->configuration.paddedSizes);
        free(plan);
        free(app);
        return 1;
    }

    const int callback_forced = app->configuration.forceCallbackVersionRealTransforms != 0;
    const int big_sequence_even_r2c = plan->bigSequenceEvenR2C != 0;
    const pfUINT scheduled_fft_len = plan->actualFFTSizePerAxis[0][0];
    const char* algorithm = big_sequence_even_r2c ? "even-half-size" : "full-complex";
    printf(
        "{\"schema_version\":%d,\"upstream_commit\":\"%s\",\"case\":\"%s\",\"kind\":\"real-shape\","
        "\"backend\":\"%s\",\"vendor\":\"%s\",\"precision\":\"%s\","
        "\"device\":{\"shared_memory_bytes\":%llu,\"shared_memory_pow2_bytes\":%llu,"
        "\"max_threads_per_block\":%llu,\"max_workgroup_size\":[%llu,%llu,%llu],"
        "\"coalesced_memory_bytes\":%llu,\"shared_banks\":32,\"supports_f64\":%s},"
        "\"payload\":{\"sequence_len\":%llu,\"batch_count\":1,\"scheduled_fft_len\":%llu,"
        "\"algorithm\":\"%s\",\"callback_forced\":%s,\"big_sequence_even_r2c\":%s}}\n",
        SNAPSHOT_SCHEMA_VERSION,
        UPSTREAM_COMMIT,
        test_case->case_name,
        test_case->backend_name,
        test_case->vendor_name,
        test_case->precision_name,
        (unsigned long long)test_case->shared_memory_bytes,
        (unsigned long long)test_case->shared_memory_pow2_bytes,
        (unsigned long long)test_case->max_threads_num,
        (unsigned long long)test_case->max_workgroup_x,
        (unsigned long long)test_case->max_workgroup_y,
        (unsigned long long)test_case->max_workgroup_z,
        (unsigned long long)test_case->device_coalesced_memory_bytes,
        test_case->supports_f64 ? "true" : "false",
        (unsigned long long)test_case->sequence_len,
        (unsigned long long)scheduled_fft_len,
        algorithm,
        callback_forced ? "true" : "false",
        big_sequence_even_r2c ? "true" : "false"
    );

    free(app->configuration.primeSizes);
    free(app->configuration.paddedSizes);
    free(plan);
    free(app);
    return 0;
}

static const char* axis0_rader_backend_name(const Axis0RaderBlockCase* test_case) {
    return test_case->backend_name != NULL ? test_case->backend_name : "vulkan";
}

static const char* axis0_rader_vendor_name(const Axis0RaderBlockCase* test_case) {
    return test_case->vendor_name != NULL ? test_case->vendor_name : "nvidia";
}

static pfUINT axis0_rader_vendor_id(const Axis0RaderBlockCase* test_case) {
    return test_case->vendor_id != 0 ? test_case->vendor_id : 0x10DE;
}

static pfUINT axis0_rader_max_threads(const Axis0RaderBlockCase* test_case) {
    return test_case->max_threads_num != 0 ? test_case->max_threads_num : 1024;
}

static pfUINT axis0_rader_max_workgroup_x(const Axis0RaderBlockCase* test_case) {
    return test_case->max_workgroup_x != 0 ? test_case->max_workgroup_x : 1024;
}

static pfUINT axis0_rader_coalesced_memory(const Axis0RaderBlockCase* test_case) {
    return test_case->coalesced_memory_bytes != 0 ? test_case->coalesced_memory_bytes : 32;
}

static pfUINT axis0_rader_warp_size(const Axis0RaderBlockCase* test_case) {
    return test_case->warp_size != 0 ? test_case->warp_size : 32;
}

static int axis0_rader_register_boost(const Axis0RaderBlockCase* test_case) {
    return test_case->register_boost != 0 ? test_case->register_boost : 4;
}

static pfUINT axis0_rader_swap_threshold(const Axis0RaderBlockCase* test_case) {
    return test_case->swap_threshold != 0 ? test_case->swap_threshold : 4194305;
}

static int axis0_rader_min_direct_prime(const Axis0RaderBlockCase* test_case) {
    return test_case->min_direct_prime != 0 ? test_case->min_direct_prime : 17;
}

static int axis0_rader_max_direct_prime(const Axis0RaderBlockCase* test_case) {
    return test_case->max_direct_prime != 0 ? test_case->max_direct_prime : 89;
}

static int axis0_rader_min_fft_prime(const Axis0RaderBlockCase* test_case) {
    return test_case->min_fft_prime != 0 ? test_case->min_fft_prime : 17;
}

static int axis0_rader_max_fft_prime(const Axis0RaderBlockCase* test_case) {
    return test_case->max_fft_prime != 0 ? test_case->max_fft_prime : 16384;
}

static void configure_axis0_rader_block_app(
    VkFFTApplication* app,
    const Axis0RaderBlockCase* test_case,
    pfUINT* temp_buffer_size
) {
    memset(app, 0, sizeof(*app));
    app->configuration.FFTdim = 1;
    app->configuration.size[0] = test_case->sequence_len;
    app->configuration.vendorID = axis0_rader_vendor_id(test_case);
    app->configuration.sharedMemorySize = test_case->shared_memory_bytes;
    app->configuration.sharedMemorySizePow2 = test_case->shared_memory_pow2_bytes;
    app->configuration.maxThreadsNum = axis0_rader_max_threads(test_case);
    app->configuration.maxComputeWorkGroupSize[0] = axis0_rader_max_workgroup_x(test_case);
    app->configuration.maxComputeWorkGroupSize[1] = axis0_rader_max_workgroup_x(test_case);
    app->configuration.maxComputeWorkGroupSize[2] = 64;
    app->configuration.coalescedMemory = axis0_rader_coalesced_memory(test_case);
    app->configuration.aimThreads = 128;
    app->configuration.numSharedBanks = 32;
    app->configuration.warpSize = axis0_rader_warp_size(test_case);
    app->configuration.registerBoost = axis0_rader_register_boost(test_case);
    app->configuration.registerBoost4Step = 1;
    app->configuration.registerBoostNonPow2 = 0;
    app->configuration.swapTo3Stage4Step = axis0_rader_swap_threshold(test_case);
    app->configuration.swapTo2Stage4Step = axis0_rader_swap_threshold(test_case);
    app->configuration.reorderFourStep = 1;
    app->configuration.fixMinRaderPrimeMult = axis0_rader_min_direct_prime(test_case);
    app->configuration.fixMaxRaderPrimeMult = axis0_rader_max_direct_prime(test_case);
    app->configuration.fixMinRaderPrimeFFT = axis0_rader_min_fft_prime(test_case);
    app->configuration.fixMaxRaderPrimeFFT = axis0_rader_max_fft_prime(test_case);
    app->configuration.coordinateFeatures = 1;
    app->configuration.numberBatches = 1;
    app->configuration.numberKernels = test_case->number_kernels != 0 ? test_case->number_kernels : 1;
    app->configuration.tempBufferSize = temp_buffer_size;
    app->actualNumBatches = test_case->batch_count;
    if (test_case->perform_convolution) {
        app->configuration.performConvolution = 1;
        app->configuration.reorderFourStep = 0;
        app->configuration.registerBoost = 1;
        app->configuration.registerBoostNonPow2 = 0;
        app->configuration.registerBoost4Step = 1;
    }
    if (test_case->half_precision) {
        app->configuration.halfPrecision = 1;
    }
    if (test_case->grouped_batch_override != 0) {
        app->configuration.groupedBatch[0] = test_case->grouped_batch_override;
    }
}

static void configure_axis0_stockham_block_app(
    VkFFTApplication* app,
    const Axis0StockhamBlockCase* test_case,
    pfUINT* temp_buffer_size
) {
    memset(app, 0, sizeof(*app));
    app->configuration.FFTdim = 1;
    app->configuration.size[0] = test_case->sequence_len;
    app->configuration.vendorID = 0x10DE;
    app->configuration.sharedMemorySize = test_case->shared_memory_bytes;
    app->configuration.sharedMemorySizePow2 = test_case->shared_memory_pow2_bytes;
    app->configuration.maxThreadsNum = 1024;
    app->configuration.maxComputeWorkGroupSize[0] = 1024;
    app->configuration.maxComputeWorkGroupSize[1] = 1024;
    app->configuration.maxComputeWorkGroupSize[2] = 64;
    app->configuration.coalescedMemory = 32;
    app->configuration.aimThreads = 128;
    app->configuration.numSharedBanks = 32;
    app->configuration.warpSize = 32;
    app->configuration.registerBoost = 4;
    app->configuration.registerBoost4Step = 1;
    app->configuration.registerBoostNonPow2 = 0;
    app->configuration.swapTo3Stage4Step = 4194305;
    app->configuration.swapTo2Stage4Step = 4194305;
    app->configuration.reorderFourStep = 1;
    app->configuration.quadDoubleDoublePrecision = 1;
    app->configuration.useLUT = 1;
    app->configuration.fixMinRaderPrimeMult = 11;
    app->configuration.fixMaxRaderPrimeMult = 29;
    app->configuration.fixMinRaderPrimeFFT = 17;
    app->configuration.fixMaxRaderPrimeFFT = 16384;
    app->configuration.coordinateFeatures = 1;
    app->configuration.numberBatches = test_case->batch_count;
    app->configuration.numberKernels = 1;
    app->configuration.tempBufferSize = temp_buffer_size;
    app->actualNumBatches = test_case->batch_count;
}

static const char* axis_block_vendor_name(const AxisBlockCase* test_case) {
    return test_case->vendor_name != NULL ? test_case->vendor_name : "nvidia";
}

static pfUINT axis_block_vendor_id(const AxisBlockCase* test_case) {
    return test_case->vendor_id != 0 ? test_case->vendor_id : 0x10DE;
}

static pfUINT axis_block_max_threads(const AxisBlockCase* test_case) {
    return test_case->max_threads_num != 0 ? test_case->max_threads_num : 1024;
}

static pfUINT axis_block_max_workgroup_x(const AxisBlockCase* test_case) {
    return test_case->max_workgroup_x != 0 ? test_case->max_workgroup_x : 1024;
}

static pfUINT axis_block_max_workgroup_y(const AxisBlockCase* test_case) {
    return test_case->max_workgroup_y != 0 ? test_case->max_workgroup_y : 1024;
}

static pfUINT axis_block_coalesced_memory(const AxisBlockCase* test_case) {
    return test_case->coalesced_memory_bytes != 0 ? test_case->coalesced_memory_bytes : 32;
}

static pfUINT axis_block_warp_size(const AxisBlockCase* test_case) {
    return test_case->warp_size > 0 ? (pfUINT)test_case->warp_size : 32;
}

static pfUINT axis_block_register_boost(const AxisBlockCase* test_case) {
    return test_case->register_boost != 0 ? test_case->register_boost : 4;
}

static pfUINT axis_block_swap_threshold(const AxisBlockCase* test_case) {
    return test_case->swap_threshold != 0 ? test_case->swap_threshold : 4194305;
}

static pfUINT axis_block_min_direct_prime(const AxisBlockCase* test_case) {
    if (test_case->min_direct_prime > 0) return (pfUINT)test_case->min_direct_prime;
    return test_case->double_double ? 11 : 17;
}

static pfUINT axis_block_max_direct_prime(const AxisBlockCase* test_case) {
    if (test_case->max_direct_prime > 0) return (pfUINT)test_case->max_direct_prime;
    return test_case->double_double ? 29 : 89;
}

static pfUINT axis_block_min_fft_prime(const AxisBlockCase* test_case) {
    return test_case->min_fft_prime > 0 ? (pfUINT)test_case->min_fft_prime : 17;
}

static pfUINT axis_block_max_fft_prime(const AxisBlockCase* test_case) {
    return test_case->max_fft_prime > 0 ? (pfUINT)test_case->max_fft_prime : 16384;
}

static pfUINT axis_block_complex_size(const AxisBlockCase* test_case) {
    if (test_case->double_double) return 4 * sizeof(double);
    if (test_case->double_precision) return 2 * sizeof(double);
    return 2 * sizeof(float);
}

static const char* axis_block_precision_name(const AxisBlockCase* test_case) {
    if (test_case->double_double) return "double-double";
    if (test_case->double_precision) return "f64";
    if (test_case->half_precision) return "f16-storage-f32-compute";
    return "f32";
}

static pfUINT axis_block_fastest_axis_len(const AxisBlockCase* test_case) {
    return test_case->fastest_axis_len != 0 ? test_case->fastest_axis_len : 8;
}

static pfUINT axis_block_outer_batches(const AxisBlockCase* test_case) {
    return test_case->outer_batches != 0 ? test_case->outer_batches : 5;
}

static void configure_axis_block_app(
    VkFFTApplication* app,
    const AxisBlockCase* test_case,
    pfUINT* temp_buffer_size
) {
    memset(app, 0, sizeof(*app));
    app->configuration.FFTdim = 2;
    app->configuration.size[0] = axis_block_fastest_axis_len(test_case);
    app->configuration.size[1] = test_case->sequence_len;
    app->configuration.vendorID = axis_block_vendor_id(test_case);
    app->configuration.sharedMemorySize = test_case->shared_memory_bytes;
    app->configuration.sharedMemorySizePow2 = test_case->shared_memory_pow2_bytes;
    app->configuration.maxThreadsNum = axis_block_max_threads(test_case);
    app->configuration.maxComputeWorkGroupSize[0] = axis_block_max_workgroup_x(test_case);
    app->configuration.maxComputeWorkGroupSize[1] = axis_block_max_workgroup_y(test_case);
    app->configuration.maxComputeWorkGroupSize[2] = 64;
    app->configuration.coalescedMemory = axis_block_coalesced_memory(test_case);
    app->configuration.aimThreads = 128;
    app->configuration.numSharedBanks = 32;
    app->configuration.warpSize = axis_block_warp_size(test_case);
    app->configuration.registerBoost = axis_block_register_boost(test_case);
    app->configuration.registerBoost4Step = 1;
    app->configuration.registerBoostNonPow2 = 0;
    app->configuration.swapTo3Stage4Step = axis_block_swap_threshold(test_case);
    app->configuration.swapTo2Stage4Step = axis_block_swap_threshold(test_case);
    app->configuration.reorderFourStep = 1;
    if (test_case->double_double) {
        app->configuration.quadDoubleDoublePrecision = 1;
        app->configuration.useLUT = 1;
    } else if (test_case->double_precision) {
        app->configuration.doublePrecision = 1;
    } else if (test_case->half_precision) {
        app->configuration.halfPrecision = 1;
    }
    app->configuration.fixMinRaderPrimeMult = axis_block_min_direct_prime(test_case);
    app->configuration.fixMaxRaderPrimeMult = axis_block_max_direct_prime(test_case);
    app->configuration.fixMinRaderPrimeFFT = axis_block_min_fft_prime(test_case);
    app->configuration.fixMaxRaderPrimeFFT = axis_block_max_fft_prime(test_case);
    app->configuration.coordinateFeatures = 1;
    app->configuration.numberBatches = 1;
    app->configuration.numberKernels = 1;
    app->configuration.tempBufferSize = temp_buffer_size;
    if (!test_case->automatic_grouping) {
        app->configuration.groupedBatch[1] = 3;
    }
    // The default 8 fastest-axis lines x 5 external batches preserves the historical
    // forty-transform snapshots; explicit cases can carry a different real higher-axis extent.
    app->actualNumBatches = axis_block_outer_batches(test_case);
}

static int emit_axis_block_case(const AxisBlockCase* test_case) {
    VkFFTApplication* app = (VkFFTApplication*)calloc(1, sizeof(*app));
    VkFFTPlan* plan = (VkFFTPlan*)calloc(1, sizeof(*plan));
    pfUINT temp_buffer_size = 0;
    if (app == NULL || plan == NULL) {
        fprintf(stderr, "failed to allocate upstream scheduler state\n");
        free(plan);
        free(app);
        return 1;
    }
    configure_axis_block_app(app, test_case, &temp_buffer_size);
    VkFFTResult result = VkFFTScheduler(app, plan, 1);
    if (result != VKFFT_SUCCESS) {
        fprintf(
            stderr,
            "VkFFTScheduler failed for %s with result %d\n",
            test_case->case_name,
            (int)result
        );
        free(plan);
        free(app);
        return 1;
    }
    const pfUINT upload_id = test_case->snapshot_upload_id >= 0
        ? (pfUINT)test_case->snapshot_upload_id
        : 0;
    if (upload_id >= plan->numAxisUploads[1]) {
        fprintf(stderr, "upstream %s is missing requested upload %llu\n", test_case->case_name, (unsigned long long)upload_id);
        free(plan);
        free(app);
        return 1;
    }
    VkFFTAxis* axis = &plan->axes[1][upload_id];
    const pfUINT complex_size = axis_block_complex_size(test_case);
    axis->specializationConstants.complexSize = complex_size;
    axis->specializationConstants.numAxisUploads = (int)plan->numAxisUploads[1];
    axis->specializationConstants.reorderFourStep = plan->numAxisUploads[1] > 1 ? 1 : 0;
    axis->specializationConstants.stageStartSize.type = 31;
    axis->specializationConstants.stageStartSize.data.i = 1;
    for (pfUINT index = 0; index < upload_id; ++index) {
        axis->specializationConstants.stageStartSize.data.i *= plan->axisSplit[1][index];
    }

    pfUINT allowed_shared = test_case->shared_memory_bytes;
    pfUINT allowed_shared_pow2 = test_case->shared_memory_pow2_bytes;
    if (axis->specializationConstants.useRaderMult > 0) {
        const pfUINT reserve =
            (pfUINT)(axis->specializationConstants.useRaderMult - 1) * complex_size;
        allowed_shared -= reserve;
        allowed_shared_pow2 -= reserve;
    }
    result = VkFFTSplitAxisBlock(
        app,
        plan,
        axis,
        1,
        upload_id,
        allowed_shared,
        allowed_shared_pow2
    );
    if (result != VKFFT_SUCCESS) {
        fprintf(
            stderr,
            "VkFFTSplitAxisBlock failed for %s with result %d\n",
            test_case->case_name,
            (int)result
        );
        free(plan);
        free(app);
        return 1;
    }

    const pfUINT higher_axis_batch =
        axis_block_fastest_axis_len(test_case) * axis_block_outer_batches(test_case);
    const pfUINT transform_count =
        test_case->sequence_len * higher_axis_batch / plan->axisSplit[1][upload_id];
    printf(
        "{\"schema_version\":%d,\"upstream_commit\":\"%s\",\"case\":\"%s\",\"kind\":\"axis-block\","
        "\"backend\":\"vulkan\",\"vendor\":\"%s\",\"precision\":\"%s\","
        "\"device\":{\"shared_memory_bytes\":%llu,\"shared_memory_pow2_bytes\":%llu,"
        "\"max_threads_per_block\":%llu,\"max_workgroup_size\":[%llu,%llu,64],"
        "\"coalesced_memory_bytes\":%llu,\"shared_banks\":32,\"supports_f64\":true},"
        "\"payload\":{\"sequence_len\":%llu,\"batch_count\":%llu,\"axis_upload_id\":",
        SNAPSHOT_SCHEMA_VERSION,
        UPSTREAM_COMMIT,
        test_case->case_name,
        axis_block_vendor_name(test_case),
        axis_block_precision_name(test_case),
        (unsigned long long)test_case->shared_memory_bytes,
        (unsigned long long)test_case->shared_memory_pow2_bytes,
        (unsigned long long)app->configuration.maxThreadsNum,
        (unsigned long long)app->configuration.maxComputeWorkGroupSize[0],
        (unsigned long long)app->configuration.maxComputeWorkGroupSize[1],
        (unsigned long long)app->configuration.coalescedMemory,
        (unsigned long long)test_case->sequence_len,
        (unsigned long long)higher_axis_batch
    );
    if (test_case->snapshot_upload_id >= 0) {
        printf("%d", test_case->snapshot_upload_id);
    } else {
        fputs("null", stdout);
    }
    printf(
        ",\"transform_count\":%llu,\"threads_per_transform\":%llu,"
        "\"grouped_batch\":%llu,\"transforms_on_x\":true,\"axis_swapped\":%s,"
        "\"local_size_x\":%llu,\"local_size_y\":%llu}}\n",
        (unsigned long long)transform_count,
        (unsigned long long)axis->axisBlock[1],
        (unsigned long long)axis->groupedBatch,
        axis->specializationConstants.axisSwapped ? "true" : "false",
        (unsigned long long)axis->axisBlock[0],
        (unsigned long long)axis->axisBlock[1]
    );

    free(plan);
    free(app);
    return 0;
}

static int emit_axis0_rader_block_case(const Axis0RaderBlockCase* test_case) {
    VkFFTApplication* app = (VkFFTApplication*)calloc(1, sizeof(*app));
    VkFFTPlan* plan = (VkFFTPlan*)calloc(1, sizeof(*plan));
    pfUINT temp_buffer_size = 0;
    if (app == NULL || plan == NULL) {
        fprintf(stderr, "failed to allocate upstream axis-0 Rader block state\n");
        free(plan);
        free(app);
        return 1;
    }
    configure_axis0_rader_block_app(app, test_case, &temp_buffer_size);
    VkFFTResult result = VkFFTScheduler(app, plan, 0);
    if (result != VKFFT_SUCCESS || plan->numAxisUploads[0] != 1) {
        fprintf(
            stderr,
            "upstream axis-0 Rader scheduler failed for %s (result=%d, uploads=%llu)\n",
            test_case->case_name,
            (int)result,
            (unsigned long long)plan->numAxisUploads[0]
        );
        free(plan);
        free(app);
        return 1;
    }
    if (test_case->expect_direct_rader
        && plan->axes[0][0].specializationConstants.useRaderMult != (int)test_case->sequence_len) {
        fprintf(
            stderr,
            "upstream scheduler did not produce expected p%llu Direct-Rader for %s (useRaderMult=%d)\n",
            (unsigned long long)test_case->sequence_len,
            test_case->case_name,
            plan->axes[0][0].specializationConstants.useRaderMult
        );
        free(plan);
        free(app);
        return 1;
    }

    // Full PlanAxis initialization carries the independent axis-0 batch extent here.
    // The scheduler-only probe must restore it before calling VkFFTSplitAxisBlock.
    plan->actualFFTSizePerAxis[0][1] = test_case->batch_count;
    VkFFTAxis* axis = &plan->axes[0][0];
    const pfUINT complex_size = 2 * sizeof(float);
    axis->specializationConstants.complexSize = complex_size;
    axis->specializationConstants.numAxisUploads = 1;
    axis->specializationConstants.reorderFourStep = 0;
    axis->specializationConstants.stageStartSize.type = 31;
    axis->specializationConstants.stageStartSize.data.i = 1;

    pfUINT allowed_shared = test_case->shared_memory_bytes;
    pfUINT allowed_shared_pow2 = test_case->shared_memory_pow2_bytes;
    if (axis->specializationConstants.useRaderMult > 0) {
        const pfUINT reserve =
            (pfUINT)(axis->specializationConstants.useRaderMult - 1) * complex_size;
        allowed_shared -= reserve;
        allowed_shared_pow2 -= reserve;
    }
    result = VkFFTSplitAxisBlock(
        app,
        plan,
        axis,
        0,
        0,
        allowed_shared,
        allowed_shared_pow2
    );
    if (result != VKFFT_SUCCESS) {
        fprintf(
            stderr,
            "VkFFTSplitAxisBlock failed for %s with result %d\n",
            test_case->case_name,
            (int)result
        );
        free(plan);
        free(app);
        return 1;
    }

    const int axis_swapped = axis->specializationConstants.axisSwapped != 0;
    const pfUINT threads_per_transform = axis_swapped ? axis->axisBlock[1] : axis->axisBlock[0];
    printf(
        "{\"schema_version\":%d,\"upstream_commit\":\"%s\",\"case\":\"%s\",\"kind\":\"axis-block\","
        "\"backend\":\"%s\",\"vendor\":\"%s\",\"precision\":\"%s\","
        "\"device\":{\"shared_memory_bytes\":%llu,\"shared_memory_pow2_bytes\":%llu,"
        "\"max_threads_per_block\":%llu,\"max_workgroup_size\":[%llu,%llu,64],"
        "\"coalesced_memory_bytes\":%llu,\"shared_banks\":32,\"supports_f64\":true},"
        "\"payload\":{\"sequence_len\":%llu,\"batch_count\":%llu,\"axis_upload_id\":null,"
        "\"transform_count\":%llu,\"threads_per_transform\":%llu,\"grouped_batch\":%llu,"
        "\"transforms_on_x\":%s,\"axis_swapped\":%s,\"local_size_x\":%llu,\"local_size_y\":%llu}}\n",
        SNAPSHOT_SCHEMA_VERSION,
        UPSTREAM_COMMIT,
        test_case->case_name,
        axis0_rader_backend_name(test_case),
        axis0_rader_vendor_name(test_case),
        test_case->half_precision ? "f16-storage-f32-compute" : "f32",
        (unsigned long long)test_case->shared_memory_bytes,
        (unsigned long long)test_case->shared_memory_pow2_bytes,
        (unsigned long long)app->configuration.maxThreadsNum,
        (unsigned long long)app->configuration.maxComputeWorkGroupSize[0],
        (unsigned long long)app->configuration.maxComputeWorkGroupSize[1],
        (unsigned long long)app->configuration.coalescedMemory,
        (unsigned long long)test_case->sequence_len,
        (unsigned long long)test_case->batch_count,
        (unsigned long long)test_case->batch_count,
        (unsigned long long)threads_per_transform,
        (unsigned long long)axis->groupedBatch,
        axis_swapped ? "true" : "false",
        axis_swapped ? "true" : "false",
        (unsigned long long)axis->axisBlock[0],
        (unsigned long long)axis->axisBlock[1]
    );

    free(plan);
    free(app);
    return 0;
}

static int emit_axis0_stockham_block_case(const Axis0StockhamBlockCase* test_case) {
    VkFFTApplication* app = (VkFFTApplication*)calloc(1, sizeof(*app));
    VkFFTPlan* plan = (VkFFTPlan*)calloc(1, sizeof(*plan));
    pfUINT temp_buffer_size = 0;
    if (app == NULL || plan == NULL) {
        fprintf(stderr, "failed to allocate upstream axis-0 DD Stockham block state\n");
        free(plan);
        free(app);
        return 1;
    }
    configure_axis0_stockham_block_app(app, test_case, &temp_buffer_size);
    VkFFTResult result = VkFFTScheduler(app, plan, 0);
    if (result != VKFFT_SUCCESS || plan->numAxisUploads[0] != 1) {
        fprintf(
            stderr,
            "upstream axis-0 DD Stockham scheduler failed for %s (result=%d, uploads=%llu)\n",
            test_case->case_name,
            (int)result,
            (unsigned long long)plan->numAxisUploads[0]
        );
        free(plan);
        free(app);
        return 1;
    }

    plan->actualFFTSizePerAxis[0][1] = test_case->batch_count;
    VkFFTAxis* axis = &plan->axes[0][0];
    axis->specializationConstants.complexSize = 4 * sizeof(double);
    axis->specializationConstants.numAxisUploads = 1;
    axis->specializationConstants.reorderFourStep = 0;
    axis->specializationConstants.stageStartSize.type = 31;
    axis->specializationConstants.stageStartSize.data.i = 1;
    result = VkFFTSplitAxisBlock(
        app,
        plan,
        axis,
        0,
        0,
        test_case->shared_memory_bytes,
        test_case->shared_memory_pow2_bytes
    );
    if (result != VKFFT_SUCCESS) {
        fprintf(
            stderr,
            "VkFFTSplitAxisBlock failed for %s with result %d\n",
            test_case->case_name,
            (int)result
        );
        free(plan);
        free(app);
        return 1;
    }

    const int axis_swapped = axis->specializationConstants.axisSwapped != 0;
    const pfUINT threads_per_transform = axis_swapped ? axis->axisBlock[1] : axis->axisBlock[0];
    printf(
        "{\"schema_version\":%d,\"upstream_commit\":\"%s\",\"case\":\"%s\",\"kind\":\"axis-block\","
        "\"backend\":\"vulkan\",\"vendor\":\"nvidia\",\"precision\":\"double-double\","
        "\"device\":{\"shared_memory_bytes\":%llu,\"shared_memory_pow2_bytes\":%llu,"
        "\"max_threads_per_block\":1024,\"max_workgroup_size\":[1024,1024,64],"
        "\"coalesced_memory_bytes\":32,\"shared_banks\":32,\"supports_f64\":true},"
        "\"payload\":{\"sequence_len\":%llu,\"batch_count\":%llu,\"axis_upload_id\":null,"
        "\"transform_count\":%llu,\"threads_per_transform\":%llu,\"grouped_batch\":%llu,"
        "\"transforms_on_x\":%s,\"axis_swapped\":%s,\"local_size_x\":%llu,\"local_size_y\":%llu}}\n",
        SNAPSHOT_SCHEMA_VERSION,
        UPSTREAM_COMMIT,
        test_case->case_name,
        (unsigned long long)test_case->shared_memory_bytes,
        (unsigned long long)test_case->shared_memory_pow2_bytes,
        (unsigned long long)test_case->sequence_len,
        (unsigned long long)test_case->batch_count,
        (unsigned long long)test_case->batch_count,
        (unsigned long long)threads_per_transform,
        (unsigned long long)axis->groupedBatch,
        axis_swapped ? "true" : "false",
        axis_swapped ? "true" : "false",
        (unsigned long long)axis->axisBlock[0],
        (unsigned long long)axis->axisBlock[1]
    );

    free(plan);
    free(app);
    return 0;
}

static pfUINT bluestein_max_threads(const BluesteinCase* test_case) {
    return test_case->max_threads_num != 0 ? test_case->max_threads_num : 1024;
}

static pfUINT bluestein_max_workgroup_x(const BluesteinCase* test_case) {
    return test_case->max_workgroup_x != 0 ? test_case->max_workgroup_x : 1024;
}

static pfUINT bluestein_coalesced_memory(const BluesteinCase* test_case) {
    return test_case->coalesced_memory_bytes != 0 ? test_case->coalesced_memory_bytes : 32;
}

static pfUINT bluestein_batch_count(const BluesteinCase* test_case) {
    return test_case->batch_count != 0 ? test_case->batch_count : 1;
}

static int select_initialized_bluestein_padding(
    const VkFFTApplication* app,
    pfUINT sequence_len,
    pfUINT* padded_len
) {
    const int arr_limit = (int)app->configuration.autoCustomBluesteinPaddingPattern;
    for (int i = 0; i < arr_limit; i++) {
        if (sequence_len < app->configuration.primeSizes[i]) continue;
        if (i != arr_limit - 1) {
            if (sequence_len < app->configuration.primeSizes[i + 1]) {
                *padded_len = app->configuration.paddedSizes[i];
                return 1;
            }
        } else if ((2 * sequence_len - 1) <= app->configuration.paddedSizes[i]) {
            *padded_len = app->configuration.paddedSizes[i];
            return 1;
        }
    }
    return 0;
}

static int emit_bluestein_case(const BluesteinCase* test_case) {
    VkFFTApplication* app = (VkFFTApplication*)calloc(1, sizeof(*app));
    VkFFTPlan* plan = (VkFFTPlan*)calloc(1, sizeof(*plan));
    pfUINT temp_buffer_size = 0;
    if (app == NULL || plan == NULL) {
        fprintf(stderr, "failed to allocate upstream scheduler state\n");
        free(plan);
        free(app);
        return 1;
    }
    app->configuration.FFTdim = 1;
    app->configuration.size[0] = test_case->sequence_len;
    app->configuration.vendorID = test_case->vendor_id;
    app->configuration.sharedMemorySize = test_case->shared_memory_bytes;
    app->configuration.sharedMemorySizePow2 = test_case->shared_memory_pow2_bytes;
    app->configuration.maxThreadsNum = bluestein_max_threads(test_case);
    app->configuration.maxComputeWorkGroupSize[0] = bluestein_max_workgroup_x(test_case);
    app->configuration.maxComputeWorkGroupSize[1] = bluestein_max_workgroup_x(test_case);
    app->configuration.maxComputeWorkGroupSize[2] = 64;
    app->configuration.coalescedMemory = bluestein_coalesced_memory(test_case);
    app->configuration.aimThreads = 128;
    app->configuration.numSharedBanks = 32;
    app->configuration.warpSize = test_case->warp_size;
    app->configuration.registerBoost = test_case->register_boost;
    app->configuration.registerBoost4Step = 1;
    app->configuration.registerBoostNonPow2 = 0;
    app->configuration.swapTo3Stage4Step = test_case->swap_to_three_stage;
    app->configuration.swapTo2Stage4Step = test_case->swap_to_three_stage;
    app->configuration.reorderFourStep = 1;
    app->configuration.fixMinRaderPrimeMult = 17;
    app->configuration.fixMaxRaderPrimeMult = 89;
    app->configuration.fixMinRaderPrimeFFT = 17;
    // Historical probes force Bluestein without changing the padding policy. The F16 p47
    // witness retains the normal FFT-Rader ceiling so the scheduler's coalescing-limited
    // Direct-Rader cap is what forces Bluestein.
    app->configuration.fixMaxRaderPrimeFFT = test_case->keep_default_rader_fft_max ? 16384 : 100;
    app->configuration.coordinateFeatures = 1;
    app->configuration.numberBatches = bluestein_batch_count(test_case);
    app->configuration.numberKernels = 1;
    app->configuration.tempBufferSize = &temp_buffer_size;
    app->actualNumBatches = bluestein_batch_count(test_case);
    if (test_case->half_precision) {
        app->configuration.halfPrecision = 1;
    }
    if (test_case->wide_precision) {
        app->configuration.doublePrecision = 1;
        app->configuration.useLUT = 1;
    }

    pfUINT padding_only_len = 0;
    VkFFTResult result = initializeBluesteinAutoPadding(app);
    if (result == VKFFT_SUCCESS) {
        if (test_case->padding_only) {
            if (!select_initialized_bluestein_padding(app, test_case->sequence_len, &padding_only_len)) {
                fprintf(stderr, "upstream Bluestein padding table did not cover %s\n", test_case->case_name);
                free(app->configuration.primeSizes);
                free(app->configuration.paddedSizes);
                free(plan);
                free(app);
                return 1;
            }
        } else {
            result = VkFFTScheduler(app, plan, 0);
        }
    }
    if (result != VKFFT_SUCCESS || (!test_case->padding_only && !app->useBluesteinFFT[0])) {
        fprintf(
            stderr,
            "upstream Bluestein scheduler/padding probe failed for %s (result=%d, enabled=%llu)\n",
            test_case->case_name,
            (int)result,
            (unsigned long long)app->useBluesteinFFT[0]
        );
        free(app->configuration.primeSizes);
        free(app->configuration.paddedSizes);
        free(plan);
        free(app);
        return 1;
    }

    printf(
        "{\"schema_version\":%d,\"upstream_commit\":\"%s\",\"case\":\"%s\",\"kind\":\"bluestein\","
        "\"backend\":\"%s\",\"vendor\":\"%s\",\"precision\":\"%s\","
        "\"device\":{\"shared_memory_bytes\":%llu,\"shared_memory_pow2_bytes\":%llu,"
        "\"max_threads_per_block\":%llu,\"max_workgroup_size\":[%llu,%llu,64],"
        "\"coalesced_memory_bytes\":%llu,\"shared_banks\":32,\"supports_f64\":true},"
        "\"payload\":{\"sequence_len\":%llu,\"batch_count\":%llu,\"padded_len\":%llu}}\n",
        SNAPSHOT_SCHEMA_VERSION,
        UPSTREAM_COMMIT,
        test_case->case_name,
        test_case->backend_name != NULL ? test_case->backend_name : "vulkan",
        test_case->vendor_name,
        test_case->half_precision ? "f16-storage-f32-compute" : (test_case->wide_precision ? "f64" : "f32"),
        (unsigned long long)test_case->shared_memory_bytes,
        (unsigned long long)test_case->shared_memory_pow2_bytes,
        (unsigned long long)bluestein_max_threads(test_case),
        (unsigned long long)bluestein_max_workgroup_x(test_case),
        (unsigned long long)bluestein_max_workgroup_x(test_case),
        (unsigned long long)bluestein_coalesced_memory(test_case),
        (unsigned long long)test_case->sequence_len,
        (unsigned long long)bluestein_batch_count(test_case),
        (unsigned long long)(test_case->padding_only ? padding_only_len : plan->actualFFTSizePerAxis[0][0])
    );

    free(app->configuration.primeSizes);
    free(app->configuration.paddedSizes);
    free(plan);
    free(app);
    return 0;
}

static void configure_device_dd_bluestein_app(
    VkFFTApplication* app,
    const DeviceDoubleDoubleBluesteinCase* test_case,
    pfUINT* temp_buffer_size
) {
    memset(app, 0, sizeof(*app));
    app->configuration.FFTdim = 1;
    app->configuration.size[0] = test_case->sequence_len;
    app->configuration.vendorID = test_case->vendor_id;
    app->configuration.sharedMemorySize = test_case->shared_memory_bytes;
    app->configuration.sharedMemorySizePow2 = test_case->shared_memory_pow2_bytes;
    app->configuration.maxThreadsNum = test_case->max_threads_num;
    app->configuration.maxComputeWorkGroupSize[0] = test_case->max_workgroup_x;
    app->configuration.maxComputeWorkGroupSize[1] = test_case->max_workgroup_x;
    app->configuration.maxComputeWorkGroupSize[2] = 64;
    app->configuration.coalescedMemory = test_case->coalesced_memory_bytes;
    app->configuration.aimThreads = 128;
    app->configuration.numSharedBanks = 32;
    app->configuration.warpSize = test_case->warp_size;
    app->configuration.registerBoost = test_case->register_boost;
    app->configuration.registerBoost4Step = 1;
    app->configuration.registerBoostNonPow2 = 0;
    app->configuration.swapTo3Stage4Step = test_case->swap_threshold;
    app->configuration.swapTo2Stage4Step = test_case->swap_threshold;
    app->configuration.reorderFourStep = 1;
    app->configuration.quadDoubleDoublePrecision = 1;
    app->configuration.useLUT = 1;
    app->configuration.fixMinRaderPrimeMult = test_case->min_direct_prime;
    app->configuration.fixMaxRaderPrimeMult = test_case->max_direct_prime;
    app->configuration.fixMinRaderPrimeFFT = test_case->min_fft_prime;
    app->configuration.fixMaxRaderPrimeFFT = test_case->max_fft_prime;
    app->configuration.coordinateFeatures = 1;
    app->configuration.numberBatches = 1;
    app->configuration.numberKernels = 1;
    app->configuration.tempBufferSize = temp_buffer_size;
    app->actualNumBatches = 1;
}

static int emit_device_dd_bluestein_case(const DeviceDoubleDoubleBluesteinCase* test_case) {
    VkFFTApplication* app = (VkFFTApplication*)calloc(1, sizeof(*app));
    VkFFTPlan* plan = (VkFFTPlan*)calloc(1, sizeof(*plan));
    pfUINT temp_buffer_size = 0;
    if (app == NULL || plan == NULL) {
        fprintf(stderr, "failed to allocate upstream DD Bluestein scheduler state\n");
        free(plan);
        free(app);
        return 1;
    }
    configure_device_dd_bluestein_app(app, test_case, &temp_buffer_size);

    VkFFTResult result = initializeBluesteinAutoPadding(app);
    if (result == VKFFT_SUCCESS) {
        result = VkFFTScheduler(app, plan, 0);
    }
    if (result != VKFFT_SUCCESS || !app->useBluesteinFFT[0]) {
        fprintf(
            stderr,
            "upstream device-default DD Bluestein scheduler failed for %s (result=%d, enabled=%llu)\n",
            test_case->case_name,
            (int)result,
            (unsigned long long)app->useBluesteinFFT[0]
        );
        free(app->configuration.primeSizes);
        free(app->configuration.paddedSizes);
        free(plan);
        free(app);
        return 1;
    }

    printf(
        "{\"schema_version\":%d,\"upstream_commit\":\"%s\",\"case\":\"%s\",\"kind\":\"bluestein\","
        "\"backend\":\"%s\",\"vendor\":\"%s\",\"precision\":\"double-double\","
        "\"device\":{\"shared_memory_bytes\":%llu,\"shared_memory_pow2_bytes\":%llu,"
        "\"max_threads_per_block\":%llu,\"max_workgroup_size\":[%llu,%llu,64],"
        "\"coalesced_memory_bytes\":%llu,\"shared_banks\":32,\"supports_f64\":true},"
        "\"payload\":{\"sequence_len\":%llu,\"batch_count\":1,\"padded_len\":%llu}}\n",
        SNAPSHOT_SCHEMA_VERSION,
        UPSTREAM_COMMIT,
        test_case->case_name,
        test_case->backend_name,
        test_case->vendor_name,
        (unsigned long long)test_case->shared_memory_bytes,
        (unsigned long long)test_case->shared_memory_pow2_bytes,
        (unsigned long long)test_case->max_threads_num,
        (unsigned long long)test_case->max_workgroup_x,
        (unsigned long long)test_case->max_workgroup_x,
        (unsigned long long)test_case->coalesced_memory_bytes,
        (unsigned long long)test_case->sequence_len,
        (unsigned long long)plan->actualFFTSizePerAxis[0][0]
    );

    free(app->configuration.primeSizes);
    free(app->configuration.paddedSizes);
    free(plan);
    free(app);
    return 0;
}

static int emit_device_dd_bluestein_stockham_reference(
    const DeviceDoubleDoubleBluesteinStockhamReference* reference
) {
    const size_t case_count = sizeof(DEVICE_DD_BLUESTEIN_CASES) / sizeof(DEVICE_DD_BLUESTEIN_CASES[0]);
    if (reference->bluestein_case_index >= case_count || reference->fastest_axis_len == 0) {
        fprintf(stderr, "invalid DD Bluestein Stockham case metadata for %s\n", reference->case_name);
        return 1;
    }
    const DeviceDoubleDoubleBluesteinCase* test_case =
        &DEVICE_DD_BLUESTEIN_CASES[reference->bluestein_case_index];
    VkFFTApplication* app = (VkFFTApplication*)calloc(1, sizeof(*app));
    VkFFTPlan* plan = (VkFFTPlan*)calloc(1, sizeof(*plan));
    pfUINT temp_buffer_size = 0;
    if (app == NULL || plan == NULL) {
        fprintf(stderr, "failed to allocate upstream DD Bluestein Stockham state\n");
        free(plan);
        free(app);
        return 1;
    }
    configure_device_dd_bluestein_app(app, test_case, &temp_buffer_size);
    app->configuration.FFTdim = 2;
    app->configuration.size[0] = reference->fastest_axis_len;
    app->configuration.size[1] = test_case->sequence_len;
    app->configuration.performBandwidthBoost = reference->perform_bandwidth_boost;

    VkFFTResult result = initializeBluesteinAutoPadding(app);
    if (result == VKFFT_SUCCESS) {
        result = VkFFTScheduler(app, plan, 1);
    }
    if (result != VKFFT_SUCCESS || !app->useBluesteinFFT[1] || plan->numAxisUploads[1] == 0) {
        fprintf(
            stderr,
            "upstream DD Bluestein Stockham scheduler failed for %s (result=%d, enabled=%llu, uploads=%llu)\n",
            reference->case_name,
            (int)result,
            (unsigned long long)app->useBluesteinFFT[1],
            (unsigned long long)plan->numAxisUploads[1]
        );
        free(app->configuration.primeSizes);
        free(app->configuration.paddedSizes);
        free(plan);
        free(app);
        return 1;
    }

    const pfUINT fft_dim = plan->actualFFTSizePerAxis[1][1];
    const pfUINT used_shared =
        (fft_dim & (fft_dim - 1)) == 0
            ? test_case->shared_memory_pow2_bytes
            : test_case->shared_memory_bytes;
    const pfUINT complex_size = 4 * sizeof(double);
    const pfUINT max_shared = used_shared / complex_size;
    const pfUINT max_strided = test_case->coalesced_memory_bytes > complex_size
        ? used_shared / test_case->coalesced_memory_bytes
        : max_shared;
    const int schedule_boost = plan->axes[1][0].specializationConstants.registerBoost;
    const pfUINT transform_count = reference->fastest_axis_len * app->actualNumBatches;

    printf(
        "{\"schema_version\":%d,\"upstream_commit\":\"%s\",\"case\":\"%s\",\"kind\":\"dd-stockham\","
        "\"backend\":\"%s\",\"vendor\":\"%s\",\"precision\":\"double-double\","
        "\"device\":{\"shared_memory_bytes\":%llu,\"shared_memory_pow2_bytes\":%llu,"
        "\"max_threads_per_block\":%llu,\"max_workgroup_size\":[%llu,%llu,64],"
        "\"coalesced_memory_bytes\":%llu,\"shared_banks\":32,\"supports_f64\":true},"
        "\"payload\":{\"sequence_len\":%llu,\"batch_count\":%llu,"
        "\"used_shared_memory_bytes\":%llu,\"max_sequence_len_shared\":%llu,"
        "\"max_sequence_len_strided\":%llu,\"register_boost\":%d,"
        "\"upload_count\":%llu,\"axis_split\":[",
        SNAPSHOT_SCHEMA_VERSION,
        UPSTREAM_COMMIT,
        reference->case_name,
        test_case->backend_name,
        test_case->vendor_name,
        (unsigned long long)test_case->shared_memory_bytes,
        (unsigned long long)test_case->shared_memory_pow2_bytes,
        (unsigned long long)test_case->max_threads_num,
        (unsigned long long)test_case->max_workgroup_x,
        (unsigned long long)test_case->max_workgroup_x,
        (unsigned long long)test_case->coalesced_memory_bytes,
        (unsigned long long)fft_dim,
        (unsigned long long)transform_count,
        (unsigned long long)used_shared,
        (unsigned long long)max_shared,
        (unsigned long long)max_strided,
        schedule_boost,
        (unsigned long long)plan->numAxisUploads[1]
    );
    for (pfUINT upload = 0; upload < plan->numAxisUploads[1]; ++upload) {
        if (upload != 0) {
            fputc(',', stdout);
        }
        printf("%llu", (unsigned long long)plan->axisSplit[1][upload]);
    }
    fputs("]}}\n", stdout);

    free(app->configuration.primeSizes);
    free(app->configuration.paddedSizes);
    free(plan);
    free(app);
    return 0;
}

static int emit_device_dd_bluestein_axis_block_reference(
    const DeviceDoubleDoubleBluesteinAxisBlockReference* reference
) {
    const size_t case_count = sizeof(DEVICE_DD_BLUESTEIN_CASES) / sizeof(DEVICE_DD_BLUESTEIN_CASES[0]);
    if (reference->bluestein_case_index >= case_count || reference->axis_id > 1) {
        fprintf(stderr, "invalid DD Bluestein axis-block case metadata for %s\n", reference->case_name);
        return 1;
    }
    const DeviceDoubleDoubleBluesteinCase* test_case =
        &DEVICE_DD_BLUESTEIN_CASES[reference->bluestein_case_index];
    VkFFTApplication* app = (VkFFTApplication*)calloc(1, sizeof(*app));
    VkFFTPlan* plan = (VkFFTPlan*)calloc(1, sizeof(*plan));
    pfUINT temp_buffer_size = 0;
    if (app == NULL || plan == NULL) {
        fprintf(stderr, "failed to allocate upstream DD Bluestein axis-block state\n");
        free(plan);
        free(app);
        return 1;
    }
    configure_device_dd_bluestein_app(app, test_case, &temp_buffer_size);
    const pfUINT axis_id = reference->axis_id;
    if (axis_id > 0) {
        app->configuration.FFTdim = 2;
        app->configuration.size[0] = reference->fastest_axis_len;
        app->configuration.size[1] = test_case->sequence_len;
    }
    VkFFTResult result = initializeBluesteinAutoPadding(app);
    if (result == VKFFT_SUCCESS) {
        result = VkFFTScheduler(app, plan, axis_id);
    }
    const pfUINT upload = reference->upload_id;
    if (result != VKFFT_SUCCESS || !app->useBluesteinFFT[axis_id] || upload >= plan->numAxisUploads[axis_id]) {
        fprintf(
            stderr,
            "upstream DD Bluestein axis-block scheduler failed for %s (result=%d, enabled=%llu, uploads=%llu)\n",
            reference->case_name,
            (int)result,
            (unsigned long long)app->useBluesteinFFT[axis_id],
            (unsigned long long)plan->numAxisUploads[axis_id]
        );
        free(app->configuration.primeSizes);
        free(app->configuration.paddedSizes);
        free(plan);
        free(app);
        return 1;
    }

    VkFFTAxis* axis = &plan->axes[axis_id][upload];
    const pfUINT complex_size = 4 * sizeof(double);
    axis->specializationConstants.complexSize = complex_size;
    axis->specializationConstants.numAxisUploads = (int)plan->numAxisUploads[axis_id];
    axis->specializationConstants.useBluesteinFFT = 1;
    axis->specializationConstants.reorderFourStep = 0;
    axis->specializationConstants.stageStartSize.type = 31;
    axis->specializationConstants.stageStartSize.data.i = 1;
    for (pfUINT prior_upload = 0; prior_upload < upload; ++prior_upload) {
        axis->specializationConstants.stageStartSize.data.i *= plan->axisSplit[axis_id][prior_upload];
    }
    pfUINT allowed_shared = test_case->shared_memory_bytes;
    pfUINT allowed_shared_pow2 = test_case->shared_memory_pow2_bytes;
    if (axis->specializationConstants.useRaderMult > 0) {
        const pfUINT reserve =
            (pfUINT)(axis->specializationConstants.useRaderMult - 1) * complex_size;
        allowed_shared -= reserve;
        allowed_shared_pow2 -= reserve;
    }
    result = VkFFTSplitAxisBlock(
        app,
        plan,
        axis,
        axis_id,
        upload,
        allowed_shared,
        allowed_shared_pow2
    );
    if (result != VKFFT_SUCCESS) {
        fprintf(stderr, "VkFFTSplitAxisBlock failed for %s with result %d\n", reference->case_name, (int)result);
        free(app->configuration.primeSizes);
        free(app->configuration.paddedSizes);
        free(plan);
        free(app);
        return 1;
    }
    const pfUINT fft_dim = (pfUINT)axis->specializationConstants.fftDim.data.i;
    const pfUINT padded_len = axis_id == 0 ? plan->actualFFTSizePerAxis[0][0] : fft_dim;
    if (fft_dim == 0 || (axis_id == 0 && padded_len % fft_dim != 0)) {
        fprintf(stderr, "invalid DD Bluestein component geometry for %s\n", reference->case_name);
        free(app->configuration.primeSizes);
        free(app->configuration.paddedSizes);
        free(plan);
        free(app);
        return 1;
    }
    const pfUINT transform_count =
        axis_id == 0 ? padded_len / fft_dim : reference->fastest_axis_len;
    const int axis_swapped = axis->specializationConstants.axisSwapped != 0;
    const int transforms_on_x = axis_id > 0 || upload > 0 || axis_swapped;
    const pfUINT threads_per_transform =
        transforms_on_x ? axis->axisBlock[1] : axis->axisBlock[0];
    printf(
        "{\"schema_version\":%d,\"upstream_commit\":\"%s\",\"case\":\"%s\",\"kind\":\"axis-block\","
        "\"backend\":\"%s\",\"vendor\":\"%s\",\"precision\":\"double-double\","
        "\"device\":{\"shared_memory_bytes\":%llu,\"shared_memory_pow2_bytes\":%llu,"
        "\"max_threads_per_block\":%llu,\"max_workgroup_size\":[%llu,%llu,64],"
        "\"coalesced_memory_bytes\":%llu,\"shared_banks\":32,\"supports_f64\":true},"
        "\"payload\":{\"sequence_len\":%llu,\"batch_count\":%llu,\"axis_upload_id\":%llu,"
        "\"transform_count\":%llu,\"threads_per_transform\":%llu,\"grouped_batch\":%llu,"
        "\"transforms_on_x\":%s,\"axis_swapped\":%s,\"local_size_x\":%llu,\"local_size_y\":%llu}}\n",
        SNAPSHOT_SCHEMA_VERSION,
        UPSTREAM_COMMIT,
        reference->case_name,
        test_case->backend_name,
        test_case->vendor_name,
        (unsigned long long)test_case->shared_memory_bytes,
        (unsigned long long)test_case->shared_memory_pow2_bytes,
        (unsigned long long)test_case->max_threads_num,
        (unsigned long long)test_case->max_workgroup_x,
        (unsigned long long)test_case->max_workgroup_x,
        (unsigned long long)test_case->coalesced_memory_bytes,
        (unsigned long long)fft_dim,
        (unsigned long long)transform_count,
        (unsigned long long)upload,
        (unsigned long long)transform_count,
        (unsigned long long)threads_per_transform,
        (unsigned long long)axis->groupedBatch,
        transforms_on_x ? "true" : "false",
        axis_swapped ? "true" : "false",
        (unsigned long long)axis->axisBlock[0],
        (unsigned long long)axis->axisBlock[1]
    );

    free(app->configuration.primeSizes);
    free(app->configuration.paddedSizes);
    free(plan);
    free(app);
    return 0;
}

static const BluesteinCase* find_bluestein_case(const char* name) {
    const size_t count = sizeof(BLUESTEIN_CASES) / sizeof(BLUESTEIN_CASES[0]);
    for (size_t index = 0; index < count; ++index) {
        if (strcmp(name, BLUESTEIN_CASES[index].case_name) == 0) {
            return &BLUESTEIN_CASES[index];
        }
    }
    return NULL;
}

static const DeviceDoubleDoubleBluesteinCase* find_device_dd_bluestein_case(const char* name) {
    const size_t count = sizeof(DEVICE_DD_BLUESTEIN_CASES) / sizeof(DEVICE_DD_BLUESTEIN_CASES[0]);
    for (size_t index = 0; index < count; ++index) {
        if (strcmp(name, DEVICE_DD_BLUESTEIN_CASES[index].case_name) == 0) {
            return &DEVICE_DD_BLUESTEIN_CASES[index];
        }
    }
    return NULL;
}

static const DeviceDoubleDoubleBluesteinAxisBlockReference*
find_device_dd_bluestein_axis_block_reference(const char* name) {
    const size_t count = sizeof(DEVICE_DD_BLUESTEIN_AXIS_BLOCK_REFERENCES) /
        sizeof(DEVICE_DD_BLUESTEIN_AXIS_BLOCK_REFERENCES[0]);
    for (size_t index = 0; index < count; ++index) {
        if (strcmp(name, DEVICE_DD_BLUESTEIN_AXIS_BLOCK_REFERENCES[index].case_name) == 0) {
            return &DEVICE_DD_BLUESTEIN_AXIS_BLOCK_REFERENCES[index];
        }
    }
    return NULL;
}

static const DeviceDoubleDoubleBluesteinStockhamReference*
find_device_dd_bluestein_stockham_reference(const char* name) {
    const size_t count = sizeof(DEVICE_DD_BLUESTEIN_STOCKHAM_REFERENCES) /
        sizeof(DEVICE_DD_BLUESTEIN_STOCKHAM_REFERENCES[0]);
    for (size_t index = 0; index < count; ++index) {
        if (strcmp(name, DEVICE_DD_BLUESTEIN_STOCKHAM_REFERENCES[index].case_name) == 0) {
            return &DEVICE_DD_BLUESTEIN_STOCKHAM_REFERENCES[index];
        }
    }
    return NULL;
}

static const Axis0StockhamBlockCase* find_axis0_stockham_block_case(const char* name) {
    const size_t count = sizeof(AXIS0_STOCKHAM_BLOCK_CASES) / sizeof(AXIS0_STOCKHAM_BLOCK_CASES[0]);
    for (size_t index = 0; index < count; ++index) {
        if (strcmp(name, AXIS0_STOCKHAM_BLOCK_CASES[index].case_name) == 0) {
            return &AXIS0_STOCKHAM_BLOCK_CASES[index];
        }
    }
    return NULL;
}

static const Axis0RaderBlockCase* find_axis0_rader_block_case(const char* name) {
    const size_t count = sizeof(AXIS0_RADER_BLOCK_CASES) / sizeof(AXIS0_RADER_BLOCK_CASES[0]);
    for (size_t index = 0; index < count; ++index) {
        if (strcmp(name, AXIS0_RADER_BLOCK_CASES[index].case_name) == 0) {
            return &AXIS0_RADER_BLOCK_CASES[index];
        }
    }
    return NULL;
}

static const AxisBlockCase* find_axis_block_case(const char* name) {
    const size_t count = sizeof(AXIS_BLOCK_CASES) / sizeof(AXIS_BLOCK_CASES[0]);
    for (size_t index = 0; index < count; ++index) {
        if (strcmp(name, AXIS_BLOCK_CASES[index].case_name) == 0) {
            return &AXIS_BLOCK_CASES[index];
        }
    }
    return NULL;
}

static const RealShapeCase* find_real_shape_case(const char* name) {
    const size_t count = sizeof(REAL_SHAPE_CASES) / sizeof(REAL_SHAPE_CASES[0]);
    for (size_t index = 0; index < count; ++index) {
        if (strcmp(name, REAL_SHAPE_CASES[index].case_name) == 0) {
            return &REAL_SHAPE_CASES[index];
        }
    }
    return NULL;
}

static const StockhamCase* find_stockham_case(const char* name) {
    const size_t count = sizeof(STOCKHAM_CASES) / sizeof(STOCKHAM_CASES[0]);
    for (size_t index = 0; index < count; ++index) {
        if (strcmp(name, STOCKHAM_CASES[index].case_name) == 0) {
            return &STOCKHAM_CASES[index];
        }
    }
    return NULL;
}

static const MixedRaderParentCase* find_mixed_rader_parent_case(const char* name) {
    const size_t count = sizeof(MIXED_RADER_PARENT_CASES) / sizeof(MIXED_RADER_PARENT_CASES[0]);
    for (size_t index = 0; index < count; ++index) {
        if (strcmp(name, MIXED_RADER_PARENT_CASES[index].case_name) == 0) {
            return &MIXED_RADER_PARENT_CASES[index];
        }
    }
    return NULL;
}

static const RaderUploadAxisBlockReference* find_rader_upload_axis_block_reference(const char* name) {
    const size_t count = sizeof(RADER_UPLOAD_AXIS_BLOCK_REFERENCES) / sizeof(RADER_UPLOAD_AXIS_BLOCK_REFERENCES[0]);
    for (size_t index = 0; index < count; ++index) {
        if (strcmp(name, RADER_UPLOAD_AXIS_BLOCK_REFERENCES[index].case_name) == 0) {
            return &RADER_UPLOAD_AXIS_BLOCK_REFERENCES[index];
        }
    }
    return NULL;
}

static const ForcedAxisBlockUploadReference* find_forced_axis_block_upload_reference(const char* name) {
    const size_t count = sizeof(FORCED_AXIS_BLOCK_UPLOAD_REFERENCES) / sizeof(FORCED_AXIS_BLOCK_UPLOAD_REFERENCES[0]);
    for (size_t index = 0; index < count; ++index) {
        if (strcmp(name, FORCED_AXIS_BLOCK_UPLOAD_REFERENCES[index].case_name) == 0) {
            return &FORCED_AXIS_BLOCK_UPLOAD_REFERENCES[index];
        }
    }
    return NULL;
}

static const ForcedAxisBlockCase* find_forced_axis_block_case(const char* name) {
    const size_t count = sizeof(FORCED_AXIS_BLOCK_CASES) / sizeof(FORCED_AXIS_BLOCK_CASES[0]);
    for (size_t index = 0; index < count; ++index) {
        if (strcmp(name, FORCED_AXIS_BLOCK_CASES[index].case_name) == 0) {
            return &FORCED_AXIS_BLOCK_CASES[index];
        }
    }
    return NULL;
}

static const RaderParentCase* find_rader_parent_case(const char* name) {
    const size_t count = sizeof(RADER_PARENT_CASES) / sizeof(RADER_PARENT_CASES[0]);
    for (size_t index = 0; index < count; ++index) {
        if (strcmp(name, RADER_PARENT_CASES[index].case_name) == 0) {
            return &RADER_PARENT_CASES[index];
        }
    }
    return NULL;
}

static const RaderUploadCase* find_rader_upload_case(const char* name) {
    const size_t count = sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]);
    for (size_t index = 0; index < count; ++index) {
        if (strcmp(name, RADER_UPLOAD_CASES[index].case_name) == 0) {
            return &RADER_UPLOAD_CASES[index];
        }
    }
    return NULL;
}

static const RaderCase* find_case(const char* name) {
    const size_t count = sizeof(RADER_CASES) / sizeof(RADER_CASES[0]);
    for (size_t index = 0; index < count; ++index) {
        if (strcmp(name, RADER_CASES[index].case_name) == 0) {
            return &RADER_CASES[index];
        }
    }
    return NULL;
}

static void list_case_names(void) {
    const size_t rader_count = sizeof(RADER_CASES) / sizeof(RADER_CASES[0]);
    for (size_t index = 0; index < rader_count; ++index) {
        puts(RADER_CASES[index].case_name);
    }
    const size_t rader_upload_count = sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]);
    for (size_t index = 0; index < rader_upload_count; ++index) {
        puts(RADER_UPLOAD_CASES[index].case_name);
    }
    const size_t rader_upload_axis_block_count =
        sizeof(RADER_UPLOAD_AXIS_BLOCK_REFERENCES) / sizeof(RADER_UPLOAD_AXIS_BLOCK_REFERENCES[0]);
    for (size_t index = 0; index < rader_upload_axis_block_count; ++index) {
        puts(RADER_UPLOAD_AXIS_BLOCK_REFERENCES[index].case_name);
    }
    const size_t rader_parent_count = sizeof(RADER_PARENT_CASES) / sizeof(RADER_PARENT_CASES[0]);
    for (size_t index = 0; index < rader_parent_count; ++index) {
        puts(RADER_PARENT_CASES[index].case_name);
    }
    const size_t mixed_rader_parent_count = sizeof(MIXED_RADER_PARENT_CASES) / sizeof(MIXED_RADER_PARENT_CASES[0]);
    for (size_t index = 0; index < mixed_rader_parent_count; ++index) {
        puts(MIXED_RADER_PARENT_CASES[index].case_name);
    }
    const size_t forced_axis_upload_count = sizeof(FORCED_AXIS_BLOCK_UPLOAD_REFERENCES) / sizeof(FORCED_AXIS_BLOCK_UPLOAD_REFERENCES[0]);
    for (size_t index = 0; index < forced_axis_upload_count; ++index) {
        puts(FORCED_AXIS_BLOCK_UPLOAD_REFERENCES[index].case_name);
    }
    const size_t stockham_count = sizeof(STOCKHAM_CASES) / sizeof(STOCKHAM_CASES[0]);
    for (size_t index = 0; index < stockham_count; ++index) {
        puts(STOCKHAM_CASES[index].case_name);
    }
    const size_t real_shape_count = sizeof(REAL_SHAPE_CASES) / sizeof(REAL_SHAPE_CASES[0]);
    for (size_t index = 0; index < real_shape_count; ++index) {
        puts(REAL_SHAPE_CASES[index].case_name);
    }
    const size_t stockham_axis_block_count =
        sizeof(STOCKHAM_AXIS_BLOCK_REFERENCES) / sizeof(STOCKHAM_AXIS_BLOCK_REFERENCES[0]);
    for (size_t index = 0; index < stockham_axis_block_count; ++index) {
        puts(STOCKHAM_AXIS_BLOCK_REFERENCES[index].case_name);
    }
    const size_t axis0_stockham_block_count =
        sizeof(AXIS0_STOCKHAM_BLOCK_CASES) / sizeof(AXIS0_STOCKHAM_BLOCK_CASES[0]);
    for (size_t index = 0; index < axis0_stockham_block_count; ++index) {
        puts(AXIS0_STOCKHAM_BLOCK_CASES[index].case_name);
    }
    const size_t axis0_rader_block_count =
        sizeof(AXIS0_RADER_BLOCK_CASES) / sizeof(AXIS0_RADER_BLOCK_CASES[0]);
    for (size_t index = 0; index < axis0_rader_block_count; ++index) {
        puts(AXIS0_RADER_BLOCK_CASES[index].case_name);
    }
    const size_t axis_block_count = sizeof(AXIS_BLOCK_CASES) / sizeof(AXIS_BLOCK_CASES[0]);
    for (size_t index = 0; index < axis_block_count; ++index) {
        puts(AXIS_BLOCK_CASES[index].case_name);
    }
    const size_t bluestein_count = sizeof(BLUESTEIN_CASES) / sizeof(BLUESTEIN_CASES[0]);
    for (size_t index = 0; index < bluestein_count; ++index) {
        puts(BLUESTEIN_CASES[index].case_name);
    }
    const size_t device_dd_bluestein_count =
        sizeof(DEVICE_DD_BLUESTEIN_CASES) / sizeof(DEVICE_DD_BLUESTEIN_CASES[0]);
    for (size_t index = 0; index < device_dd_bluestein_count; ++index) {
        puts(DEVICE_DD_BLUESTEIN_CASES[index].case_name);
    }
    const size_t device_dd_bluestein_axis_block_count =
        sizeof(DEVICE_DD_BLUESTEIN_AXIS_BLOCK_REFERENCES) /
        sizeof(DEVICE_DD_BLUESTEIN_AXIS_BLOCK_REFERENCES[0]);
    for (size_t index = 0; index < device_dd_bluestein_axis_block_count; ++index) {
        puts(DEVICE_DD_BLUESTEIN_AXIS_BLOCK_REFERENCES[index].case_name);
    }
    const size_t device_dd_bluestein_stockham_count =
        sizeof(DEVICE_DD_BLUESTEIN_STOCKHAM_REFERENCES) /
        sizeof(DEVICE_DD_BLUESTEIN_STOCKHAM_REFERENCES[0]);
    for (size_t index = 0; index < device_dd_bluestein_stockham_count; ++index) {
        puts(DEVICE_DD_BLUESTEIN_STOCKHAM_REFERENCES[index].case_name);
    }
}

static int parse_sweep_uint(const char* text, pfUINT* value) {
    if ((text == NULL) || (text[0] == '\0') || (text[0] == '-')) return 1;
    char* end = NULL;
    const unsigned long long parsed = strtoull(text, &end, 10);
    if ((end == text) || (end == NULL) || (*end != '\0')) return 1;
    *value = (pfUINT)parsed;
    return 0;
}

static int configure_sweep_profile(RaderUploadCase* test_case, const char* profile) {
    if (strcmp(profile, "nv-vk") == 0) {
        test_case->profile_kind = 0;
        return 0;
    }
    if (strcmp(profile, "amd-vk") == 0) {
        test_case->profile_kind = 1;
        return 0;
    }
    if (strcmp(profile, "intel-cl") == 0) {
        test_case->profile_kind = 2;
        return 0;
    }
    if (strcmp(profile, "intel-vk") == 0) {
        test_case->profile_kind = 3;
        return 0;
    }
    if (strcmp(profile, "intel-l0") == 0) {
        test_case->profile_kind = 4;
        return 0;
    }
    return 1;
}

static int configure_sweep_precision(RaderUploadCase* test_case, const char* precision) {
    if (strcmp(precision, "f32") == 0) return 0;
    if (strcmp(precision, "f16") == 0) {
        test_case->half_precision = 1;
        return 0;
    }
    if (strcmp(precision, "f64f32") == 0) {
        test_case->double_precision_float_memory = 1;
        return 0;
    }
    if (strcmp(precision, "dd") == 0) {
        test_case->double_double = 1;
        return 0;
    }
    return 1;
}

static int emit_rader_upload_probe_file(const char* path, int mode) {
    FILE* input = fopen(path, "r");
    if (input == NULL) {
        fprintf(stderr, "failed to open Rader-upload probe file: %s\n", path);
        return 2;
    }
    char line[2048];
    size_t line_number = 0;
    size_t emitted = 0;
    while (fgets(line, sizeof(line), input) != NULL) {
        ++line_number;
        char* cursor = line;
        while ((*cursor == ' ') || (*cursor == '\t')) ++cursor;
        if ((*cursor == '\0') || (*cursor == '\n') || (*cursor == '#')) continue;

        char* fields[21] = {0};
        size_t field_count = 0;
        char* token = strtok(cursor, "\t\r\n");
        while ((token != NULL) && (field_count < 21)) {
            fields[field_count++] = token;
            token = strtok(NULL, "\t\r\n");
        }
        if (((field_count != 15) && (field_count != 17) && (field_count != 18) && (field_count != 21)) || (token != NULL)) {
            fprintf(stderr, "%s:%zu: expected 15, 17, 18, or 21 tab-separated fields\n", path, line_number);
            fclose(input);
            return 2;
        }

        RaderUploadCase test_case;
        memset(&test_case, 0, sizeof(test_case));
        test_case.case_name = fields[0];
        if (configure_sweep_profile(&test_case, fields[1]) != 0) {
            fprintf(stderr, "%s:%zu: unsupported profile %s\n", path, line_number, fields[1]);
            fclose(input);
            return 2;
        }
        if (configure_sweep_precision(&test_case, fields[2]) != 0) {
            fprintf(stderr, "%s:%zu: unsupported precision %s\n", path, line_number, fields[2]);
            fclose(input);
            return 2;
        }

        pfUINT common[8] = {0};
        for (size_t index = 0; index < 8; ++index) {
            const size_t field_index = index + 3;
            if (parse_sweep_uint(fields[field_index], &common[index]) != 0) {
                fprintf(
                    stderr,
                    "%s:%zu: invalid integer field %zu: %s\n",
                    path,
                    line_number,
                    field_index + 1,
                    fields[field_index]
                );
                fclose(input);
                return 2;
            }
        }
        test_case.sequence_len = common[0];
        test_case.shared_memory_bytes = common[1];
        test_case.shared_memory_pow2_bytes = common[2];
        test_case.max_threads_num = common[3];
        test_case.max_workgroup_x = common[4];
        test_case.max_workgroup_y = common[4];
        test_case.strided_axis = (int)common[5];
        test_case.fastest_axis_len = common[6];
        test_case.bandwidth_boost = (int)common[7];

        size_t tuning_offset = 11;
        test_case.batch_count = 1;
        if ((field_count == 17) || (field_count == 18) || (field_count == 21)) {
            if ((parse_sweep_uint(fields[11], &test_case.batch_count) != 0) ||
                (parse_sweep_uint(fields[12], &test_case.grouped_batch_override) != 0)) {
                fprintf(stderr, "%s:%zu: invalid batch/groupedBatch field\n", path, line_number);
                fclose(input);
                return 2;
            }
            if (test_case.batch_count == 0) {
                fprintf(stderr, "%s:%zu: batch_count must be non-zero\n", path, line_number);
                fclose(input);
                return 2;
            }
            tuning_offset = 13;
        }
        if ((field_count == 18) || (field_count == 21)) {
            pfUINT perform_zero_padding = 0;
            if ((parse_sweep_uint(fields[13], &perform_zero_padding) != 0) ||
                (perform_zero_padding > 1)) {
                fprintf(stderr, "%s:%zu: zero_padding must be 0 or 1\n", path, line_number);
                fclose(input);
                return 2;
            }
            test_case.perform_zero_padding = (int)perform_zero_padding;
            tuning_offset = 14;
        }
        if (field_count == 21) {
            if ((parse_sweep_uint(fields[14], &test_case.axis_id_override) != 0) ||
                (parse_sweep_uint(fields[15], &test_case.middle_axis_len) != 0) ||
                (parse_sweep_uint(fields[16], &test_case.axis1_grouped_batch_override) != 0)) {
                fprintf(stderr, "%s:%zu: invalid axis2/groupedBatch[1] context field\n", path, line_number);
                fclose(input);
                return 2;
            }
            if ((test_case.axis_id_override != 2) || (test_case.middle_axis_len == 0) ||
                (test_case.strided_axis == 0)) {
                fprintf(stderr, "%s:%zu: 21-field probe requires strided axis_id=2 and middle_axis_len > 0\n", path, line_number);
                fclose(input);
                return 2;
            }
            tuning_offset = 17;
        }
        pfUINT tuning[4] = {0};
        for (size_t index = 0; index < 4; ++index) {
            const size_t field_index = tuning_offset + index;
            if (parse_sweep_uint(fields[field_index], &tuning[index]) != 0) {
                fprintf(
                    stderr,
                    "%s:%zu: invalid integer field %zu: %s\n",
                    path,
                    line_number,
                    field_index + 1,
                    fields[field_index]
                );
                fclose(input);
                return 2;
            }
        }
        test_case.min_direct_prime = (int)tuning[0];
        test_case.max_direct_prime = (int)tuning[1];
        test_case.min_fft_prime = (int)tuning[2];
        test_case.max_fft_prime = (int)tuning[3];
        if ((test_case.strided_axis != 0) && (test_case.fastest_axis_len == 0)) {
            fprintf(stderr, "%s:%zu: strided probe requires fastest_axis_len > 0\n", path, line_number);
            fclose(input);
            return 2;
        }
        const int emit_result = (mode == 1)
            ? emit_parameterized_axis_classification_case(&test_case)
            : ((mode == 2)
                ? emit_parameterized_axis_block_case(&test_case)
                : emit_parameterized_rader_upload_case(&test_case));
        if (emit_result == 3) {
            continue;
        }
        if (emit_result != 0) {
            fclose(input);
            return 1;
        }
        ++emitted;
    }
    if (ferror(input)) {
        fprintf(stderr, "failed while reading Rader-upload probe file: %s\n", path);
        fclose(input);
        return 2;
    }
    fclose(input);
    if (emitted == 0) {
        fprintf(stderr, "Rader-upload probe file contains no cases: %s\n", path);
        return 2;
    }
    return 0;
}

static void print_help(const char* executable) {
    printf(
        "Usage: %s [--case NAME | --list-cases | --rader-upload-file PATH | --classification-file PATH | --axis-block-file PATH]\n"
        "Emit scheduler JSONL directly from pinned upstream VkFFT headers.\n"
        "Currently supported cases include Rader/upload/parent/AxisBlock references, Stockham references, forced F32 Bluestein references, device-default DD Bluestein references, and DD higher-axis Bluestein Stockham references.\n",
        executable
    );
}

int main(int argc, char** argv) {
    if (argc == 1) {
        const size_t rader_count = sizeof(RADER_CASES) / sizeof(RADER_CASES[0]);
        for (size_t index = 0; index < rader_count; ++index) {
            if (emit_rader_case(&RADER_CASES[index]) != 0) {
                return 1;
            }
        }
        const size_t rader_upload_count = sizeof(RADER_UPLOAD_CASES) / sizeof(RADER_UPLOAD_CASES[0]);
        for (size_t index = 0; index < rader_upload_count; ++index) {
            if (emit_rader_upload_case(&RADER_UPLOAD_CASES[index]) != 0) {
                return 1;
            }
        }
        const size_t rader_upload_axis_block_count =
            sizeof(RADER_UPLOAD_AXIS_BLOCK_REFERENCES) / sizeof(RADER_UPLOAD_AXIS_BLOCK_REFERENCES[0]);
        for (size_t index = 0; index < rader_upload_axis_block_count; ++index) {
            if (emit_rader_upload_axis_block_reference(&RADER_UPLOAD_AXIS_BLOCK_REFERENCES[index]) != 0) {
                return 1;
            }
        }
        const size_t rader_parent_count = sizeof(RADER_PARENT_CASES) / sizeof(RADER_PARENT_CASES[0]);
        for (size_t index = 0; index < rader_parent_count; ++index) {
            if (emit_rader_parent_case(&RADER_PARENT_CASES[index]) != 0) {
                return 1;
            }
        }
        const size_t mixed_rader_parent_count = sizeof(MIXED_RADER_PARENT_CASES) / sizeof(MIXED_RADER_PARENT_CASES[0]);
        for (size_t index = 0; index < mixed_rader_parent_count; ++index) {
            if (emit_mixed_rader_parent_case(&MIXED_RADER_PARENT_CASES[index]) != 0) {
                return 1;
            }
        }
        const size_t forced_axis_upload_count = sizeof(FORCED_AXIS_BLOCK_UPLOAD_REFERENCES) / sizeof(FORCED_AXIS_BLOCK_UPLOAD_REFERENCES[0]);
        for (size_t index = 0; index < forced_axis_upload_count; ++index) {
            if (emit_forced_axis_block_upload_reference(&FORCED_AXIS_BLOCK_UPLOAD_REFERENCES[index]) != 0) {
                return 1;
            }
        }
        const size_t stockham_count = sizeof(STOCKHAM_CASES) / sizeof(STOCKHAM_CASES[0]);
        for (size_t index = 0; index < stockham_count; ++index) {
            if (emit_stockham_case(&STOCKHAM_CASES[index]) != 0) {
                return 1;
            }
        }
        const size_t real_shape_count = sizeof(REAL_SHAPE_CASES) / sizeof(REAL_SHAPE_CASES[0]);
        for (size_t index = 0; index < real_shape_count; ++index) {
            if (emit_real_shape_case(&REAL_SHAPE_CASES[index]) != 0) {
                return 1;
            }
        }
        const size_t stockham_axis_block_count =
            sizeof(STOCKHAM_AXIS_BLOCK_REFERENCES) / sizeof(STOCKHAM_AXIS_BLOCK_REFERENCES[0]);
        for (size_t index = 0; index < stockham_axis_block_count; ++index) {
            if (emit_stockham_axis_block_reference(&STOCKHAM_AXIS_BLOCK_REFERENCES[index]) != 0) {
                return 1;
            }
        }
        const size_t axis0_stockham_block_count =
            sizeof(AXIS0_STOCKHAM_BLOCK_CASES) / sizeof(AXIS0_STOCKHAM_BLOCK_CASES[0]);
        for (size_t index = 0; index < axis0_stockham_block_count; ++index) {
            if (emit_axis0_stockham_block_case(&AXIS0_STOCKHAM_BLOCK_CASES[index]) != 0) {
                return 1;
            }
        }
        const size_t axis0_rader_block_count =
            sizeof(AXIS0_RADER_BLOCK_CASES) / sizeof(AXIS0_RADER_BLOCK_CASES[0]);
        for (size_t index = 0; index < axis0_rader_block_count; ++index) {
            if (emit_axis0_rader_block_case(&AXIS0_RADER_BLOCK_CASES[index]) != 0) {
                return 1;
            }
        }
        const size_t axis_block_count = sizeof(AXIS_BLOCK_CASES) / sizeof(AXIS_BLOCK_CASES[0]);
        for (size_t index = 0; index < axis_block_count; ++index) {
            if (emit_axis_block_case(&AXIS_BLOCK_CASES[index]) != 0) {
                return 1;
            }
        }
        const size_t bluestein_count = sizeof(BLUESTEIN_CASES) / sizeof(BLUESTEIN_CASES[0]);
        for (size_t index = 0; index < bluestein_count; ++index) {
            if (emit_bluestein_case(&BLUESTEIN_CASES[index]) != 0) {
                return 1;
            }
        }
        const size_t device_dd_bluestein_count =
            sizeof(DEVICE_DD_BLUESTEIN_CASES) / sizeof(DEVICE_DD_BLUESTEIN_CASES[0]);
        for (size_t index = 0; index < device_dd_bluestein_count; ++index) {
            if (emit_device_dd_bluestein_case(&DEVICE_DD_BLUESTEIN_CASES[index]) != 0) {
                return 1;
            }
        }
        const size_t device_dd_bluestein_axis_block_count =
            sizeof(DEVICE_DD_BLUESTEIN_AXIS_BLOCK_REFERENCES) /
            sizeof(DEVICE_DD_BLUESTEIN_AXIS_BLOCK_REFERENCES[0]);
        for (size_t index = 0; index < device_dd_bluestein_axis_block_count; ++index) {
            if (emit_device_dd_bluestein_axis_block_reference(
                    &DEVICE_DD_BLUESTEIN_AXIS_BLOCK_REFERENCES[index]
                ) != 0) {
                return 1;
            }
        }
        const size_t device_dd_bluestein_stockham_count =
            sizeof(DEVICE_DD_BLUESTEIN_STOCKHAM_REFERENCES) /
            sizeof(DEVICE_DD_BLUESTEIN_STOCKHAM_REFERENCES[0]);
        for (size_t index = 0; index < device_dd_bluestein_stockham_count; ++index) {
            if (emit_device_dd_bluestein_stockham_reference(
                    &DEVICE_DD_BLUESTEIN_STOCKHAM_REFERENCES[index]
                ) != 0) {
                return 1;
            }
        }
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--list-cases") == 0) {
        list_case_names();
        return 0;
    }
    if (argc == 3 && strcmp(argv[1], "--rader-upload-file") == 0) {
        return emit_rader_upload_probe_file(argv[2], 0);
    }
    if (argc == 3 && strcmp(argv[1], "--classification-file") == 0) {
        return emit_rader_upload_probe_file(argv[2], 1);
    }
    if (argc == 3 && strcmp(argv[1], "--axis-block-file") == 0) {
        return emit_rader_upload_probe_file(argv[2], 2);
    }
    if (argc == 2 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        print_help(argv[0]);
        return 0;
    }
    if (argc == 3 && strcmp(argv[1], "--case") == 0) {
        const RaderCase* rader_case = find_case(argv[2]);
        if (rader_case != NULL) {
            return emit_rader_case(rader_case);
        }
        const RaderUploadCase* rader_upload_case = find_rader_upload_case(argv[2]);
        if (rader_upload_case != NULL) {
            return emit_rader_upload_case(rader_upload_case);
        }
        const RaderUploadAxisBlockReference* rader_upload_axis_block_reference =
            find_rader_upload_axis_block_reference(argv[2]);
        if (rader_upload_axis_block_reference != NULL) {
            return emit_rader_upload_axis_block_reference(rader_upload_axis_block_reference);
        }
        const ForcedAxisBlockUploadReference* forced_axis_upload_reference = find_forced_axis_block_upload_reference(argv[2]);
        if (forced_axis_upload_reference != NULL) {
            return emit_forced_axis_block_upload_reference(forced_axis_upload_reference);
        }
        const ForcedAxisBlockCase* forced_axis_block_case = find_forced_axis_block_case(argv[2]);
        if (forced_axis_block_case != NULL) {
            return emit_forced_axis_block_case(forced_axis_block_case);
        }
        const RaderParentCase* rader_parent_case = find_rader_parent_case(argv[2]);
        if (rader_parent_case != NULL) {
            return emit_rader_parent_case(rader_parent_case);
        }
        const MixedRaderParentCase* mixed_rader_parent_case = find_mixed_rader_parent_case(argv[2]);
        if (mixed_rader_parent_case != NULL) {
            return emit_mixed_rader_parent_case(mixed_rader_parent_case);
        }
        const StockhamCase* stockham_case = find_stockham_case(argv[2]);
        if (stockham_case != NULL) {
            return emit_stockham_case(stockham_case);
        }
        const RealShapeCase* real_shape_case = find_real_shape_case(argv[2]);
        if (real_shape_case != NULL) {
            return emit_real_shape_case(real_shape_case);
        }
        const size_t stockham_axis_block_count =
            sizeof(STOCKHAM_AXIS_BLOCK_REFERENCES) / sizeof(STOCKHAM_AXIS_BLOCK_REFERENCES[0]);
        for (size_t index = 0; index < stockham_axis_block_count; ++index) {
            if (strcmp(argv[2], STOCKHAM_AXIS_BLOCK_REFERENCES[index].case_name) == 0) {
                return emit_stockham_axis_block_reference(&STOCKHAM_AXIS_BLOCK_REFERENCES[index]);
            }
        }
        const Axis0StockhamBlockCase* axis0_stockham_block_case =
            find_axis0_stockham_block_case(argv[2]);
        if (axis0_stockham_block_case != NULL) {
            return emit_axis0_stockham_block_case(axis0_stockham_block_case);
        }
        const Axis0RaderBlockCase* axis0_rader_block_case =
            find_axis0_rader_block_case(argv[2]);
        if (axis0_rader_block_case != NULL) {
            return emit_axis0_rader_block_case(axis0_rader_block_case);
        }
        const AxisBlockCase* axis_block_case = find_axis_block_case(argv[2]);
        if (axis_block_case != NULL) {
            return emit_axis_block_case(axis_block_case);
        }
        const BluesteinCase* bluestein_case = find_bluestein_case(argv[2]);
        if (bluestein_case != NULL) {
            return emit_bluestein_case(bluestein_case);
        }
        const DeviceDoubleDoubleBluesteinCase* device_dd_bluestein_case =
            find_device_dd_bluestein_case(argv[2]);
        if (device_dd_bluestein_case != NULL) {
            return emit_device_dd_bluestein_case(device_dd_bluestein_case);
        }
        const DeviceDoubleDoubleBluesteinAxisBlockReference* device_dd_bluestein_axis_block_reference =
            find_device_dd_bluestein_axis_block_reference(argv[2]);
        if (device_dd_bluestein_axis_block_reference != NULL) {
            return emit_device_dd_bluestein_axis_block_reference(
                device_dd_bluestein_axis_block_reference
            );
        }
        const DeviceDoubleDoubleBluesteinStockhamReference* device_dd_bluestein_stockham_reference =
            find_device_dd_bluestein_stockham_reference(argv[2]);
        if (device_dd_bluestein_stockham_reference != NULL) {
            return emit_device_dd_bluestein_stockham_reference(
                device_dd_bluestein_stockham_reference
            );
        }
        fprintf(stderr, "unsupported upstream scheduler reference case: %s\n", argv[2]);
        return 2;
    }
    print_help(argv[0]);
    return 2;
}
