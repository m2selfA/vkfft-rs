// Policy-only reference extractor for the pinned upstream VkFFT initialization source.
//
// The runner extracts the policy assignment statements verbatim from
// vkFFT_InitializeApp.h into generated *.inc files. This helper supplies only the
// minimal fake device/application state needed to execute those upstream expressions;
// scheduler policy constants are intentionally not duplicated here.
#include <cstdint>
#include <cstdio>
#include <cstring>

#define SNAPSHOT_SCHEMA_VERSION 1
#define UPSTREAM_COMMIT "066a17c17068c0f11c9298d848c2976c71fad1c1"

struct ReferenceConfiguration {
    std::uint64_t sharedMemorySize = 0;
    std::uint64_t coalescedMemory = 0;
    std::uint64_t warpSize = 0;
    std::uint64_t registerBoost = 0;
    std::uint64_t registerBoost4Step = 0;
    std::uint64_t swapTo2Stage4Step = 0;
    std::uint64_t swapTo3Stage4Step = 0;
    std::uint64_t vendorID = 0;
    int halfPrecision = 0;
    int doublePrecision = 0;
    int doublePrecisionFloatMemory = 0;
    int quadDoubleDoublePrecision = 0;
    int quadDoubleDoublePrecisionDoubleMemory = 0;
    int useLUT = 0;
    int useLUT_4step = 0;
    int useRaderUintLUT = 0;
    int registerBoostNonPow2 = 0;
    int reorderFourStep = 0;
};

struct ReferenceApplication {
    ReferenceConfiguration configuration;
};

struct ReferenceInputConfiguration {
    int useLUT = 0;
    int useLUT_4step = 0;
    int disableReorderFourStep = 0;
};

struct ReferenceVulkanLimits {
    std::uint64_t maxComputeSharedMemorySize = 0;
};

struct ReferenceVulkanProperties {
    std::uint32_t vendorID = 0;
    ReferenceVulkanLimits limits;
};

struct ReferenceLevelZeroProperties {
    std::uint32_t physicalEUSimdWidth = 0;
};

struct ReferenceMetalState {
    std::uint32_t width = 0;
    std::uint32_t threadExecutionWidth() const { return width; }
};

enum class ReferenceBackend {
    Vulkan,
    Cuda,
    Hip,
    OpenCl,
    LevelZero,
    Metal,
};

struct PolicyCase {
    const char* case_name;
    const char* backend_name;
    const char* vendor_name;
    ReferenceBackend backend;
    std::uint32_t vendor_id;
    bool f64;
    std::uint64_t shared_memory_bytes;
    std::uint64_t shared_memory_pow2_bytes;
    std::uint64_t max_threads_per_block;
    std::uint64_t max_workgroup_x;
    std::uint64_t max_workgroup_y;
    std::uint64_t max_workgroup_z;
    std::uint64_t device_coalesced_memory_bytes;
    bool supports_f64;
    std::uint32_t simulated_subgroup_width;
    bool f16 = false;
    bool f64_f32_storage = false;
    bool double_double = false;
    bool double_double_f64_storage = false;
};

static const PolicyCase POLICY_CASES[] = {
    {"policy-nvidia-vulkan-f32-48k", "vulkan", "nvidia", ReferenceBackend::Vulkan,
     0x10DE, false, 48 * 1024, 32 * 1024, 1024, 1024, 1024, 64, 32, true, 32},
    {"policy-nvidia-vulkan-f64-48k", "vulkan", "nvidia", ReferenceBackend::Vulkan,
     0x10DE, true, 48 * 1024, 32 * 1024, 1024, 1024, 1024, 64, 32, true, 32},
    {"policy-nvidia-vulkan-f16-48k", "vulkan", "nvidia", ReferenceBackend::Vulkan,
     0x10DE, false, 48 * 1024, 32 * 1024, 1024, 1024, 1024, 64, 32, true, 32, true},
    {"policy-nvidia-cuda-f32-48k", "cuda", "nvidia", ReferenceBackend::Cuda,
     0x10DE, false, 48 * 1024, 32 * 1024, 1024, 1024, 1024, 64, 32, true, 32},
    {"policy-nvidia-opencl-f32-48k", "opencl", "nvidia", ReferenceBackend::OpenCl,
     0x10DE, false, 48 * 1024, 32 * 1024, 1024, 1024, 1024, 64, 32, true, 32},
    {"policy-amd-vulkan-f32-48k", "vulkan", "amd", ReferenceBackend::Vulkan,
     0x1002, false, 48 * 1024, 32 * 1024, 1024, 1024, 1024, 64, 32, true, 64},
    {"policy-amd-vulkan-f16-48k", "vulkan", "amd", ReferenceBackend::Vulkan,
     0x1002, false, 48 * 1024, 32 * 1024, 1024, 1024, 1024, 64, 32, true, 64, true},
    {"policy-amd-vulkan-f32-64k", "vulkan", "amd", ReferenceBackend::Vulkan,
     0x1002, false, 64 * 1024, 64 * 1024, 1024, 1024, 1024, 64, 32, true, 64},
    {"policy-amd-vulkan-f64-f32-storage-64k", "vulkan", "amd", ReferenceBackend::Vulkan,
     0x1002, false, 64 * 1024, 64 * 1024, 1024, 1024, 1024, 64, 32, true, 64, false, true},
    {"policy-intel-vulkan-f64-f32-storage-64k", "vulkan", "intel", ReferenceBackend::Vulkan,
     0x8086, false, 64 * 1024, 64 * 1024, 1024, 1024, 1024, 64, 64, true, 32, false, true},
    {"policy-nvidia-vulkan-dd-64k", "vulkan", "nvidia", ReferenceBackend::Vulkan,
     0x10DE, false, 64 * 1024, 64 * 1024, 1024, 1024, 1024, 64, 32, true, 32, false, false, true},
    {"policy-amd-vulkan-dd-64k", "vulkan", "amd", ReferenceBackend::Vulkan,
     0x1002, false, 64 * 1024, 64 * 1024, 1024, 1024, 1024, 64, 32, true, 64, false, false, true},
    {"policy-intel-vulkan-dd-64k", "vulkan", "intel", ReferenceBackend::Vulkan,
     0x8086, false, 64 * 1024, 64 * 1024, 1024, 1024, 1024, 64, 64, true, 32, false, false, true},
    {"policy-amd-vulkan-dd-f64-storage-64k", "vulkan", "amd", ReferenceBackend::Vulkan,
     0x1002, false, 64 * 1024, 64 * 1024, 1024, 1024, 1024, 64, 32, true, 64, false, false, false, true},
    {"policy-amd-hip-f32-64k", "hip", "amd", ReferenceBackend::Hip,
     0x1002, false, 64 * 1024, 64 * 1024, 1024, 1024, 1024, 64, 32, true, 64},
    {"policy-amd-hip-f32-64k-wave32", "hip", "amd", ReferenceBackend::Hip,
     0x1002, false, 64 * 1024, 64 * 1024, 1024, 1024, 1024, 64, 32, true, 32},
    {"policy-amd-hip-f64-64k", "hip", "amd", ReferenceBackend::Hip,
     0x1002, true, 64 * 1024, 64 * 1024, 1024, 1024, 1024, 64, 32, true, 64},
    {"policy-amd-hip-f64-f32-storage-64k", "hip", "amd", ReferenceBackend::Hip,
     0x1002, false, 64 * 1024, 64 * 1024, 1024, 1024, 1024, 64, 32, true, 64, false, true},
    {"policy-amd-hip-dd-64k", "hip", "amd", ReferenceBackend::Hip,
     0x1002, false, 64 * 1024, 64 * 1024, 1024, 1024, 1024, 64, 32, true, 64, false, false, true},
    {"policy-amd-hip-dd-f64-storage-64k", "hip", "amd", ReferenceBackend::Hip,
     0x1002, false, 64 * 1024, 64 * 1024, 1024, 1024, 1024, 64, 32, true, 64, false, false, false, true},
    {"policy-intel-level-zero-f32", "level-zero", "intel", ReferenceBackend::LevelZero,
     0x8086, false, 32 * 1024, 32 * 1024, 256, 256, 256, 256, 32, true, 1},
    {"policy-intel-level-zero-f16", "level-zero", "intel", ReferenceBackend::LevelZero,
     0x8086, false, 32 * 1024, 32 * 1024, 256, 256, 256, 256, 32, true, 1, true},
    {"policy-intel-level-zero-f64-f32-storage-64k", "level-zero", "intel", ReferenceBackend::LevelZero,
     0x8086, false, 64 * 1024, 64 * 1024, 256, 256, 256, 256, 64, true, 1, false, true},
    {"policy-intel-level-zero-dd-64k", "level-zero", "intel", ReferenceBackend::LevelZero,
     0x8086, false, 64 * 1024, 64 * 1024, 256, 256, 256, 256, 64, true, 1, false, false, true},
    {"policy-intel-opencl-f32", "opencl", "intel", ReferenceBackend::OpenCl,
     0x8086, false, 32 * 1024, 32 * 1024, 256, 256, 256, 256, 32, true, 32},
    {"policy-intel-opencl-f16", "opencl", "intel", ReferenceBackend::OpenCl,
     0x8086, false, 32 * 1024, 32 * 1024, 256, 256, 256, 256, 32, true, 32, true},
    {"policy-intel-opencl-f64-f32-storage-64k", "opencl", "intel", ReferenceBackend::OpenCl,
     0x8086, false, 64 * 1024, 64 * 1024, 256, 256, 256, 256, 64, true, 32, false, true},
    {"policy-amd-opencl-f64-f32-storage-64k", "opencl", "amd", ReferenceBackend::OpenCl,
     0x1002, false, 64 * 1024, 64 * 1024, 1024, 1024, 1024, 64, 32, true, 64, false, true},
    {"policy-amd-opencl-dd-64k", "opencl", "amd", ReferenceBackend::OpenCl,
     0x1002, false, 64 * 1024, 64 * 1024, 1024, 1024, 1024, 64, 32, true, 64, false, false, true},
    {"policy-intel-opencl-dd-64k", "opencl", "intel", ReferenceBackend::OpenCl,
     0x8086, false, 64 * 1024, 64 * 1024, 256, 256, 256, 256, 64, true, 32, false, false, true},
    {"policy-apple-metal-f64-f32-storage-32k", "metal", "apple", ReferenceBackend::Metal,
     0x1027f00, false, 32 * 1024, 32 * 1024, 256, 256, 256, 256, 32, false, 1, false, true},
    {"policy-apple-metal-dd-32k", "metal", "apple", ReferenceBackend::Metal,
     0x1027f00, false, 32 * 1024, 32 * 1024, 256, 256, 256, 256, 32, false, 1, false, false, true},
    {"policy-apple-metal-f32", "metal", "apple", ReferenceBackend::Metal,
     0x1027f00, false, 32 * 1024, 32 * 1024, 256, 256, 256, 256, 32, false, 1},
};

static const PolicyCase* find_case(const char* name) {
    for (const PolicyCase& test_case : POLICY_CASES) {
        if (std::strcmp(name, test_case.case_name) == 0) {
            return &test_case;
        }
    }
    return nullptr;
}

static const char* twiddle_source(int use_lut) {
    return use_lut > 0 ? "lookup-table" : "on-the-fly";
}

static int emit_case(const PolicyCase& test_case) {
    ReferenceApplication storage{};
    ReferenceApplication* app = &storage;
    ReferenceInputConfiguration inputLaunchConfiguration{};
    app->configuration.sharedMemorySize = test_case.shared_memory_bytes;
    app->configuration.doublePrecision = test_case.f64 ? 1 : 0;
    app->configuration.doublePrecisionFloatMemory = test_case.f64_f32_storage ? 1 : 0;
    app->configuration.halfPrecision = test_case.f16 ? 1 : 0;
    app->configuration.quadDoubleDoublePrecision = test_case.double_double ? 1 : 0;
    app->configuration.quadDoubleDoublePrecisionDoubleMemory =
        test_case.double_double_f64_storage ? 1 : 0;

    switch (test_case.backend) {
        case ReferenceBackend::Vulkan: {
            ReferenceVulkanProperties physicalDeviceProperties{};
            physicalDeviceProperties.vendorID = test_case.vendor_id;
            physicalDeviceProperties.limits.maxComputeSharedMemorySize =
                test_case.shared_memory_bytes;
#include "upstream_policy_vulkan.inc"
            break;
        }
        case ReferenceBackend::Cuda: {
            // CUDA obtains warpSize through a device attribute before the fixed policy
            // assignments extracted below.
            app->configuration.warpSize = test_case.simulated_subgroup_width;
#include "upstream_policy_cuda.inc"
            break;
        }
        case ReferenceBackend::Hip: {
            // HIP likewise queries warpSize before the fixed policy assignments.
            app->configuration.warpSize = test_case.simulated_subgroup_width;
#include "upstream_policy_hip.inc"
            break;
        }
        case ReferenceBackend::OpenCl: {
            std::uint32_t vendorID = test_case.vendor_id;
            std::uint64_t sharedMemorySize = test_case.shared_memory_bytes;
#include "upstream_policy_opencl.inc"
            break;
        }
        case ReferenceBackend::LevelZero: {
            ReferenceLevelZeroProperties device_properties{};
            device_properties.physicalEUSimdWidth = test_case.simulated_subgroup_width;
#include "upstream_policy_level_zero.inc"
            break;
        }
        case ReferenceBackend::Metal: {
            ReferenceMetalState metal_state{test_case.simulated_subgroup_width};
            ReferenceMetalState* dummy_state = &metal_state;
#include "upstream_policy_metal.inc"
            break;
        }
    }

#include "upstream_policy_lut_finalize.inc"
#include "upstream_policy_reorder.inc"

    const char* precision_name = test_case.f16
        ? "f16-storage-f32-compute"
        : (test_case.f64_f32_storage
            ? "f64-compute-f32-storage"
            : (test_case.double_double
                ? "double-double"
                : (test_case.double_double_f64_storage
                    ? "double-double-f64-storage"
                    : (test_case.f64 ? "f64" : "f32"))));
    std::printf(
        "{\"schema_version\":%d,\"upstream_commit\":\"%s\",\"case\":\"%s\","
        "\"kind\":\"policy\",\"backend\":\"%s\",\"vendor\":\"%s\","
        "\"precision\":\"%s\",\"device\":{\"shared_memory_bytes\":%llu,"
        "\"shared_memory_pow2_bytes\":%llu,\"max_threads_per_block\":%llu,"
        "\"max_workgroup_size\":[%llu,%llu,%llu],\"coalesced_memory_bytes\":%llu,"
        "\"shared_banks\":32,\"supports_f64\":%s},\"payload\":{"
        "\"register_boost\":%llu,\"register_boost_four_step\":%llu,"
        "\"register_boost_non_power_of_two\":%s,\"reorder_four_step\":%s,"
        "\"swap_to_two_stage_four_step\":%llu,\"swap_to_three_stage_four_step\":%llu,"
        "\"subgroup_width\":%llu,\"coalesced_memory_bytes\":%llu,"
        "\"stockham_twiddle_source\":\"%s\",\"four_step_twiddle_source\":\"%s\"}}\n",
        SNAPSHOT_SCHEMA_VERSION,
        UPSTREAM_COMMIT,
        test_case.case_name,
        test_case.backend_name,
        test_case.vendor_name,
        precision_name,
        static_cast<unsigned long long>(test_case.shared_memory_bytes),
        static_cast<unsigned long long>(test_case.shared_memory_pow2_bytes),
        static_cast<unsigned long long>(test_case.max_threads_per_block),
        static_cast<unsigned long long>(test_case.max_workgroup_x),
        static_cast<unsigned long long>(test_case.max_workgroup_y),
        static_cast<unsigned long long>(test_case.max_workgroup_z),
        static_cast<unsigned long long>(test_case.device_coalesced_memory_bytes),
        test_case.supports_f64 ? "true" : "false",
        static_cast<unsigned long long>(app->configuration.registerBoost),
        static_cast<unsigned long long>(app->configuration.registerBoost4Step),
        app->configuration.registerBoostNonPow2 ? "true" : "false",
        app->configuration.reorderFourStep ? "true" : "false",
        static_cast<unsigned long long>(app->configuration.swapTo2Stage4Step),
        static_cast<unsigned long long>(app->configuration.swapTo3Stage4Step),
        static_cast<unsigned long long>(app->configuration.warpSize),
        static_cast<unsigned long long>(app->configuration.coalescedMemory),
        twiddle_source(app->configuration.useLUT),
        twiddle_source(app->configuration.useLUT_4step));
    return 0;
}

static void list_cases() {
    for (const PolicyCase& test_case : POLICY_CASES) {
        std::puts(test_case.case_name);
    }
}

int main(int argc, char** argv) {
    if (argc == 1) {
        for (const PolicyCase& test_case : POLICY_CASES) {
            if (emit_case(test_case) != 0) {
                return 1;
            }
        }
        return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--list-cases") == 0) {
        list_cases();
        return 0;
    }
    if (argc == 3 && std::strcmp(argv[1], "--case") == 0) {
        const PolicyCase* test_case = find_case(argv[2]);
        if (test_case == nullptr) {
            std::fprintf(stderr, "unsupported upstream scheduler policy case: %s\n", argv[2]);
            return 2;
        }
        return emit_case(*test_case);
    }
    std::fprintf(stderr, "Usage: %s [--case NAME | --list-cases]\n", argv[0]);
    return 2;
}
