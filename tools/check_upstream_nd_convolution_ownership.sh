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

plan="$upstream_dir/vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_Plans/vkFFT_Plan_FFT.h"
run_app="$upstream_dir/vkFFT/vkFFT/vkFFT_AppManagement/vkFFT_RunApp.h"
initialize="$upstream_dir/vkFFT/vkFFT/vkFFT_AppManagement/vkFFT_InitializeApp.h"
convolution="$upstream_dir/vkFFT/vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel1/PrePostProcessing/vkFFT_Convolution.h"
read_write="$upstream_dir/vkFFT/vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel1/vkFFT_ReadWrite.h"
zeropad="$upstream_dir/vkFFT/vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_Zeropad.h"
r2c_plan="$upstream_dir/vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_Plans/vkFFT_Plan_R2C.h"
structs="$upstream_dir/vkFFT/vkFFT/vkFFT_Structs/vkFFT_Structs.h"
update_buffers="$upstream_dir/vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_UpdateBuffers.h"
api_params="$upstream_dir/vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_InitAPIParameters.h"
sample15="$upstream_dir/benchmark_scripts/vkFFT_scripts/src/sample_15_precision_VkFFT_single_r2c.cpp"
sample52="$upstream_dir/benchmark_scripts/vkFFT_scripts/src/sample_52_convolution_VkFFT_single_2d_batched_r2c.cpp"
sample51="$upstream_dir/benchmark_scripts/vkFFT_scripts/src/sample_51_convolution_VkFFT_single_3d_matrix_zeropadding_r2c.cpp"

# Fixed-commit ownership contract used by NdConvolutionIr:
# 1. only the final FFT dimension/upload0 is tagged convolutionStep;
# 2. RunApp walks the complete forward ND transform before that midpoint and then
#    walks the remaining inverse dimensions backwards;
# 3. the convolution kernel address incorporates non-current FFT dimensions before
#    kernel data is loaded, so the midpoint multiply owns a complete ND frequency tensor;
# 4. sequence conjugation happens before multiplication and cross-power normalization
#    uses the pinned norm -> reciprocal-square-root -> scale order at that same midpoint;
# 5. matrix mode promotes coordinateFeatures to the matrix size, runs the last forward
#    upload0 as one fused coordinate system, and the convolution body performs the full
#    row-by-column coordinate sum before the inverse side uses numberKernels ownership;
# 6. numberKernels is a one-input fan-out: sample_52 fixes convolution numberBatches=1,
#    maps its prepared-kernel batch count to numberKernels, and RunApp switches the
#    inverse/output side from numberBatches ownership to numberKernels ownership.
# 7. sample_52 is a real 2D application-convolution witness: performR2C is enabled,
#    the caller input is tightly packed full-real data, kernel/midpoint storage uses the
#    compact complex (x/2+1) spectrum, the application normalizes the inverse, and the
#    output keeps upstream's (x+2)-padded real physical ABI for each kernel slab.
# 8. sample_52's native real surface uses coordinateFeatures=2 with matrixConvolution=1.
#    appendKernelOffset extracts the current coordinate into the kernel coordinate stride,
#    then adds batchID on the next kernel-set stride for K>1. Sample kernel/output loops are
#    kernel-set outer, coordinate inner, so this is independent-coordinate fan-out rather
#    than matrix row-sum ownership.
# 9. sample_51 is the pinned real-matrix witness: a 3D R2C application prepares nine
#    nonsymmetric 3x3 kernel planes, then applies them to three real coordinates with
#    matrixConvolution=3, coordinateFeatures=3, K=1 and normalized inverse ownership.
#    The sample also enables 3D zero-padding and copies the kernel-preparation configuration
#    into the application configuration before setting performConvolution/matrix ownership;
#    all three logical axes retain their half-to-end zero ranges.
sed -n '555,580p' "$plan" | grep -F 'app->configuration.FFTdim - 1 == axis_id' >/dev/null
sed -n '555,580p' "$plan" | grep -F 'axis->specializationConstants.convolutionStep = 1;' >/dev/null
sed -n '225,485p' "$run_app" | grep -F 'for (int i = 1; i <app->configuration.FFTdim; i++){' >/dev/null
sed -n '225,485p' "$run_app" | grep -F 'if ((app->configuration.FFTdim == (i+1)) && (app->configuration.performConvolution)) {' >/dev/null
sed -n '225,485p' "$run_app" | grep -F 'for (int i = (int)app->configuration.FFTdim-1; i > 0; i--){' >/dev/null
grep -F 'appendKernelOffset(sc, 0, (int)strideType);' "$convolution" >/dev/null
sed -n '320,405p' "$read_write" | grep -F 'for (int i = 1; i < sc->numFFTdims; i++){' >/dev/null
sed -n '320,405p' "$read_write" | grep -F 'bufferStride[locStrideOrder]' >/dev/null
grep -F 'if (sc->conjugateConvolution == 1) {' "$convolution" >/dev/null
grep -F 'PfConjugate(sc, &sc->regIDs[i + l * sc->registers_per_thread], &sc->regIDs[i + l * sc->registers_per_thread]);' "$convolution" >/dev/null
grep -F 'if (sc->crossPowerSpectrumNormalization) {' "$convolution" >/dev/null
grep -F 'PfNorm(sc, &sc->tempFloat, &sc->temp_conv[0]);' "$convolution" >/dev/null
grep -F 'PfRsqrt(sc, &sc->tempFloat, &sc->tempFloat);' "$convolution" >/dev/null
sed -n '1360,1380p' "$initialize" | grep -F 'if (app->configuration.matrixConvolution > 1) app->configuration.coordinateFeatures = app->configuration.matrixConvolution;' >/dev/null
sed -n '233,262p' "$run_app" | grep -F 'pfUINT maxCoordinate = ((app->configuration.matrixConvolution > 1) && (l == 0)) ? 1 : app->configuration.coordinateFeatures;' >/dev/null
sed -n '233,262p' "$run_app" | grep -F 'dispatchBlock[2] = maxCoordinate * app->configuration.numberBatches;' >/dev/null
sed -n '323,430p' "$run_app" | grep -F 'dispatchBlock[2] = app->configuration.coordinateFeatures * app->configuration.numberKernels;' >/dev/null
grep -F 'for (pfUINT j = 0; j < sc->matrixConvolution; j++) {' "$convolution" >/dev/null
grep -F 'for (pfUINT l = 0; l < sc->matrixConvolution; l++) {' "$convolution" >/dev/null
grep -F 'temp_int.data.i = k * sc->inputStride[sc->numFFTdims].data.i;' "$convolution" >/dev/null
grep -F 'if (sc->symmetricKernel) {' "$convolution" >/dev/null
grep -F 'k = (l < j) ? (l * sc->matrixConvolution - l * l + j) : (j * sc->matrixConvolution - j * j + l);' "$convolution" >/dev/null
grep -F 'k = (j * sc->matrixConvolution + l);' "$convolution" >/dev/null
sed -n '200,215p' "$sample52" | grep -F 'convolution_configuration.numberBatches = 1;//one batch - numberKernels convolutions' >/dev/null
sed -n '200,215p' "$sample52" | grep -F 'convolution_configuration.numberKernels = configuration.numberBatches;// number of convolutions on a single input' >/dev/null
sed -n '300,325p' "$sample52" | grep -F 'for (uint64_t f = 0; f < convolution_configuration.numberKernels; f++) {' >/dev/null
sed -n '264,320p' "$run_app" | grep -F 'dispatchBlock[2] = app->configuration.coordinateFeatures * app->configuration.numberBatches;' >/dev/null
sed -n '323,430p' "$run_app" | grep -F 'dispatchBlock[2] = app->configuration.coordinateFeatures * app->configuration.numberKernels;' >/dev/null
grep -F 'if (sc->numKernels.data.i > 1) {' "$convolution" >/dev/null
grep -F 'sc->regIDs_copy[i + l * sc->registers_per_thread]' "$convolution" >/dev/null
sed -n '75,90p' "$sample52" | grep -F 'configuration.performR2C = true;' >/dev/null
sed -n '75,90p' "$sample52" | grep -F 'configuration.normalize = 1;' >/dev/null
sed -n '105,130p' "$sample52" | grep -F 'configuration.size[0] / 2 + 1' >/dev/null
sed -n '185,215p' "$sample52" | grep -F 'convolution_configuration.performConvolution = true;' >/dev/null
sed -n '205,220p' "$sample52" | grep -F 'uint64_t inputBufferSize = ((uint64_t)convolution_configuration.coordinateFeatures) * sizeof(float) * (convolution_configuration.size[0])' >/dev/null
sed -n '205,220p' "$sample52" | grep -F 'uint64_t bufferSize = convolution_configuration.numberKernels * convolution_configuration.coordinateFeatures * sizeof(float) * 2 * (convolution_configuration.size[0] / 2 + 1)' >/dev/null
sed -n '205,220p' "$sample52" | grep -F 'convolution_configuration.isInputFormatted = true;' >/dev/null
sed -n '285,335p' "$sample52" | grep -F 'buffer_output[i + j * (convolution_configuration.size[0] + 2)' >/dev/null
sed -n '75,90p' "$sample52" | grep -F 'configuration.coordinateFeatures = 2;' >/dev/null
sed -n '150,175p' "$sample52" | grep -F 'for (uint64_t f = 0; f < configuration.numberBatches; f++) {' >/dev/null
sed -n '150,175p' "$sample52" | grep -F 'for (uint64_t v = 0; v < configuration.coordinateFeatures; v++) {' >/dev/null
sed -n '150,175p' "$sample52" | grep -F 'f * configuration.coordinateFeatures * (configuration.size[0] + 2)' >/dev/null
sed -n '300,335p' "$sample52" | grep -F 'for (uint64_t v = 0; v < convolution_configuration.coordinateFeatures; v++) {' >/dev/null
sed -n '368,390p' "$read_write" | grep -F 'maxCoordinate = sc->numCoordinates * sc->matrixConvolution;' >/dev/null
sed -n '368,390p' "$read_write" | grep -F 'PfMul(sc, &sc->tempInt, &sc->tempInt, &bufferStride[sc->numFFTdims], 0);' >/dev/null
sed -n '384,395p' "$read_write" | grep -F 'PfMul(sc, &sc->tempInt, &sc->batchID, &sc->inputStride[sc->numFFTdims+1], 0);' >/dev/null

sed -n '70,100p' "$sample51" | grep -F 'configuration.FFTdim = 3;' >/dev/null
sed -n '70,105p' "$sample51" | grep -F 'configuration.normalize = 1;' >/dev/null
sed -n '70,105p' "$sample51" | grep -F 'configuration.performR2C = true;' >/dev/null
sed -n '70,105p' "$sample51" | grep -F 'configuration.coordinateFeatures = 9;' >/dev/null
grep -F 'uint64_t kernelSize = ((uint64_t)configuration.coordinateFeatures) * sizeof(float) * 2 * (configuration.size[0] / 2 + 1)' "$sample51" >/dev/null
sed -n '198,212p' "$sample51" | grep -F 'convolution_configuration.performConvolution = true;' >/dev/null
sed -n '198,212p' "$sample51" | grep -F 'convolution_configuration.matrixConvolution = 3;' >/dev/null
sed -n '198,212p' "$sample51" | grep -F 'convolution_configuration.coordinateFeatures = 3;' >/dev/null
grep -F 'uint64_t bufferSize = ((uint64_t)convolution_configuration.coordinateFeatures) * sizeof(float) * 2 * (convolution_configuration.size[0] / 2 + 1)' "$sample51" >/dev/null
sed -n '80,95p' "$sample51" | grep -F 'configuration.performZeropadding[0] = true;' >/dev/null
sed -n '80,95p' "$sample51" | grep -F 'configuration.performZeropadding[1] = true;' >/dev/null
sed -n '80,95p' "$sample51" | grep -F 'configuration.performZeropadding[2] = true;' >/dev/null
sed -n '80,95p' "$sample51" | grep -F 'configuration.fft_zeropad_left[0] = (uint64_t)ceil(configuration.size[0] / 2.0);' >/dev/null
sed -n '80,95p' "$sample51" | grep -F 'configuration.fft_zeropad_right[0] = configuration.size[0];' >/dev/null
sed -n '80,95p' "$sample51" | grep -F 'configuration.fft_zeropad_left[1] = (uint64_t)ceil(configuration.size[1] / 2.0);' >/dev/null
sed -n '80,95p' "$sample51" | grep -F 'configuration.fft_zeropad_right[1] = configuration.size[1];' >/dev/null
sed -n '80,95p' "$sample51" | grep -F 'configuration.fft_zeropad_left[2] = (uint64_t)ceil(configuration.size[2] / 2.0);' >/dev/null
sed -n '80,95p' "$sample51" | grep -F 'configuration.fft_zeropad_right[2] = configuration.size[2];' >/dev/null
sed -n '198,205p' "$sample51" | grep -F 'convolution_configuration = configuration;' >/dev/null

# 10. Matrix convolution and numberKernels are orthogonal ownership dimensions in the
#     pinned convolutionStep: batched-kernel preparation snapshots every matrix input
#     coordinate, the row/column plane index is selected first, and batchID then selects
#     the kernel set. Read/write addressing likewise collapses matrix upload0 to one
#     coordinate before applying the numberKernels batchID stride.
sed -n '55,65p' "$convolution" | grep -F '((sc->convolutionStep) && ((sc->matrixConvolution > 1) || (sc->numKernels.data.i > 1)))' >/dev/null
sed -n '112,120p' "$convolution" | grep -F 'for (pfUINT j = 0; j < sc->matrixConvolution; j++) {' >/dev/null
sed -n '112,120p' "$convolution" | grep -F 'PfMov(sc, &sc->regIDs_copy[i + j * sc->registers_per_thread], &sc->regIDs[i + j * sc->registers_per_thread]);' >/dev/null
sed -n '350,370p' "$convolution" | grep -F 'if (sc->numKernels.data.i > 1) {' >/dev/null
sed -n '350,370p' "$convolution" | grep -F 'PfMul(sc, &sc->w, &sc->temp, &sc->regIDs_copy[i + l * sc->registers_per_thread], 0);' >/dev/null
sed -n '378,398p' "$read_write" | grep -F 'if ((sc->matrixConvolution > 1) && (sc->convolutionStep)) {' >/dev/null
sed -n '378,398p' "$read_write" | grep -F 'maxCoordinate = 1;' >/dev/null
sed -n '378,398p' "$read_write" | grep -F 'if (sc->convolutionStep && (sc->numKernels.data.i > 1)) {' >/dev/null

# 11. R2C application policy and logical zero-padding are orthogonal to kernel-set
#     ownership. The pinned R2C plan forwards Sequence conjugation and cross-power
#     normalization into specialization constants. Zero-padding helpers inspect only
#     logical coordinates plus left/right ranges and do not reference batchID or
#     numKernels; kernel-set ownership remains in the separate convolution/read-write
#     offsets pinned above.
grep -F 'axis->specializationConstants.conjugateConvolution = (int)app->configuration.conjugateConvolution;' "$r2c_plan" >/dev/null
grep -F 'axis->specializationConstants.crossPowerSpectrumNormalization = (int)app->configuration.crossPowerSpectrumNormalization;' "$r2c_plan" >/dev/null
grep -F 'static inline void checkZeropad_otherAxes(VkFFTSpecializationConstantsLayout* sc, PfContainer* location, int axisCheck)' "$zeropad" >/dev/null
grep -F 'if (sc->performZeropaddingFull[i]) {' "$zeropad" >/dev/null
grep -F 'if (sc->fft_zeropad_left_full[i].data.i < sc->fft_zeropad_right_full[i].data.i) {' "$zeropad" >/dev/null
grep -F 'PfIf_ge_start(sc, location, &sc->fft_zeropad_left_full[i]);' "$zeropad" >/dev/null
grep -F 'PfIf_lt_start(sc, location, &sc->fft_zeropad_right_full[i]);' "$zeropad" >/dev/null
grep -F 'static inline void checkZeropadStart_currentFFTAxis(VkFFTSpecializationConstantsLayout* sc, int readWrite, int type, PfContainer* inoutID)' "$zeropad" >/dev/null
grep -F 'PfIf_lt_start(sc, inoutID, &sc->fft_zeropad_left_read[sc->axis_id]);' "$zeropad" >/dev/null
grep -F 'PfIf_ge_start(sc, inoutID, &sc->fft_zeropad_right_read[sc->axis_id]);' "$zeropad" >/dev/null
if grep -F 'batchID' "$zeropad" >/dev/null; then echo 'zero-pad helper unexpectedly references batchID' >&2; exit 3; fi
if grep -F 'numKernels' "$zeropad" >/dev/null; then echo 'zero-pad helper unexpectedly references numKernels' >&2; exit 3; fi

# 12. Custom formatted R2C caller strides are a distinct batch1/K1 boundary in the
#     pinned API. The configuration documents that restriction, sample_15 proves the
#     R2C input-stride mechanism, the FFT plan selects inputBufferStride on the forward
#     first boundary and outputBufferStride on the inverse first boundary even when
#     performConvolution owns the midpoint, and descriptor selection binds the matching
#     external input/output buffers rather than the dense internal buffer. Initialization
#     preserves explicit user strides instead of normalizing them back to bufferStride.
grep -F 'pfUINT isInputFormatted; //specify if input buffer is padded - 0 - padded, 1 - not padded. For example if it is not padded for R2C if out-of-place mode is selected (only if numberBatches==1 and numberKernels==1)' "$structs" >/dev/null
grep -F 'pfUINT isOutputFormatted; //specify if output buffer is padded - 0 - padded, 1 - not padded. For example if it is not padded for R2C if out-of-place mode is selected (only if numberBatches==1 and numberKernels==1)' "$structs" >/dev/null
grep -F 'pfUINT inputBufferStride[VKFFT_MAX_FFT_DIMENSIONS];//input buffer strides. Used if isInputFormatted is enabled. Default set to bufferStride values' "$structs" >/dev/null
grep -F 'pfUINT outputBufferStride[VKFFT_MAX_FFT_DIMENSIONS];//output buffer strides. Used if isInputFormatted is enabled. Default set to bufferStride values' "$structs" >/dev/null
grep -F 'configuration.performR2C = 1;' "$sample15" >/dev/null
grep -F 'configuration.isInputFormatted = 1;' "$sample15" >/dev/null
grep -F 'configuration.inputBufferStride[0] = configuration.size[0];' "$sample15" >/dev/null
grep -F 'if ((!inverse) && (axis_id == app->firstAxis) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (app->configuration.isInputFormatted)) usedStride = app->configuration.inputBufferStride;' "$plan" >/dev/null
grep -F 'if ((inverse) && (axis_id == app->firstAxis) && (((axis_upload_id == 0) && (!app->configuration.performConvolution)) || ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && ((reverseBluesteinMultiUpload == 1) || (app->configuration.performConvolution)))) && ((app->configuration.isOutputFormatted))) usedStride = app->configuration.outputBufferStride;' "$plan" >/dev/null
sed -n '28,48p' "$update_buffers" | grep -F '(app->configuration.isInputFormatted)' >/dev/null
sed -n '28,48p' "$update_buffers" | grep -F '((axis_id == app->firstAxis) && (!inverse))' >/dev/null
sed -n '28,48p' "$update_buffers" | grep -F 'locBufferNum = app->configuration.inputBufferNum;' >/dev/null
sed -n '112,138p' "$update_buffers" | grep -F '(app->configuration.isOutputFormatted' >/dev/null
sed -n '112,138p' "$update_buffers" | grep -F '((axis_id == app->firstAxis) && (inverse))' >/dev/null
sed -n '112,138p' "$update_buffers" | grep -F 'locBufferNum = app->configuration.outputBufferNum;' >/dev/null
sed -n '1000,1012p' "$initialize" | grep -F 'app->configuration.inputBufferStride[0] = inputLaunchConfiguration.inputBufferStride[0];' >/dev/null
sed -n '1012,1022p' "$initialize" | grep -F 'app->configuration.outputBufferStride[0] = inputLaunchConfiguration.outputBufferStride[0];' >/dev/null

# 13. Do not over-generalize the formatted boundary. The pinned API explicitly says
#     omitDimension does not work with convolutions. sample_52 does set isInputFormatted
#     for its separate full-real input while K>1, but it never supplies a custom
#     inputBufferStride; initialization therefore takes the dense size[0] default. This is
#     evidence for a separate unpadded input buffer, not for custom pitched K>1 ownership.
grep -F 'pfUINT omitDimension[VKFFT_MAX_FFT_DIMENSIONS];//disable FFT for this dimension (0 - FFT enabled, 1 - FFT disabled). Default 0. Doesn'"'"'t work for R2C dimension 0 for now. Doesn'"'"'t work with convolutions.' "$structs" >/dev/null
grep -F 'convolution_configuration.isInputFormatted = true; //if input is a different buffer, it doesn'"'"'t have to be zeropadded/R2C padded' "$sample52" >/dev/null
if grep -F 'convolution_configuration.inputBufferStride' "$sample52" >/dev/null; then echo 'sample_52 unexpectedly supplies a custom inputBufferStride' >&2; exit 3; fi
sed -n '998,1010p' "$initialize" | grep -F 'if (inputLaunchConfiguration.performR2C && (!app->configuration.isInputFormatted))' >/dev/null
sed -n '998,1010p' "$initialize" | grep -F 'app->configuration.inputBufferStride[0] = app->configuration.size[0];' >/dev/null

# 14. Formatted K1 and spatial zero-padding are orthogonal at the application boundary.
#     performConvolution keeps the spatial write mask at the final application boundary,
#     while address generation checks logical coordinates before multiplying by the
#     selected physical buffer stride. Caller pitch therefore cannot change the mask.
grep -F 'if ((!app->configuration.frequencyZeroPadding) && (((axis_upload_id == 0) && (!((axis->specializationConstants.useBluesteinFFT) || (app->configuration.performConvolution)))) || ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && ((((reverseBluesteinMultiUpload == 1) || (FFTPlan->numAxisUploads[axis_id] == 1)) || (app->configuration.performConvolution)))))) {' "$plan" >/dev/null
grep -F 'if (((app->configuration.frequencyZeroPadding) && (((axis_upload_id == 0) && (!axis->specializationConstants.useBluesteinFFT)) || ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (axis->specializationConstants.useBluesteinFFT && ((reverseBluesteinMultiUpload == 1) || (FFTPlan->numAxisUploads[axis_id] == 1)))))) || (((!app->configuration.frequencyZeroPadding) && (app->configuration.FFTdim - 1 == axis_id) && (axis_upload_id == 0) && (FFTPlan->numAxisUploads[axis_id] == 1) && (app->configuration.performConvolution)))) {' "$plan" >/dev/null
sed -n '171,260p' "$read_write" | grep -F 'PfContainer* bufferStride = (readWrite) ? sc->outputStride : sc->inputStride;' >/dev/null
sed -n '171,260p' "$read_write" | grep -F 'checkZeropad_otherAxes(sc, &sc->inoutID_y, i);' >/dev/null
sed -n '171,260p' "$read_write" | grep -F 'PfMul(sc, &sc->inoutID_y, &sc->inoutID_y, &bufferStride[locStrideOrder], 0);' >/dev/null

# 15. Mixed-storage precision flags are general VkFFT storage/compute contracts and are
#     not excluded by performConvolution initialization. halfPrecisionMemoryOnly keeps
#     compute/kernel/temp storage in F32 while narrowing only the first forward caller input
#     and final inverse caller output to F16. doublePrecisionFloatMemory selects FP64
#     arithmetic with FP32 memory storage. The public configuration copies both flags before
#     independently copying performConvolution, so the pinned source has no precision-vs-
#     convolution rejection to justify keeping this surface ordinary-only.
grep -F 'pfUINT halfPrecisionMemoryOnly; //use half precision only as input/output buffer. Input/Output have to be allocated as half, buffer/tempBuffer have to be allocated as float (out of place mode only). Specify isInputFormatted and isOutputFormatted to use (0 - off, 1 - on)' "$structs" >/dev/null
grep -F 'pfUINT doublePrecisionFloatMemory; //use FP64 precision for all calculations, while all memory storage is done in FP32.' "$structs" >/dev/null
sed -n '150,173p' "$api_params" | grep -F 'if (app->configuration.halfPrecisionMemoryOnly) {' >/dev/null
sed -n '150,173p' "$api_params" | grep -F 'sc->floatTypeKernelMemoryCode = 12;' >/dev/null
sed -n '150,173p' "$api_params" | grep -F 'if ((sc->axis_id == app->firstAxis) && (sc->axis_upload_id == sc->numAxisUploads - 1) && (!sc->actualInverse)) {' >/dev/null
sed -n '150,173p' "$api_params" | grep -F 'if ((sc->axis_id == app->firstAxis) && (((!sc->reorderFourStep) && (sc->axis_upload_id == sc->numAxisUploads - 1)) || ((sc->reorderFourStep) && (sc->axis_upload_id == 0))) && (sc->actualInverse)) {' >/dev/null
sed -n '220,232p' "$api_params" | grep -F 'if (app->configuration.doublePrecisionFloatMemory) {' >/dev/null
sed -n '220,232p' "$api_params" | grep -F 'sc->floatTypeCode = 22;' >/dev/null
sed -n '220,232p' "$api_params" | grep -F 'sc->floatTypeKernelMemoryCode = 12;' >/dev/null
grep -F 'if (inputLaunchConfiguration.doublePrecisionFloatMemory != 0)' "$initialize" >/dev/null
grep -F 'if (inputLaunchConfiguration.halfPrecisionMemoryOnly != 0)' "$initialize" >/dev/null
grep -F 'app->configuration.performConvolution = inputLaunchConfiguration.performConvolution;' "$initialize" >/dev/null

# 16. Spatial zero-padding composes independently with the mixed-storage type contract.
#     Initialization copies the logical mask/ranges without inspecting memory precision, and
#     the R2C specialization forwards the full-dimensional mask without reading either
#     mixed-storage flag. The already-pinned performConvolution boundary rules in section 14
#     therefore select the same logical caller edges while section 15 independently selects
#     F16/F32 or F32/F64 storage/compute types. This proves composition, not custom pitching.
sed -n '1315,1325p' "$initialize" | grep -F 'app->configuration.performZeropadding[i] = inputLaunchConfiguration.performZeropadding[i];' >/dev/null
sed -n '1315,1325p' "$initialize" | grep -F 'app->configuration.fft_zeropad_left[i] = inputLaunchConfiguration.fft_zeropad_left[i];' >/dev/null
sed -n '1315,1325p' "$initialize" | grep -F 'app->configuration.fft_zeropad_right[i] = inputLaunchConfiguration.fft_zeropad_right[i];' >/dev/null
grep -F 'axis->specializationConstants.performZeropaddingFull[i] = (int)app->configuration.performZeropadding[i];' "$r2c_plan" >/dev/null
if sed -n '1315,1325p' "$initialize" | grep -E 'halfPrecisionMemoryOnly|doublePrecisionFloatMemory' >/dev/null; then echo 'zero-padding initialization unexpectedly branches on mixed storage precision' >&2; exit 3; fi
if sed -n '210,260p' "$r2c_plan" | grep -E 'halfPrecisionMemoryOnly|doublePrecisionFloatMemory' >/dev/null; then echo 'R2C zero-padding specialization unexpectedly branches on mixed storage precision' >&2; exit 3; fi

# 17. Custom formatted K1 ownership also composes independently with mixed storage when
#     spatial zero-padding is not part of the same new boundary. halfPrecisionMemoryOnly's
#     public contract explicitly calls for isInputFormatted/isOutputFormatted to use separate
#     half caller buffers. Initialization preserves those flags and explicit strides without
#     inspecting mixed precision, FFT planning selects the custom strides at the same true
#     application boundaries without inspecting precision, and descriptor selection binds the
#     matching external buffers without inspecting precision. Conversely, the memory type
#     selector in section 15 does not inspect formatted state.
grep -F 'app->configuration.isInputFormatted = inputLaunchConfiguration.isInputFormatted;' "$initialize" >/dev/null
grep -F 'app->configuration.isOutputFormatted = inputLaunchConfiguration.isOutputFormatted;' "$initialize" >/dev/null
grep -F 'app->configuration.inputBufferStride[0] = inputLaunchConfiguration.inputBufferStride[0];' "$initialize" >/dev/null
grep -F 'app->configuration.outputBufferStride[0] = inputLaunchConfiguration.outputBufferStride[0];' "$initialize" >/dev/null
if sed -n '984,1042p' "$initialize" | grep -E 'halfPrecisionMemoryOnly|doublePrecisionFloatMemory' >/dev/null; then echo 'formatted stride initialization unexpectedly branches on mixed storage precision' >&2; exit 3; fi
grep -F 'if ((!inverse) && (axis_id == app->firstAxis) && (axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && (app->configuration.isInputFormatted)) usedStride = app->configuration.inputBufferStride;' "$plan" >/dev/null
grep -F 'if ((inverse) && (axis_id == app->firstAxis) && (((axis_upload_id == 0) && (!app->configuration.performConvolution)) || ((axis_upload_id == FFTPlan->numAxisUploads[axis_id] - 1) && ((reverseBluesteinMultiUpload == 1) || (app->configuration.performConvolution)))) && ((app->configuration.isOutputFormatted))) usedStride = app->configuration.outputBufferStride;' "$plan" >/dev/null
if sed -n '245,345p' "$plan" | grep -E 'halfPrecisionMemoryOnly|doublePrecisionFloatMemory' >/dev/null; then echo 'formatted stride selection unexpectedly branches on mixed storage precision' >&2; exit 3; fi
sed -n '28,48p' "$update_buffers" | grep -F 'locBufferNum = app->configuration.inputBufferNum;' >/dev/null
sed -n '112,138p' "$update_buffers" | grep -F 'locBufferNum = app->configuration.outputBufferNum;' >/dev/null
if sed -n '28,158p' "$update_buffers" | grep -E 'halfPrecisionMemoryOnly|doublePrecisionFloatMemory' >/dev/null; then echo 'formatted descriptor selection unexpectedly branches on mixed storage precision' >&2; exit 3; fi
if sed -n '150,232p' "$api_params" | grep -E 'isInputFormatted|isOutputFormatted' >/dev/null; then echo 'mixed storage type selection unexpectedly branches on formatted state' >&2; exit 3; fi

# 18. The three-way mixed-storage + formatted + Spatial-padding boundary is represented by
#     the same upstream code path rather than a fourth hidden policy branch. Read/write first
#     checks the logical zero-pad coordinate and only then multiplies by the selected physical
#     buffer stride. That combined address path does not inspect mixed precision, while the
#     memory-type selector does not inspect formatted or zero-pad state.
sed -n '171,260p' "$read_write" | grep -F 'PfContainer* bufferStride = (readWrite) ? sc->outputStride : sc->inputStride;' >/dev/null
sed -n '171,260p' "$read_write" | grep -F 'checkZeropad_otherAxes(sc, &sc->inoutID_y, i);' >/dev/null
sed -n '171,260p' "$read_write" | grep -F 'PfMul(sc, &sc->inoutID_y, &sc->inoutID_y, &bufferStride[locStrideOrder], 0);' >/dev/null
mask_line=$(grep -n -F 'checkZeropad_otherAxes(sc, &sc->inoutID_y, i);' "$read_write" | head -1 | cut -d: -f1)
stride_line=$(grep -n -F 'PfMul(sc, &sc->inoutID_y, &sc->inoutID_y, &bufferStride[locStrideOrder], 0);' "$read_write" | head -1 | cut -d: -f1)
if [ "$mask_line" -ge "$stride_line" ]; then echo 'logical zero-pad check no longer precedes physical stride mapping' >&2; exit 3; fi
if sed -n '171,260p' "$read_write" | grep -E 'halfPrecisionMemoryOnly|doublePrecisionFloatMemory' >/dev/null; then echo 'combined zero-pad/stride addressing unexpectedly branches on mixed storage precision' >&2; exit 3; fi
if sed -n '150,232p' "$api_params" | grep -E 'isInputFormatted|isOutputFormatted|performZeropadding|fft_zeropad' >/dev/null; then echo 'mixed storage type selection unexpectedly branches on formatted/padding state' >&2; exit 3; fi

# 19. Scalar numberKernels fan-out is orthogonal to the mixed-storage memory-type contract.
#     Pinned sample_52 fixes one input batch and maps prepared-kernel batches to numberKernels;
#     RunApp keeps the forward side on numberBatches but switches inverse/output dispatch to
#     numberKernels. convolutionStep reuses the saved forward registers for each kernel. The
#     mixed-memory selector has no batch/kernel ownership state and no joint precision/K branch.
grep -F 'if (inputLaunchConfiguration.numberKernels != 0)' "$initialize" | grep -F 'app->configuration.numberKernels = inputLaunchConfiguration.numberKernels;' >/dev/null
sed -n '200,215p' "$sample52" | grep -F 'convolution_configuration.numberBatches = 1;//one batch - numberKernels convolutions' >/dev/null
sed -n '200,215p' "$sample52" | grep -F 'convolution_configuration.numberKernels = configuration.numberBatches;// number of convolutions on a single input' >/dev/null
grep -F 'dispatchBlock[2] = app->configuration.coordinateFeatures * app->configuration.numberBatches;' "$run_app" >/dev/null
grep -F 'dispatchBlock[2] = app->configuration.coordinateFeatures * app->configuration.numberKernels;' "$run_app" >/dev/null
grep -F 'if (sc->numKernels.data.i > 1) {' "$convolution" >/dev/null
grep -F 'PfMul(sc, &sc->w, &sc->temp, &sc->regIDs_copy[i + l * sc->registers_per_thread], 0);' "$convolution" >/dev/null
if grep -R -E '(halfPrecisionMemoryOnly|doublePrecisionFloatMemory).*(numberKernels|numKernels)|(numberKernels|numKernels).*(halfPrecisionMemoryOnly|doublePrecisionFloatMemory)' "$upstream_dir/vkFFT/vkFFT" >/dev/null; then echo 'numberKernels unexpectedly has a mixed-precision joint branch' >&2; exit 3; fi
if sed -n '150,232p' "$api_params" | grep -E 'numberKernels|numKernels|numberBatches|batchID' >/dev/null; then echo 'mixed storage type selection unexpectedly depends on kernel/batch ownership' >&2; exit 3; fi

# 20. Scalar K fan-out composes with Spatial zero-padding independently of mixed storage.
#     The zero-pad helper contains neither kernel/batch ownership nor mixed-memory policy;
#     RunApp kernel fan-out contains no mixed-memory policy either. Initialization copies K,
#     the logical padding mask, and both mixed-precision flags independently, while R2C keeps
#     forwarding the same full-dimensional zero-pad mask.
if grep -E 'batchID|numKernels|numberKernels' "$zeropad" >/dev/null; then echo 'zero-pad helper unexpectedly depends on kernel ownership' >&2; exit 3; fi
if grep -E 'halfPrecisionMemoryOnly|doublePrecisionFloatMemory' "$zeropad" >/dev/null; then echo 'zero-pad helper unexpectedly depends on mixed storage precision' >&2; exit 3; fi
if grep -E 'halfPrecisionMemoryOnly|doublePrecisionFloatMemory' "$run_app" >/dev/null; then echo 'RunApp kernel ownership unexpectedly depends on mixed storage precision' >&2; exit 3; fi
grep -F 'app->configuration.numberKernels = inputLaunchConfiguration.numberKernels;' "$initialize" >/dev/null
grep -F 'app->configuration.performZeropadding[i] = inputLaunchConfiguration.performZeropadding[i];' "$initialize" >/dev/null
grep -F 'app->configuration.halfPrecisionMemoryOnly = inputLaunchConfiguration.halfPrecisionMemoryOnly;' "$initialize" >/dev/null
grep -F 'app->configuration.doublePrecisionFloatMemory = inputLaunchConfiguration.doublePrecisionFloatMemory;' "$initialize" >/dev/null
grep -F 'axis->specializationConstants.performZeropaddingFull[i] = (int)app->configuration.performZeropadding[i];' "$r2c_plan" >/dev/null

# 21. sample_52 independent-coordinate fan-out is orthogonal to mixed storage. The sample
#     stores C independent full-real inputs, prepares K*C compact kernel spectra, and writes
#     K*C outputs. RunApp dispatches C*numberBatches forward and C*numberKernels inverse;
#     ReadWrite applies the coordinate stride independently of precision. No mixed-memory
#     branch references coordinateFeatures/numCoordinates and the type selector has no C/K state.
sed -n '75,90p' "$sample52" | grep -F 'configuration.coordinateFeatures = 2;' >/dev/null
grep -F 'uint64_t inputBufferSize = ((uint64_t)convolution_configuration.coordinateFeatures) * sizeof(float) * (convolution_configuration.size[0])' "$sample52" >/dev/null
grep -F 'uint64_t bufferSize = convolution_configuration.numberKernels * convolution_configuration.coordinateFeatures * sizeof(float) * 2 * (convolution_configuration.size[0] / 2 + 1)' "$sample52" >/dev/null
grep -F 'convolution_configuration.numberKernels = configuration.numberBatches;// number of convolutions on a single input' "$sample52" >/dev/null
grep -F 'dispatchBlock[2] = app->configuration.coordinateFeatures * app->configuration.numberBatches;' "$run_app" >/dev/null
grep -F 'dispatchBlock[2] = app->configuration.coordinateFeatures * app->configuration.numberKernels;' "$run_app" >/dev/null
grep -F 'maxCoordinate = sc->numCoordinates * sc->matrixConvolution;' "$read_write" >/dev/null
if grep -R -E '(halfPrecisionMemoryOnly|doublePrecisionFloatMemory).*(coordinateFeatures|numCoordinates)|(coordinateFeatures|numCoordinates).*(halfPrecisionMemoryOnly|doublePrecisionFloatMemory)' "$upstream_dir/vkFFT/vkFFT" >/dev/null; then echo 'independent-coordinate ownership unexpectedly has a mixed-precision joint branch' >&2; exit 3; fi
if sed -n '150,232p' "$api_params" | grep -E 'coordinateFeatures|numCoordinates|numberKernels|numKernels' >/dev/null; then echo 'mixed storage type selection unexpectedly depends on coordinate/kernel ownership' >&2; exit 3; fi

# 22. Independent-coordinate K fan-out composes with Spatial zero-padding without adding a
#     coordinate-specific mask path. Initialization copies coordinateFeatures and the logical
#     zero-pad ranges independently; the zero-pad helper contains no C/K/batch/precision state,
#     and R2C propagates the same full-dimensional mask before RunApp expands inverse ownership
#     from C inputs to K*C outputs.
grep -F 'app->configuration.coordinateFeatures = inputLaunchConfiguration.coordinateFeatures;' "$initialize" >/dev/null
grep -F 'app->configuration.performZeropadding[i] = inputLaunchConfiguration.performZeropadding[i];' "$initialize" >/dev/null
grep -F 'axis->specializationConstants.performZeropaddingFull[i] = (int)app->configuration.performZeropadding[i];' "$r2c_plan" >/dev/null
if grep -E 'numCoordinates|coordinateFeatures' "$zeropad" >/dev/null; then echo 'zero-pad helper unexpectedly depends on coordinate ownership' >&2; exit 3; fi
if grep -E 'numKernels|numberKernels|batchID' "$zeropad" >/dev/null; then echo 'zero-pad helper unexpectedly depends on K/batch ownership' >&2; exit 3; fi
if grep -E 'halfPrecisionMemoryOnly|doublePrecisionFloatMemory' "$zeropad" >/dev/null; then echo 'zero-pad helper unexpectedly depends on mixed precision' >&2; exit 3; fi
grep -F 'dispatchBlock[2] = app->configuration.coordinateFeatures * app->configuration.numberBatches;' "$run_app" >/dev/null
grep -F 'dispatchBlock[2] = app->configuration.coordinateFeatures * app->configuration.numberKernels;' "$run_app" >/dev/null

# 23. Dense nonsymmetric 3x3 matrix K1 ownership is orthogonal to mixed storage. sample_51
#     pins matrixConvolution=3/coordinateFeatures=3, RunApp collapses the forward convolution
#     upload to one matrix system before restoring three inverse coordinates, and the convolution
#     body performs the row-by-column sum. None of those paths inspect mixed-memory precision;
#     conversely the memory type selector has no matrix/coordinate/K ownership state.
sed -n '198,212p' "$sample51" | grep -F 'convolution_configuration.matrixConvolution = 3;' >/dev/null
sed -n '198,212p' "$sample51" | grep -F 'convolution_configuration.coordinateFeatures = 3;' >/dev/null
sed -n '233,262p' "$run_app" | grep -F 'pfUINT maxCoordinate = ((app->configuration.matrixConvolution > 1) && (l == 0)) ? 1 : app->configuration.coordinateFeatures;' >/dev/null
grep -F 'dispatchBlock[2] = app->configuration.coordinateFeatures * app->configuration.numberKernels;' "$run_app" >/dev/null
grep -F 'for (pfUINT j = 0; j < sc->matrixConvolution; j++) {' "$convolution" >/dev/null
grep -F 'for (pfUINT l = 0; l < sc->matrixConvolution; l++) {' "$convolution" >/dev/null
if grep -R -E '(halfPrecisionMemoryOnly|doublePrecisionFloatMemory).*(matrixConvolution)|(matrixConvolution).*(halfPrecisionMemoryOnly|doublePrecisionFloatMemory)' "$upstream_dir/vkFFT/vkFFT" >/dev/null; then echo 'matrix ownership unexpectedly has a mixed-precision joint branch' >&2; exit 3; fi
if sed -n '150,232p' "$api_params" | grep -E 'matrixConvolution|coordinateFeatures|numCoordinates|numberKernels|numKernels' >/dev/null; then echo 'mixed storage type selection unexpectedly depends on matrix ownership' >&2; exit 3; fi

# 24. sample_51 directly composes nonsymmetric 3x3 matrix K1 with Spatial zero-padding:
#     the padded kernel-preparation configuration is copied into the application before matrix
#     ownership is selected. Initialization and R2C propagate that logical mask independently,
#     while the zero-pad helper contains no matrix/coordinate/K/batch/precision policy and the
#     mixed-memory type selector contains no matrix/padding state.
sed -n '198,205p' "$sample51" | grep -F 'convolution_configuration = configuration;' >/dev/null
sed -n '80,95p' "$sample51" | grep -F 'configuration.performZeropadding[0] = true;' >/dev/null
sed -n '80,95p' "$sample51" | grep -F 'configuration.performZeropadding[1] = true;' >/dev/null
sed -n '80,95p' "$sample51" | grep -F 'configuration.performZeropadding[2] = true;' >/dev/null
grep -F 'app->configuration.performZeropadding[i] = inputLaunchConfiguration.performZeropadding[i];' "$initialize" >/dev/null
grep -F 'axis->specializationConstants.performZeropaddingFull[i] = (int)app->configuration.performZeropadding[i];' "$r2c_plan" >/dev/null
if grep -E 'matrixConvolution|numCoordinates|coordinateFeatures|numKernels|numberKernels|batchID|halfPrecisionMemoryOnly|doublePrecisionFloatMemory' "$zeropad" >/dev/null; then echo 'matrix padding helper unexpectedly depends on ownership or mixed precision' >&2; exit 3; fi
if sed -n '150,232p' "$api_params" | grep -E 'matrixConvolution|coordinateFeatures|performZeropadding|fft_zeropad' >/dev/null; then echo 'mixed storage type selection unexpectedly depends on matrix padding state' >&2; exit 3; fi

# 25. Matrix row-sum and numberKernels fan-out are orthogonal to mixed storage. The pinned
#     convolutionStep snapshots every matrix coordinate when either matrixConvolution or K>1 is
#     active, selects the matrix plane first, then uses batchID for the kernel set; read/write
#     likewise collapses matrix upload0 before applying the K batch stride. None of these paths
#     inspect mixed precision, and the type selector contains neither matrix nor K ownership.
sed -n '55,65p' "$convolution" | grep -F '((sc->convolutionStep) && ((sc->matrixConvolution > 1) || (sc->numKernels.data.i > 1)))' >/dev/null
sed -n '112,120p' "$convolution" | grep -F 'PfMov(sc, &sc->regIDs_copy[i + j * sc->registers_per_thread], &sc->regIDs[i + j * sc->registers_per_thread]);' >/dev/null
sed -n '350,370p' "$convolution" | grep -F 'if (sc->numKernels.data.i > 1) {' >/dev/null
sed -n '350,370p' "$convolution" | grep -F 'PfMul(sc, &sc->w, &sc->temp, &sc->regIDs_copy[i + l * sc->registers_per_thread], 0);' >/dev/null
sed -n '378,398p' "$read_write" | grep -F 'if ((sc->matrixConvolution > 1) && (sc->convolutionStep)) {' >/dev/null
sed -n '378,398p' "$read_write" | grep -F 'if (sc->convolutionStep && (sc->numKernels.data.i > 1)) {' >/dev/null
if grep -R -E '(halfPrecisionMemoryOnly|doublePrecisionFloatMemory).*(matrixConvolution|numberKernels|numKernels)|(matrixConvolution|numberKernels|numKernels).*(halfPrecisionMemoryOnly|doublePrecisionFloatMemory)' "$upstream_dir/vkFFT/vkFFT" >/dev/null; then echo 'matrix K fan-out unexpectedly has a mixed-precision joint branch' >&2; exit 3; fi
if sed -n '150,232p' "$api_params" | grep -E 'matrixConvolution|coordinateFeatures|numberKernels|numKernels|batchID' >/dev/null; then echo 'mixed storage type selection unexpectedly depends on matrix K fan-out ownership' >&2; exit 3; fi

# 26. Matrix K fan-out and Spatial zero-padding remain independent in the same application graph.
#     K ownership lives in convolution/read-write offsets, while logical zero-padding lives in a
#     helper that references neither matrix nor K/batch state; both are also independent of mixed
#     precision. Thus one padded 3-coordinate input may fan out to K*3 independently masked outputs.
grep -F 'if (sc->convolutionStep && (sc->numKernels.data.i > 1)) {' "$read_write" >/dev/null
grep -F 'maxCoordinate = 1;' "$read_write" >/dev/null
grep -F 'axis->specializationConstants.performZeropaddingFull[i] = (int)app->configuration.performZeropadding[i];' "$r2c_plan" >/dev/null
if grep -E 'matrixConvolution|numCoordinates|coordinateFeatures|numKernels|numberKernels|batchID' "$zeropad" >/dev/null; then echo 'matrix K padding helper unexpectedly depends on matrix/K ownership' >&2; exit 3; fi
if grep -E 'halfPrecisionMemoryOnly|doublePrecisionFloatMemory' "$zeropad" >/dev/null; then echo 'matrix K padding helper unexpectedly depends on mixed precision' >&2; exit 3; fi
if grep -E 'halfPrecisionMemoryOnly|doublePrecisionFloatMemory' "$run_app" >/dev/null; then echo 'matrix K RunApp ownership unexpectedly depends on mixed precision' >&2; exit 3; fi
if sed -n '150,232p' "$api_params" | grep -E 'matrixConvolution|coordinateFeatures|numberKernels|numKernels|performZeropadding|fft_zeropad' >/dev/null; then echo 'mixed storage type selection unexpectedly depends on matrix K padding ownership' >&2; exit 3; fi

# 27. Sequence conjugation and cross-power normalization are compute-stage policy and remain
#     orthogonal to mixed caller storage. The R2C plan forwards both flags into the convolution
#     specialization, convolutionStep applies conjugation/norm/rsqrt there, and neither policy
#     path has a joint branch with halfPrecisionMemoryOnly/doublePrecisionFloatMemory.
grep -F 'axis->specializationConstants.conjugateConvolution = (int)app->configuration.conjugateConvolution;' "$r2c_plan" >/dev/null
grep -F 'axis->specializationConstants.crossPowerSpectrumNormalization = (int)app->configuration.crossPowerSpectrumNormalization;' "$r2c_plan" >/dev/null
grep -F 'if (sc->conjugateConvolution == 1) {' "$convolution" >/dev/null
grep -F 'if (sc->crossPowerSpectrumNormalization) {' "$convolution" >/dev/null
if grep -R -E '(halfPrecisionMemoryOnly|doublePrecisionFloatMemory).*(conjugateConvolution|crossPowerSpectrumNormalization)|(conjugateConvolution|crossPowerSpectrumNormalization).*(halfPrecisionMemoryOnly|doublePrecisionFloatMemory)' "$upstream_dir/vkFFT/vkFFT" >/dev/null; then echo 'convolution policy unexpectedly has a mixed-precision joint branch' >&2; exit 3; fi
if sed -n '150,232p' "$api_params" | grep -E 'conjugateConvolution|crossPowerSpectrumNormalization' >/dev/null; then echo 'mixed storage type selection unexpectedly depends on convolution policy' >&2; exit 3; fi

printf 'upstream ND convolution ownership/policy/matrix/fanout/real/coordinates/matrix/padding/matrix-kernels/real-orthogonality/formatted-k1/negative-boundaries/formatted-padding/mixed-storage/mixed-padding/mixed-formatted/mixed-formatted-padding/mixed-fanout/mixed-fanout-padding/mixed-independent-coordinates/mixed-independent-padding/mixed-matrix-k1/mixed-matrix-padding/mixed-matrix-fanout/mixed-matrix-fanout-padding/mixed-policy: 219/219 source contracts matched at %s\n' "$expected_commit"
