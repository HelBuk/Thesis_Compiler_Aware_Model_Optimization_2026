#include "c2f_m2_plugin.hpp"

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>

#include <cuda_runtime.h>
#include <cudnn.h>

#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

using namespace nvinfer1;

namespace {
char const* kPLUGIN_NAME{"YoloC2fM2_TRT"};
char const* kPLUGIN_VERSION{"1"};

template <typename T>
void writeToBuffer(char*& buffer, T const& value) {
    std::memcpy(buffer, &value, sizeof(T));
    buffer += sizeof(T);
}

template <typename T>
T readFromBuffer(char const*& buffer) {
    T value;
    std::memcpy(&value, buffer, sizeof(T));
    buffer += sizeof(T);
    return value;
}
} // namespace

// ============================
// Plugin
// ============================

YoloC2fM2Plugin::YoloC2fM2Plugin(std::string weightsPath)
    : mWeightsPath(std::move(weightsPath)) {}

YoloC2fM2Plugin::YoloC2fM2Plugin(void const* data, size_t) {
    char const* d = static_cast<char const*>(data);

    int32_t weightsLen = readFromBuffer<int32_t>(d);
    mWeightsPath.assign(d, d + weightsLen);
    d += weightsLen;

    int32_t nsLen = readFromBuffer<int32_t>(d);
    mNamespace.assign(d, d + nsLen);
    d += nsLen;
}

YoloC2fM2Plugin::~YoloC2fM2Plugin() {
    terminate();
}

void YoloC2fM2Plugin::destroyWeights() noexcept {
    auto freePtr = [](float*& p) {
        if (p) {
            cudaFree(p);
            p = nullptr;
        }
    };

    freePtr(d_cv1_w);
    freePtr(d_cv1_b);
    freePtr(d_m0_cv1_w);
    freePtr(d_m0_cv1_b);
    freePtr(d_m0_cv2_w);
    freePtr(d_m0_cv2_b);
    freePtr(d_cv2_w);
    freePtr(d_cv2_b);
}

bool YoloC2fM2Plugin::loadWeightsToDevice() {
    if (mWeightsPath.empty()) {
        return false;
    }

    // Replace this with your real loader.
    // Expected tensor shapes:
    // cv1_w      [C,     C,     1,1]
    // cv1_b      [C]
    // m0_cv1_w   [C/2,   C/2,   3,3]
    // m0_cv1_b   [C/2]
    // m0_cv2_w   [C/2,   C/2,   3,3]
    // m0_cv2_b   [C/2]
    // cv2_w      [C,   3*C/2,   1,1]
    // cv2_b      [C]

    // For now, just fail clearly if not implemented.
    return false;
}

const char* YoloC2fM2Plugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}

const char* YoloC2fM2Plugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

int YoloC2fM2Plugin::getNbOutputs() const noexcept {
    return 1;
}

int YoloC2fM2Plugin::initialize() noexcept {
    if (!mCudnn) {
        if (cudnnCreate(&mCudnn) != CUDNN_STATUS_SUCCESS) {
            mCudnn = nullptr;
            return 1;
        }
    }

    if (!loadWeightsToDevice()) {
        return 1;
    }

    return 0;
}

void YoloC2fM2Plugin::terminate() noexcept {
    destroyWeights();

    if (mCudnn) {
        cudnnDestroy(mCudnn);
        mCudnn = nullptr;
    }
}

size_t YoloC2fM2Plugin::getSerializationSize() const noexcept {
    return sizeof(int32_t) + mWeightsPath.size()
         + sizeof(int32_t) + mNamespace.size();
}

void YoloC2fM2Plugin::serialize(void* buffer) const noexcept {
    char* d = static_cast<char*>(buffer);

    int32_t weightsLen = static_cast<int32_t>(mWeightsPath.size());
    writeToBuffer(d, weightsLen);
    std::memcpy(d, mWeightsPath.data(), mWeightsPath.size());
    d += mWeightsPath.size();

    int32_t nsLen = static_cast<int32_t>(mNamespace.size());
    writeToBuffer(d, nsLen);
    std::memcpy(d, mNamespace.data(), mNamespace.size());
    d += mNamespace.size();
}

void YoloC2fM2Plugin::destroy() noexcept {
    delete this;
}

void YoloC2fM2Plugin::setPluginNamespace(char const* pluginNamespace) noexcept {
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

char const* YoloC2fM2Plugin::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

DataType YoloC2fM2Plugin::getOutputDataType(
    int,
    DataType const* inputTypes,
    int) const noexcept {
    return inputTypes[0];
}

IPluginV2DynamicExt* YoloC2fM2Plugin::clone() const noexcept {
    auto* p = new YoloC2fM2Plugin(mWeightsPath);
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

DimsExprs YoloC2fM2Plugin::getOutputDimensions(
    int,
    DimsExprs const* inputs,
    int,
    IExprBuilder&) noexcept {
    return inputs[0];
}

bool YoloC2fM2Plugin::supportsFormatCombination(
    int pos,
    PluginTensorDesc const* inOut,
    int,
    int) noexcept {

    auto const& desc = inOut[pos];
    return desc.format == TensorFormat::kLINEAR &&
           desc.type == DataType::kFLOAT;
}

void YoloC2fM2Plugin::configurePlugin(
    DynamicPluginTensorDesc const*,
    int,
    DynamicPluginTensorDesc const*,
    int) noexcept {}

size_t YoloC2fM2Plugin::getMaxConvWorkspaceBytes(int N, int C, int H, int W) const noexcept {
    (void)N; (void)C; (void)H; (void)W;
    return 64ULL * 1024ULL * 1024ULL; // 64 MB
}

size_t YoloC2fM2Plugin::getWorkspaceSize(
    PluginTensorDesc const* inputs,
    int nbInputs,
    PluginTensorDesc const*,
    int) const noexcept {

    if (nbInputs != 1) return 0;

    auto const& d = inputs[0].dims;
    if (d.nbDims != 4) return 0;

    int N = d.d[0];
    int C = d.d[1];
    int H = d.d[2];
    int W = d.d[3];

    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return 0;
    if ((C % 2) != 0) return 0;

    int halfC = C / 2;
    size_t HW = static_cast<size_t>(H) * static_cast<size_t>(W);

    size_t featureElems =
        static_cast<size_t>(N) * C * HW +              // cv1_out
        static_cast<size_t>(N) * halfC * HW +          // m0_out
        static_cast<size_t>(N) * (3 * halfC) * HW;    // concat_out

    size_t featureBytes = featureElems * sizeof(float);
    size_t convWsBytes = getMaxConvWorkspaceBytes(N, C, H, W);

    return featureBytes + convWsBytes;
}

bool YoloC2fM2Plugin::runConv1x1(
    const float* x,
    float* y,
    int N, int Cin, int Cout, int H, int W,
    const float* w,
    const float* b,
    void* workspace,
    size_t workspaceBytes,
    cudaStream_t stream) const noexcept {

    if (!mCudnn || !x || !y || !w || !b) return false;

    cudnnTensorDescriptor_t xDesc{}, yDesc{}, bDesc{};
    cudnnFilterDescriptor_t wDesc{};
    cudnnConvolutionDescriptor_t convDesc{};

    cudnnCreateTensorDescriptor(&xDesc);
    cudnnCreateTensorDescriptor(&yDesc);
    cudnnCreateTensorDescriptor(&bDesc);
    cudnnCreateFilterDescriptor(&wDesc);
    cudnnCreateConvolutionDescriptor(&convDesc);

    cudnnSetStream(mCudnn, stream);

    cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, Cin, H, W);
    cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, Cout, H, W);
    cudnnSetTensor4dDescriptor(bDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, Cout, 1, 1);
    cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, Cout, Cin, 1, 1);
    cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

    float alpha = 1.f, beta = 0.f;
    bool ok = true;

    ok &= (cudnnConvolutionForward(
        mCudnn,
        &alpha,
        xDesc, x,
        wDesc, w,
        convDesc,
        algo,
        workspace, workspaceBytes,
        &beta,
        yDesc, y) == CUDNN_STATUS_SUCCESS);

    if (ok) {
        ok &= (cudnnAddTensor(
            mCudnn,
            &alpha,
            bDesc, b,
            &alpha,
            yDesc, y) == CUDNN_STATUS_SUCCESS);
    }

    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroyTensorDescriptor(bDesc);
    cudnnDestroyFilterDescriptor(wDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);

    return ok;
}

bool YoloC2fM2Plugin::runConv3x3(
    const float* x,
    float* y,
    int N, int Cin, int Cout, int H, int W,
    const float* w,
    const float* b,
    void* workspace,
    size_t workspaceBytes,
    cudaStream_t stream) const noexcept {

    if (!mCudnn || !x || !y || !w || !b) return false;

    cudnnTensorDescriptor_t xDesc{}, yDesc{}, bDesc{};
    cudnnFilterDescriptor_t wDesc{};
    cudnnConvolutionDescriptor_t convDesc{};

    cudnnCreateTensorDescriptor(&xDesc);
    cudnnCreateTensorDescriptor(&yDesc);
    cudnnCreateTensorDescriptor(&bDesc);
    cudnnCreateFilterDescriptor(&wDesc);
    cudnnCreateConvolutionDescriptor(&convDesc);

    cudnnSetStream(mCudnn, stream);

    cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, Cin, H, W);
    cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, Cout, H, W);
    cudnnSetTensor4dDescriptor(bDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, Cout, 1, 1);
    cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, Cout, Cin, 3, 3);
    cudnnSetConvolution2dDescriptor(convDesc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

    float alpha = 1.f, beta = 0.f;
    bool ok = true;

    ok &= (cudnnConvolutionForward(
        mCudnn,
        &alpha,
        xDesc, x,
        wDesc, w,
        convDesc,
        algo,
        workspace, workspaceBytes,
        &beta,
        yDesc, y) == CUDNN_STATUS_SUCCESS);

    if (ok) {
        ok &= (cudnnAddTensor(
            mCudnn,
            &alpha,
            bDesc, b,
            &alpha,
            yDesc, y) == CUDNN_STATUS_SUCCESS);
    }

    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroyTensorDescriptor(bDesc);
    cudnnDestroyFilterDescriptor(wDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);

    return ok;
}

void YoloC2fM2Plugin::attachToContext(
    cudnnContext* cudnn,
    cublasContext*,
    IGpuAllocator*) noexcept {
    if (cudnn) {
        mCudnn = cudnn;
    }
}

void YoloC2fM2Plugin::detachFromContext() noexcept {
    // Do not destroy here if TRT owns it.
}

// ============================
// Creator
// ============================

YoloC2fM2PluginCreator::YoloC2fM2PluginCreator() {
    mPluginAttributes.emplace_back(
        PluginField{"weights_path", nullptr, PluginFieldType::kCHAR, 1});
    mFC.nbFields = static_cast<int>(mPluginAttributes.size());
    mFC.fields = mPluginAttributes.data();
}

const char* YoloC2fM2PluginCreator::getPluginName() const noexcept {
    return kPLUGIN_NAME;
}

const char* YoloC2fM2PluginCreator::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

const PluginFieldCollection* YoloC2fM2PluginCreator::getFieldNames() noexcept {
    return &mFC;
}

IPluginV2* YoloC2fM2PluginCreator::createPlugin(
    const char*,
    PluginFieldCollection const* fc) noexcept {

    std::string weightsPath;
    if (fc != nullptr) {
        for (int i = 0; i < fc->nbFields; ++i) {
            auto const& f = fc->fields[i];
            if (std::string(f.name) == "weights_path" && f.data != nullptr) {
                weightsPath = static_cast<const char*>(f.data);
            }
        }
    }

    auto* p = new YoloC2fM2Plugin(weightsPath);
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

IPluginV2* YoloC2fM2PluginCreator::deserializePlugin(
    const char*,
    void const* serialData,
    size_t serialLength) noexcept {

    auto* p = new YoloC2fM2Plugin(serialData, serialLength);
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

void YoloC2fM2PluginCreator::setPluginNamespace(const char* libNamespace) noexcept {
    mNamespace = libNamespace ? libNamespace : "";
}

const char* YoloC2fM2PluginCreator::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

REGISTER_TENSORRT_PLUGIN(YoloC2fM2PluginCreator);