#include "c2f_m2_plugin.hpp"

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include <iostream>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace nvinfer1;

namespace {
char const* kPLUGIN_NAME    = "YoloC2fM2_TRT";
char const* kPLUGIN_VERSION = "1";

// -------------------------------------------------------------------------
// Serialisation helpers
// -------------------------------------------------------------------------
template <typename T>
void writeToBuffer(char*& buf, T const& v) {
    std::memcpy(buf, &v, sizeof(T));
    buf += sizeof(T);
}
template <typename T>
T readFromBuffer(char const*& buf) {
    T v;
    std::memcpy(&v, buf, sizeof(T));
    buf += sizeof(T);
    return v;
}

// -------------------------------------------------------------------------
// Binary weight file reader
// Parses the format written by export_model2_weights.py:
//   magic(4)  num_tensors(u32)
//   [ name_len(u32) name(char[]) ndim(u32) dims(u32[]) data_len(u32) data(f32[]) ]*
// -------------------------------------------------------------------------
struct WeightEntry {
    std::string           name;
    std::vector<int>      shape;
    std::vector<float>    data;
};

static std::vector<WeightEntry> loadBinWeights(std::string const& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return {};

    char magic[4] = {};
    f.read(magic, 4);
    if (std::memcmp(magic, "C2FW", 4) != 0) return {};

    uint32_t num = 0;
    f.read(reinterpret_cast<char*>(&num), 4);

    std::vector<WeightEntry> entries(num);
    for (auto& e : entries) {
        uint32_t nameLen = 0;
        f.read(reinterpret_cast<char*>(&nameLen), 4);
        e.name.resize(nameLen);
        f.read(e.name.data(), nameLen);

        uint32_t ndim = 0;
        f.read(reinterpret_cast<char*>(&ndim), 4);
        e.shape.resize(ndim);
        for (auto& d : e.shape) {
            uint32_t tmp = 0;
            f.read(reinterpret_cast<char*>(&tmp), 4);
            d = static_cast<int>(tmp);
        }

        uint32_t dataBytes = 0;
        f.read(reinterpret_cast<char*>(&dataBytes), 4);
        e.data.resize(dataBytes / sizeof(float));
        f.read(reinterpret_cast<char*>(e.data.data()), dataBytes);

        if (!f) return {};   // truncated file
    }
    return entries;
}

// -------------------------------------------------------------------------
// GPU upload helpers
// -------------------------------------------------------------------------
static bool uploadToDevice(float** dst, std::vector<float> const& src) {
    if (src.empty()) return false;
    size_t bytes = src.size() * sizeof(float);
    if (cudaMalloc(dst, bytes) != cudaSuccess) return false;
    if (cudaMemcpy(*dst, src.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(*dst);
        *dst = nullptr;
        return false;
    }
    return true;
}
} // anonymous namespace


// ===========================================================================
// ConvDescSet
// ===========================================================================
void ConvDescSet::destroy() noexcept {
    if (xDesc) { cudnnDestroyTensorDescriptor(xDesc);      xDesc = nullptr; }
    if (wDesc) { cudnnDestroyFilterDescriptor(wDesc);      wDesc = nullptr; }
    if (cDesc) { cudnnDestroyConvolutionDescriptor(cDesc); cDesc = nullptr; }
    if (yDesc) { cudnnDestroyTensorDescriptor(yDesc);      yDesc = nullptr; }
    if (bDesc) { cudnnDestroyTensorDescriptor(bDesc);      bDesc = nullptr; }
    workspaceBytes = 0;
}


// ===========================================================================
// YoloC2fM2Plugin
// ===========================================================================

YoloC2fM2Plugin::YoloC2fM2Plugin(std::string weightsPath)
    : mWeightsPath(std::move(weightsPath)) {}

// Deserialise constructor
YoloC2fM2Plugin::YoloC2fM2Plugin(void const* data, size_t) {
    char const* d = static_cast<char const*>(data);

    int32_t wLen = readFromBuffer<int32_t>(d);
    mWeightsPath.assign(d, d + wLen);  d += wLen;

    int32_t nsLen = readFromBuffer<int32_t>(d);
    mNamespace.assign(d, d + nsLen);   d += nsLen;

    mCin    = readFromBuffer<int32_t>(d);
    mCout   = readFromBuffer<int32_t>(d);
    mHalfC  = readFromBuffer<int32_t>(d);
    mN      = readFromBuffer<int32_t>(d);
    mH      = readFromBuffer<int32_t>(d);
    mW      = readFromBuffer<int32_t>(d);
    mShortcut    = readFromBuffer<int32_t>(d) != 0;
    mInputFormat = static_cast<TensorFormat>(readFromBuffer<int32_t>(d));
}

YoloC2fM2Plugin::~YoloC2fM2Plugin() { terminate(); }

// ---------------------------------------------------------------------------
void YoloC2fM2Plugin::destroyWeights() noexcept {
    auto freePtr = [](float*& p) { if (p) { cudaFree(p); p = nullptr; } };
    freePtr(d_cv1_w);    freePtr(d_cv1_b);
    freePtr(d_m0_cv1_w); freePtr(d_m0_cv1_b);
    freePtr(d_m0_cv2_w); freePtr(d_m0_cv2_b);
    freePtr(d_cv2_w);    freePtr(d_cv2_b);
    mWinogradReady = false;
}

// ---------------------------------------------------------------------------
bool YoloC2fM2Plugin::loadWeightsToDevice() {
    std::cout << "[loadWeightsToDevice] path=" << mWeightsPath << std::endl;
    destroyWinogradFilters();

    if (mWeightsPath.empty()) {
        std::cout << "[loadWeightsToDevice] empty weights path" << std::endl;
        return false;
    }

    auto entries = loadBinWeights(mWeightsPath);
    if (entries.empty()) {
        std::cout << "[loadWeightsToDevice] failed to read weights file or file empty" << std::endl;
        return false;
    }

    std::unordered_map<std::string, WeightEntry*> idx;
    for (auto& e : entries) idx[e.name] = &e;

    auto get = [&](std::string const& name) -> WeightEntry* {
        auto it = idx.find(name);
        return (it != idx.end()) ? it->second : nullptr;
    };

    auto* meta_cin  = get("meta_cin");
    auto* meta_cout = get("meta_cout");
    if (meta_cin  && !meta_cin->data.empty())  mCin  = static_cast<int>(meta_cin->data[0]);
    if (meta_cout && !meta_cout->data.empty()) mCout = static_cast<int>(meta_cout->data[0]);
    mHalfC = mCout / 2;

    auto* sc = get("shortcut");
    if (sc && !sc->data.empty()) mShortcut = sc->data[0] > 0.5f;

    auto* cv1_w    = get("cv1_w");
    auto* cv1_b    = get("cv1_b");
    auto* m0cv1_w  = get("m0_cv1_w");
    auto* m0cv1_b  = get("m0_cv1_b");
    auto* m0cv2_w  = get("m0_cv2_w");
    auto* m0cv2_b  = get("m0_cv2_b");
    auto* cv2_w    = get("cv2_w");
    auto* cv2_b    = get("cv2_b");

    if (!cv1_w || !cv1_b || !m0cv1_w || !m0cv1_b ||
        !m0cv2_w || !m0cv2_b || !cv2_w || !cv2_b) {
        std::cout << "[loadWeightsToDevice] missing required tensors" << std::endl;
        std::cout << "  cv1_w=" << (cv1_w != nullptr)
                  << " cv1_b=" << (cv1_b != nullptr)
                  << " m0_cv1_w=" << (m0cv1_w != nullptr)
                  << " m0_cv1_b=" << (m0cv1_b != nullptr)
                  << " m0_cv2_w=" << (m0cv2_w != nullptr)
                  << " m0_cv2_b=" << (m0cv2_b != nullptr)
                  << " cv2_w=" << (cv2_w != nullptr)
                  << " cv2_b=" << (cv2_b != nullptr)
                  << std::endl;
        return false;
    }

    if (cv1_w->shape.size() == 4) {
        mCout  = cv1_w->shape[0];
        mCin   = cv1_w->shape[1];
        mHalfC = mCout / 2;
    }

    std::cout << "[loadWeightsToDevice] derived dims"
              << " mCin=" << mCin
              << " mCout=" << mCout
              << " mHalfC=" << mHalfC
              << " shortcut=" << mShortcut
              << std::endl;

    h_cv1_w    = cv1_w->data;    h_cv1_b    = cv1_b->data;
    h_m0_cv1_w = m0cv1_w->data;  h_m0_cv1_b = m0cv1_b->data;
    h_m0_cv2_w = m0cv2_w->data;  h_m0_cv2_b = m0cv2_b->data;
    h_cv2_w    = cv2_w->data;    h_cv2_b    = cv2_b->data;

    bool ok = true;
    ok &= uploadToDevice(&d_cv1_w,    h_cv1_w);
    ok &= uploadToDevice(&d_cv1_b,    h_cv1_b);
    ok &= uploadToDevice(&d_m0_cv1_w, h_m0_cv1_w);
    ok &= uploadToDevice(&d_m0_cv1_b, h_m0_cv1_b);
    ok &= uploadToDevice(&d_m0_cv2_w, h_m0_cv2_w);
    ok &= uploadToDevice(&d_m0_cv2_b, h_m0_cv2_b);
    ok &= uploadToDevice(&d_cv2_w,    h_cv2_w);
    ok &= uploadToDevice(&d_cv2_b,    h_cv2_b);

    std::cout << "[loadWeightsToDevice] upload ok=" << ok << std::endl;
    mWinogradReady = false;
    return ok;
}

void YoloC2fM2Plugin::destroyWinogradFilters() noexcept {
    auto freePtr = [](float*& p) { if (p) { cudaFree(p); p = nullptr; } };
    freePtr(d_m0_cv1_wino);
    freePtr(d_m0_cv2_wino);
    mWinogradReady = false;
}

bool YoloC2fM2Plugin::precomputeWinogradFilters() noexcept {
    destroyWinogradFilters();

    if (h_m0_cv1_w.empty() || h_m0_cv2_w.empty()) {
        std::cerr << "[precomputeWinogradFilters] missing host weights" << std::endl;
        return false;
    }

    bool ok = true;
    ok &= precomputeWinoFilterTransform(
        h_m0_cv1_w.data(), mHalfC, mHalfC, &d_m0_cv1_wino);
    ok &= precomputeWinoFilterTransform(
        h_m0_cv2_w.data(), mHalfC, mHalfC, &d_m0_cv2_wino);

    if (!ok) {
        std::cerr << "[precomputeWinogradFilters] transform failed; using cuDNN fallback" << std::endl;
        destroyWinogradFilters();
        return false;
    }

    mWinogradReady = true;
    return true;
}

bool YoloC2fM2Plugin::canUseWinograd(
    int N, int C, int K, int H, int W,
    int kH, int kW, int stride, int pad) const noexcept
{
    if (!mWinogradReady) return false;
    if (N != 1 || C != 16 || K != 16) return false;
    if (H <= 0 || W <= 0) return false;
    if (kH != 3 || kW != 3) return false;
    if (stride != 1 || pad != 1) return false;
    return true;
}

// ---------------------------------------------------------------------------
// Build a single cuDNN convolution descriptor set.
// Enables TF32 tensor-op math on Ampere (sm87 Orin GA10B) for ~4× speedup.
// Uses cudnnGetConvolutionForwardAlgorithm_v7 for heuristic-based algo
// selection without benchmarking (safe at engine-build time).
// ---------------------------------------------------------------------------
bool YoloC2fM2Plugin::buildOneDescSet(
    ConvDescSet& d,
    int N, int Cin, int Cout, int H, int W,
    int kH, int kW, int padH, int padW,
    cudnnTensorFormat_t xFmt, cudnnTensorFormat_t yFmt) noexcept
{
    std::cout << "[buildOneDescSet] enter"
              << " N=" << N
              << " Cin=" << Cin
              << " Cout=" << Cout
              << " H=" << H
              << " W=" << W
              << " k=" << kH << "x" << kW
              << " pad=" << padH << "," << padW
              << std::endl;

    if (cudnnCreateTensorDescriptor(&d.xDesc) != CUDNN_STATUS_SUCCESS) {
        std::cout << "[buildOneDescSet] cudnnCreateTensorDescriptor x FAILED" << std::endl;
        return false;
    }
    if (cudnnCreateFilterDescriptor(&d.wDesc) != CUDNN_STATUS_SUCCESS) {
        std::cout << "[buildOneDescSet] cudnnCreateFilterDescriptor FAILED" << std::endl;
        return false;
    }
    if (cudnnCreateConvolutionDescriptor(&d.cDesc) != CUDNN_STATUS_SUCCESS) {
        std::cout << "[buildOneDescSet] cudnnCreateConvolutionDescriptor FAILED" << std::endl;
        return false;
    }
    if (cudnnCreateTensorDescriptor(&d.yDesc) != CUDNN_STATUS_SUCCESS) {
        std::cout << "[buildOneDescSet] cudnnCreateTensorDescriptor y FAILED" << std::endl;
        return false;
    }
    if (cudnnCreateTensorDescriptor(&d.bDesc) != CUDNN_STATUS_SUCCESS) {
        std::cout << "[buildOneDescSet] cudnnCreateTensorDescriptor b FAILED" << std::endl;
        return false;
    }

    if (cudnnSetTensor4dDescriptor(
            d.xDesc, xFmt, CUDNN_DATA_FLOAT, N, Cin, H, W) != CUDNN_STATUS_SUCCESS) {
        std::cout << "[buildOneDescSet] cudnnSetTensor4dDescriptor x FAILED"
                  << " N=" << N << " Cin=" << Cin << " H=" << H << " W=" << W
                  << " xFmt=" << xFmt << std::endl;
        return false;
    }

    if (cudnnSetTensor4dDescriptor(
            d.yDesc, yFmt, CUDNN_DATA_FLOAT, N, Cout, H, W) != CUDNN_STATUS_SUCCESS) {
        std::cout << "[buildOneDescSet] cudnnSetTensor4dDescriptor y FAILED"
                  << " N=" << N << " Cout=" << Cout << " H=" << H << " W=" << W
                  << " yFmt=" << yFmt << std::endl;
        return false;
    }

    if (cudnnSetTensor4dDescriptor(
            d.bDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, Cout, 1, 1) != CUDNN_STATUS_SUCCESS) {
        std::cout << "[buildOneDescSet] cudnnSetTensor4dDescriptor b FAILED"
                  << " Cout=" << Cout << std::endl;
        return false;
    }

    if (cudnnSetFilter4dDescriptor(
            d.wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, Cout, Cin, kH, kW) != CUDNN_STATUS_SUCCESS) {
        std::cout << "[buildOneDescSet] cudnnSetFilter4dDescriptor FAILED"
                  << " Cout=" << Cout << " Cin=" << Cin
                  << " kH=" << kH << " kW=" << kW << std::endl;
        return false;
    }

    if (cudnnSetConvolution2dDescriptor(
            d.cDesc, padH, padW, 1, 1, 1, 1,
            CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT) != CUDNN_STATUS_SUCCESS) {
        std::cout << "[buildOneDescSet] cudnnSetConvolution2dDescriptor FAILED"
                  << " padH=" << padH << " padW=" << padW << std::endl;
        return false;
    }

    if (cudnnSetConvolutionMathType(
            d.cDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION) != CUDNN_STATUS_SUCCESS) {
        std::cout << "[buildOneDescSet] cudnnSetConvolutionMathType FAILED" << std::endl;
        return false;
    }

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    int retCount = 0;
    cudnnConvolutionFwdAlgoPerf_t perf{};
    if (cudnnGetConvolutionForwardAlgorithm_v7(
            mCudnn, d.xDesc, d.wDesc, d.cDesc, d.yDesc,
            1, &retCount, &perf) != CUDNN_STATUS_SUCCESS || retCount == 0) {
        std::cout << "[buildOneDescSet] cudnnGetConvolutionForwardAlgorithm_v7 fallback" << std::endl;
        d.algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    } else {
        d.algo = perf.algo;
    }
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

    if (cudnnGetConvolutionForwardWorkspaceSize(
            mCudnn, d.xDesc, d.wDesc, d.cDesc, d.yDesc, d.algo, &d.workspaceBytes) != CUDNN_STATUS_SUCCESS) {
        std::cout << "[buildOneDescSet] cudnnGetConvolutionForwardWorkspaceSize FAILED" << std::endl;
        return false;
    }

    std::cout << "[buildOneDescSet] OK"
              << " workspaceBytes=" << d.workspaceBytes
              << std::endl;

    return true;
}

bool YoloC2fM2Plugin::buildDescSets() noexcept {
    std::cout << "[buildDescSets] enter"
              << " mN=" << mN
              << " mCin=" << mCin
              << " mCout=" << mCout
              << " mHalfC=" << mHalfC
              << " mH=" << mH
              << " mW=" << mW
              << std::endl;

    if (!mCudnn || mCin <= 0 || mCout <= 0 || mH <= 0 || mW <= 0) {
        std::cout << "[buildDescSets] invalid state"
                  << " mCudnn=" << (mCudnn != nullptr)
                  << " mCin=" << mCin
                  << " mCout=" << mCout
                  << " mH=" << mH
                  << " mW=" << mW
                  << std::endl;
        return false;
    }

    destroyDescSets();

    // kCHW32 (= 5) is TRT's Ampere FP32 NHWC-vectorised layout.
    // For our tensors (C = 32 exactly) it is byte-equivalent to NHWC.
    // Using CUDNN_TENSOR_NHWC for the plugin I/O descriptors lets cuDNN consume
    // and produce the tensor directly without any format-conversion copy node.
    // All intermediate (workspace) tensors stay NCHW — slice / add kernels only
    // work with NCHW, and workspace buffers have no TRT format constraints.
    cudnnTensorFormat_t ioFmt =
        (mInputFormat == TensorFormat::kCHW32) ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;

    bool ok = true;
    // cv1: reads plugin input (ioFmt), writes workspace (NCHW)
    ok &= buildOneDescSet(mCv1Desc,   mN, mCin,     mCout,  mH, mW, 1, 1, 0, 0, ioFmt,             CUDNN_TENSOR_NCHW);
    // m0.cv1 / m0.cv2: workspace → workspace, always NCHW
    ok &= buildOneDescSet(mM0Cv1Desc, mN, mHalfC,   mHalfC, mH, mW, 3, 3, 1, 1, CUDNN_TENSOR_NCHW, CUDNN_TENSOR_NCHW);
    ok &= buildOneDescSet(mM0Cv2Desc, mN, mHalfC,   mHalfC, mH, mW, 3, 3, 1, 1, CUDNN_TENSOR_NCHW, CUDNN_TENSOR_NCHW);
    // cv2: reads workspace (NCHW), writes plugin output (ioFmt)
    ok &= buildOneDescSet(mCv2Desc,   mN, 3*mHalfC, mCout,  mH, mW, 1, 1, 0, 0, CUDNN_TENSOR_NCHW, ioFmt);

    mDescsCached = ok;
    std::cout << "[buildDescSets] result=" << ok << std::endl;
    return ok;
}

void YoloC2fM2Plugin::destroyDescSets() noexcept {
    mCv1Desc.destroy();
    mM0Cv1Desc.destroy();
    mM0Cv2Desc.destroy();
    mCv2Desc.destroy();
    mDescsCached = false;
}

// ---------------------------------------------------------------------------
bool YoloC2fM2Plugin::runConv(
    ConvDescSet const& d,
    float const* x, float* y,
    float const* w, float const* b,
    void* workspace, cudaStream_t stream,
    bool addBias) const noexcept
{
    if (!mCudnn || !d.xDesc) {
        std::cerr << "[runConv] invalid state"
                  << " mCudnn=" << (mCudnn != nullptr)
                  << " d.xDesc=" << (d.xDesc != nullptr)
                  << std::endl;
        return false;
    }

    cudnnStatus_t st = cudnnSetStream(mCudnn, stream);
    if (st != CUDNN_STATUS_SUCCESS) {
        std::cerr << "[runConv] cudnnSetStream FAILED: "
                  << cudnnGetErrorString(st) << std::endl;
        return false;
    }

    float alpha = 1.f, beta = 0.f;

    st = cudnnConvolutionForward(
        mCudnn, &alpha,
        d.xDesc, x,
        d.wDesc, w,
        d.cDesc, d.algo,
        workspace, d.workspaceBytes,
        &beta,
        d.yDesc, y);

    if (st != CUDNN_STATUS_SUCCESS) {
        std::cerr << "[runConv] cudnnConvolutionForward FAILED: "
                  << cudnnGetErrorString(st)
                  << " workspaceBytes=" << d.workspaceBytes
                  << std::endl;
        return false;
    }

    if (addBias) {
        st = cudnnAddTensor(
            mCudnn, &alpha,
            d.bDesc, b,
            &alpha,
            d.yDesc, y);

        if (st != CUDNN_STATUS_SUCCESS) {
            std::cerr << "[runConv] cudnnAddTensor FAILED: "
                      << cudnnGetErrorString(st) << std::endl;
            return false;
        }
    }

    return true;
}


// ===========================================================================
// IPluginV2 interface
// ===========================================================================

const char* YoloC2fM2Plugin::getPluginType()    const noexcept { return kPLUGIN_NAME;    }
const char* YoloC2fM2Plugin::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
int         YoloC2fM2Plugin::getNbOutputs()     const noexcept { return 1; }

int YoloC2fM2Plugin::initialize() noexcept {
    std::cerr << "[initialize] enter"
              << " mCin=" << mCin
              << " mCout=" << mCout
              << " mHalfC=" << mHalfC
              << " mN=" << mN
              << " mH=" << mH
              << " mW=" << mW
              << " path=" << mWeightsPath
              << std::endl;

    if (!mCudnn) {
        if (cudnnCreate(&mCudnn) != CUDNN_STATUS_SUCCESS) {
            std::cerr << "[initialize] cudnnCreate FAILED" << std::endl;
            mCudnn = nullptr;
            return 1;
        }
        mCudnnOwned = true;
    }

    if (!d_cv1_w || !d_cv1_b || !d_m0_cv1_w || !d_m0_cv1_b ||
        !d_m0_cv2_w || !d_m0_cv2_b || !d_cv2_w || !d_cv2_b) {
        if (!loadWeightsToDevice()) {
            std::cerr << "[initialize] loadWeightsToDevice FAILED" << std::endl;
            return 1;
        }
    }

    if (!mDescsCached) {
        if (!buildDescSets()) {
            std::cerr << "[initialize] buildDescSets FAILED" << std::endl;
            return 1;
        }
    }

    // Optional fast path. If this fails, enqueue() will fall back to cuDNN.
    if (!mWinogradReady) {
        precomputeWinogradFilters();
    }

    std::cerr << "[initialize] OK" << std::endl;
    return 0;
}

void YoloC2fM2Plugin::terminate() noexcept {
    destroyDescSets();
    destroyWinogradFilters();
    destroyWeights();

    if (mCudnn && mCudnnOwned) {
        cudnnDestroy(mCudnn);
        mCudnn      = nullptr;
        mCudnnOwned = false;
    }
}

// ---------------------------------------------------------------------------
size_t YoloC2fM2Plugin::getSerializationSize() const noexcept {
    return sizeof(int32_t) + mWeightsPath.size()
         + sizeof(int32_t) + mNamespace.size()
         + sizeof(int32_t) * 8;   // Cin, Cout, halfC, N, H, W, shortcut, inputFormat
}

void YoloC2fM2Plugin::serialize(void* buffer) const noexcept {
    char* d = static_cast<char*>(buffer);

    int32_t wLen = static_cast<int32_t>(mWeightsPath.size());
    writeToBuffer(d, wLen);
    std::memcpy(d, mWeightsPath.data(), mWeightsPath.size()); d += mWeightsPath.size();

    int32_t nsLen = static_cast<int32_t>(mNamespace.size());
    writeToBuffer(d, nsLen);
    std::memcpy(d, mNamespace.data(), mNamespace.size()); d += mNamespace.size();

    writeToBuffer(d, static_cast<int32_t>(mCin));
    writeToBuffer(d, static_cast<int32_t>(mCout));
    writeToBuffer(d, static_cast<int32_t>(mHalfC));
    writeToBuffer(d, static_cast<int32_t>(mN));
    writeToBuffer(d, static_cast<int32_t>(mH));
    writeToBuffer(d, static_cast<int32_t>(mW));
    writeToBuffer(d, static_cast<int32_t>(mShortcut ? 1 : 0));
    writeToBuffer(d, static_cast<int32_t>(mInputFormat));

    std::cout << "[serialize] mCin=" << mCin
          << " mCout=" << mCout
          << " mHalfC=" << mHalfC
          << " mH=" << mH
          << " mW=" << mW
          << std::endl;
}

void YoloC2fM2Plugin::destroy() noexcept { delete this; }

void YoloC2fM2Plugin::setPluginNamespace(char const* ns) noexcept {
    mNamespace = ns ? ns : "";
}
char const* YoloC2fM2Plugin::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

DataType YoloC2fM2Plugin::getOutputDataType(int, DataType const* inputTypes, int) const noexcept {
    return inputTypes[0];
}

IPluginV2DynamicExt* YoloC2fM2Plugin::clone() const noexcept {
    std::cerr << "[clone] mCin=" << mCin
              << " mCout=" << mCout
              << " mHalfC=" << mHalfC
              << " mN=" << mN
              << " mH=" << mH
              << " mW=" << mW
              << " path=" << mWeightsPath
              << std::endl;

    auto* p = new YoloC2fM2Plugin(mWeightsPath);
    p->mCin      = mCin;
    p->mCout     = mCout;
    p->mHalfC    = mHalfC;
    p->mN        = mN;
    p->mH        = mH;
    p->mW        = mW;
    p->mShortcut     = mShortcut;
    p->mInputFormat  = mInputFormat;
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

// ---------------------------------------------------------------------------
// FIX: output has Cout channels (e.g. 64), NOT Cin (e.g. 32).
// If mCout is not yet populated (build-time before weights are loaded),
// we cannot return the right dims — caller must supply cin/cout via
// plugin fields or peek the weight file before getOutputDimensions.
// ---------------------------------------------------------------------------
DimsExprs YoloC2fM2Plugin::getOutputDimensions(
    int, DimsExprs const* inputs, int, IExprBuilder& eb) noexcept
{
    std::cout << "[getOutputDimensions] mCin=" << mCin
              << " mCout=" << mCout
              << " mHalfC=" << mHalfC
              << std::endl;

    DimsExprs out = inputs[0];
    if (mCout > 0) {
        out.d[1] = eb.constant(mCout);
    }
    return out;
}

bool YoloC2fM2Plugin::supportsFormatCombination(
    int pos, PluginTensorDesc const* inOut, int, int) noexcept
{
    auto const& desc = inOut[pos];
    // Only FP32
    if (desc.type != DataType::kFLOAT) return false;
    // Accept kCHW32 (Ampere NHWC-vectorised, eliminates the reformat copy node)
    // and kLINEAR (NCHW, safe fallback on non-Ampere / older cuDNN).
    if (desc.format != TensorFormat::kCHW32 && desc.format != TensorFormat::kLINEAR)
        return false;
    // Require all I/O tensors to share the same format so TRT doesn't insert
    // an additional inter-tensor reformat between input and output.
    if (pos > 0) return desc.format == inOut[0].format;
    return true;
}

void YoloC2fM2Plugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int nbInputs,
    DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
    auto printDims = [](char const* label, Dims const& d) {
        std::cout << label << " nbDims=" << d.nbDims << " (";
        for (int i = 0; i < d.nbDims; ++i) {
            std::cout << d.d[i];
            if (i + 1 < d.nbDims) std::cout << ", ";
        }
        std::cout << ")";
    };

    if (nbInputs < 1) {
        std::cout << "[configurePlugin] nbInputs < 1" << std::endl;
        return;
    }

    auto const& descDims = in[0].desc.dims;
    auto const& minDims  = in[0].min;
    auto const& optDims  = in[0].opt;
    auto const& maxDims  = in[0].max;

    std::cout << "[configurePlugin] ";
    printDims("desc", descDims);
    std::cout << " ";
    printDims("min", minDims);
    std::cout << " ";
    printDims("opt", optDims);
    std::cout << " ";
    printDims("max", maxDims);
    std::cout << std::endl;

    auto pickDims = [&]() -> Dims const* {
        if (descDims.nbDims > 0) return &descDims;
        if (optDims.nbDims > 0)  return &optDims;
        if (maxDims.nbDims > 0)  return &maxDims;
        if (minDims.nbDims > 0)  return &minDims;
        return nullptr;
    };

    Dims const* chosen = pickDims();
    if (!chosen) {
        std::cout << "[configurePlugin] no valid dims available" << std::endl;
        return;
    }

    // For explicit-batch networks, plugins often see CHW here.
    // If TRT gives NCHW, we handle that too.
    if (chosen->nbDims == 3) {
        // [C, H, W]
        if (mCin == 0 && chosen->d[0] > 0) mCin = chosen->d[0];
        if (chosen->d[1] > 0) mH = chosen->d[1];
        if (chosen->d[2] > 0) mW = chosen->d[2];
        mN = 1;
    } else if (chosen->nbDims == 4) {
        // [N, C, H, W]
        if (chosen->d[0] > 0) mN = chosen->d[0];
        else mN = 1;
        if (mCin == 0 && chosen->d[1] > 0) mCin = chosen->d[1];
        if (chosen->d[2] > 0) mH = chosen->d[2];
        if (chosen->d[3] > 0) mW = chosen->d[3];
    } else {
        std::cout << "[configurePlugin] unsupported chosen->nbDims="
                  << chosen->nbDims << std::endl;
        return;
    }

    std::cout << "[configurePlugin] resolved"
              << " mN=" << mN
              << " mCin=" << mCin
              << " mCout=" << mCout
              << " mHalfC=" << mHalfC
              << " mH=" << mH
              << " mW=" << mW
              << std::endl;

    // Store the format TRT negotiated (kCHW32 or kLINEAR).
    // This must be done before buildDescSets so the right cuDNN tensor format
    // (NHWC vs NCHW) is used for the I/O descriptors.
    mInputFormat = in[0].desc.format;
    mDescsCached = false;   // force rebuild with the correct format

    std::cout << "[configurePlugin] mInputFormat=" << static_cast<int>(mInputFormat)
              << " (5=kCHW32/NHWC, 0=kLINEAR/NCHW)" << std::endl;

    // Create cuDNN handle here (if not already present) so that buildDescSets()
    // can compute accurate cuDNN workspace sizes before getWorkspaceSize() is
    // called by TRT during engine build.
    if (!mCudnn) {
        cudnnStatus_t cs = cudnnCreate(&mCudnn);
        if (cs != CUDNN_STATUS_SUCCESS) {
            std::cerr << "[configurePlugin] cudnnCreate FAILED: "
                      << cudnnGetErrorString(cs) << std::endl;
            return;
        }
        mCudnnOwned = true;
    }
    buildDescSets();
}

// ---------------------------------------------------------------------------
// Workspace layout (enqueue cuDNN path — all intermediate buffers NCHW):
//   [0] cv1_out    : N * Cout     * H * W  floats  (cv1 output, pre-slice)
//   [1] m0cv1_out  : N * halfC    * H * W  floats  (m0.cv1 output)
//   [2] concat     : N * 3*halfC  * H * W  floats  (cv2 input, 3 segments)
//         seg0 [     0 ..  halfC) : x1  — sliced from cv1_out[:,  0:halfC]
//         seg1 [  halfC.. 2*halfC): x2  — sliced from cv1_out[:,halfC:Cout]
//                                        also the shortcut source for m0
//         seg2 [2*halfC.. 3*halfC): m0out — m0.cv2 output + x2 residual
//   [3] conv_ws    : max cuDNN workspace across the 4 convolutions
// ---------------------------------------------------------------------------
size_t YoloC2fM2Plugin::getWorkspaceSize(
    PluginTensorDesc const*,
    int,
    PluginTensorDesc const*,
    int) const noexcept
{
    if (mH <= 0 || mW <= 0 || mCout <= 0 || mHalfC <= 0) return 0;

    size_t HW       = static_cast<size_t>(mN) * mH * mW;
    // cv1_out + m0cv1_out + concat  = (Cout + halfC + 3*halfC) channels
    size_t tensorWs = (static_cast<size_t>(mCout) + mHalfC + 3 * mHalfC)
                      * HW * sizeof(float);

    // cuDNN algorithm workspace — exact if descriptors already built,
    // conservative 64 MB if called before initialize() (engine-build time).
    size_t convWs = 0;
    if (mDescsCached) {
        convWs = std::max(
            std::max(mCv1Desc.workspaceBytes,   mM0Cv1Desc.workspaceBytes),
            std::max(mM0Cv2Desc.workspaceBytes, mCv2Desc.workspaceBytes));
    } else {
        convWs = 64ULL * 1024 * 1024;  // 64 MB upper bound for heuristic algo
    }

    return tensorWs + convWs;
}

// ---------------------------------------------------------------------------
// attachToContext / detachFromContext — TRT 8.x only.
// TRT 10.x (JetPack 6.x) removed these from IPluginV2DynamicExt.
// The plugin always creates its own cuDNN handle in initialize(), so these
// are an optional optimisation (share TRT's handle) rather than required.
#if NV_TENSORRT_MAJOR < 10
void YoloC2fM2Plugin::attachToContext(
    cudnnContext* cudnn, cublasContext*, IGpuAllocator*) noexcept
{
    if (cudnn) {
        if (mCudnnOwned && mCudnn) {
            cudnnDestroy(mCudnn);
            mCudnnOwned = false;
        }
        mCudnn = cudnn;
        if (mCin > 0 && mCout > 0 && mH > 0 && mW > 0)
            buildDescSets();
    }
}

void YoloC2fM2Plugin::detachFromContext() noexcept {
    // Do not destroy TRT's shared handle
}
#endif


// ===========================================================================
// Creator
// ===========================================================================

YoloC2fM2PluginCreator::YoloC2fM2PluginCreator() {
    mPluginAttributes.emplace_back(
        PluginField{"weights_path", nullptr, PluginFieldType::kCHAR, 1});
    mPluginAttributes.emplace_back(
        PluginField{"cin", nullptr, PluginFieldType::kINT32, 1});
    mPluginAttributes.emplace_back(
        PluginField{"cout", nullptr, PluginFieldType::kINT32, 1});
    mPluginAttributes.emplace_back(
        PluginField{"halfc", nullptr, PluginFieldType::kINT32, 1});

    mFC.nbFields = static_cast<int>(mPluginAttributes.size());
    mFC.fields   = mPluginAttributes.data();
}

const char* YoloC2fM2PluginCreator::getPluginName()    const noexcept { return kPLUGIN_NAME;    }
const char* YoloC2fM2PluginCreator::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
const PluginFieldCollection* YoloC2fM2PluginCreator::getFieldNames() noexcept { return &mFC; }

IPluginV2* YoloC2fM2PluginCreator::createPlugin(
    const char*, PluginFieldCollection const* fc) noexcept
{
    std::string weightsPath;
    int cin = 0, cout = 0, halfc = 0;

    if (fc) {
        for (int i = 0; i < fc->nbFields; ++i) {
            auto const& f = fc->fields[i];
            if (!f.data || !f.name) continue;

            std::string name = f.name;

            if (name == "weights_path") {
                weightsPath = static_cast<char const*>(f.data);
            } else if (name == "cin") {
                if (f.type == PluginFieldType::kINT32) {
                    cin = *static_cast<int const*>(f.data);
                } else if (f.type == PluginFieldType::kINT64) {
                    cin = static_cast<int>(*static_cast<int64_t const*>(f.data));
                }
            } else if (name == "cout") {
                if (f.type == PluginFieldType::kINT32) {
                    cout = *static_cast<int const*>(f.data);
                } else if (f.type == PluginFieldType::kINT64) {
                    cout = static_cast<int>(*static_cast<int64_t const*>(f.data));
                }
            } else if (name == "halfc") {
                if (f.type == PluginFieldType::kINT32) {
                    halfc = *static_cast<int const*>(f.data);
                } else if (f.type == PluginFieldType::kINT64) {
                    halfc = static_cast<int>(*static_cast<int64_t const*>(f.data));
                }
            }
        }
    }

    std::cout << "[createPlugin] weightsPath=" << weightsPath
          << " cin=" << cin
          << " cout=" << cout
          << " halfc=" << halfc
          << std::endl;

    auto* p = new YoloC2fM2Plugin(weightsPath);
    p->setStaticDims(cin, cout, (halfc > 0) ? halfc : (cout / 2));
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

IPluginV2* YoloC2fM2PluginCreator::deserializePlugin(
    const char*, void const* data, size_t len) noexcept
{
    auto* p = new YoloC2fM2Plugin(data, len);
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

void        YoloC2fM2PluginCreator::setPluginNamespace(char const* ns) noexcept { mNamespace = ns ? ns : ""; }
const char* YoloC2fM2PluginCreator::getPluginNamespace() const noexcept         { return mNamespace.c_str(); }

REGISTER_TENSORRT_PLUGIN(YoloC2fM2PluginCreator);
