#include "c2f_m2_plugin.hpp"

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>
#include <cuda_runtime.h>
#include <cudnn.h>

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
    mShortcut = readFromBuffer<int32_t>(d) != 0;
}

YoloC2fM2Plugin::~YoloC2fM2Plugin() { terminate(); }

// ---------------------------------------------------------------------------
void YoloC2fM2Plugin::destroyWeights() noexcept {
    auto freePtr = [](float*& p) { if (p) { cudaFree(p); p = nullptr; } };
    freePtr(d_cv1_w);    freePtr(d_cv1_b);
    freePtr(d_m0_cv1_w); freePtr(d_m0_cv1_b);
    freePtr(d_m0_cv2_w); freePtr(d_m0_cv2_b);
    freePtr(d_cv2_w);    freePtr(d_cv2_b);
}

// ---------------------------------------------------------------------------
bool YoloC2fM2Plugin::loadWeightsToDevice() {
    if (mWeightsPath.empty()) return false;

    auto entries = loadBinWeights(mWeightsPath);
    if (entries.empty()) return false;

    // Index by name for easy lookup
    std::unordered_map<std::string, WeightEntry*> idx;
    for (auto& e : entries) idx[e.name] = &e;

    auto get = [&](std::string const& name) -> WeightEntry* {
        auto it = idx.find(name);
        return (it != idx.end()) ? it->second : nullptr;
    };

    // Read channel metadata written by the Python exporter
    auto* meta_cin  = get("meta_cin");
    auto* meta_cout = get("meta_cout");
    if (meta_cin  && !meta_cin->data.empty())  mCin   = static_cast<int>(meta_cin->data[0]);
    if (meta_cout && !meta_cout->data.empty()) mCout  = static_cast<int>(meta_cout->data[0]);
    mHalfC = mCout / 2;

    auto* sc = get("shortcut");
    if (sc && !sc->data.empty()) mShortcut = sc->data[0] > 0.5f;

    // Validate expected tensors exist
    auto* cv1_w    = get("cv1_w");
    auto* cv1_b    = get("cv1_b");
    auto* m0cv1_w  = get("m0_cv1_w");
    auto* m0cv1_b  = get("m0_cv1_b");
    auto* m0cv2_w  = get("m0_cv2_w");
    auto* m0cv2_b  = get("m0_cv2_b");
    auto* cv2_w    = get("cv2_w");
    auto* cv2_b    = get("cv2_b");

    if (!cv1_w || !cv1_b || !m0cv1_w || !m0cv1_b ||
        !m0cv2_w || !m0cv2_b || !cv2_w || !cv2_b)
        return false;

    // Derive dimensions from weight shapes (overrides meta if present)
    // cv1_w : [Cout, Cin, 1, 1]
    if (cv1_w->shape.size() == 4) {
        mCout  = cv1_w->shape[0];
        mCin   = cv1_w->shape[1];
        mHalfC = mCout / 2;
    }

    // Cache host copies for serialisation
    h_cv1_w    = cv1_w->data;    h_cv1_b    = cv1_b->data;
    h_m0_cv1_w = m0cv1_w->data;  h_m0_cv1_b = m0cv1_b->data;
    h_m0_cv2_w = m0cv2_w->data;  h_m0_cv2_b = m0cv2_b->data;
    h_cv2_w    = cv2_w->data;    h_cv2_b    = cv2_b->data;

    // Upload to GPU
    bool ok = true;
    ok &= uploadToDevice(&d_cv1_w,    h_cv1_w);
    ok &= uploadToDevice(&d_cv1_b,    h_cv1_b);
    ok &= uploadToDevice(&d_m0_cv1_w, h_m0_cv1_w);
    ok &= uploadToDevice(&d_m0_cv1_b, h_m0_cv1_b);
    ok &= uploadToDevice(&d_m0_cv2_w, h_m0_cv2_w);
    ok &= uploadToDevice(&d_m0_cv2_b, h_m0_cv2_b);
    ok &= uploadToDevice(&d_cv2_w,    h_cv2_w);
    ok &= uploadToDevice(&d_cv2_b,    h_cv2_b);
    return ok;
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
    int kH, int kW, int padH, int padW) noexcept
{
    if (cudnnCreateTensorDescriptor(&d.xDesc)      != CUDNN_STATUS_SUCCESS) return false;
    if (cudnnCreateFilterDescriptor(&d.wDesc)      != CUDNN_STATUS_SUCCESS) return false;
    if (cudnnCreateConvolutionDescriptor(&d.cDesc) != CUDNN_STATUS_SUCCESS) return false;
    if (cudnnCreateTensorDescriptor(&d.yDesc)      != CUDNN_STATUS_SUCCESS) return false;
    if (cudnnCreateTensorDescriptor(&d.bDesc)      != CUDNN_STATUS_SUCCESS) return false;

    cudnnSetTensor4dDescriptor(d.xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, Cin,  H, W);
    cudnnSetTensor4dDescriptor(d.yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, Cout, H, W);
    cudnnSetTensor4dDescriptor(d.bDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, Cout, 1, 1);
    cudnnSetFilter4dDescriptor(d.wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                               Cout, Cin, kH, kW);
    cudnnSetConvolution2dDescriptor(d.cDesc, padH, padW, 1, 1, 1, 1,
                                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    // Enable TF32 — lets Ampere TensorCores handle FP32 convolutions at
    // TF32 precision (negligible accuracy loss for inference).
    cudnnSetConvolutionMathType(d.cDesc, CUDNN_TF32_TENSOR_OP_MATH_ALLOW_CONVERSION);

    // Heuristic algo selection (no benchmark allocation needed).
    // cudnnGetConvolutionForwardAlgorithm_v7 is deprecated in cuDNN 9.x but
    // still present and functional — suppress the deprecation warning.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    int retCount = 0;
    cudnnConvolutionFwdAlgoPerf_t perf{};
    if (cudnnGetConvolutionForwardAlgorithm_v7(
            mCudnn, d.xDesc, d.wDesc, d.cDesc, d.yDesc,
            1, &retCount, &perf) != CUDNN_STATUS_SUCCESS || retCount == 0) {
        d.algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    } else {
        d.algo = perf.algo;
    }
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

    // Query exact workspace bytes for the chosen algorithm
    cudnnGetConvolutionForwardWorkspaceSize(
        mCudnn, d.xDesc, d.wDesc, d.cDesc, d.yDesc, d.algo, &d.workspaceBytes);

    return true;
}

bool YoloC2fM2Plugin::buildDescSets() noexcept {
    if (!mCudnn || mCin <= 0 || mCout <= 0 || mH <= 0 || mW <= 0) return false;
    destroyDescSets();

    bool ok = true;
    ok &= buildOneDescSet(mCv1Desc,   mN, mCin,    mCout,        mH, mW, 1, 1, 0, 0);
    ok &= buildOneDescSet(mM0Cv1Desc, mN, mHalfC,  mHalfC,       mH, mW, 3, 3, 1, 1);
    ok &= buildOneDescSet(mM0Cv2Desc, mN, mHalfC,  mHalfC,       mH, mW, 3, 3, 1, 1);
    ok &= buildOneDescSet(mCv2Desc,   mN, 3*mHalfC, mCout,       mH, mW, 1, 1, 0, 0);

    mDescsCached = ok;
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
    void* workspace, cudaStream_t stream) const noexcept
{
    if (!mCudnn || !d.xDesc) return false;

    cudnnSetStream(mCudnn, stream);

    float alpha = 1.f, beta = 0.f;
    if (cudnnConvolutionForward(
            mCudnn, &alpha,
            d.xDesc, x,
            d.wDesc, w,
            d.cDesc, d.algo,
            workspace, d.workspaceBytes,
            &beta,
            d.yDesc, y) != CUDNN_STATUS_SUCCESS) return false;

    // Fused bias add
    if (cudnnAddTensor(
            mCudnn, &alpha,
            d.bDesc, b,
            &alpha,
            d.yDesc, y) != CUDNN_STATUS_SUCCESS) return false;

    return true;
}


// ===========================================================================
// IPluginV2 interface
// ===========================================================================

const char* YoloC2fM2Plugin::getPluginType()    const noexcept { return kPLUGIN_NAME;    }
const char* YoloC2fM2Plugin::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
int         YoloC2fM2Plugin::getNbOutputs()     const noexcept { return 1; }

int YoloC2fM2Plugin::initialize() noexcept {
    // Create our own cuDNN handle if TRT hasn't provided one via attachToContext
    if (!mCudnn) {
        if (cudnnCreate(&mCudnn) != CUDNN_STATUS_SUCCESS) {
            mCudnn = nullptr;
            return 1;
        }
        mCudnnOwned = true;
    }

    if (!loadWeightsToDevice()) return 1;
    if (!buildDescSets())        return 1;

    return 0;
}

void YoloC2fM2Plugin::terminate() noexcept {
    destroyDescSets();
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
         + sizeof(int32_t) * 7;   // Cin, Cout, halfC, N, H, W, shortcut
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
    auto* p = new YoloC2fM2Plugin(mWeightsPath);
    p->mCin       = mCin;
    p->mCout      = mCout;
    p->mHalfC     = mHalfC;
    p->mN         = mN;
    p->mH         = mH;
    p->mW         = mW;
    p->mShortcut  = mShortcut;
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

// ---------------------------------------------------------------------------
// FIX: output has Cout channels (e.g. 64), NOT Cin (e.g. 32).
// If mCout is not yet populated (build-time before weights are loaded),
// we cannot return the right dims — caller must supply cin/cout via
// plugin fields or peek the weight file before getOutputDimensions.
// For a static-batch engine we use mCout set from the weight file header.
// ---------------------------------------------------------------------------
DimsExprs YoloC2fM2Plugin::getOutputDimensions(
    int, DimsExprs const* inputs, int, IExprBuilder& eb) noexcept
{
    DimsExprs out = inputs[0];
    if (mCout > 0) {
        out.d[1] = eb.constant(mCout);
    }
    // If mCout == 0 (pre-weight load), output channel dim is left as input's.
    // This is resolved after initialize() when the engine is serialised.
    return out;
}

bool YoloC2fM2Plugin::supportsFormatCombination(
    int pos, PluginTensorDesc const* inOut, int, int) noexcept
{
    auto const& desc = inOut[pos];
    return desc.format == TensorFormat::kLINEAR
        && desc.type   == DataType::kFLOAT;
}

void YoloC2fM2Plugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int,
    DynamicPluginTensorDesc const*, int) noexcept
{
    // Store the optimal-profile spatial dims for descriptor caching
    auto const& opt = in[0].opt;
    if (opt.nbDims == 4) {
        mN = opt.d[0] > 0 ? opt.d[0] : 1;
        mH = opt.d[2];
        mW = opt.d[3];
        // Input channel count from ONNX graph (mCin may already be set by weight loader)
        if (mCin == 0) mCin = opt.d[1];
    }
}

// ---------------------------------------------------------------------------
// Workspace layout:
//   [cv1_out   : N * Cout     * H * W * sizeof(float)]
//   [m0cv1_out : N * halfC    * H * W * sizeof(float)]
//   [concat    : N * 3*halfC  * H * W * sizeof(float)]
//     concat[0   ..halfC  ]: x1 (first split of cv1_out)
//     concat[halfC..2*halfC]: x2 (second split, also shortcut source)
//     concat[2*halfC..3*halfC]: m0.cv2 output + shortcut
//   [conv_ws   : max workspace across all 4 convolutions]
// ---------------------------------------------------------------------------
size_t YoloC2fM2Plugin::getWorkspaceSize(
    PluginTensorDesc const* inputs, int nbInputs,
    PluginTensorDesc const*, int) const noexcept
{
    if (nbInputs != 1 || inputs[0].dims.nbDims != 4) return 0;

    int N = inputs[0].dims.d[0];
    int H = inputs[0].dims.d[2];
    int W = inputs[0].dims.d[3];
    if (N <= 0 || H <= 0 || W <= 0 || mCout <= 0) return 0;

    size_t HW      = static_cast<size_t>(H) * W;
    size_t cv1Sz   = static_cast<size_t>(N) * mCout    * HW * sizeof(float);
    size_t halfSz  = static_cast<size_t>(N) * mHalfC   * HW * sizeof(float);
    size_t concatSz= static_cast<size_t>(N) * 3*mHalfC * HW * sizeof(float);

    // Max cuDNN workspace across all 4 convolutions
    size_t convWs = 0;
    convWs = std::max(convWs, mCv1Desc.workspaceBytes);
    convWs = std::max(convWs, mM0Cv1Desc.workspaceBytes);
    convWs = std::max(convWs, mM0Cv2Desc.workspaceBytes);
    convWs = std::max(convWs, mCv2Desc.workspaceBytes);

    // If descriptors not yet built, use a conservative upper bound
    if (convWs == 0) convWs = 64ULL << 20;   // 64 MB fallback

    return cv1Sz + halfSz + concatSz + convWs;
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
    if (fc) {
        for (int i = 0; i < fc->nbFields; ++i) {
            auto const& f = fc->fields[i];
            if (f.data && std::string(f.name) == "weights_path")
                weightsPath = static_cast<char const*>(f.data);
        }
    }
    auto* p = new YoloC2fM2Plugin(weightsPath);
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
