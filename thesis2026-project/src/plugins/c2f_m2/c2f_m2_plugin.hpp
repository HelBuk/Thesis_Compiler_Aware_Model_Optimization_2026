#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Winograd / fused activation runtime helpers (implemented in .cu files)
// ---------------------------------------------------------------------------
bool precomputeWinoFilterTransform(
    float const* hFilter, int outC, int inC, float** dTransformed) noexcept;

bool launchWinogradConv3x3SiLU(
    float const* x,
    float* y,
    float const* winoFilter,
    float const* bias,
    int N, int C, int K, int H, int W,
    cudaStream_t stream) noexcept;

bool launchBiasSiLUInplace(
    float* x,
    float const* bias,
    int N, int C, int H, int W,
    bool isNHWC,
    cudaStream_t stream) noexcept;

// ---------------------------------------------------------------------------
// Per-convolution cached cuDNN state.
// Created once in initialize(), destroyed in terminate().
// Re-using descriptors across enqueue() calls avoids ~20 descriptor
// create/destroy ops per inference — a critical performance fix.
// ---------------------------------------------------------------------------
struct ConvDescSet {
    cudnnTensorDescriptor_t    xDesc{nullptr};
    cudnnFilterDescriptor_t    wDesc{nullptr};
    cudnnConvolutionDescriptor_t cDesc{nullptr};
    cudnnTensorDescriptor_t    yDesc{nullptr};
    cudnnTensorDescriptor_t    bDesc{nullptr};
    cudnnConvolutionFwdAlgo_t  algo{CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM};
    size_t                     workspaceBytes{0};

    void destroy() noexcept;
};

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------
class YoloC2fM2Plugin final : public nvinfer1::IPluginV2DynamicExt {
public:
    explicit YoloC2fM2Plugin(std::string weightsPath = "");
    YoloC2fM2Plugin(void const* data, size_t length);
    ~YoloC2fM2Plugin() override;

    // IPluginV2
    const char* getPluginType()    const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int  getNbOutputs()            const noexcept override;
    int  initialize()                    noexcept override;
    void terminate()                     noexcept override;
    size_t getSerializationSize()  const noexcept override;
    void serialize(void* buffer)   const noexcept override;
    void destroy()                       noexcept override;
    void setPluginNamespace(char const* ns) noexcept override;
    char const* getPluginNamespace()    const noexcept override;

    void setStaticDims(int cin, int cout, int halfc) noexcept {
        mCin = cin;
        mCout = cout;
        mHalfC = halfc;
    }

    // IPluginV2Ext
    nvinfer1::DataType getOutputDataType(
        int index,
        nvinfer1::DataType const* inputTypes,
        int nbInputs) const noexcept override;

    // IPluginV2DynamicExt
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex,
        nvinfer1::DimsExprs const* inputs,
        int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(
        int pos,
        nvinfer1::PluginTensorDesc const* inOut,
        int nbInputs,
        int nbOutputs) noexcept override;

    void configurePlugin(
        nvinfer1::DynamicPluginTensorDesc const* in,
        int nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out,
        int nbOutputs) noexcept override;

    size_t getWorkspaceSize(
        nvinfer1::PluginTensorDesc const* inputs,
        int nbInputs,
        nvinfer1::PluginTensorDesc const* outputs,
        int nbOutputs) const noexcept override;

    int enqueue(
        nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs,
        void* const*       outputs,
        void*              workspace,
        cudaStream_t       stream) noexcept override;

    // attachToContext / detachFromContext exist only in TRT 8.x.
    // TRT 10.x (JetPack 6.x) removed them from IPluginV2DynamicExt entirely.
#if NV_TENSORRT_MAJOR < 10
    void attachToContext(
        cudnnContext*            cudnn,
        cublasContext*           cublas,
        nvinfer1::IGpuAllocator* allocator) noexcept override;

    void detachFromContext() noexcept override;
#endif

private:
    // -------------------------------------------------------------------
    // Weight management
    // -------------------------------------------------------------------
    void destroyWeights()       noexcept;
    bool loadWeightsToDevice();             // reads .bin file, uploads to GPU

    // -------------------------------------------------------------------
    // cuDNN descriptor helpers
    // -------------------------------------------------------------------
    bool buildDescSets()        noexcept;   // called from initialize()
    void destroyDescSets()      noexcept;   // called from terminate()

    bool buildOneDescSet(
        ConvDescSet& d,
        int N, int Cin, int Cout, int H, int W,
        int kH, int kW, int padH, int padW,
        cudnnTensorFormat_t xFmt = CUDNN_TENSOR_NCHW,
        cudnnTensorFormat_t yFmt = CUDNN_TENSOR_NCHW) noexcept;

    bool runConv(
        ConvDescSet const& d,
        float const* x,
        float*       y,
        float const* w,
        float const* b,
        void*        workspace,
        cudaStream_t stream,
        bool         addBias = true) const noexcept;

    bool precomputeWinogradFilters() noexcept;
    void destroyWinogradFilters() noexcept;
    bool canUseWinograd(
        int N, int C, int K, int H, int W,
        int kH, int kW, int stride, int pad) const noexcept;

private:
    std::string mNamespace;
    std::string mWeightsPath;

    // ---------------------------------------------------------------
    // Channel / spatial dimensions — populated from weight shapes and
    // configurePlugin().  Needed for getOutputDimensions() & workspace.
    // ---------------------------------------------------------------
    int mCin{0};    // input channels to C2f  (e.g. 32 for model.2)
    int mCout{0};   // output channels of C2f (e.g. 64 for model.2)
    int mHalfC{0};  // mCout / 2              (e.g. 32)
    int mN{1};      // batch size (from configurePlugin optimal profile)
    int mH{0};      // spatial height
    int mW{0};      // spatial width

    bool mShortcut{true};   // bottleneck residual add flag

    // cuDNN handle — either our own (created in initialize) or from TRT
    cudnnHandle_t mCudnn{nullptr};
    bool          mCudnnOwned{false};   // true → we destroy it in terminate

    // ---------------------------------------------------------------
    // Cached cuDNN descriptor sets (one per conv in the C2f block)
    // ---------------------------------------------------------------
    ConvDescSet mCv1Desc;     // cv1   : Cin  → Cout,   1×1
    ConvDescSet mM0Cv1Desc;   // m0.cv1: halfC→ halfC,  3×3
    ConvDescSet mM0Cv2Desc;   // m0.cv2: halfC→ halfC,  3×3
    ConvDescSet mCv2Desc;     // cv2   : 3*halfC→Cout,  1×1
    bool        mDescsCached{false};

    // Format negotiated with TRT in supportsFormatCombination / configurePlugin.
    // kCHW32 (= 5) is TRT's Ampere FP32 NHWC-vectorised format (C padded to 32).
    // For our tensors (C = 32) kCHW32 is byte-equivalent to plain NHWC.
    // Accepting kCHW32 avoids the 50–900 µs NCHW↔NHWC reformat copy node.
    nvinfer1::TensorFormat mInputFormat{nvinfer1::TensorFormat::kLINEAR};

    // ---------------------------------------------------------------
    // Device weight pointers
    // ---------------------------------------------------------------
    float* d_cv1_w{nullptr};
    float* d_cv1_b{nullptr};
    float* d_m0_cv1_w{nullptr};
    float* d_m0_cv1_b{nullptr};
    float* d_m0_cv2_w{nullptr};
    float* d_m0_cv2_b{nullptr};
    float* d_cv2_w{nullptr};
    float* d_cv2_b{nullptr};
    float* d_m0_cv1_wino{nullptr};  // [K, C, 16] Winograd filter transform
    float* d_m0_cv2_wino{nullptr};  // [K, C, 16] Winograd filter transform
    bool   mWinogradReady{false};

    // Host-side raw weights for serialisation round-trip
    std::vector<float> h_cv1_w,    h_cv1_b;
    std::vector<float> h_m0_cv1_w, h_m0_cv1_b;
    std::vector<float> h_m0_cv2_w, h_m0_cv2_b;
    std::vector<float> h_cv2_w,    h_cv2_b;
};


// ---------------------------------------------------------------------------
// Creator
// ---------------------------------------------------------------------------
class YoloC2fM2PluginCreator : public nvinfer1::IPluginCreator {
public:
    YoloC2fM2PluginCreator();

    const char* getPluginName()    const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(
        const char* name,
        const nvinfer1::PluginFieldCollection* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        const char* name,
        const void* serialData,
        size_t      serialLength) noexcept override;

    void        setPluginNamespace(const char* ns) noexcept override;
    const char* getPluginNamespace()         const noexcept override;

private:
    std::string mNamespace;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    nvinfer1::PluginFieldCollection    mFC{};
};
