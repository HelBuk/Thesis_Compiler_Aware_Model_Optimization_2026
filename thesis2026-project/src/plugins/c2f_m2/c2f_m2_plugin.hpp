#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include <string>
#include <vector>

class YoloC2fM2Plugin final : public nvinfer1::IPluginV2DynamicExt {
public:
    explicit YoloC2fM2Plugin(std::string weightsPath = "");
    YoloC2fM2Plugin(void const* data, size_t length);
    ~YoloC2fM2Plugin() override;

    // IPluginV2
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

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
        void* const* outputs,
        void* workspace,
        cudaStream_t stream) noexcept override;

    void attachToContext(
        cudnnContext* cudnn,
        cublasContext* cublas,
        nvinfer1::IGpuAllocator* allocator) noexcept override;

    void detachFromContext() noexcept override;

private:
    void destroyWeights() noexcept;
    bool loadWeightsToDevice();
    size_t getMaxConvWorkspaceBytes(int N, int C, int H, int W) const noexcept;

    bool runConv1x1(
        const float* x,
        float* y,
        int N, int Cin, int Cout, int H, int W,
        const float* w,
        const float* b,
        void* workspace,
        size_t workspaceBytes,
        cudaStream_t stream) const noexcept;

    bool runConv3x3(
        const float* x,
        float* y,
        int N, int Cin, int Cout, int H, int W,
        const float* w,
        const float* b,
        void* workspace,
        size_t workspaceBytes,
        cudaStream_t stream) const noexcept;

private:
    std::string mNamespace;
    std::string mWeightsPath;

    // C2f(m=2) module-2 assumptions:
    // input  : [N, C,   H, W]
    // cv1    : [N, C,   H, W]
    // split  : x1=[N,C/2,H,W], x2=[N,C/2,H,W]
    // m0.cv1 : [N,C/2,H,W] -> [N,C/2,H,W]
    // m0.cv2 : [N,C/2,H,W] -> [N,C/2,H,W]
    // concat : [N,3C/2,H,W]
    // cv2    : [N,3C/2,H,W] -> [N,C,H,W]

    cudnnHandle_t mCudnn{nullptr};

    // Device weights
    float* d_cv1_w{nullptr};
    float* d_cv1_b{nullptr};

    float* d_m0_cv1_w{nullptr};
    float* d_m0_cv1_b{nullptr};

    float* d_m0_cv2_w{nullptr};
    float* d_m0_cv2_b{nullptr};

    float* d_cv2_w{nullptr};
    float* d_cv2_b{nullptr};

    // Host-side cached raw weights if you want serialization later
    std::vector<float> h_cv1_w, h_cv1_b;
    std::vector<float> h_m0_cv1_w, h_m0_cv1_b;
    std::vector<float> h_m0_cv2_w, h_m0_cv2_b;
    std::vector<float> h_cv2_w, h_cv2_b;
};


// ============================
// Creator
// ============================

class YoloC2fM2PluginCreator : public nvinfer1::IPluginCreator {
public:
    YoloC2fM2PluginCreator();

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(
        const char* name,
        const nvinfer1::PluginFieldCollection* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        const char* name,
        const void* serialData,
        size_t serialLength) noexcept override;

    void setPluginNamespace(const char* libNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    std::string mNamespace;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    nvinfer1::PluginFieldCollection mFC{};
};