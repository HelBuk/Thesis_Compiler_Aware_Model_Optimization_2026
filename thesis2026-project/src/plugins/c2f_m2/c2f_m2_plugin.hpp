#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <string>
#include <vector>

class YoloC2fM2Plugin final : public nvinfer1::IPluginV2DynamicExt {
public:
    explicit YoloC2fM2Plugin(std::string weightsPath = "");
    YoloC2fM2Plugin(void const* data, size_t length);
    ~YoloC2fM2Plugin() override = default;

    // IPluginV2
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
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
        cudnnContext* cudnnContext,
        cublasContext* cublasContext,
        nvinfer1::IGpuAllocator* gpuAllocator) noexcept override;

    void detachFromContext() noexcept override;

private:
    std::string mWeightsPath;
    std::string mNamespace;
};

class YoloC2fM2PluginCreator final : public nvinfer1::IPluginCreator {
public:
    YoloC2fM2PluginCreator();

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(
        char const* name,
        nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        char const* name,
        void const* serialData,
        size_t serialLength) noexcept override;

    void setPluginNamespace(char const* libNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

private:
    std::string mNamespace;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    nvinfer1::PluginFieldCollection mFC{};
};