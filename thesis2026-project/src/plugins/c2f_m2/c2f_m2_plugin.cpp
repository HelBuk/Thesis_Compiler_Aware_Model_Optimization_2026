#include "c2f_m2_plugin.hpp"

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>

#include <cstring>
#include <string>

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

size_t volume(Dims const& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) {
        v *= static_cast<size_t>(d.d[i]);
    }
    return v;
}
} // namespace

// ============================
// YoloC2fM2Plugin
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

char const* YoloC2fM2Plugin::getPluginType() const noexcept {
    return kPLUGIN_NAME;
}

char const* YoloC2fM2Plugin::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

int YoloC2fM2Plugin::getNbOutputs() const noexcept {
    return 1;
}

int YoloC2fM2Plugin::initialize() noexcept {
    return 0;
}

void YoloC2fM2Plugin::terminate() noexcept {}

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
    return desc.format == TensorFormat::kLINEAR
        && desc.type == DataType::kFLOAT;
}

void YoloC2fM2Plugin::configurePlugin(
    DynamicPluginTensorDesc const*,
    int,
    DynamicPluginTensorDesc const*,
    int) noexcept {}

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

    size_t elems =
        static_cast<size_t>(N) * C * HW +              // cv1_out
        static_cast<size_t>(N) * halfC * HW +          // m0_out
        static_cast<size_t>(N) * (3 * halfC) * HW;    // concat_out

    return elems * sizeof(float);
}

void YoloC2fM2Plugin::attachToContext(
    cudnnContext*,
    cublasContext*,
    IGpuAllocator*) noexcept {}

void YoloC2fM2Plugin::detachFromContext() noexcept {}

// ============================
// Creator
// ============================

YoloC2fM2PluginCreator::YoloC2fM2PluginCreator() {
    mPluginAttributes.emplace_back(
        PluginField{"weights_path", nullptr, PluginFieldType::kCHAR, 1});
    mFC.nbFields = static_cast<int>(mPluginAttributes.size());
    mFC.fields = mPluginAttributes.data();
}

char const* YoloC2fM2PluginCreator::getPluginName() const noexcept {
    return kPLUGIN_NAME;
}

char const* YoloC2fM2PluginCreator::getPluginVersion() const noexcept {
    return kPLUGIN_VERSION;
}

PluginFieldCollection const* YoloC2fM2PluginCreator::getFieldNames() noexcept {
    return &mFC;
}

IPluginV2* YoloC2fM2PluginCreator::createPlugin(
    char const*,
    PluginFieldCollection const* fc) noexcept {

    std::string weightsPath;
    if (fc != nullptr) {
        for (int i = 0; i < fc->nbFields; ++i) {
            auto const& f = fc->fields[i];
            if (std::string(f.name) == "weights_path" && f.data != nullptr) {
                weightsPath = static_cast<char const*>(f.data);
            }
        }
    }

    auto* p = new YoloC2fM2Plugin(weightsPath);
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

IPluginV2* YoloC2fM2PluginCreator::deserializePlugin(
    char const*,
    void const* serialData,
    size_t serialLength) noexcept {

    auto* p = new YoloC2fM2Plugin(serialData, serialLength);
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

void YoloC2fM2PluginCreator::setPluginNamespace(char const* libNamespace) noexcept {
    mNamespace = libNamespace ? libNamespace : "";
}

char const* YoloC2fM2PluginCreator::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

REGISTER_TENSORRT_PLUGIN(YoloC2fM2PluginCreator);