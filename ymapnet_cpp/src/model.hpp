#pragma once

// ---------------------------------------------------------------------------
// ggml headers – standalone ggml repo uses flat includes (no ggml/ prefix)
// ---------------------------------------------------------------------------
#if __has_include(<ggml/ggml.h>)
#  include <ggml/ggml.h>
#  include <ggml/ggml-alloc.h>
#  include <ggml/ggml-backend.h>
#  include <ggml/ggml-cpu.h>
#  include <ggml/gguf.h>
#else
#  include <ggml.h>
#  include <ggml-alloc.h>
#  include <ggml-backend.h>
#  include <ggml-cpu.h>
#  include <gguf.h>
#endif

// CUDA backend – ggml_backend_cuda_init() lives in ggml-cuda.h
#if defined(GGML_USE_CUDA)
#  if __has_include(<ggml/ggml-cuda.h>)
#    include <ggml/ggml-cuda.h>
#  elif __has_include(<ggml-cuda.h>)
#    include <ggml-cuda.h>
#  endif
#endif

#include <string>
#include <vector>
#include <unordered_map>

// ---------------------------------------------------------------------------
// Metadata read from the GGUF KV store
// ---------------------------------------------------------------------------
struct ModelMeta {
    uint32_t    input_w        = 256;
    uint32_t    input_h        = 256;
    uint32_t    input_ch       = 3;
    uint32_t    output_w       = 256;
    uint32_t    output_h       = 256;
    uint32_t    output_ch      = 73;
    float       heatmap_scale  = 120.f;
    uint32_t    base_channels  = 48;
    uint32_t    enc_reps       = 7;
    uint32_t    dec_reps       = 7;
    std::string model_name;
    std::string activation;
    bool        bn_folded      = true;
    std::string weight_dtype;
    std::vector<std::string> heatmap_names;
    std::string layer_arch_json;    // full layer list – used by Phase 3 graph builder
};

// ---------------------------------------------------------------------------
// YMAPNetModel
// ---------------------------------------------------------------------------
class YMAPNetModel {
public:
    // Load from a GGUF file.  Uses CUDA backend when available (and when
    // GGML_USE_CUDA was defined at compile time), otherwise CPU.
    // Returns false on failure.
    bool load(const std::string & gguf_path, int cuda_device = 0);

    // Release all ggml/backend resources.
    void free();

    // Run inference.
    // input_rgb : float32 [H * W * 3], layout [H, W, C], values in [0, 1]
    // Returns   : float32 [H * W * output_ch], layout [H, W, C]
    //
    // Phase 2: stub – returns zeros of the correct shape.
    // Phase 3: replace body with the full U-Net graph.
    std::vector<float> forward(const float * input_rgb, int W, int H);

    void print_info() const;

    const ModelMeta& meta() const { return meta_; }

private:
    ModelMeta               meta_;
    ggml_backend_t          backend_    = nullptr;
    ggml_backend_buffer_t   weight_buf_ = nullptr;
    ggml_context          * weight_ctx_ = nullptr;

    // All named weight tensors loaded from GGUF
    std::unordered_map<std::string, ggml_tensor*> weights_;

    // Build the computation graph once after load (called by load()).
    // W, H must match meta_.input_w / input_h.
    bool build_graph(int W, int H);

    // Helper: look up a weight tensor (asserts on failure)
    ggml_tensor* w(const std::string & name) const;

    // Persistent computation graph (built once in load, reused every forward call)
    ggml_context  * compute_ctx_    = nullptr;
    ggml_cgraph   * graph_          = nullptr;
    ggml_gallocr_t  allocr_         = nullptr;
    ggml_tensor   * input_tensor_   = nullptr;   // graph input node
    ggml_tensor   * output_tensor_  = nullptr;   // graph output node (heatmap)
};
