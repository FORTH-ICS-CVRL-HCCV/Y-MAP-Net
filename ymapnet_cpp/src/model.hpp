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
    uint32_t    output_ch      = 76;
    uint32_t    output16_ch    = 0;       // 16-bit (depth) head channels, 0 = none
    float       heatmap_scale  = 120.f;   // value range of the 8-bit head ([-scale, scale])
    float       output_scale   = 120.f;   // final Rescaling of the 8-bit head (incl. tanh headroom)
    float       output16_scale = 32767.f; // final Rescaling of the 16-bit head
    float       layernorm_eps  = 1e-3f;
    uint32_t    base_channels  = 70;
    uint32_t    enc_reps       = 7;
    uint32_t    dec_reps       = 7;
    std::string model_name;
    std::string serial;             // training run number ("288"), empty for older GGUFs
    std::string activation;
    bool        bn_folded      = true;
    std::string weight_dtype;
    std::vector<std::string> heatmap_names;
    std::vector<int> keypoint_parent_ids;   // configuration.json keypoint_parents as joint ids (empty: older GGUF)
    std::string paf_encoding;               // configuration.json heatmapPAFEncoding ("signed" for older GGUFs)
    std::string bridge_encoding;            // configuration.json heatmapBridgeEncoding ("signed" for older GGUFs)
    std::vector<float> skeleton_score_weights; // configuration.json skeletonScoreWeights (empty: default person score)
    std::string layer_arch_json;    // full layer list (informational)

    // Architecture as discovered from the tensor names in build_graph()
    bool             coord_conv     = false;
    bool             bottleneck_attn = false;
    std::vector<int> enc_filters;
    std::vector<int> dec_filters;
    int              block_reps     = 0;
    int              bridge_layers  = 0;
};

// ---------------------------------------------------------------------------
// YMAPNetModel
// ---------------------------------------------------------------------------
class YMAPNetModel {
public:
    // Load from a GGUF file.  Backend preference: Vulkan (if vulkan_device >= 0
    // and built with GGML_USE_VULKAN), then CUDA (if cuda_device >= 0 and built
    // with GGML_USE_CUDA), otherwise CPU.
    // Returns false on failure.
    // n_threads: CPU backend threads (0 = all hardware threads).
    // upcast_f16: store F16 weights as F32 on the backend (small file, F32 compute – ~2x faster
    //             on GPUs without fast F16 matmul, e.g. Samsung Xclipse 920).
    bool load(const std::string & gguf_path, int cuda_device = 0, int vulkan_device = -1, int n_threads = 0,
              bool upcast_f16 = false);

    // Release all ggml/backend resources.
    void free();

    // Run inference.
    // input_rgb : float32 [3 * H * W], CHW planar (see preprocess_frame), values in [0, 1]
    // Returns   : float32 [output_ch * H * W], CHW planar, the 8-bit head ("hm")
    std::vector<float> forward(const float * input_rgb, int W, int H);

    // 16-bit head ("hm_16b") of the last forward() call, CHW planar; empty if the model has none.
    std::vector<float> heatmap_16bit() const;

    // Token head ("Description"), present when the GGUF was exported with its token graph
    // (convertModelToGGUF.py --heatmaps-only --tokens, or a full export).
    bool has_tokens() const { return tokens_tensor_ != nullptr; }
    // Sigmoid scores of the last forward() call, one per vocabulary class; empty without a token head
    std::vector<float> token_scores() const;
    // Top-K words scoring above threshold as "word:0.87 word:0.55 ..." – mirrors
    // decodeOneHotDescriptionToString(vocabulary, multihot, K=21) in ymapnet/tokens/TokenEstimator.py
    std::string describe(int k = 21, float threshold = 0.24f) const;

    void print_info() const;

    const ModelMeta& meta() const { return meta_; }

    // CPU backend only: change the number of worker threads (takes effect on the next forward())
    bool is_cpu() const { return backend_ && ggml_backend_is_cpu(backend_); }
    void set_threads(int n) { if (is_cpu() && n > 0) ggml_backend_cpu_set_n_threads(backend_, n); }

    // Name of the active ggml backend ("CUDA0", "Vulkan0", "CPU"), or "" before load()
    std::string backend_name() const { return backend_ ? ggml_backend_name(backend_) : ""; }

    // The active backend, e.g. to run a second graph on it (tenpamy.hpp); owned by this model
    ggml_backend_t backend() const { return backend_; }

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

    // Persistent computation graph (built once in load, reused every forward call)
    ggml_context  * compute_ctx_    = nullptr;
    ggml_cgraph   * graph_          = nullptr;
    ggml_gallocr_t  allocr_         = nullptr;
    ggml_tensor   * input_tensor_   = nullptr;   // graph input node (RGB [+ CoordConv x,y planes])
    ggml_tensor   * output_tensor_  = nullptr;   // graph output node (8-bit heatmap)
    ggml_tensor   * output16_tensor_ = nullptr;  // graph output node (16-bit heatmap) or null
    ggml_tensor   * tokens_tensor_  = nullptr;   // graph output node (token multihot scores) or null
    std::string              token_graph_;       // serialised token head (see _token_head_graph)
    std::string              token_source_;      // layer the token head reads from
    std::vector<std::string> vocabulary_;        // class id → word
    std::vector<float> coord_planes_;            // constant CoordConv planes, uploaded every frame
};
