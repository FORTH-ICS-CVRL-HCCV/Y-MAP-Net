#include "model.hpp"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <vector>

// ---------------------------------------------------------------------------
// Small JSON helpers (no external dependency)
// Parse a JSON array-of-strings: ["a","b",...] → vector<string>
// ---------------------------------------------------------------------------
static std::vector<std::string> parse_json_str_array(const std::string & json) {
    std::vector<std::string> out;
    size_t pos = 0;
    while ((pos = json.find('"', pos)) != std::string::npos) {
        ++pos;
        size_t end = json.find('"', pos);
        if (end == std::string::npos) break;
        out.push_back(json.substr(pos, end - pos));
        pos = end + 1;
    }
    return out;
}

// ---------------------------------------------------------------------------
// GGUF KV convenience wrappers
// ---------------------------------------------------------------------------
static std::string kv_str(gguf_context* ctx, const char* key, const char* def = "") {
    int id = gguf_find_key(ctx, key);
    return id >= 0 ? gguf_get_val_str(ctx, id) : def;
}
static uint32_t kv_u32(gguf_context* ctx, const char* key, uint32_t def = 0) {
    int id = gguf_find_key(ctx, key);
    return id >= 0 ? static_cast<uint32_t>(gguf_get_val_u32(ctx, id)) : def;
}
static float kv_f32(gguf_context* ctx, const char* key, float def = 0.f) {
    int id = gguf_find_key(ctx, key);
    return id >= 0 ? gguf_get_val_f32(ctx, id) : def;
}
static bool kv_bool(gguf_context* ctx, const char* key, bool def = false) {
    int id = gguf_find_key(ctx, key);
    return id >= 0 ? gguf_get_val_bool(ctx, id) : def;
}

// ---------------------------------------------------------------------------
// Graph-building helpers
// ---------------------------------------------------------------------------

// Broadcast bias [C] over conv output [W, H, C, N]
static ggml_tensor* add_bias(ggml_context* ctx, ggml_tensor* x, ggml_tensor* bias) {
    return ggml_add(ctx, x, ggml_reshape_4d(ctx, bias, 1, 1, bias->ne[0], 1));
}

// Leaky ReLU with Keras 3 default slope
static ggml_tensor* lrelu(ggml_context* ctx, ggml_tensor* x) {
    return ggml_leaky_relu(ctx, x, 0.2f, false);
}

// Conv2D with "same" padding, optional dilation.
// weight: [KW, KH, IC, OC] (ggml ne layout from GGUF)
// input:  [W, H, IC, N]
static ggml_tensor* conv2d_same(ggml_context* ctx,
                                ggml_tensor* weight, ggml_tensor* bias,
                                ggml_tensor* x,
                                int stride = 1, int dilation = 1)
{
    // same padding = dilation * (KW-1)/2 for odd kernels
    int kw  = (int)weight->ne[0];
    int pad = dilation * (kw - 1) / 2;
    ggml_tensor* out = ggml_conv_2d(ctx, weight, x, stride, stride, pad, pad, dilation, dilation);
    return add_bias(ctx, out, bias);
}

// Conv2D 1x1 (no padding)
static ggml_tensor* conv1x1(ggml_context* ctx,
                             ggml_tensor* weight, ggml_tensor* bias,
                             ggml_tensor* x)
{
    ggml_tensor* out = ggml_conv_2d(ctx, weight, x, 1, 1, 0, 0, 1, 1);
    return add_bias(ctx, out, bias);
}

// conv_block: 2x (conv3x3 + leaky_relu) + shortcut residual (optional 1x1 proj)
// r0name, r1name: weight base names for the two 3x3 convs
// projname: base name for 1x1 shortcut projection (empty = direct add)
static ggml_tensor* conv_block(ggml_context* ctx,
                                const std::unordered_map<std::string, ggml_tensor*>& W,
                                ggml_tensor* x,
                                const std::string& r0, const std::string& r1,
                                const std::string& proj)
{
    ggml_tensor* shortcut = x;

    x = conv2d_same(ctx, W.at(r0 + ".weight"), W.at(r0 + ".bias"), x);
    x = lrelu(ctx, x);
    x = conv2d_same(ctx, W.at(r1 + ".weight"), W.at(r1 + ".bias"), x);
    x = lrelu(ctx, x);

    if (proj.empty()) {
        x = ggml_add(ctx, shortcut, x);
    } else {
        ggml_tensor* s = conv1x1(ctx, W.at(proj + ".weight"), W.at(proj + ".bias"), shortcut);
        x = ggml_add(ctx, s, x);
    }
    return x;
}

// encoder_block: conv_block + avg_pool 2x2 → returns {skip, pooled}
static std::pair<ggml_tensor*, ggml_tensor*>
enc_block(ggml_context* ctx,
          const std::unordered_map<std::string, ggml_tensor*>& W,
          ggml_tensor* x, int d, int nf, const std::string& proj)
{
    char r0[64], r1[64];
    std::snprintf(r0, sizeof(r0), "conv2D_encoder_d%d_%d_leaky_relu_r0", d, nf);
    std::snprintf(r1, sizeof(r1), "conv2D_encoder_d%d_%d_leaky_relu_r1", d, nf);

    ggml_tensor* skip = conv_block(ctx, W, x, r0, r1, proj);
    ggml_tensor* pooled = ggml_pool_2d(ctx, skip, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0.f, 0.f);
    return {skip, pooled};
}

// ASPP bridge (4 parallel branches → concat → project + shortcut residual)
static ggml_tensor* aspp_bridge(ggml_context* ctx,
                                 const std::unordered_map<std::string, ggml_tensor*>& W,
                                 ggml_tensor* x)
{
    // Branch 1: 1x1
    ggml_tensor* b1 = conv1x1(ctx, W.at("aspp_b1.weight"), W.at("aspp_b1.bias"), x);
    b1 = lrelu(ctx, b1);

    // Branch 2: 3x3 dilation=1
    ggml_tensor* b2 = conv2d_same(ctx, W.at("aspp_b2.weight"), W.at("aspp_b2.bias"), x, 1, 1);
    b2 = lrelu(ctx, b2);

    // Branch 3: 3x3 dilation=2
    ggml_tensor* b3 = conv2d_same(ctx, W.at("aspp_b3.weight"), W.at("aspp_b3.bias"), x, 1, 2);
    b3 = lrelu(ctx, b3);

    // Branch global: avg_pool(full spatial) → 1x1 → conv1x1 → upsample back
    int spatial = (int)x->ne[0]; // W == H == 2 at bridge
    ggml_tensor* bg = ggml_pool_2d(ctx, x, GGML_OP_POOL_AVG, spatial, spatial, spatial, spatial, 0.f, 0.f);
    bg = conv1x1(ctx, W.at("aspp_global_conv.weight"), W.at("aspp_global_conv.bias"), bg);
    bg = lrelu(ctx, bg);
    bg = ggml_upscale(ctx, bg, spatial, GGML_SCALE_MODE_NEAREST);

    // Concat all branches along channel dim (dim=2)
    ggml_tensor* cat = ggml_concat(ctx, b1, b2, 2);
    cat = ggml_concat(ctx, cat, b3, 2);
    cat = ggml_concat(ctx, cat, bg, 2);

    // Project concatenated branches → 101 channels + leaky_relu
    ggml_tensor* proj = conv1x1(ctx, W.at("aspp_project.weight"), W.at("aspp_project.bias"), cat);
    proj = lrelu(ctx, proj);

    // Shortcut residual (plain 1x1 conv, no activation)
    ggml_tensor* sc = conv1x1(ctx, W.at("aspp_shortcut.weight"), W.at("aspp_shortcut.bias"), x);

    return ggml_add(ctx, proj, sc);
}

// decoder_block:
//   conv_transpose + bias (no activation)
//   concat with skip
//   conv_block (with internal shortcut)
//   add(conv_transpose_out, conv_block_out)  ← decoder residual
static ggml_tensor* dec_block(ggml_context* ctx,
                               const std::unordered_map<std::string, ggml_tensor*>& W,
                               ggml_tensor* x, ggml_tensor* skip,
                               int level, int nf, const std::string& proj)
{
    char dt[64], r0[64], r1[64];
    std::snprintf(dt, sizeof(dt), "conv2DT_decoder_L%d_%d",                              level, nf);
    std::snprintf(r0, sizeof(r0), "conv2D_decoder_L%d_%d_%d_leaky_relu_r0", level, nf, nf);
    std::snprintf(r1, sizeof(r1), "conv2D_decoder_L%d_%d_%d_leaky_relu_r1", level, nf, nf);

    ggml_tensor* up = ggml_conv_transpose_2d_p0(ctx, W.at(std::string(dt) + ".weight"), x, 2);
    up = add_bias(ctx, up, W.at(std::string(dt) + ".bias"));

    ggml_tensor* cat    = ggml_concat(ctx, up, skip, 2);
    ggml_tensor* cblock = conv_block(ctx, W, cat, r0, r1, proj);
    return ggml_add(ctx, up, cblock);
}

// ---------------------------------------------------------------------------
// YMAPNetModel::load
// ---------------------------------------------------------------------------
bool YMAPNetModel::load(const std::string & gguf_path, int cuda_device) {
#if defined(GGML_USE_CUDA)
    if (cuda_device >= 0) {
        backend_ = ggml_backend_cuda_init(cuda_device);
        if (backend_) printf("[model] CUDA backend (device %d)\n", cuda_device);
        else          fprintf(stderr, "[model] CUDA init failed, falling back to CPU\n");
    }
#endif
    if (!backend_) {
        backend_ = ggml_backend_cpu_init();
        printf("[model] CPU backend\n");
    }

    gguf_context* gguf_ctx = nullptr;
    {
        struct gguf_init_params p = { true, &weight_ctx_ };
        gguf_ctx = gguf_init_from_file(gguf_path.c_str(), p);
    }
    if (!gguf_ctx) {
        fprintf(stderr, "[model] Failed to open GGUF: %s\n", gguf_path.c_str());
        return false;
    }

    meta_.input_w       = kv_u32(gguf_ctx, "ymapnet.input_width",        256);
    meta_.input_h       = kv_u32(gguf_ctx, "ymapnet.input_height",       256);
    meta_.input_ch      = kv_u32(gguf_ctx, "ymapnet.input_channels",       3);
    meta_.output_w      = meta_.input_w;
    meta_.output_h      = meta_.input_h;
    meta_.output_ch     = kv_u32(gguf_ctx, "ymapnet.output_channels",     73);
    meta_.heatmap_scale = kv_f32(gguf_ctx, "ymapnet.heatmap_scale",     120.f);
    meta_.base_channels = kv_u32(gguf_ctx, "ymapnet.base_channels",       48);
    meta_.enc_reps      = kv_u32(gguf_ctx, "ymapnet.encoder_repetitions",  7);
    meta_.dec_reps      = kv_u32(gguf_ctx, "ymapnet.decoder_repetitions",  7);
    meta_.model_name    = kv_str(gguf_ctx, "general.name");
    meta_.activation    = kv_str(gguf_ctx, "ymapnet.activation", "leaky_relu");
    meta_.bn_folded     = kv_bool(gguf_ctx, "ymapnet.bn_folded", true);
    meta_.weight_dtype  = kv_str(gguf_ctx, "ymapnet.weight_dtype", "f32");
    meta_.heatmap_names = parse_json_str_array(kv_str(gguf_ctx, "ymapnet.heatmap_names", "[]"));
    meta_.layer_arch_json = kv_str(gguf_ctx, "ymapnet.layer_architecture", "[]");

    weight_buf_ = ggml_backend_alloc_ctx_tensors(weight_ctx_, backend_);

    if (!weight_buf_) {
        fprintf(stderr, "[model] Weight buffer allocation failed\n");
        gguf_free(gguf_ctx);
        return false;
    }

    FILE* fp = std::fopen(gguf_path.c_str(), "rb");
    if (!fp) {
        fprintf(stderr, "[model] Cannot reopen file: %s\n", gguf_path.c_str());
        gguf_free(gguf_ctx);
        return false;
    }

    const size_t data_base = gguf_get_data_offset(gguf_ctx);
    const int    n_tensors = gguf_get_n_tensors(gguf_ctx);

    for (int i = 0; i < n_tensors; ++i) {
        const char*  name   = gguf_get_tensor_name(gguf_ctx, i);
        ggml_tensor* tensor = ggml_get_tensor(weight_ctx_, name);
        size_t       offset = gguf_get_tensor_offset(gguf_ctx, i);
        size_t       nbytes = ggml_nbytes(tensor);

        std::fseek(fp, static_cast<long>(data_base + offset), SEEK_SET);
        std::vector<uint8_t> buf(nbytes);
        if (std::fread(buf.data(), 1, nbytes, fp) != nbytes) {
            fprintf(stderr, "[model] Short read for '%s'\n", name);
            std::fclose(fp);
            gguf_free(gguf_ctx);
            return false;
        }
        ggml_backend_tensor_set(tensor, buf.data(), 0, nbytes);
        weights_[name] = tensor;
    }

    std::fclose(fp);
    gguf_free(gguf_ctx);

    printf("[model] Loaded %d tensors  (%s  %s)\n",
           n_tensors, meta_.model_name.c_str(), meta_.weight_dtype.c_str());

    if (!build_graph(static_cast<int>(meta_.input_w), static_cast<int>(meta_.input_h))) {
        fprintf(stderr, "[model] build_graph failed\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// YMAPNetModel::free
// ---------------------------------------------------------------------------
void YMAPNetModel::free() {
    if (allocr_)      { ggml_gallocr_free(allocr_);            allocr_      = nullptr; }
    if (compute_ctx_) { ggml_free(compute_ctx_);               compute_ctx_ = nullptr;
                        graph_ = nullptr; input_tensor_ = nullptr; output_tensor_ = nullptr; }
    weights_.clear();
    if (weight_buf_) { ggml_backend_buffer_free(weight_buf_); weight_buf_ = nullptr; }
    if (weight_ctx_) { ggml_free(weight_ctx_);                weight_ctx_ = nullptr; }
    if (backend_)    { ggml_backend_free(backend_);           backend_    = nullptr; }
}

// ---------------------------------------------------------------------------
// YMAPNetModel::w  (weight lookup with assert)
// ---------------------------------------------------------------------------
ggml_tensor* YMAPNetModel::w(const std::string & name) const {
    auto it = weights_.find(name);
    if (it == weights_.end()) {
        fprintf(stderr, "[model] Missing tensor: '%s'\n", name.c_str());
        assert(false);
    }
    return it->second;
}

// ---------------------------------------------------------------------------
// YMAPNetModel::build_graph  — called once from load()
// ---------------------------------------------------------------------------
bool YMAPNetModel::build_graph(int W, int H) {
    const auto& wt = weights_;

    static const struct { int nf; const char* proj; } ENC[7] = {
        { 73,  "conv2d"   }, { 73,  ""         }, { 94,  "conv2d_1" },
        { 131, "conv2d_2" }, { 184, "conv2d_3" }, { 258, "conv2d_4" }, { 361, "conv2d_5" },
    };
    static const struct { int nf; const char* proj; } DEC[7] = {
        { 1632, "conv2d_6" }, { 906, "conv2d_7"  }, { 503, "conv2d_8"  },
        { 279,  "conv2d_9" }, { 155, "conv2d_10" }, { 86,  "conv2d_11" }, { 73, "conv2d_12" },
    };

    struct ggml_init_params cp = { 2 * 1024 * 1024, nullptr, true };
    compute_ctx_ = ggml_init(cp);
    if (!compute_ctx_) return false;

    // Input  [W, H, 3, 1]
    ggml_tensor* x = ggml_new_tensor_4d(compute_ctx_, GGML_TYPE_F32, W, H, 3, 1);
    ggml_set_name(x, "input");
    input_tensor_ = x;

    // Encoder
    ggml_tensor* skips[7];
    for (int i = 0; i < 7; ++i) {
        auto [skip, pooled] = enc_block(compute_ctx_, wt, x, i, ENC[i].nf, ENC[i].proj);
        skips[i] = skip;
        x = pooled;
    }

    // ASPP bridge
    x = aspp_bridge(compute_ctx_, wt, x);

    // Decoder
    for (int i = 0; i < 7; ++i)
        x = dec_block(compute_ctx_, wt, x, skips[6 - i], i, DEC[i].nf, DEC[i].proj);

    // Output head: pixelwise 1x1 + DRH + split tanh + scale
    x = conv1x1(compute_ctx_, wt.at("pixelwise.weight"), wt.at("pixelwise.bias"), x);
    x = lrelu(compute_ctx_, x);

    ggml_tensor* drh = conv1x1(compute_ctx_, wt.at("drh_entry.weight"), wt.at("drh_entry.bias"), x);
    drh = lrelu(compute_ctx_, drh);
    drh = conv2d_same(compute_ctx_, wt.at("drh_conv1.weight"), wt.at("drh_conv1.bias"), drh, 1, 2);
    drh = lrelu(compute_ctx_, drh);
    drh = conv2d_same(compute_ctx_, wt.at("drh_conv2.weight"), wt.at("drh_conv2.bias"), drh, 1, 4);
    drh = lrelu(compute_ctx_, drh);

    ggml_tensor* pre   = conv1x1(compute_ctx_, wt.at("hm_tanh_pre.weight"),   wt.at("hm_tanh_pre.bias"),   x);
    pre   = ggml_tanh(compute_ctx_, pre);
    ggml_tensor* depth = conv1x1(compute_ctx_, wt.at("hm_tanh_depth.weight"), wt.at("hm_tanh_depth.bias"), drh);
    depth = ggml_tanh(compute_ctx_, depth);
    ggml_tensor* post  = conv1x1(compute_ctx_, wt.at("hm_tanh_post.weight"),  wt.at("hm_tanh_post.bias"),  x);
    post  = ggml_tanh(compute_ctx_, post);

    ggml_tensor* hm = ggml_concat(compute_ctx_, pre, depth, 2);
    hm = ggml_concat(compute_ctx_, hm, post, 2);
    hm = ggml_scale(compute_ctx_, hm, meta_.heatmap_scale);
    output_tensor_ = hm;

    // Build and allocate
    graph_ = ggml_new_graph_custom(compute_ctx_, 4096, false);
    ggml_build_forward_expand(graph_, hm);

    allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (!ggml_gallocr_alloc_graph(allocr_, graph_)) {
        fprintf(stderr, "[model] build_graph: gallocr_alloc_graph failed\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// YMAPNetModel::forward  — upload input, run cached graph, download output
// ---------------------------------------------------------------------------
std::vector<float> YMAPNetModel::forward(const float* input_rgb, int W, int H) {
    ggml_backend_tensor_set(input_tensor_, input_rgb, 0, (size_t)W * H * 3 * sizeof(float));
    ggml_backend_graph_compute(backend_, graph_);

    const size_t n_out = (size_t)W * H * meta_.output_ch;
    std::vector<float> result(n_out);
    ggml_backend_tensor_get(output_tensor_, result.data(), 0, n_out * sizeof(float));
    return result;
}

// ---------------------------------------------------------------------------
// YMAPNetModel::print_info
// ---------------------------------------------------------------------------
void YMAPNetModel::print_info() const {
    printf("\n=== YMAPNet Model ===\n");
    printf("  name           : %s\n",   meta_.model_name.c_str());
    printf("  input          : %ux%ux%u\n", meta_.input_w, meta_.input_h, meta_.input_ch);
    printf("  output         : %ux%ux%u\n", meta_.output_w, meta_.output_h, meta_.output_ch);
    printf("  heatmap scale  : %.0f\n", meta_.heatmap_scale);
    printf("  base channels  : %u\n",   meta_.base_channels);
    printf("  enc/dec reps   : %u / %u\n", meta_.enc_reps, meta_.dec_reps);
    printf("  BN folded      : %s\n",   meta_.bn_folded ? "yes" : "no");
    printf("  weight dtype   : %s\n",   meta_.weight_dtype.c_str());
    printf("  tensors loaded : %zu\n",  weights_.size());
    printf("  backend        : %s\n",   ggml_backend_name(backend_) ? ggml_backend_name(backend_) : "?");
    printf("=====================\n\n");
}
