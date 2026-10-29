#include "model.hpp"

// CUDA backend – ggml_backend_cuda_init() lives in ggml-cuda.h
#if defined(GGML_USE_CUDA)
#  if __has_include(<ggml/ggml-cuda.h>)
#    include <ggml/ggml-cuda.h>
#  elif __has_include(<ggml-cuda.h>)
#    include <ggml-cuda.h>
#  endif
#endif

// Vulkan backend – ggml_backend_vk_init() lives in ggml-vulkan.h
#if defined(GGML_USE_VULKAN)
#  include <ggml-vulkan.h>
#  include <vulkan/vulkan.h>
#endif

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <thread>
#include <unordered_map>
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
//
// All feature maps are kept channel-first, CWH: ne=[C, W, H, N], contiguous. That makes LayerNorm
// (ggml_norm over ne0) and 1x1 convs (one matmul) direct, and ggml's depthwise conv accepts a
// channel-contiguous view and returns the same memory layout – so the ConvNeXt blocks need no
// layout copies at all. Only the network input/outputs (planar WHC) and the im2col inputs of the
// few 3x3 convs are converted.
// ---------------------------------------------------------------------------
using WeightMap = std::unordered_map<std::string, ggml_tensor*>;

static ggml_tensor* get(const WeightMap& W, const std::string& name) {
    auto it = W.find(name);
    if (it == W.end()) throw std::runtime_error("missing tensor '" + name + "'");
    return it->second;
}
static bool has(const WeightMap& W, const std::string& name) { return W.count(name) != 0; }

// Channel count N of a tensor named "<prefix><N><suffix>", or -1 if none exists.
static int find_filters(const WeightMap& W, const std::string& prefix, const std::string& suffix) {
    for (const auto& kv : W) {
        const std::string& n = kv.first;
        if (n.size() <= prefix.size() + suffix.size()) continue;
        if (n.compare(0, prefix.size(), prefix) != 0) continue;
        if (n.compare(n.size() - suffix.size(), suffix.size(), suffix) != 0) continue;
        std::string mid = n.substr(prefix.size(), n.size() - prefix.size() - suffix.size());
        if (!mid.empty() && mid.find_first_not_of("0123456789") == std::string::npos)
            return std::stoi(mid);
    }
    return -1;
}

static ggml_tensor* to_cwh(ggml_context* ctx, ggml_tensor* x) { return ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3)); }
static ggml_tensor* to_whc(ggml_context* ctx, ggml_tensor* x) { return ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3)); }

// PReLU (Keras shared_axes=[1,2]: one slope per channel): max(x,0) + alpha * min(x,0); alpha [C]
// Keep this op sequence: ggml-vulkan fuses it, together with a directly preceding bias ADD, into one
// shader (patches/ggml-vulkan-bias-prelu.patch).
static ggml_tensor* prelu(ggml_context* ctx, ggml_tensor* x, ggml_tensor* alpha) {
    ggml_tensor* pos = ggml_relu(ctx, x);
    ggml_tensor* neg = ggml_sub(ctx, x, pos);
    return ggml_add(ctx, pos, ggml_mul(ctx, neg, alpha));
}

// 1x1 Conv2D on a CWH map as a single matmul. weight ne=[1, 1, Cin, Cout]
static ggml_tensor* pw_conv(ggml_context* ctx, const WeightMap& W, const std::string& name, ggml_tensor* x) {
    ggml_tensor* w = get(W, name + ".weight");
    w = ggml_reshape_2d(ctx, w, w->ne[2], w->ne[3]);
    ggml_tensor* x2 = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1] * x->ne[2] * x->ne[3]);
    ggml_tensor* y  = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, w, x2), w->ne[1], x->ne[1], x->ne[2], x->ne[3]);
    return ggml_add(ctx, y, get(W, name + ".bias"));   // last, so a following PReLU can fuse with it on Vulkan
}

// KxK Conv2D, "same" padding, CWH in and out: im2col + one matmul with the weights as src0 (the
// orientation ggml's GPU matmul kernels are tuned for); the result is already CWH, so unlike
// ggml_conv_2d there is no permute+copy afterwards. weight ne=[KW, KH, Cin, Cout]
// direct: use ggml's implicit-GEMM CONV_2D instead, which skips the KxK-times-larger im2col buffer
// (~300 MB at 256x256). Faster on Vulkan (S22 Ultra 1180 -> 1090 ms/frame), 2x slower on CUDA.
static ggml_tensor* conv_same(ggml_context* ctx, const WeightMap& W, const std::string& name,
                              ggml_tensor* x, bool direct, int dilation = 1)
{
    ggml_tensor* w = get(W, name + ".weight");
    const int pad = dilation * ((int)w->ne[0] - 1) / 2;
    if (direct) {
        ggml_tensor* y = to_cwh(ctx, ggml_conv_2d_direct(ctx, w, to_whc(ctx, x), 1, 1, pad, pad, dilation, dilation));
        return ggml_add(ctx, y, get(W, name + ".bias"));
    }
    ggml_tensor* col = ggml_im2col(ctx, w, to_whc(ctx, x), 1, 1, pad, pad, dilation, dilation, true,
                                   w->type == GGML_TYPE_F16 ? GGML_TYPE_F16 : GGML_TYPE_F32);   // [K, OW, OH, N]
    ggml_tensor* y = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1] * w->ne[2], w->ne[3]),
        ggml_reshape_2d(ctx, col, col->ne[0], col->ne[1] * col->ne[2] * col->ne[3]));          // [Cout, OW*OH]
    y = ggml_reshape_4d(ctx, y, w->ne[3], col->ne[1], col->ne[2], col->ne[3]);
    return ggml_add(ctx, y, get(W, name + ".bias"));
}

// Depthwise KxK conv on a CWH map with no feature-map copy: ggml_conv_2d_dw_direct takes a
// channel-contiguous ("CWHN") view and returns its result in the same memory order. The kernel must
// be channel-contiguous too (same transform as test_conv_2d_dw in ggml's test-backend-ops); it is
// tiny (KxKxC), so it is re-laid out in the graph.
static ggml_tensor* dw_conv(ggml_context* ctx, ggml_tensor* k, ggml_tensor* x) {
    const int pad = ((int)k->ne[0] - 1) / 2;
    k = ggml_permute(ctx, ggml_cont(ctx, ggml_permute(ctx, k, 2, 3, 1, 0)), 3, 2, 0, 1);
    ggml_tensor* v = ggml_permute(ctx, x, 2, 0, 1, 3);                    // logical WHC, CWH memory
    ggml_tensor* y = ggml_conv_2d_dw_direct(ctx, k, v, 1, 1, pad, pad, 1, 1);
    return ggml_permute(ctx, y, 1, 2, 0, 3);                                 // back to logical CWH
}

// 2x2 average pooling (stride 2) on a CWH map: mean of four strided views, no layout copy
static ggml_tensor* avg_pool2(ggml_context* ctx, ggml_tensor* x) {
    const int64_t C = x->ne[0], OW = x->ne[1] / 2, OH = x->ne[2] / 2;
    auto tap = [&](int dx, int dy) {
        return ggml_view_3d(ctx, x, C, OW, OH, 2 * x->nb[1], 2 * x->nb[2], dx * x->nb[1] + dy * x->nb[2]);
    };
    ggml_tensor* s = ggml_add(ctx, ggml_add(ctx, tap(0, 0), tap(1, 0)), ggml_add(ctx, tap(0, 1), tap(1, 1)));
    return ggml_reshape_4d(ctx, ggml_scale(ctx, s, 0.25f), C, OW, OH, 1);
}

// LayerNorm over channels of a CWH map
static ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* gamma, ggml_tensor* beta, float eps) {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps), gamma), beta);
}

// convnext_block (NNModel.py): DW 7x7 → LN → PW 4C → PReLU → PW C, + residual
// (1x1 shortcut projection when the channel count changes). CWH in, CWH out.
static ggml_tensor* convnext_block(ggml_context* ctx, const WeightMap& W, ggml_tensor* x,
                                   const std::string& tag, float eps)
{
    const std::string dw = "dwconv_" + tag;
    ggml_tensor* h = ggml_add(ctx, dw_conv(ctx, get(W, dw + ".weight"), x), get(W, dw + ".bias"));
    h = layer_norm(ctx, h, get(W, dw + ".ln_gamma"), get(W, dw + ".ln_beta"), eps);
    h = pw_conv(ctx, W, "pwconv1_" + tag, h);
    h = prelu(ctx, h, get(W, "pwconv1_" + tag + ".alpha"));
    h = pw_conv(ctx, W, "pwconv2_" + tag, h);

    const std::string proj = "shortcut_proj_" + tag;
    return ggml_add(ctx, has(W, proj + ".weight") ? pw_conv(ctx, W, proj, x) : x, h);
}

// feature_conv with useDepthwiseSeparable: `reps` convnext blocks named "<name>_r<i>_<nf>"
static ggml_tensor* feature_conv(ggml_context* ctx, const WeightMap& W, ggml_tensor* x,
                                 const std::string& name, int nf, int reps, float eps)
{
    for (int r = 0; r < reps; ++r)
        x = convnext_block(ctx, W, x, name + "_r" + std::to_string(r) + "_" + std::to_string(nf), eps);
    return x;
}

// Bottleneck self-attention (pre-norm LN → MHA → residual) over the W*H tokens of a CWH map.
static ggml_tensor* bottleneck_attention(ggml_context* ctx, const WeightMap& W, ggml_tensor* x, float eps) {
    const int64_t C = x->ne[0], Wd = x->ne[1], Hd = x->ne[2], T = Wd * Hd;
    const std::string mha = "bottleneck_attn_mha";

    ggml_tensor* seq = ggml_reshape_2d(ctx, x, C, T);                  // [C, T]
    ggml_tensor* ln  = layer_norm(ctx, seq, get(W, "bottleneck_attn_ln.gamma"),
                                  get(W, "bottleneck_attn_ln.beta"), eps);

    const ggml_tensor* qb = get(W, mha + ".q.bias");                  // [kd, heads]
    const int64_t kd = qb->ne[0], heads = qb->ne[1];

    // q/k/v: [C, kd*heads] matmul → [kd, heads, T] → [kd, T, heads]
    auto proj = [&](const char* p) {
        ggml_tensor* w = get(W, mha + "." + p + ".weight");
        ggml_tensor* b = get(W, mha + "." + p + ".bias");
        ggml_tensor* y = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, w, C, kd * heads), ln);
        y = ggml_add(ctx, y, ggml_reshape_1d(ctx, b, kd * heads));
        return ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, y, kd, heads, T), 0, 2, 1, 3));
    };
    ggml_tensor* q = proj("q");
    ggml_tensor* k = proj("k");
    ggml_tensor* v = proj("v");

    ggml_tensor* scores = ggml_mul_mat(ctx, k, q);                                     // [Tk, Tq, heads]
    scores = ggml_soft_max_ext(ctx, scores, nullptr, 1.0f / std::sqrt((float)kd), 0.0f);
    ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));                // [Tk, kd, heads]
    ggml_tensor* o  = ggml_mul_mat(ctx, vt, scores);                                    // [kd, Tq, heads]
    o = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3)), kd * heads, T);

    ggml_tensor* ow = get(W, mha + ".o.weight");                                        // [kd, heads, C]
    o = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, ow, kd * heads, C), o);                  // [C, T]
    o = ggml_add(ctx, o, get(W, mha + ".o.bias"));

    return ggml_reshape_4d(ctx, ggml_add(ctx, seq, o), C, Wd, Hd, 1);
}

// decoder_block with upsampling_mode='nearest' (CWH):
//   up = PReLU(BN(conv3x3(nearest_upsample(x))))   (BN folded into conv_up)
//   out = up + feature_conv(concat(up, skip))
static ggml_tensor* dec_block(ggml_context* ctx, const WeightMap& W, ggml_tensor* x, ggml_tensor* skip,
                              int level, int nf, int reps, float eps, bool direct)
{
    const std::string tag = "decoder_L" + std::to_string(level) + "_" + std::to_string(nf);

    ggml_tensor* up = ggml_interpolate(ctx, x, x->ne[0], 2 * x->ne[1], 2 * x->ne[2], x->ne[3],
                                       GGML_SCALE_MODE_NEAREST);
    up = conv_same(ctx, W, "conv_up_" + tag, up, direct);
    up = prelu(ctx, up, get(W, "conv_up_" + tag + ".alpha"));

    ggml_tensor* cat = ggml_concat(ctx, up, skip, 0);                 // channels are ne0
    return ggml_add(ctx, up, feature_conv(ctx, W, cat, tag, nf, reps, eps));
}

#if defined(GGML_USE_VULKAN)
// ggml-vulkan only checks the *instance* (loader) version. On a Vulkan 1.1 *device* – e.g. Mali-G72
// on Android 13 – it then reads the (uninitialised) 1.2 feature struct and can call the missing
// vkGetBufferDeviceAddress → SIGSEGV. So check the device's own API version, matched by name.
static bool vk_device_is_1_2(int ggml_index) {
    char name[256] = {};
    ggml_backend_vk_get_device_description(ggml_index, name, sizeof(name));

    VkApplicationInfo app = {};
    app.sType      = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ci = {};
    ci.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ci, nullptr, &inst) != VK_SUCCESS) return false;

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(inst, &n, devs.data());
    bool ok = false;
    for (VkPhysicalDevice d : devs) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(d, &p);
        if (std::strcmp(p.deviceName, name) == 0) {
            ok = p.apiVersion >= VK_API_VERSION_1_2;
            if (!ok) fprintf(stderr, "[model] Vulkan device '%s' is Vulkan %u.%u; ggml needs 1.2\n",
                             p.deviceName, VK_API_VERSION_MAJOR(p.apiVersion), VK_API_VERSION_MINOR(p.apiVersion));
            break;
        }
    }
    vkDestroyInstance(inst, nullptr);
    return ok;
}
#endif

// ---------------------------------------------------------------------------
// Token head: built from the text graph written by convertModelToGGUF.py (_token_head_graph),
// one layer per line "type|name|in1,in2|key=value;...". All tensors are flat vectors.
// ---------------------------------------------------------------------------
static std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    size_t start = 0;
    for (size_t i = 0; i <= s.size(); ++i)
        if (i == s.size() || s[i] == sep) { out.push_back(s.substr(start, i - start)); start = i + 1; }
    return out;
}

static ggml_tensor* activation(ggml_context* ctx, ggml_tensor* x, const std::string& act) {
    if (act == "linear")     return x;
    if (act == "tanh")       return ggml_tanh(ctx, x);
    if (act == "sigmoid")    return ggml_sigmoid(ctx, x);
    if (act == "relu")       return ggml_relu(ctx, x);
    if (act == "leaky_relu") return ggml_leaky_relu(ctx, x, 0.2f, false);   // Keras 3 default slope
    throw std::runtime_error("token head: unsupported activation '" + act + "'");
}

static ggml_tensor* build_token_head(ggml_context* ctx, const WeightMap& W, const std::string& graph,
                                     const std::string& source, ggml_tensor* source_tensor)
{
    std::unordered_map<std::string, ggml_tensor*> t = { { source, source_tensor } };
    ggml_tensor* last = nullptr;
    for (const std::string& line : split(graph, '\n')) {
        if (line.empty()) continue;
        std::vector<std::string> f = split(line, '|');
        if (f.size() != 4) throw std::runtime_error("token head: bad line '" + line + "'");
        const std::string& op = f[0], & name = f[1];
        std::vector<ggml_tensor*> in;
        for (const std::string& i : split(f[2], ',')) {
            auto it = t.find(i);
            if (it == t.end()) throw std::runtime_error("token head: '" + name + "' reads unknown '" + i + "'");
            in.push_back(it->second);
        }
        std::unordered_map<std::string, std::string> kv;
        if (!f[3].empty())
            for (const std::string& p : split(f[3], ';')) {
                size_t eq = p.find('=');
                kv[p.substr(0, eq)] = p.substr(eq + 1);
            }

        ggml_tensor* y = nullptr;
        if (op == "identity") {
            y = in[0];
        } else if (op == "dense") {
            ggml_tensor* w = get(W, name + ".weight");                        // ne = [in, out]
            y = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, w, in[0]), w->ne[1]);
            y = activation(ctx, ggml_add(ctx, y, get(W, name + ".bias")), kv["act"]);
        } else if (op == "prelu") {
            y = prelu(ctx, in[0], get(W, kv["alpha"]));
        } else if (op == "layernorm") {
            y = layer_norm(ctx, in[0], get(W, kv["gamma"]), get(W, kv["beta"]), std::stof(kv["eps"]));
        } else if (op == "rescale") {
            if (std::stof(kv["offset"]) != 0.f) throw std::runtime_error("token head: Rescaling offset unsupported");
            y = ggml_scale(ctx, in[0], std::stof(kv["scale"]));
        } else if (op == "add") {
            y = in[0];
            for (size_t i = 1; i < in.size(); ++i) y = ggml_add(ctx, y, in[i]);
        } else if (op == "concat") {
            y = in[0];
            for (size_t i = 1; i < in.size(); ++i) y = ggml_concat(ctx, y, in[i], 0);
        } else if (op == "activation") {
            y = activation(ctx, in[0], kv["act"]);
        } else {
            throw std::runtime_error("token head: unsupported op '" + op + "'");
        }
        t[name] = y;
        last = y;
    }
    return last;   // the final layer (tokens_multihot)
}

// ---------------------------------------------------------------------------
// YMAPNetModel::load
// ---------------------------------------------------------------------------
bool YMAPNetModel::load(const std::string & gguf_path, int cuda_device, int vulkan_device, int n_threads,
                        bool upcast_f16) {
#if defined(GGML_USE_VULKAN)
    if (vulkan_device >= 0) {
        if (vulkan_device < ggml_backend_vk_get_device_count() && vk_device_is_1_2(vulkan_device)) {
            char desc[256];
            ggml_backend_vk_get_device_description(vulkan_device, desc, sizeof(desc));
            backend_ = ggml_backend_vk_init(vulkan_device);
            if (backend_) printf("[model] Vulkan backend (device %d: %s)\n", vulkan_device, desc);
        }
        if (!backend_) fprintf(stderr, "[model] Vulkan init failed for device %d\n", vulkan_device);
    }
#else
    if (vulkan_device >= 0)
        fprintf(stderr, "[model] Built without Vulkan (configure with -DYMAPNET_VULKAN=ON)\n");
#endif
#if defined(GGML_USE_CUDA)
    if (!backend_ && cuda_device >= 0) {
        backend_ = ggml_backend_cuda_init(cuda_device);
        if (backend_) printf("[model] CUDA backend (device %d)\n", cuda_device);
        else          fprintf(stderr, "[model] CUDA init failed, falling back to CPU\n");
    }
#endif
    if (!backend_) {
        backend_ = ggml_backend_cpu_init();
        if (n_threads <= 0) n_threads = (int)std::max(1u, std::thread::hardware_concurrency());
        ggml_backend_cpu_set_n_threads(backend_, n_threads);   // ggml's default is 4
        printf("[model] CPU backend (%d threads)\n", n_threads);
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
    meta_.output_ch     = kv_u32(gguf_ctx, "ymapnet.output_channels",     76);
    meta_.output16_ch   = kv_u32(gguf_ctx, "ymapnet.output_16bit_channels", 0);
    meta_.heatmap_scale = kv_f32(gguf_ctx, "ymapnet.heatmap_scale",     120.f);
    meta_.output_scale  = kv_f32(gguf_ctx, "ymapnet.heatmap_output_scale", meta_.heatmap_scale);
    meta_.output16_scale = kv_f32(gguf_ctx, "ymapnet.heatmap_16bit_scale", 32767.f);
    meta_.layernorm_eps = kv_f32(gguf_ctx, "ymapnet.layernorm_eps",      1e-3f);
    meta_.base_channels = kv_u32(gguf_ctx, "ymapnet.base_channels",       70);
    meta_.enc_reps      = kv_u32(gguf_ctx, "ymapnet.encoder_repetitions",  7);
    meta_.dec_reps      = kv_u32(gguf_ctx, "ymapnet.decoder_repetitions",  7);
    meta_.model_name    = kv_str(gguf_ctx, "general.name");
    meta_.serial        = kv_str(gguf_ctx, "ymapnet.serial");
    meta_.activation    = kv_str(gguf_ctx, "ymapnet.activation", "prelu");
    meta_.bn_folded     = kv_bool(gguf_ctx, "ymapnet.bn_folded", true);
    meta_.weight_dtype  = kv_str(gguf_ctx, "ymapnet.weight_dtype", "f32");
    meta_.heatmap_names = parse_json_str_array(kv_str(gguf_ctx, "ymapnet.heatmap_names", "[]"));
    meta_.paf_encoding  = kv_str(gguf_ctx, "ymapnet.paf_encoding", "signed");
    meta_.bridge_encoding = kv_str(gguf_ctx, "ymapnet.bridge_encoding", "signed");
    meta_.keypoint_parent_ids.clear();
    if (int id = gguf_find_key(gguf_ctx, "ymapnet.keypoint_parent_ids");
        id >= 0 && gguf_get_arr_type(gguf_ctx, id) == GGUF_TYPE_INT32) {
        const int32_t* p = static_cast<const int32_t*>(gguf_get_arr_data(gguf_ctx, id));
        meta_.keypoint_parent_ids.assign(p, p + gguf_get_arr_n(gguf_ctx, id));
    }
    meta_.skeleton_score_weights.clear();
    if (int id = gguf_find_key(gguf_ctx, "ymapnet.skeleton_score_weights");
        id >= 0 && gguf_get_arr_type(gguf_ctx, id) == GGUF_TYPE_FLOAT32) {
        const float* p = static_cast<const float*>(gguf_get_arr_data(gguf_ctx, id));
        meta_.skeleton_score_weights.assign(p, p + gguf_get_arr_n(gguf_ctx, id));
    }
    meta_.layer_arch_json = kv_str(gguf_ctx, "ymapnet.layer_architecture", "[]");
    token_graph_  = kv_str(gguf_ctx, "ymapnet.token_graph");
    token_source_ = kv_str(gguf_ctx, "ymapnet.token_source");
    vocabulary_.clear();
    if (int id = gguf_find_key(gguf_ctx, "ymapnet.vocabulary"); id >= 0) {
        const size_t n = gguf_get_arr_n(gguf_ctx, id);
        vocabulary_.reserve(n);
        for (size_t i = 0; i < n; ++i) vocabulary_.emplace_back(gguf_get_arr_str(gguf_ctx, id, i));
    }

    // Retype F16 weights to F32 before allocation; the data is converted while reading below
    if (upcast_f16) {
        for (ggml_tensor* t = ggml_get_first_tensor(weight_ctx_); t; t = ggml_get_next_tensor(weight_ctx_, t)) {
            if (t->type != GGML_TYPE_F16) continue;
            t->type  = GGML_TYPE_F32;
            t->nb[0] = sizeof(float);
            for (int d = 1; d < GGML_MAX_DIMS; ++d) t->nb[d] = t->nb[d - 1] * t->ne[d - 1];
        }
        meta_.weight_dtype += " (as f32)";
    }

    // Q8_0/Q4_0 kernels (convertModelToGGUF.py _quantize_rows) are stored as row-padded 2D blocks
    // for size only: retype them to F32 in their original shape ('ymapnet.qshape.<name>'); the
    // data is dequantized (and multiplied by 'ymapnet.qscale.<name>') while reading below.
    // qrows[name] = padded row length in the file
    std::unordered_map<std::string, int64_t> qrows;
    for (ggml_tensor* t = ggml_get_first_tensor(weight_ctx_); t; t = ggml_get_next_tensor(weight_ctx_, t)) {
        if (!ggml_is_quantized(t->type)) continue;
        const int id = gguf_find_key(gguf_ctx, (std::string("ymapnet.qshape.") + t->name).c_str());
        if (id < 0 || gguf_get_arr_type(gguf_ctx, id) != GGUF_TYPE_INT32 || gguf_get_arr_n(gguf_ctx, id) > GGML_MAX_DIMS) {
            fprintf(stderr, "[model] Quantized tensor '%s' has no ymapnet.qshape\n", t->name);
            gguf_free(gguf_ctx);
            return false;
        }
        qrows[t->name] = t->ne[0];
        const int32_t* ne = static_cast<const int32_t*>(gguf_get_arr_data(gguf_ctx, id));
        for (int d = 0; d < GGML_MAX_DIMS; ++d) t->ne[d] = d < (int)gguf_get_arr_n(gguf_ctx, id) ? ne[d] : 1;
        t->type  = GGML_TYPE_F32;
        t->nb[0] = sizeof(float);
        for (int d = 1; d < GGML_MAX_DIMS; ++d) t->nb[d] = t->nb[d - 1] * t->ne[d - 1];
    }
    if (!qrows.empty() && !upcast_f16) meta_.weight_dtype += " (as f32)";

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
        size_t       nbytes = gguf_get_tensor_size(gguf_ctx, i);   // bytes in the file

        std::fseek(fp, static_cast<long>(data_base + offset), SEEK_SET);
        std::vector<uint8_t> buf(nbytes);
        if (std::fread(buf.data(), 1, nbytes, fp) != nbytes) {
            fprintf(stderr, "[model] Short read for '%s'\n", name);
            std::fclose(fp);
            gguf_free(gguf_ctx);
            return false;
        }
        const ggml_type ftype = gguf_get_tensor_type(gguf_ctx, i);
        if (ggml_is_quantized(ftype)) {
            const int64_t padded = qrows.at(name), nrows = (int64_t)(nbytes / ggml_row_size(ftype, padded));
            const int64_t n = ggml_nelements(tensor), row = n / nrows;
            std::vector<float> full(padded * nrows), f32(n);
            ggml_get_type_traits(ftype)->to_float(buf.data(), full.data(), (int64_t)full.size());
            const float scale = kv_f32(gguf_ctx, (std::string("ymapnet.qscale.") + name).c_str(), 1.f);
            for (int64_t r = 0; r < nrows; ++r)
                for (int64_t c = 0; c < row; ++c) f32[r * row + c] = full[r * padded + c] * scale;
            ggml_backend_tensor_set(tensor, f32.data(), 0, f32.size() * sizeof(float));
        } else if (ftype == GGML_TYPE_F16 && tensor->type == GGML_TYPE_F32) {
            std::vector<float> f32(ggml_nelements(tensor));
            ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t*>(buf.data()), f32.data(), (int64_t)f32.size());
            ggml_backend_tensor_set(tensor, f32.data(), 0, f32.size() * sizeof(float));
        } else {
            ggml_backend_tensor_set(tensor, buf.data(), 0, nbytes);
        }
        weights_[name] = tensor;
    }

    std::fclose(fp);
    gguf_free(gguf_ctx);

    printf("[model] Loaded %d tensors  (%s  %s)\n",
           n_tensors, meta_.model_name.c_str(), meta_.weight_dtype.c_str());

    bool ok = false;
    try {
        ok = build_graph(static_cast<int>(meta_.input_w), static_cast<int>(meta_.input_h));
    } catch (const std::exception& e) {
        fprintf(stderr, "[model] %s\n", e.what());
    }
    if (!ok) {
        fprintf(stderr, "[model] build_graph failed – is this GGUF from the current model "
                        "(re-export with ymapnet/conversion/convertModelToGGUF.py)?\n");
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
                        graph_ = nullptr; input_tensor_ = nullptr; output_tensor_ = nullptr;
                        output16_tensor_ = nullptr; }
    weights_.clear();
    if (weight_buf_) { ggml_backend_buffer_free(weight_buf_); weight_buf_ = nullptr; }
    if (weight_ctx_) { ggml_free(weight_ctx_);                weight_ctx_ = nullptr; }
    if (backend_)    { ggml_backend_free(backend_);           backend_    = nullptr; }
}

// ---------------------------------------------------------------------------
// YMAPNetModel::build_graph  — called once from load()
//
// Mirrors build_unet() in ymapnet/core/NNModel.py for the configuration
// useDepthwiseSeparable=true, deconCheckerboardSmoothing="nearest",
// useASPPBridge=false, useDepthRefinementHead=false, activation="prelu".
// Level widths and optional parts (CoordConv, bridge depth, bottleneck
// attention, pixelwise, 16-bit head) are discovered from the tensor names.
// ---------------------------------------------------------------------------
bool YMAPNetModel::build_graph(int W, int H) {
    const auto& wt  = weights_;
    const float eps = meta_.layernorm_eps;
#if defined(GGML_USE_VULKAN)
    const bool direct_conv = ggml_backend_is_vk(backend_);   // see conv_same
#else
    const bool direct_conv = false;
#endif

    // --- discover architecture ---
    meta_.enc_filters.clear();
    meta_.dec_filters.clear();
    for (int i = 0;; ++i) {
        int nf = find_filters(wt, "dwconv_encoder_d" + std::to_string(i) + "_r0_", ".weight");
        if (nf < 0) break;
        meta_.enc_filters.push_back(nf);
    }
    for (int i = 0;; ++i) {
        int nf = find_filters(wt, "conv_up_decoder_L" + std::to_string(i) + "_", ".weight");
        if (nf < 0) break;
        meta_.dec_filters.push_back(nf);
    }
    if (meta_.enc_filters.empty() || meta_.dec_filters.size() != meta_.enc_filters.size()) {
        fprintf(stderr, "[model] Unsupported architecture: expected ConvNeXt encoder (dwconv_encoder_*) and "
                        "nearest-upsampling decoder (conv_up_decoder_*) tensors, found %zu / %zu levels\n",
                meta_.enc_filters.size(), meta_.dec_filters.size());
        return false;
    }
    meta_.block_reps = 0;
    while (has(wt, "dwconv_encoder_d0_r" + std::to_string(meta_.block_reps) + "_" +
                   std::to_string(meta_.enc_filters[0]) + ".weight"))
        ++meta_.block_reps;

    const int in_ch = (int)get(wt, "dwconv_encoder_d0_r0_" + std::to_string(meta_.enc_filters[0]) + ".weight")->ne[3];
    meta_.coord_conv = (in_ch == (int)meta_.input_ch + 2);
    if (!meta_.coord_conv && in_ch != (int)meta_.input_ch) {
        fprintf(stderr, "[model] First block expects %d input channels (input_channels=%u)\n", in_ch, meta_.input_ch);
        return false;
    }
    meta_.bridge_layers   = has(wt, "BridgeFinish.weight") ? 3 : has(wt, "BridgeMid.weight") ? 2 : 1;
    meta_.bottleneck_attn = has(wt, "bottleneck_attn_mha.q.weight");

    const int levels = (int)meta_.enc_filters.size();
    const int reps   = meta_.block_reps;

    // --- graph ---
    const size_t max_nodes = 8192;
    struct ggml_init_params cp = {
        ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false), nullptr, true };
    compute_ctx_ = ggml_init(cp);
    if (!compute_ctx_) return false;
    ggml_context* ctx = compute_ctx_;

    // Input [W, H, 3 (+2 CoordConv planes), 1]
    ggml_tensor* x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, in_ch, 1);
    ggml_set_name(x, "input");
    ggml_set_input(x);
    input_tensor_ = x;

    // CoordConv (NNModel.py CoordConv2D): x = linspace(-1,1,W) along width, y = linspace(-1,1,H) along height
    coord_planes_.clear();
    if (meta_.coord_conv) {
        coord_planes_.resize((size_t)2 * W * H);
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w) {
                coord_planes_[(size_t)h * W + w]         = W > 1 ? -1.f + 2.f * w / (W - 1) : -1.f;
                coord_planes_[(size_t)W * H + h * W + w] = H > 1 ? -1.f + 2.f * h / (H - 1) : -1.f;
            }
    }

    // The input is uploaded as planar WHC; everything from here on is CWH
    x = to_cwh(ctx, x);

    // Encoder
    std::vector<ggml_tensor*> skips(levels);
    for (int i = 0; i < levels; ++i) {
        x = feature_conv(ctx, wt, x, "encoder_d" + std::to_string(i), meta_.enc_filters[i], reps, eps);
        skips[i] = x;
        x = avg_pool2(ctx, x);
    }

    // Bridge: 3x3 convs with dilation 1, 2, 4 (BN folded) + PReLU
    static const char* BRIDGE[3] = { "BridgeStart", "BridgeMid", "BridgeFinish" };
    for (int i = 0; i < meta_.bridge_layers; ++i) {
        x = conv_same(ctx, wt, BRIDGE[i], x, direct_conv, 1 << i);
        x = prelu(ctx, x, get(wt, std::string(BRIDGE[i]) + ".alpha"));
    }

    if (meta_.bottleneck_attn)
        x = bottleneck_attention(ctx, wt, x, eps);

    // Token head reads the flattened bottleneck (Keras Flatten of [H,W,C] = our CWH memory order)
    tokens_tensor_ = nullptr;
    if (!token_graph_.empty()) {
        if (token_source_ == "bottleneck_attn_unflatten" && meta_.bottleneck_attn) {
            ggml_tensor* flat = ggml_reshape_1d(ctx, x, ggml_nelements(x));
            tokens_tensor_ = build_token_head(ctx, wt, token_graph_, token_source_, flat);
            ggml_set_output(tokens_tensor_);
        } else {
            fprintf(stderr, "[model] token head reads '%s', which this build cannot provide – skipped\n",
                    token_source_.c_str());
        }
    }

    // Decoder
    for (int i = 0; i < levels; ++i)
        x = dec_block(ctx, wt, x, skips[levels - 1 - i], i, meta_.dec_filters[i], reps, eps, direct_conv);

    ggml_tensor* dec = x;

    // 8-bit head: [pixelwise 1x1 + PReLU] → 1x1 → tanh → Rescaling
    ggml_tensor* p = dec;
    if (has(wt, "pixelwise.weight"))
        p = prelu(ctx, pw_conv(ctx, wt, "pixelwise", dec), get(wt, "pixelwise.alpha"));
    ggml_tensor* hm = ggml_tanh(ctx, pw_conv(ctx, wt, "hm_tanh_8bit", p));
    hm = to_whc(ctx, ggml_scale(ctx, hm, meta_.output_scale));
    ggml_set_output(hm);
    output_tensor_ = hm;

    // 16-bit head: taps the decoder output directly (not pixelwise)
    output16_tensor_ = nullptr;
    if (has(wt, "hm_tanh_16bit.weight")) {
        ggml_tensor* hm16 = ggml_tanh(ctx, pw_conv(ctx, wt, "hm_tanh_16bit", dec));
        hm16 = to_whc(ctx, ggml_scale(ctx, hm16, meta_.output16_scale));
        ggml_set_output(hm16);
        output16_tensor_ = hm16;
        meta_.output16_ch = (uint32_t)hm16->ne[2];
    }
    meta_.output_ch = (uint32_t)hm->ne[2];

    // Build and allocate
    graph_ = ggml_new_graph_custom(ctx, max_nodes, false);
    ggml_build_forward_expand(graph_, hm);
    if (output16_tensor_) ggml_build_forward_expand(graph_, output16_tensor_);
    if (tokens_tensor_)   ggml_build_forward_expand(graph_, tokens_tensor_);

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
    const size_t rgb_bytes = (size_t)W * H * 3 * sizeof(float);
    ggml_backend_tensor_set(input_tensor_, input_rgb, 0, rgb_bytes);
    if (!coord_planes_.empty())
        ggml_backend_tensor_set(input_tensor_, coord_planes_.data(), rgb_bytes,
                                coord_planes_.size() * sizeof(float));
    ggml_backend_graph_compute(backend_, graph_);

    const size_t n_out = (size_t)W * H * meta_.output_ch;
    std::vector<float> result(n_out);
    ggml_backend_tensor_get(output_tensor_, result.data(), 0, n_out * sizeof(float));
    return result;
}

// ---------------------------------------------------------------------------
// YMAPNetModel::heatmap_16bit  — 16-bit head of the last forward() call
// ---------------------------------------------------------------------------
std::vector<float> YMAPNetModel::heatmap_16bit() const {
    if (!output16_tensor_) return {};
    std::vector<float> result(ggml_nelements(output16_tensor_));
    ggml_backend_tensor_get(output16_tensor_, result.data(), 0, ggml_nbytes(output16_tensor_));
    return result;
}

// ---------------------------------------------------------------------------
// YMAPNetModel::token_scores / describe
// ---------------------------------------------------------------------------
std::vector<float> YMAPNetModel::token_scores() const {
    if (!tokens_tensor_) return {};
    std::vector<float> result(ggml_nelements(tokens_tensor_));
    ggml_backend_tensor_get(tokens_tensor_, result.data(), 0, ggml_nbytes(tokens_tensor_));
    return result;
}

std::string YMAPNetModel::describe(int k, float threshold) const {
    std::vector<float> sc = token_scores();
    if (sc.empty()) return "";
    std::vector<int> idx(sc.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int)i;
    k = std::min<int>(k, (int)idx.size());
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return sc[a] > sc[b]; });

    std::string out;
    char buf[32];
    for (int i = 0; i < k; ++i) {
        const int id = idx[i];
        if (!(sc[id] > threshold)) break;
        // ids without a word (e.g. the model has one more class than vocabulary.json) are skipped
        if (id >= (int)vocabulary_.size() || vocabulary_[id].empty()) continue;
        std::snprintf(buf, sizeof(buf), ":%0.2f", sc[id]);
        if (!out.empty()) out += ' ';
        out += vocabulary_[id] + buf;
    }
    return out;
}

// ---------------------------------------------------------------------------
// YMAPNetModel::print_info
// ---------------------------------------------------------------------------
void YMAPNetModel::print_info() const {
    printf("\n=== YMAPNet Model ===\n");
    printf("  name           : %s\n",   meta_.model_name.c_str());
    printf("  serial         : %s\n",   meta_.serial.empty() ? "?" : meta_.serial.c_str());
    printf("  input          : %ux%ux%u\n", meta_.input_w, meta_.input_h, meta_.input_ch);
    printf("  output         : %ux%ux%u  (x%.0f)\n", meta_.output_w, meta_.output_h, meta_.output_ch, meta_.output_scale);
    printf("  output 16-bit  : %u ch  (x%.0f)\n", meta_.output16_ch, meta_.output16_scale);
    printf("  activation     : %s\n",   meta_.activation.c_str());
    printf("  CoordConv      : %s\n",   meta_.coord_conv ? "yes" : "no");
    printf("  encoder        :");
    for (int f : meta_.enc_filters) printf(" %d", f);
    printf("  (x%d ConvNeXt blocks)\n", meta_.block_reps);
    printf("  bridge         : %d conv(s)%s\n", meta_.bridge_layers, meta_.bottleneck_attn ? " + attention" : "");
    printf("  token head     : %s\n", tokens_tensor_ ? ("yes, " + std::to_string(vocabulary_.size()) + " words").c_str() : "no");
    printf("  decoder        :");
    for (int f : meta_.dec_filters) printf(" %d", f);
    printf("\n");
    printf("  BN folded      : %s\n",   meta_.bn_folded ? "yes" : "no");
    printf("  weight dtype   : %s\n",   meta_.weight_dtype.c_str());
    printf("  tensors loaded : %zu\n",  weights_.size());
    printf("  backend        : %s\n",   ggml_backend_name(backend_) ? ggml_backend_name(backend_) : "?");
    printf("=====================\n\n");
}
