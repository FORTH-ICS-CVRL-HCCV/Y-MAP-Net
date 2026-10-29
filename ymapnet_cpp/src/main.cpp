#include "model.hpp"
#include "preprocess.hpp"
#include "skeleton.hpp"
#include "heatmap_vis.hpp"

#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
struct Config {
    std::string model_path  = "ymapnet_model/model_f32.gguf";
    std::string input_src   = "0";       // "0" = webcam 0, or path to video/image
    int         input_size  = 256;
    float       threshold   = 0.3f;
    int         cuda_device = 0;
    int         vulkan_device = -1;      // >= 0 selects the Vulkan backend
    int         threads     = 0;         // CPU backend threads, 0 = all
    bool        upcast_f16  = false;     // load F16 weights as F32
    bool        cpu_only    = false;
    bool        headless    = false;
    bool        centre_crop = true;
    bool        info_only   = false;     // print model info and exit
    std::string dump_path;               // if non-empty, write raw heatmap float32 here and exit
};

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  --model PATH      GGUF model file (default: %s)\n", "ymapnet_model/model_f32.gguf");
    printf("  --from SRC        Input: 0=webcam, path to video/image (default: 0)\n");
    printf("  --threshold T     Joint peak threshold, fraction of the 0..240 heatmap range (default: 0.30)\n");
    printf("  --cuda DEVICE     CUDA device index (default: 0)\n");
    printf("  --vulkan DEVICE   Use Vulkan device index (build with -DYMAPNET_VULKAN=ON)\n");
    printf("  --cpu             Force CPU backend\n");
    printf("  --threads N       CPU backend threads (default: all hardware threads)\n");
    printf("  --upcast          Load F16 weights as F32 (faster on GPUs with slow F16 matmul)\n");
    printf("  --headless        No display window\n");
    printf("  --nocrop          Disable centre-crop\n");
    printf("  --info            Print model info and exit\n");
    printf("  --dump PATH       Save raw float32 heatmap of first frame to PATH (16-bit head to PATH.16bit) and exit\n");
}

static Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--model")     && i+1 < argc) cfg.model_path  = argv[++i];
        else if (!strcmp(argv[i], "--from")      && i+1 < argc) cfg.input_src   = argv[++i];
        else if (!strcmp(argv[i], "--threshold") && i+1 < argc) cfg.threshold   = std::stof(argv[++i]);
        else if (!strcmp(argv[i], "--cuda")      && i+1 < argc) cfg.cuda_device = std::stoi(argv[++i]);
        else if (!strcmp(argv[i], "--vulkan")    && i+1 < argc) cfg.vulkan_device = std::stoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads")   && i+1 < argc) cfg.threads     = std::stoi(argv[++i]);
        else if (!strcmp(argv[i], "--upcast")) cfg.upcast_f16  = true;
        else if (!strcmp(argv[i], "--cpu"))    cfg.cpu_only    = true;
        else if (!strcmp(argv[i], "--headless"))cfg.headless   = true;
        else if (!strcmp(argv[i], "--nocrop")) cfg.centre_crop = false;
        else if (!strcmp(argv[i], "--info"))   cfg.info_only   = true;
        else if (!strcmp(argv[i], "--dump")      && i+1 < argc) cfg.dump_path   = argv[++i];
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            print_usage(argv[0]); std::exit(0);
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]); std::exit(1);
        }
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------
using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    Config cfg = parse_args(argc, argv);

    // Force CPU if requested (override CUDA device selection in model.load)
    if (cfg.cpu_only) {
        printf("[main] Forcing CPU backend\n");
#if defined(GGML_USE_CUDA)
        // pass cuda_device = -1 as a sentinel; model.cpp will skip cuda_init
        cfg.cuda_device = -1;
#endif
        cfg.vulkan_device = -1;
    }

    // -----------------------------------------------------------------------
    // Load model
    // -----------------------------------------------------------------------
    YMAPNetModel model;
    {
        auto t0 = Clock::now();
        if (!model.load(cfg.model_path, cfg.cuda_device, cfg.vulkan_device, cfg.threads, cfg.upcast_f16)) {
            fprintf(stderr, "[main] Model load failed.\n");
            return 1;
        }
        printf("[main] Model loaded in %.1f ms\n", ms_since(t0));
    }

    model.print_info();

    if (cfg.info_only) return 0;

    // -----------------------------------------------------------------------
    // Open input
    // -----------------------------------------------------------------------
    cv::VideoCapture cap;
    bool is_image = false;

    // Try as integer (webcam index)
    bool src_is_int = !cfg.input_src.empty() &&
        cfg.input_src.find_first_not_of("0123456789") == std::string::npos;
    if (src_is_int) {
        cap.open(std::stoi(cfg.input_src));
    } else {
        // Check for image extensions
        const char* img_exts[] = {".jpg",".jpeg",".png",".bmp",".tiff",".webp"};
        for (auto ext : img_exts) {
            if (cfg.input_src.size() >= strlen(ext) &&
                cfg.input_src.compare(cfg.input_src.size()-strlen(ext), strlen(ext), ext) == 0) {
                is_image = true; break;
            }
        }
        if (!is_image) cap.open(cfg.input_src);
    }

    if (!is_image && !cap.isOpened()) {
        fprintf(stderr, "[main] Cannot open input: %s\n", cfg.input_src.c_str());
        model.free();
        return 1;
    }

    const int  W   = cfg.input_size;
    const int  H   = cfg.input_size;
    const int  OCH = static_cast<int>(model.meta().output_ch);
    const skel::AuxChannels aux = skel::find_aux_channels(model.meta().heatmap_names);
    const skel::Topology topo = skel::make_topology(model.meta().keypoint_parent_ids, model.meta().paf_encoding,
                                                    model.meta().bridge_encoding, model.meta().skeleton_score_weights);

    printf("[main] Processing %s  →  heatmap %dx%dx%d\n",
           cfg.input_src.c_str(), W, H, OCH);

    // -----------------------------------------------------------------------
    // Main loop
    // -----------------------------------------------------------------------
    cv::Mat      frame;
    int          frame_count = 0;
    double       total_preprocess_ms = 0.0;
    double       total_forward_ms    = 0.0;
    double       total_post_ms       = 0.0;
    double       total_nn_ms         = 0.0;  // pre + fwd + post (no display)

    auto loop_start = Clock::now();

    while (true) {
        // --- grab frame ---
        if (is_image) {
            frame = cv::imread(cfg.input_src);
            if (frame.empty()) { fprintf(stderr, "[main] Cannot read image\n"); break; }
        } else {
            if (!cap.read(frame) || frame.empty()) break;
        }

        // --- preprocess ---
        auto t_pre = Clock::now();
        std::vector<float> input_buf = preprocess_frame(frame, W, cfg.centre_crop);
        total_preprocess_ms += ms_since(t_pre);

        // --- forward pass ---
        auto t_fwd = Clock::now();
        std::vector<float> heatmap = model.forward(input_buf.data(), W, H);
        total_forward_ms += ms_since(t_fwd);

        if (model.has_tokens()) printf("[main] Description: %s\n", model.describe().c_str());

        // --- dump raw heatmap (validation mode) ---
        if (!cfg.dump_path.empty()) {
            FILE* fp = std::fopen(cfg.dump_path.c_str(), "wb");
            if (!fp) { fprintf(stderr, "[main] Cannot write dump: %s\n", cfg.dump_path.c_str()); }
            else {
                std::fwrite(heatmap.data(), sizeof(float), heatmap.size(), fp);
                std::fclose(fp);
                printf("[main] Wrote raw heatmap (%zu floats) to %s\n", heatmap.size(), cfg.dump_path.c_str());
            }
            std::vector<float> tok = model.token_scores();
            if (!tok.empty()) {
                std::string pt = cfg.dump_path + ".tokens";
                if (FILE* ft = std::fopen(pt.c_str(), "wb")) {
                    std::fwrite(tok.data(), sizeof(float), tok.size(), ft);
                    std::fclose(ft);
                    printf("[main] Wrote token scores (%zu floats) to %s\n", tok.size(), pt.c_str());
                }
            }
            std::vector<float> hm16 = model.heatmap_16bit();
            if (!hm16.empty()) {
                std::string p16 = cfg.dump_path + ".16bit";
                FILE* fp16 = std::fopen(p16.c_str(), "wb");
                if (!fp16) { fprintf(stderr, "[main] Cannot write dump: %s\n", p16.c_str()); }
                else {
                    std::fwrite(hm16.data(), sizeof(float), hm16.size(), fp16);
                    std::fclose(fp16);
                    printf("[main] Wrote raw 16-bit heatmap (%zu floats) to %s\n", hm16.size(), p16.c_str());
                }
            }
            break;
        }

        // --- postprocess ---
        auto t_post = Clock::now();
        float sx = static_cast<float>(frame.cols) / W;
        float sy = static_cast<float>(frame.rows) / H;
        std::vector<Person> persons = resolve_skeletons(heatmap.data(), W, H, OCH, cfg.threshold, aux, topo);
        total_post_ms += ms_since(t_post);
        total_nn_ms   += ms_since(t_pre);  // pre + fwd + post

        ++frame_count;

        // --- display / overlay ---
        if (!cfg.headless) {
            cv::Mat hm_grid = make_heatmap_grid(
                heatmap.data(), W, H, OCH, model.meta().heatmap_scale);
            cv::imshow("YMAPNet Heatmaps", hm_grid);

            cv::Mat vis = frame.clone();
            draw_skeletons(vis, persons, sx, sy);

            // FPS overlay: NN fps (pre+fwd+post) and loop fps (wall)
            double elapsed   = ms_since(loop_start);
            double loop_fps  = frame_count * 1000.0 / (elapsed > 0.0 ? elapsed : 1.0);
            double nn_fps    = frame_count * 1000.0 / (total_nn_ms > 0.0 ? total_nn_ms : 1.0);
            char buf[64];
            std::snprintf(buf, sizeof(buf), "NN %.1f  loop %.1f fps", nn_fps, loop_fps);
            cv::putText(vis, buf, {10, 25}, cv::FONT_HERSHEY_SIMPLEX,
                        0.7, {0, 255, 0}, 2, cv::LINE_AA);

            cv::imshow("YMAPNet", vis);
            int key = cv::waitKey(is_image ? 0 : 1);
            if (key == 27 || key == 'q') break;
        }

        if (is_image) break;   // single frame
    }

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    if (frame_count > 0) {
        double elapsed_s = ms_since(loop_start) / 1000.0;
        double nn_fps    = frame_count * 1000.0 / (total_nn_ms > 0.0 ? total_nn_ms : 1.0);
        printf("\n--- Performance (%d frames) ---\n", frame_count);
        printf("  NN fps         : %.1f  (pre+fwd+post only)\n", nn_fps);
        printf("  Loop fps       : %.1f  (wall, incl. display)\n", frame_count / elapsed_s);
        printf("  Wall time      : %.2f s\n", elapsed_s);
        printf("  Preprocess     : %.2f ms/frame\n", total_preprocess_ms / frame_count);
        printf("  Forward pass   : %.2f ms/frame\n", total_forward_ms / frame_count);
        printf("  Postprocess    : %.2f ms/frame\n", total_post_ms / frame_count);
    }

    model.free();
    return 0;
}
