#include "model.hpp"
#include "preprocess.hpp"
#include "postprocess.hpp"
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
    std::string model_path  = "../2d_pose_estimation/model_f32.gguf";
    std::string input_src   = "0";       // "0" = webcam 0, or path to video/image
    int         input_size  = 256;
    float       threshold   = 0.3f;
    int         cuda_device = 0;
    bool        cpu_only    = false;
    bool        headless    = false;
    bool        centre_crop = true;
    bool        info_only   = false;     // print model info and exit
    std::string dump_path;               // if non-empty, write raw heatmap float32 here and exit
};

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  --model PATH      GGUF model file (default: %s)\n", "../2d_pose_estimation/model_f32.gguf");
    printf("  --from SRC        Input: 0=webcam, path to video/image (default: 0)\n");
    printf("  --threshold T     Keypoint confidence threshold 0..1 (default: 0.30)\n");
    printf("  --cuda DEVICE     CUDA device index (default: 0)\n");
    printf("  --cpu             Force CPU backend\n");
    printf("  --headless        No display window\n");
    printf("  --nocrop          Disable centre-crop\n");
    printf("  --info            Print model info and exit\n");
    printf("  --dump PATH       Save raw float32 heatmap of first frame to PATH and exit\n");
}

static Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--model")     && i+1 < argc) cfg.model_path  = argv[++i];
        else if (!strcmp(argv[i], "--from")      && i+1 < argc) cfg.input_src   = argv[++i];
        else if (!strcmp(argv[i], "--threshold") && i+1 < argc) cfg.threshold   = std::stof(argv[++i]);
        else if (!strcmp(argv[i], "--cuda")      && i+1 < argc) cfg.cuda_device = std::stoi(argv[++i]);
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
    }

    // -----------------------------------------------------------------------
    // Load model
    // -----------------------------------------------------------------------
    YMAPNetModel model;
    {
        auto t0 = Clock::now();
        if (!model.load(cfg.model_path, cfg.cuda_device)) {
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

        // --- dump raw heatmap (validation mode) ---
        if (!cfg.dump_path.empty()) {
            FILE* fp = std::fopen(cfg.dump_path.c_str(), "wb");
            if (!fp) { fprintf(stderr, "[main] Cannot write dump: %s\n", cfg.dump_path.c_str()); }
            else {
                std::fwrite(heatmap.data(), sizeof(float), heatmap.size(), fp);
                std::fclose(fp);
                printf("[main] Wrote raw heatmap (%zu floats) to %s\n", heatmap.size(), cfg.dump_path.c_str());
            }
            break;
        }

        // --- postprocess ---
        auto t_post = Clock::now();
        float sx = static_cast<float>(frame.cols) / W;
        float sy = static_cast<float>(frame.rows) / H;
        std::vector<Keypoint> kps = detect_keypoints(
            heatmap.data(), W, H, OCH, cfg.threshold, sx, sy);
        total_post_ms += ms_since(t_post);
        total_nn_ms   += ms_since(t_pre);  // pre + fwd + post

        ++frame_count;

        // --- display / overlay ---
        if (!cfg.headless) {
            cv::Mat hm_grid = make_heatmap_grid(
                heatmap.data(), W, H, OCH, model.meta().heatmap_scale);
            cv::imshow("YMAPNet Heatmaps", hm_grid);

            cv::Mat vis = frame.clone();
            draw_skeleton(vis, kps, cfg.threshold);

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
