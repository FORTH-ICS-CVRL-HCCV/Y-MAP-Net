#pragma once
#include <opencv2/imgproc.hpp>
#include <vector>
#include <cstdint>

// Preprocess a BGR frame for YMAPNet inference.
//
// Steps:
//   1. Centre-crop to square (largest inscribed square) – optional
//   2. Resize to target_size × target_size
//   3. BGR → RGB
//   4. Scale uint8 [0,255] → float32 [0,1]
//
// Returns a contiguous float32 buffer of length target_size*target_size*3
// laid out as [H, W, C] = [R, G, B] per pixel (matches Keras input).
inline std::vector<float> preprocess_frame(
        const cv::Mat & bgr,
        int   target_size = 256,
        bool  centre_crop = true)
{
    cv::Mat src = bgr;

    if (centre_crop) {
        int side = std::min(src.cols, src.rows);
        int x0   = (src.cols - side) / 2;
        int y0   = (src.rows - side) / 2;
        src = src(cv::Rect(x0, y0, side, side));
    }

    cv::Mat resized;
    cv::resize(src, resized, {target_size, target_size}, 0, 0, cv::INTER_LINEAR);

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    cv::Mat fp;
    rgb.convertTo(fp, CV_32FC3, 1.0 / 255.0);

    // ggml tensor ne=[W,H,C,N] with default strides has CHW planar layout:
    //   element(w,h,c) = data[w + W*h + W*H*c]
    // OpenCV stores [H,W,C] interleaved, so we must de-interleave to CHW.
    std::vector<cv::Mat> planes;
    cv::split(fp, planes);   // planes[0]=R, planes[1]=G, planes[2]=B  [H,W] each

    std::vector<float> buf(target_size * target_size * 3);
    const size_t plane_sz = (size_t)target_size * target_size;
    for (int c = 0; c < 3; ++c)
        std::memcpy(buf.data() + c * plane_sz,
                    planes[c].ptr<float>(),
                    plane_sz * sizeof(float));
    return buf;
}

// Copy a preprocessed float32 [H, W, 3] buffer into an OpenCV Mat for display.
inline cv::Mat float_buf_to_mat(const float * buf, int W, int H) {
    cv::Mat fp(H, W, CV_32FC3, const_cast<float*>(buf));
    cv::Mat u8;
    fp.convertTo(u8, CV_8UC3, 255.0);
    cv::cvtColor(u8, u8, cv::COLOR_RGB2BGR);
    return u8;
}
