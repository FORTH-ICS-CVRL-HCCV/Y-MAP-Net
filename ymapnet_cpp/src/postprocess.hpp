#pragma once
#include "skeleton.hpp"
#include <opencv2/imgproc.hpp>
#include <vector>
#include <cmath>

// Dequantise a raw heatmap channel from float32 (range ≈ [-120, 120])
// to uint8 [0, 240] the same way YMAPNet.py does it.
inline void dequantise_heatmap(
        const float * channel_f32,
        uint8_t     * out_u8,
        int           n_pixels,
        float         scale = 120.f)
{
    for (int i = 0; i < n_pixels; ++i) {
        float v = channel_f32[i] + scale;   // shift [-scale, scale] → [0, 2*scale]
        v = std::max(0.f, std::min(2.f * scale, v));
        out_u8[i] = static_cast<uint8_t>(v);
    }
}

// Find the single peak in a heatmap channel using connected-component centroids.
// Mirrors find_peak_points_from_convertIO() in YMAPNet.py.
//
// heatmap_u8: single channel uint8 [H, W], values in [0, 240]
// Returns {x, y, confidence} or {0,0,0,false} if nothing exceeds threshold.
inline Keypoint find_peak(
        const uint8_t * heatmap_u8,
        int W, int H,
        float threshold = 0.3f)
{
    // threshold in the same [0,1] scale as YMAPNet.py
    auto thresh_u8 = static_cast<uint8_t>(threshold * 240.f);

    cv::Mat hm(H, W, CV_8UC1, const_cast<uint8_t*>(heatmap_u8));
    cv::Mat binary;
    cv::threshold(hm, binary, thresh_u8, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    if (contours.empty()) return {};

    // Pick the contour with the highest summed activation
    int   best_idx    = 0;
    float best_energy = 0.f;
    for (int i = 0; i < static_cast<int>(contours.size()); ++i) {
        cv::Mat mask = cv::Mat::zeros(H, W, CV_8UC1);
        cv::drawContours(mask, contours, i, 255, cv::FILLED);
        float energy = static_cast<float>(cv::sum(hm & mask)[0]);
        if (energy > best_energy) { best_energy = energy; best_idx = i; }
    }

    // Subpixel centroid of the winning blob
    cv::Moments m = cv::moments(contours[best_idx]);
    if (std::abs(m.m00) < 1e-6) return {};

    float cx   = static_cast<float>(m.m10 / m.m00);
    float cy   = static_cast<float>(m.m01 / m.m00);
    float conf = best_energy / (240.f * static_cast<float>(W * H));

    return { cx, cy, conf, true };
}

// Detect all NUM_JOINTS keypoints from the network heatmap output.
//
// heatmap_f32: float32 buffer [H, W, n_channels] (row-major, channels last)
//              channels 0..NUM_JOINTS-1 are body joint heatmaps
// scale_to_w / scale_to_h: scale factors to map from heatmap coords back to
//              the original image resolution (pass 1.f to keep heatmap coords)
inline std::vector<Keypoint> detect_keypoints(
        const float * heatmap_f32,
        int           hm_w,
        int           hm_h,
        int           n_channels,
        float         threshold  = 0.3f,
        float         scale_to_w = 1.f,
        float         scale_to_h = 1.f)
{
    std::vector<Keypoint> result(NUM_JOINTS);
    std::vector<uint8_t>  chan_u8(hm_w * hm_h);

    for (int j = 0; j < NUM_JOINTS && j < n_channels; ++j) {
        const float * ch = heatmap_f32 + j * hm_w * hm_h;
        dequantise_heatmap(ch, chan_u8.data(), hm_w * hm_h);
        Keypoint kp = find_peak(chan_u8.data(), hm_w, hm_h, threshold);
        kp.x *= scale_to_w;
        kp.y *= scale_to_h;
        result[j] = kp;
    }
    return result;
}
