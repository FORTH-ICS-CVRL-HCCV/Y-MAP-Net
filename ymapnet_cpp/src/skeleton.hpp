#pragma once
#include <array>
#include <string>
#include <vector>
#include <opencv2/imgproc.hpp>

// COCO-17 joint indices (matches configuration.json heatmap channels 0-16)
enum Joint : int {
    NOSE=0, L_EYE, R_EYE, L_EAR, R_EAR,
    L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW,
    L_WRIST, R_WRIST, L_HIP, R_HIP,
    L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
    NUM_JOINTS
};

inline const char* joint_name(int j) {
    static const char* names[] = {
        "nose","left_eye","right_eye","left_ear","right_ear",
        "left_shoulder","right_shoulder","left_elbow","right_elbow",
        "left_wrist","right_wrist","left_hip","right_hip",
        "left_knee","right_knee","left_ankle","right_ankle"
    };
    return (j >= 0 && j < NUM_JOINTS) ? names[j] : "?";
}

// Directed skeleton edges (parent → child), matches keypoint_parents in configuration.json
struct SkeletonEdge { int from, to; cv::Scalar color; };
inline const std::array<SkeletonEdge, 16>& skeleton_edges() {
    static const std::array<SkeletonEdge, 16> edges = {{
        {NOSE,       L_EYE,      {255, 200,   0}},
        {NOSE,       R_EYE,      {255, 200,   0}},
        {L_EYE,      L_EAR,      {255, 150,   0}},
        {R_EYE,      R_EAR,      {255, 150,   0}},
        {NOSE,       L_SHOULDER, {  0, 255, 100}},
        {NOSE,       R_SHOULDER, {  0, 255, 100}},
        {L_SHOULDER, L_ELBOW,    {  0, 200, 255}},
        {R_SHOULDER, R_ELBOW,    {  0, 200, 255}},
        {L_ELBOW,    L_WRIST,    {  0, 100, 255}},
        {R_ELBOW,    R_WRIST,    {  0, 100, 255}},
        {NOSE,       L_HIP,      {200,   0, 255}},
        {NOSE,       R_HIP,      {200,   0, 255}},
        {L_HIP,      L_KNEE,     {255,   0, 200}},
        {R_HIP,      R_KNEE,     {255,   0, 200}},
        {L_KNEE,     L_ANKLE,    {255,   0, 100}},
        {R_KNEE,     R_ANKLE,    {255,   0, 100}},
    }};
    return edges;
}

struct Keypoint {
    float x = 0.f, y = 0.f, confidence = 0.f;
    bool  valid = false;
};

// Draw detected skeleton onto `img` (in-place).
// `kps` has NUM_JOINTS entries, scaled to `img` dimensions.
inline void draw_skeleton(cv::Mat& img, const std::vector<Keypoint>& kps, float conf_thresh = 0.3f) {
    if (kps.size() < NUM_JOINTS) return;

    for (const auto& edge : skeleton_edges()) {
        const auto& a = kps[edge.from];
        const auto& b = kps[edge.to];
        if (!a.valid || !b.valid) continue;
        if (a.confidence < conf_thresh || b.confidence < conf_thresh) continue;
        cv::line(img,
            {static_cast<int>(a.x), static_cast<int>(a.y)},
            {static_cast<int>(b.x), static_cast<int>(b.y)},
            edge.color, 2, cv::LINE_AA);
    }

    for (int j = 0; j < NUM_JOINTS; ++j) {
        const auto& kp = kps[j];
        if (!kp.valid || kp.confidence < conf_thresh) continue;
        cv::circle(img, {static_cast<int>(kp.x), static_cast<int>(kp.y)},
                   4, {255, 255, 255}, -1, cv::LINE_AA);
        cv::circle(img, {static_cast<int>(kp.x), static_cast<int>(kp.y)},
                   3, {0, 0, 0}, 1, cv::LINE_AA);
    }
}
