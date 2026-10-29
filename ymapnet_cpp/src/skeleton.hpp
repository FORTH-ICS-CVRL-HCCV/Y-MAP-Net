#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <tuple>
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

// ===========================================================================
// Multi-person skeleton resolution — C++ port of ymapnet/skeletons/ (Python).
//
// Bottom-up OpenPose-style association adapted to Y-MAPNet's single-channel PAFs.  Each
// PAF channel holds a ramp along the bone drawn by the C DataLoader, t=0 at the child joint,
// t=1 at the parent joint, on a -120 background.  The ramp depends on the model's
// configuration.json 'heatmapPAFEncoding' (see Topology):
//     signed    (serials <= 288): value = -127 + 254 t if child AND parent are right_* joints,
//                                 +127 - 254 t otherwise
//     child_lit:                  value = +127 -> +64 for every limb
// person_lr_bridge ('heatmapBridgeEncoding'): signed -127 (left) -> +127 (right), or child_lit
// +127 (left) -> +64 (right).
// and which parent each joint's PAF runs to follows its 'keypoint_parents' (serials <= 288:
// hips -> nose; DataLoader PAF_HIPS_TO_SHOULDERS: hips -> same-side shoulder).
// Steps (see ymapnet/skeletons/resolve.py for the rationale and the COCO benchmark behind
// every constant):
//   1. joint peaks: Gaussian blur (sigma 2.5, background border) + 5x5 NMS + parabolic
//      sub-pixel refinement
//   2. optional: weight left/right joint confidences by the rightjoints/leftjoints channels,
//      and wrist / ankle / nose+eye confidences by the Hand / Foot / Face masks around the peak
//   3. score every candidate pair of every limb against the expected ramp
//   4. greedy bipartite matching per limb: torso limbs first, then the other limbs, then the
//      left<->right bridges; priority adds joint confidence and a bonus when the parent already
//      belongs to a person, and drops with the largest depth jump along the limb (links between
//      people cross depth edges); a pair the assembly refuses does not use up its two peaks
//   5. join upper-body fragments (shoulders, no hips) with lower-body fragments (hips, no
//      shoulders) by torso geometry, attach eyes/ears geometrically (to the neck when there is
//      no nose/eye), drop weak persons
// Keep in sync with the Python implementation (parity: scripts/testSkeletonParity.py).
// ===========================================================================

namespace skel {

struct Peak { float x, y, score; };

// PAF channel (absolute, -1 = none) of each joint's limb, configuration.json paf_parents
inline int paf_channel(int j) {
    static const int ch[NUM_JOINTS] = {-1, -1, -1, -1, -1, 28, 19, 27, 18, 26, 17, 25, 22, 24, 21, 23, 20};
    return ch[j];
}
inline int joint_side(int j) {   // +1 left, -1 right, 0 centre
    if (j == NOSE) return 0;
    return (j % 2 == 1) ? +1 : -1;
}

constexpr float kBackground     = -120.f;
constexpr float kChildLitParentQ = (64.f - kBackground) / 240.f;   // DataLoader PAF_PARENT_VALUE 64

// Limb layout of a model: configuration.json keypoint_parents, heatmapPAFEncoding and
// heatmapBridgeEncoding, stored in the GGUF as ymapnet.keypoint_parent_ids / ymapnet.paf_encoding /
// ymapnet.bridge_encoding.  The defaults are the layout of serials <= 288 (hips -> nose, signed
// ramps), which is what GGUFs without those keys contain.
struct Topology {
    std::array<int, NUM_JOINTS> parent = {NOSE, NOSE, NOSE, L_EYE, R_EYE, NOSE, NOSE, L_SHOULDER, R_SHOULDER,
                                          L_ELBOW, R_ELBOW, NOSE, NOSE, L_HIP, R_HIP, L_KNEE, R_KNEE};
    bool child_lit = false;     // limb PAFs
    bool bridge_lit = false;    // person_lr_bridge
    std::vector<float> score_weights;   // learned person score [bias, w...] (ymapnet/skeletons/score.py); empty = default
};
inline Topology make_topology(const std::vector<int>& parent_ids, const std::string& paf_encoding,
                              const std::string& bridge_encoding = "signed",
                              const std::vector<float>& score_weights = {}) {
    Topology t;
    t.score_weights = score_weights;
    if ((int)parent_ids.size() == NUM_JOINTS &&
        std::all_of(parent_ids.begin(), parent_ids.end(), [](int p) { return p >= 0 && p < NUM_JOINTS; }))
        std::copy(parent_ids.begin(), parent_ids.end(), t.parent.begin());
    t.child_lit = paf_encoding == "child_lit";
    t.bridge_lit = bridge_encoding == "child_lit";
    return t;
}

constexpr int   kPafStart       = 17;
constexpr int   kLimbSamples    = 12;
constexpr float kLimbTolerance  = 0.7f;
constexpr float kMinLimbScore   = 0.15f;
constexpr float kPeakWeight     = 1.5f;
constexpr float kOwnedBonus     = 0.5f;
constexpr float kMaxLimbLength  = 0.6f;   // fraction of max(W,H)
constexpr float kFaceGate       = 1.3f;
constexpr float kNeckGate       = 1.0f;   // eyes/ears anchored at the neck: within kNeckGate * longest bone
constexpr float kTorsoRatio     = 2.0f;   // torso <= kTorsoRatio * longest bone when linking body halves
constexpr int   kMinJoints      = 3;
constexpr float kMinPersonScore = 0.1f;   // sum(joint confidence) / NUM_JOINTS
constexpr float kDepthWeight    = 0.5f;   // priority penalty per unit of the largest depth jump along a limb
constexpr float kPartMaskFloor  = 0.3f;   // Hand/Foot/Face weighting: floor + (1-floor)*mask
constexpr int   kPartMaskRadius = 4;

// Bones that give a person's scale (longest present one survives foreshortening)
constexpr int kScaleBones[10][2] = {{L_SHOULDER, R_SHOULDER}, {L_HIP, R_HIP}, {L_SHOULDER, L_ELBOW},
                                    {R_SHOULDER, R_ELBOW},    {L_ELBOW, L_WRIST}, {R_ELBOW, R_WRIST},
                                    {L_HIP, L_KNEE},          {R_HIP, R_KNEE},  {L_KNEE, L_ANKLE},
                                    {R_KNEE, R_ANKLE}};

// Joint candidates of one channel (raw values in [-120, 120]), strongest first.
inline std::vector<Peak> find_peaks(const cv::Mat& raw, float threshold_raw, int nms = 5, double sigma = 2.5) {
    // blur against a background border: the default reflected border mirrors a joint near the
    // image edge onto itself and drags its peak up to 2px onto the edge pixel
    cv::Mat hm;
    cv::GaussianBlur(raw + 120.f, hm, cv::Size(0, 0), sigma, 0, cv::BORDER_CONSTANT);
    hm -= 120.f;
    cv::Mat dil;
    cv::dilate(hm, dil, cv::Mat::ones(nms, nms, CV_8U));
    const int H = hm.rows, W = hm.cols;

    std::vector<std::pair<int, int>> cand;   // (y, x) row-major like np.nonzero
    for (int y = 0; y < H; ++y) {
        const float* r = hm.ptr<float>(y);
        const float* d = dil.ptr<float>(y);
        for (int x = 0; x < W; ++x)
            if (r[x] >= d[x] && r[x] > threshold_raw) cand.emplace_back(y, x);
    }
    std::stable_sort(cand.begin(), cand.end(), [&](const auto& a, const auto& b) {
        return hm.at<float>(a.first, a.second) > hm.at<float>(b.first, b.second);
    });

    std::vector<Peak> out;
    cv::Mat taken = cv::Mat::zeros(H, W, CV_8U);
    const int r = nms / 2;
    for (const auto& [y, x] : cand) {
        if (taken.at<uint8_t>(y, x)) continue;   // plateau: one peak per plateau
        taken(cv::Range(std::max(0, y - r), std::min(H, y + r + 1)),
              cv::Range(std::max(0, x - r), std::min(W, x + r + 1))).setTo(1);
        const float c = hm.at<float>(y, x);
        const float l = hm.at<float>(y, std::max(0, x - 1)), rr = hm.at<float>(y, std::min(W - 1, x + 1));
        const float u = hm.at<float>(std::max(0, y - 1), x), dd = hm.at<float>(std::min(H - 1, y + 1), x);
        const float dxd = l - 2 * c + rr, dyd = u - 2 * c + dd;
        float dx = dxd < 0 ? 0.5f * (l - rr) / dxd : 0.f;
        float dy = dyd < 0 ? 0.5f * (u - dd) / dyd : 0.f;
        dx = std::clamp(dx, -0.5f, 0.5f);
        dy = std::clamp(dy, -0.5f, 0.5f);
        out.push_back({x + dx, y + dy, (c + 120.f) / 240.f});
    }
    return out;
}

// N x M agreement (0..1) of each (child, parent) segment with the expected ramp; the expected lit
// level q = (value + 120) / 240 goes linearly from q0 at the child to q1 at the parent.
inline std::vector<float> score_limb(const cv::Mat& ch, float q0, float q1, const std::vector<Peak>& A,
                                     const std::vector<Peak>& B, float max_len) {
    const int n = (int)A.size(), m = (int)B.size(), W = ch.cols, H = ch.rows;
    std::vector<float> s(n * m, 0.f);
    float qsum = 0.f;
    float t[kLimbSamples], q[kLimbSamples];
    for (int k = 0; k < kLimbSamples; ++k) {
        t[k] = k / float(kLimbSamples - 1);
        q[k] = q0 + (q1 - q0) * t[k];
        qsum += q[k];
    }
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < m; ++k) {
            const float vx = B[k].x - A[i].x, vy = B[k].y - A[i].y;
            const float len = std::sqrt(vx * vx + vy * vy);
            if (len > max_len || len < 1.f) continue;
            float acc = 0.f;
            for (int u = 0; u < kLimbSamples; ++u) {
                const float px = std::clamp(A[i].x + t[u] * vx, 0.f, W - 1.001f);
                const float py = std::clamp(A[i].y + t[u] * vy, 0.f, H - 1.001f);
                const int x0 = (int)px, y0 = (int)py;
                const float fx = px - x0, fy = py - y0;
                const float obs = (ch.at<float>(y0, x0) * (1 - fx) + ch.at<float>(y0, x0 + 1) * fx) * (1 - fy) +
                                  (ch.at<float>(y0 + 1, x0) * (1 - fx) + ch.at<float>(y0 + 1, x0 + 1) * fx) * fy;
                const float p = std::clamp((obs - kBackground) / 240.f, 0.f, 1.f);
                acc += std::clamp(1.f - std::fabs(p - q[u]) / kLimbTolerance, 0.f, 1.f) * q[u];
            }
            s[i * m + k] = acc / qsum;
        }
    return s;
}

// Channel indices of the optional auxiliary heatmaps (-1 when the model lacks them)
struct AuxChannels { int bridge = -1, right = -1, left = -1, depth = -1, hand = -1, foot = -1, face = -1; };
inline AuxChannels find_aux_channels(const std::vector<std::string>& heatmap_names) {
    AuxChannels a;
    for (int i = 0; i < (int)heatmap_names.size(); ++i) {
        if (heatmap_names[i] == "person_lr_bridge") a.bridge = i;
        else if (heatmap_names[i] == "rightjoints") a.right = i;
        else if (heatmap_names[i] == "leftjoints") a.left = i;
        else if (heatmap_names[i] == "depthmap") a.depth = i;
        else if (heatmap_names[i] == "Hand") a.hand = i;
        else if (heatmap_names[i] == "Foot") a.foot = i;
        else if (heatmap_names[i] == "Face") a.face = i;
    }
    return a;
}

}  // namespace skel

struct Person {
    std::array<Keypoint, NUM_JOINTS> joints{};   // heatmap pixel coordinates
    float score = 0.f;   // learned person score (Topology::score_weights), else mean confidence * joints / NUM_JOINTS
};

// hm          : float32 [n_channels * H * W], CHW planar, raw values (~[-120, 120])
// threshold   : joint peak threshold as a fraction of the 0..240 range (0.21 == Python's 50)
// topo        : the model's limb layout and PAF encoding (default: serials <= 288)
inline std::vector<Person> resolve_skeletons(const float* hm, int W, int H, int n_channels, float threshold,
                                             const skel::AuxChannels& aux = {}, const skel::Topology& topo = {}) {
    using namespace skel;
    auto channel = [&](int c) {
        cv::Mat m(H, W, CV_32F, const_cast<float*>(hm + (size_t)c * W * H));
        cv::Mat out;
        cv::min(cv::max(m, -120.f), 120.f, out);
        return out;
    };
    if (n_channels < kPafStart + 12) return {};

    // 1. peaks
    std::vector<std::vector<Peak>> peaks(NUM_JOINTS);
    for (int j = 0; j < NUM_JOINTS; ++j) peaks[j] = find_peaks(channel(j), threshold * 240.f - 120.f);

    // 2. left/right aggregate weighting
    if (aux.right >= 0 && aux.left >= 0 && aux.right < n_channels && aux.left < n_channels) {
        cv::Mat R = channel(aux.right), L = channel(aux.left);
        for (int j = 0; j < NUM_JOINTS; ++j) {
            const int side = joint_side(j);
            if (!side) continue;
            for (auto& pk : peaks[j]) {
                const int xi = std::clamp((int)std::nearbyint(pk.x), 0, W - 1);
                const int yi = std::clamp((int)std::nearbyint(pk.y), 0, H - 1);
                const float agree = side * (L.at<float>(yi, xi) - R.at<float>(yi, xi)) / 240.f;
                pk.score *= std::clamp(0.5f + 0.5f * agree, 0.f, 1.f);
            }
        }
    }

    // 2b. Hand/Foot/Face masks: strongest mask value within kPartMaskRadius px of each peak
    const std::pair<int, std::vector<int>> parts[3] = {{aux.hand, {L_WRIST, R_WRIST}},
                                                        {aux.foot, {L_ANKLE, R_ANKLE}},
                                                        {aux.face, {NOSE, L_EYE, R_EYE}}};
    for (const auto& [c, joints] : parts) {
        if (c < 0 || c >= n_channels) continue;
        cv::Mat m;
        cv::dilate(channel(c), m, cv::Mat::ones(2 * kPartMaskRadius + 1, 2 * kPartMaskRadius + 1, CV_8U));
        for (int j : joints)
            for (auto& pk : peaks[j]) {
                const int xi = std::clamp((int)std::nearbyint(pk.x), 0, W - 1);
                const int yi = std::clamp((int)std::nearbyint(pk.y), 0, H - 1);
                pk.score *= kPartMaskFloor + (1.f - kPartMaskFloor) *
                            std::clamp((m.at<float>(yi, xi) + 120.f) / 240.f, 0.f, 1.f);
            }
    }

    // 3-4. limbs: torso (parent = a root joint) first, then the rest, then the bridges, once each
    // side's chain is built; joint order within
    struct Limb { int child, parent, channel; float q0, q1; };
    std::vector<Limb> limbs;
    for (int pass = 0; pass < 3; ++pass) {
        if (pass == 2) {   // person_lr_bridge: signed (-127 at the left joint) or lit (+127 there)
            if (aux.bridge >= 0 && aux.bridge < n_channels) {
                const float q0 = topo.bridge_lit ? 1.f : 0.f, q1 = topo.bridge_lit ? kChildLitParentQ : 1.f;
                limbs.push_back({L_SHOULDER, R_SHOULDER, aux.bridge, q0, q1});
                limbs.push_back({L_HIP, R_HIP, aux.bridge, q0, q1});
            }
            continue;
        }
        for (int j = 0; j < NUM_JOINTS; ++j) {
            const int p = topo.parent[j];
            if (paf_channel(j) < 0 || p == j) continue;
            if ((pass == 0) != (topo.parent[p] == p)) continue;
            if (topo.child_lit)
                limbs.push_back({j, p, paf_channel(j), 1.f, kChildLitParentQ});
            else if (joint_side(j) < 0 && joint_side(p) < 0)
                limbs.push_back({j, p, paf_channel(j), 0.f, 1.f});
            else
                limbs.push_back({j, p, paf_channel(j), 1.f, 0.f});
        }
    }

    std::vector<std::array<int, NUM_JOINTS>> persons;
    std::vector<std::vector<int>> owner(NUM_JOINTS);
    for (int j = 0; j < NUM_JOINTS; ++j) owner[j].assign(peaks[j].size(), -1);
    auto new_person = [&]() { std::array<int, NUM_JOINTS> p; p.fill(-1); persons.push_back(p); return (int)persons.size() - 1; };
    auto set = [&](int p, int j, int i) { persons[p][j] = i; owner[j][i] = p; };
    // move person pb into pa unless both own the same joint type (two different people)
    auto merge = [&](int pa, int pb) {
        for (int j = 0; j < NUM_JOINTS; ++j)
            if (persons[pa][j] >= 0 && persons[pb][j] >= 0) return false;
        for (int j = 0; j < NUM_JOINTS; ++j)
            if (persons[pb][j] >= 0) { set(pa, j, persons[pb][j]); persons[pb][j] = -1; }
        return true;
    };
    // false when the pair contradicts the persons its peaks already belong to
    auto connect = [&](int ja, int ia, int jb, int ib) {
        const int pa = owner[ja][ia], pb = owner[jb][ib];
        if (pa < 0 && pb < 0) { const int p = new_person(); set(p, ja, ia); set(p, jb, ib); }
        else if (pa >= 0 && pb < 0) { if (persons[pa][jb] >= 0) return false; set(pa, jb, ib); }
        else if (pa < 0 && pb >= 0) { if (persons[pb][ja] >= 0) return false; set(pb, ja, ia); }
        else if (pa != pb) return merge(pa, pb);
        return true;
    };

    const float max_len = kMaxLimbLength * std::max(W, H);
    cv::Mat depth;
    if (aux.depth >= 0 && aux.depth < n_channels) depth = channel(aux.depth);
    auto depth_jump = [&](float ax, float ay, float bx, float by) {   // largest step between consecutive samples
        float prev = 0.f, jump = 0.f;
        for (int u = 0; u < kLimbSamples; ++u) {
            const float t = u / float(kLimbSamples - 1);
            const int xi = std::clamp((int)std::nearbyint(ax + t * (bx - ax)), 0, W - 1);
            const int yi = std::clamp((int)std::nearbyint(ay + t * (by - ay)), 0, H - 1);
            const float d = depth.at<float>(yi, xi);
            if (u > 0) jump = std::max(jump, std::fabs(d - prev));
            prev = d;
        }
        return jump / 240.f;
    };
    for (const Limb& l : limbs) {
        const auto& A = peaks[l.child];
        const auto& B = peaks[l.parent];
        if (A.empty() || B.empty()) continue;
        cv::Mat ch;
        cv::dilate(channel(l.channel), ch, cv::Mat::ones(3, 3, CV_8U));   // tolerate 1px off the 2px line
        const std::vector<float> s = score_limb(ch, l.q0, l.q1, A, B, max_len);
        const int m = (int)B.size();
        std::vector<std::tuple<float, int, int>> cand;   // (-priority, i, k), row-major for stable ties
        for (int i = 0; i < (int)A.size(); ++i)
            for (int k = 0; k < m; ++k) {
                if (s[i * m + k] <= kMinLimbScore) continue;
                float pr = s[i * m + k] + kPeakWeight * 0.5f * (A[i].score + B[k].score);
                if (owner[l.parent][k] >= 0) pr += kOwnedBonus;
                if (!depth.empty()) pr -= kDepthWeight * depth_jump(A[i].x, A[i].y, B[k].x, B[k].y);
                if (pr <= 0.f) continue;   // a large depth jump vetoes the limb
                cand.emplace_back(-pr, i, k);
            }
        std::stable_sort(cand.begin(), cand.end(),
                         [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });
        std::vector<char> usedA(A.size(), 0), usedB(B.size(), 0);
        for (const auto& [np, i, k] : cand) {
            if (usedA[i] || usedB[k]) continue;
            if (connect(l.child, i, l.parent, k)) usedA[i] = usedB[k] = 1;   // a refused pair keeps its peaks
        }
    }

    auto longest_bone = [&](const std::array<int, NUM_JOINTS>& P) {
        float best = 0.f;
        for (const auto& b : kScaleBones)
            if (P[b[0]] >= 0 && P[b[1]] >= 0) {
                const Peak& a = peaks[b[0]][P[b[0]]];
                const Peak& c = peaks[b[1]][P[b[1]]];
                best = std::max(best, std::hypot(a.x - c.x, a.y - c.y));
            }
        return best;
    };

    // 5a. upper-body fragments (shoulders, no hips) + lower-body fragments (hips, no shoulders):
    // nothing joins them when the hips' PAF runs to an undetected nose; linked when the torso is at
    // most kTorsoRatio longest bones and left->right directions agree, shortest relative torso first
    {
        auto centre = [&](const std::array<int, NUM_JOINTS>& P, int a, int b, float& x, float& y) {
            int n = 0;
            x = y = 0.f;
            for (int j : {a, b})
                if (P[j] >= 0) { x += peaks[j][P[j]].x; y += peaks[j][P[j]].y; ++n; }
            if (n) { x /= n; y /= n; }
            return n;
        };
        std::vector<int> uppers, lowers;
        for (int p = 0; p < (int)persons.size(); ++p) {
            const auto& P = persons[p];
            const bool sh = P[L_SHOULDER] >= 0 || P[R_SHOULDER] >= 0, hp = P[L_HIP] >= 0 || P[R_HIP] >= 0;
            if (sh && !hp) uppers.push_back(p);
            if (hp && !sh) lowers.push_back(p);
        }
        std::vector<std::tuple<float, int, int>> cand;
        for (int u : uppers) {
            const auto& U = persons[u];
            float sx, sy;
            centre(U, L_SHOULDER, R_SHOULDER, sx, sy);
            for (int lo : lowers) {
                const auto& L = persons[lo];
                bool overlap = false;
                for (int j = 0; j < NUM_JOINTS; ++j) overlap |= U[j] >= 0 && L[j] >= 0;
                if (overlap) continue;
                const float ref = std::max(longest_bone(U), longest_bone(L));
                if (ref <= 0.f) continue;
                float hx, hy;
                centre(L, L_HIP, R_HIP, hx, hy);
                const float ratio = std::hypot(sx - hx, sy - hy) / ref;
                if (ratio > kTorsoRatio) continue;
                if (U[L_SHOULDER] >= 0 && U[R_SHOULDER] >= 0 && L[L_HIP] >= 0 && L[R_HIP] >= 0) {
                    const Peak &ls = peaks[L_SHOULDER][U[L_SHOULDER]], &rs = peaks[R_SHOULDER][U[R_SHOULDER]];
                    const Peak &lh = peaks[L_HIP][L[L_HIP]], &rh = peaks[R_HIP][L[R_HIP]];
                    const float vsx = rs.x - ls.x, vsy = rs.y - ls.y, vhx = rh.x - lh.x, vhy = rh.y - lh.y;
                    // directions only measurable when neither pair is foreshortened (side views)
                    if (std::min(std::hypot(vsx, vsy), std::hypot(vhx, vhy)) > 0.3f * ref &&
                        vsx * vhx + vsy * vhy <= 0.f)
                        continue;
                }
                float cost = ratio;
                if (!depth.empty()) cost += kDepthWeight * depth_jump(sx, sy, hx, hy);
                cand.emplace_back(cost, u, lo);
            }
        }
        std::sort(cand.begin(), cand.end());
        std::vector<char> used(persons.size(), 0);
        for (const auto& [cost, u, lo] : cand) {
            if (used[u] || used[lo]) continue;
            if (merge(u, lo)) used[u] = used[lo] = 1;
        }
    }

    // noses no limb claimed still seed (face-only) persons
    for (int i = 0; i < (int)peaks[NOSE].size(); ++i)
        if (owner[NOSE][i] < 0) set(new_person(), NOSE, i);

    // 5b. eyes / ears: global greedy by distance, gated by each person's own scale; the neck (shoulder
    // midpoint) is the anchor of last resort, gated by the longest bone (shoulder width collapses in
    // side views)
    constexpr int NECK = -2;
    auto dist = [](float ax, float ay, float bx, float by) { return std::hypot(ax - bx, ay - by); };
    const int face[4][4] = {{L_EYE, NOSE, NECK, -1}, {R_EYE, NOSE, NECK, -1},
                            {L_EAR, L_EYE, NOSE, NECK}, {R_EAR, R_EYE, NOSE, NECK}};
    for (const auto& f : face) {
        const int j = f[0];
        std::vector<std::tuple<float, int, int>> cand;
        for (int p = 0; p < (int)persons.size(); ++p) {
            const auto& P = persons[p];
            if (P[j] >= 0) continue;
            float ref = 0.08f * std::max(W, H);
            const bool hl = P[L_SHOULDER] >= 0, hr = P[R_SHOULDER] >= 0, hn = P[NOSE] >= 0;
            const Peak* ls = hl ? &peaks[L_SHOULDER][P[L_SHOULDER]] : nullptr;
            const Peak* rs = hr ? &peaks[R_SHOULDER][P[R_SHOULDER]] : nullptr;
            if (hl && hr) ref = dist(ls->x, ls->y, rs->x, rs->y);
            else if (hn && (hl || hr)) {
                const Peak& n = peaks[NOSE][P[NOSE]];
                const Peak* s = hl ? ls : rs;
                ref = dist(n.x, n.y, s->x, s->y);
            }
            for (int a = 1; a < 4 && f[a] != -1; ++a) {
                float gate = kFaceGate * ref, ax, ay;
                if (f[a] == NECK) {
                    if (!hl && !hr) continue;
                    ax = hl && hr ? (ls->x + rs->x) / 2 : (hl ? ls->x : rs->x);
                    ay = hl && hr ? (ls->y + rs->y) / 2 : (hl ? ls->y : rs->y);
                    const float bone = longest_bone(P);
                    if (bone > 0.f) gate = kNeckGate * bone;
                } else {
                    if (P[f[a]] < 0) continue;
                    ax = peaks[f[a]][P[f[a]]].x;
                    ay = peaks[f[a]][P[f[a]]].y;
                }
                for (int i = 0; i < (int)peaks[j].size(); ++i) {
                    const float d = dist(peaks[j][i].x, peaks[j][i].y, ax, ay);
                    if (d <= gate) cand.emplace_back(d, p, i);
                }
                break;
            }
        }
        std::sort(cand.begin(), cand.end());
        for (const auto& [d, p, i] : cand)
            if (owner[j][i] < 0 && persons[p][j] < 0) set(p, j, i);
    }

    // 6. output
    std::vector<Person> out;
    for (const auto& P : persons) {
        int n = 0;
        float sum = 0.f;
        for (int j = 0; j < NUM_JOINTS; ++j)
            if (P[j] >= 0) { ++n; sum += peaks[j][P[j]].score; }
        if (n < kMinJoints || sum / NUM_JOINTS < kMinPersonScore) continue;
        Person person;
        for (int j = 0; j < NUM_JOINTS; ++j)
            if (P[j] >= 0) {
                const Peak& pk = peaks[j][P[j]];
                person.joints[j] = {pk.x, pk.y, pk.score, true};
            }
        person.score = sum / NUM_JOINTS;
        out.push_back(person);
    }

    // 7. learned person score (ymapnet/skeletons/score.py: same features, same order, keep in sync), best first
    constexpr int kAux = 6, kScoreFeatures = 7 + NUM_JOINTS + kAux;
    if ((int)topo.score_weights.size() == 1 + kScoreFeatures) {
        const int auxCh[kAux] = {aux.face, aux.hand, aux.foot, aux.right, aux.left, aux.bridge};
        cv::Mat auxM[kAux];
        for (int a = 0; a < kAux; ++a)
            if (auxCh[a] >= 0 && auxCh[a] < n_channels) auxM[a] = channel(auxCh[a]);
        const std::vector<float>& w = topo.score_weights;
        for (Person& p : out) {
            double f[kScoreFeatures] = {0};
            int n = 0;
            double sum = 0, mn = 1e9, mx = -1e9, x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9, auxSum[kAux] = {0};
            for (int j = 0; j < NUM_JOINTS; ++j) {
                const Keypoint& k = p.joints[j];
                if (!k.valid) continue;
                ++n; sum += k.confidence;
                mn = std::min(mn, (double)k.confidence); mx = std::max(mx, (double)k.confidence);
                x0 = std::min(x0, (double)k.x); x1 = std::max(x1, (double)k.x);
                y0 = std::min(y0, (double)k.y); y1 = std::max(y1, (double)k.y);
                f[7 + j] = k.confidence;
                const int xi = std::clamp((int)k.x, 0, W - 1), yi = std::clamp((int)k.y, 0, H - 1);
                for (int a = 0; a < kAux; ++a)
                    if (!auxM[a].empty()) auxSum[a] += auxM[a].at<float>(yi, xi);
            }
            f[0] = n; f[1] = sum / n; f[2] = mn; f[3] = mx;
            f[4] = std::sqrt((x1 - x0 + 1.0) * (y1 - y0 + 1.0));
            f[5] = sum / NUM_JOINTS; f[6] = (double)out.size();
            for (int a = 0; a < kAux; ++a) f[7 + NUM_JOINTS + a] = auxSum[a] / n;
            double s = w[0];
            for (int i = 0; i < kScoreFeatures; ++i) s += (double)w[1 + i] * f[i];
            p.score = (float)s;
        }
        std::stable_sort(out.begin(), out.end(), [](const Person& a, const Person& b) { return a.score > b.score; });
    }
    return out;
}

// Draw every resolved person; joints are scaled from heatmap to image coordinates by (sx, sy).
inline void draw_skeletons(cv::Mat& img, const std::vector<Person>& persons, float sx, float sy) {
    for (const Person& p : persons) {
        std::vector<Keypoint> kps(p.joints.begin(), p.joints.end());
        for (auto& k : kps) { k.x *= sx; k.y *= sy; }
        draw_skeleton(img, kps, 0.f);
    }
}
