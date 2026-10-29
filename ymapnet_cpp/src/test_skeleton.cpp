// Runs resolve_skeletons() on a raw heatmap dump and prints one line per person:
//   x y confidence valid   (x NUM_JOINTS, heatmap pixel coordinates)
// Used by scripts/testSkeletonParity.py to check the port against ymapnet/skeletons (Python).
//
// usage: test_skeleton heatmap.f32 W H C threshold bridge_ch right_ch left_ch depth_ch hand_ch foot_ch face_ch
//                      [parent_ids paf_encoding [bridge_encoding [score_weights]]]
//        heatmap.f32 = float32 [C*H*W] CHW planar (the layout of Model::forward / --dump)
//        parent_ids  = comma separated parent joint id per joint (configuration.json keypoint_parents),
//        paf_encoding, bridge_encoding = signed | child_lit; all default to the layout of serials <= 288
//        score_weights = comma separated learned person score [bias, w...] (configuration.json skeletonScoreWeights)
// Each line ends with the person's score.
#include "skeleton.hpp"
#include <cstdio>
#include <cstdlib>
#include <sstream>

int main(int argc, char** argv) {
    if (argc != 13 && argc != 15 && argc != 16 && argc != 17) {
        fprintf(stderr, "usage: %s heatmap.f32 W H C threshold bridge_ch right_ch left_ch depth_ch hand_ch foot_ch face_ch"
                        " [parent_ids paf_encoding [bridge_encoding [score_weights]]]\n", argv[0]);
        return 1;
    }
    const int W = atoi(argv[2]), H = atoi(argv[3]), C = atoi(argv[4]);
    std::vector<float> hm((size_t)W * H * C);
    FILE* f = fopen(argv[1], "rb");
    if (!f || fread(hm.data(), sizeof(float), hm.size(), f) != hm.size()) {
        fprintf(stderr, "cannot read %s\n", argv[1]);
        return 1;
    }
    fclose(f);

    skel::AuxChannels aux;
    aux.bridge = atoi(argv[6]);
    aux.right  = atoi(argv[7]);
    aux.left   = atoi(argv[8]);
    aux.depth  = atoi(argv[9]);
    aux.hand   = atoi(argv[10]);
    aux.foot   = atoi(argv[11]);
    aux.face   = atoi(argv[12]);
    skel::Topology topo;
    if (argc >= 15) {
        std::vector<int> parents;
        std::stringstream ss(argv[13]);
        for (std::string tok; std::getline(ss, tok, ',');) parents.push_back(atoi(tok.c_str()));
        std::vector<float> weights;
        if (argc == 17) {
            std::stringstream ws(argv[16]);
            for (std::string tok; std::getline(ws, tok, ',');) weights.push_back((float)atof(tok.c_str()));
        }
        topo = skel::make_topology(parents, argv[14], argc >= 16 ? argv[15] : "signed", weights);
    }
    for (const Person& p : resolve_skeletons(hm.data(), W, H, C, (float)atof(argv[5]), aux, topo)) {
        for (const Keypoint& k : p.joints) printf("%.4f %.4f %.4f %d ", k.x, k.y, k.confidence, k.valid ? 1 : 0);
        printf("%.6f\n", p.score);
    }
    return 0;
}
