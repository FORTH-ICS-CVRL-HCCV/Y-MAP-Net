#TODO:
#This is a stub it should take the unordered points and return a list of skeletons with populated joints or 0,0 if they don't exist
import sys
import os
import json
import cv2
import numpy as np

"""
"keypoint_names": [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle"
    ]

    "keypoint_parents": {
        "nose": "nose",
        "left_eye": "nose",
        "right_eye": "nose",
        "left_ear": "left_eye",
        "right_ear": "right_eye",
        "left_shoulder": "nose",
        "right_shoulder": "nose",
        "left_elbow": "left_shoulder",
        "right_elbow": "right_shoulder",
        "left_wrist": "left_elbow",
        "right_wrist": "right_elbow",
        "left_hip": "nose",
        "right_hip": "nose",
        "left_knee": "left_hip",
        "right_knee": "right_hip",
        "left_ankle": "left_knee",
        "right_ankle": "right_knee"
    }

      "keypoint_children": {
                           "nose":      ["left_eye","right_eye","left_shoulder","right_shoulder","left_hip","right_hip"],
                                  "left_eye":  ["left_ear"],
                                  "right_eye": ["right_ear"],
                                  "left_ear":  [],
                                  "right_ear": [],
                                  "left_shoulder":  ["left_elbow"],
                                  "right_shoulder": ["right_elbow"],
                                  "left_elbow": ["left_wrist"],
                                  "right_elbow":["right_wrist"],
                                  "left_wrist": [],
                                  "right_wrist":[],
                                  "left_hip":   ["left_knee"],
                                  "right_hip":  ["right_knee"],
                                  "left_knee":  ["left_ankle"],
                                  "right_knee": ["right_ankle"],
                                  "left_ankle": [],
                                  "right_ankle":[]
                              }

 'paf_parents': [
     0,
     0,
     0,
     0,
     0,
     28,
     19,
     27,
     18,
     26,
     17,
     25,
     22,
     24,
     21,
     23,
     20
    ] 


-----------------------------
          Keypoints
-----------------------------
hm  0       head
hm  1       endsite_eye.l
hm  2       endsite_eye.r
hm  3       lear
hm  4       rear
hm  5       lshoulder
hm  6       rshoulder
hm  7       lelbow
hm  8       relbow
hm  9       lhand
hm 10       rhand
hm 11       lhip
hm 12       rhip
hm 13       lknee
hm 14       rknee
hm 15       lfoot
hm 16       rfoot
-----------------------------
           PAFs
-----------------------------
hm 17       10->8         0
hm 18       8->6          1
hm 19       6->0          2
hm 20       16->14        3
hm 21       14->12        4
hm 22       12->0         5
hm 23       15->13        6
hm 24       13->11        7
hm 25       11->0         8
hm 26       9->7          9
hm 27       7->5          10
hm 28       5->0          11
-----------------------------
hm 29       depthmap
...
hm 34       person


"""

def resolveJointHierarchyS(keypoint_results,keypoint_depths,keypoint_names,keypoint_parents,depth_map):
 skeletons = list()
 return skeletons
 print("resolveJointHierarchy called with ",len(keypoint_results)," keypoint_results")
 print("resolveJointHierarchy called with ",len(keypoint_depths)," keypoint_depths")
 print("resolveJointHierarchy called with ",len(keypoint_names)," keypoint_names")
 print("resolveJointHierarchy called with ",len(keypoint_parents)," keypoint_parents")

 for jID in range(len(keypoint_names)):
     print(" There are ",len(keypoint_results[jID])," joints of type ",keypoint_names[jID]," #",jID)

 image = np.array(depth_map,dtype=np.uint8)
 cv2.imshow('DEPTHMAP', image)


 return skeletons

def sample_paf_along_line(PAF_heatmaps, paf_index,
                          x1, y1, x2, y2,
                          num_samples=20):
    """
    Sample a 1-channel PAF along the segment (x1,y1)->(x2,y2).
    Accepts either:
      • numpy array [H,W,C]
      • list of numpy arrays [C][H,W]
    """

    if PAF_heatmaps is None or paf_index is None:
        return 0.0

    # -------- handle both formats --------
    if isinstance(PAF_heatmaps, list):
        # list of [H,W] numpy arrays
        C = len(PAF_heatmaps)
        if paf_index < 0 or paf_index >= C:
            return 0.0
        channel = PAF_heatmaps[paf_index]
        H, W = channel.shape
    else:
        # numpy array [H,W,C]
        H, W, C = PAF_heatmaps.shape
        if paf_index < 0 or paf_index >= C:
            return 0.0
        channel = PAF_heatmaps[..., paf_index]

    # Limb length check
    dx = x2 - x1
    dy = y2 - y1
    seg_len = (dx * dx + dy * dy) ** 0.5
    if seg_len < 1e-6:
        return 0.0

    acc = 0.0
    valid = 0

    for i in range(num_samples + 1):
        t = i / float(num_samples)
        xn = x1 + t * dx
        yn = y1 + t * dy

        px = int(round(xn * (W - 1)))
        py = int(round(yn * (H - 1)))

        if 0 <= px < W and 0 <= py < H:
            v = float(channel[py, px])
            acc += abs(v)
            valid += 1

    if valid == 0:
        return 0.0

    return (acc / valid) / 120.0  # normalize to 0..1



def filter_joint_heatmaps_with_segmentation_and_pafs(
    keypoint_heatmaps,     # HxWx17
    paf_heatmaps,          # HxWx12
    person_label_map,      # HxW
    paf_parents,
    paf_children,
    seg_required=True,
    paf_threshold=0.20
):
    H, W, K = keypoint_heatmaps.shape
    filtered = keypoint_heatmaps.copy()

    BACKGROUND = np.float32(-120.0)   # deactivated value → falls below peak threshold

    # 1. Segmentation mask -------------------------------------
    if seg_required and person_label_map is not None:
        seg_mask = (person_label_map > 0)[..., None]   # HxWx1
        filtered = np.where(seg_mask, filtered, BACKGROUND)

    # 2. PAF threshold mask ------------------------------------
    paf_abs_threshold = paf_threshold * 127.0

    # Start with everything masked OFF (False)
    paf_mask = np.zeros((H, W, K), dtype=bool)

    # Parent and child PAF masks – vectorized
    for jID in range(K):
        masks = []

        parent_paf = paf_parents[jID]
        if parent_paf >= 0:
            mask_parent = np.abs(paf_heatmaps[:, :, parent_paf]) >= paf_abs_threshold
            masks.append(mask_parent)

        child_paf = paf_children[jID]
        if child_paf >= 0:
            mask_child = np.abs(paf_heatmaps[:, :, child_paf]) >= paf_abs_threshold
            masks.append(mask_child)

        if len(masks) == 0:   # no PAF for this joint → allow all
            paf_mask[:, :, jID] = True
        else:
            paf_mask[:, :, jID] = np.any(np.stack(masks, axis=2), axis=2)

    # 3. Apply PAF filter ----------------------------------------
    filtered = np.where(paf_mask, filtered, BACKGROUND)

    return filtered



# Function to calculate Euclidean distance between two points
def distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Function to find the closest joint to a given point
def closest_joint(point):
        min_dist = float('inf')
        closest_idx = None
        for i, joint in enumerate(keypoint_results):
            if not visited[i]:
                dist = distance(point, joint)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
        return closest_idx


def findClosestJointDistanceOnly(keypoint_candidates, skeleton, joint_index):
    min_distance = float('inf')
    closest_joint = None
    joint_x, joint_y, _ = skeleton[joint_index * 3:joint_index * 3 + 3]
    
    for candidate in keypoint_candidates:
        candidate_x, candidate_y, _ = candidate
        distance = ((candidate_x - joint_x) ** 2 + (candidate_y - joint_y) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            closest_joint = candidate
    
    return closest_joint

def calculateAverageDistance(skeleton):
    total_distance = 0
    count = 0
    num_joints = len(skeleton) // 3
    
    for i in range(num_joints):
        for j in range(i + 1, num_joints):
            x1, y1, _ = skeleton[i * 3: i * 3 + 3]
            x2, y2, _ = skeleton[j * 3: j * 3 + 3]
            if (x1 != 0 or y1 != 0) and (x2 != 0 or y2 != 0):
                distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                total_distance += distance
                count += 1
    
    return total_distance / count if count > 0 else 0

def assign_best_candidate_for_person(joint_candidates,
                                     person_label_map,
                                     person_id,
                                     W, H):
    """
    From all candidates for a given joint type, pick the best one for this person.

    joint_candidates: list of (x_norm, y_norm, score)
    person_label_map: HxW int32, 0 = background, 1..N = persons
    person_id: label ID in person_label_map
    W, H: heatmap width and height

    If no candidate falls inside this person's mask, falls back to
    the globally best candidate (by score).
    """
    if not joint_candidates:
        return None

    # No segmentation? just pick best by score
    if person_label_map is None or W is None or H is None:
        return max(joint_candidates, key=lambda c: c[2])

    best = None
    best_score = -1e9

    for (xn, yn, score) in joint_candidates:
        px = int(round(xn * (W - 1)))
        py = int(round(yn * (H - 1)))

        if 0 <= px < W and 0 <= py < H:
            pid = person_label_map[py, px]
        else:
            pid = 0

        # prefer candidates that lie in this person's region
        if pid == person_id and score > best_score:
            best_score = score
            best = (xn, yn, score)

    # If none belonged to this person, optionally fall back to global best
    if best is None:
        best = max(joint_candidates, key=lambda c: c[2])

    return best


# ---------------------------------------------------------------------------
# Anatomical limb-length heuristics
# ---------------------------------------------------------------------------

# Expected limb length as a multiple of the body “scale” (the adaptive
# avg_distance used in growSkeletons).  Tuned for COCO 17-joint topology
# where nose is the root and connects directly to shoulders and hips.
_LIMB_SCALE_FACTOR = {
    'left_eye':       0.20,   # nose → left_eye   (very short)
    'right_eye':      0.20,   # nose → right_eye
    'left_ear':       0.28,   # left_eye  → left_ear
    'right_ear':      0.28,   # right_eye → right_ear
    'left_shoulder':  0.55,   # nose → left_shoulder
    'right_shoulder': 0.55,   # nose → right_shoulder
    'left_elbow':     0.70,   # left_shoulder  → left_elbow
    'right_elbow':    0.70,   # right_shoulder → right_elbow
    'left_wrist':     0.65,   # left_elbow  → left_wrist
    'right_wrist':    0.65,   # right_elbow → right_wrist
    'left_hip':       0.65,   # nose → left_hip
    'right_hip':      0.65,   # nose → right_hip
    'left_knee':      0.85,   # left_hip  → left_knee
    'right_knee':     0.85,   # right_hip → right_knee
    'left_ankle':     0.85,   # left_knee  → left_ankle
    'right_ankle':    0.85,   # right_knee → right_ankle
}

# Symmetric counterpart for each joint (used to cross-check limb lengths).
_SYMMETRIC_LIMB = {
    'left_eye':       'right_eye',
    'right_eye':      'left_eye',
    'left_ear':       'right_ear',
    'right_ear':      'left_ear',
    'left_shoulder':  'right_shoulder',
    'right_shoulder': 'left_shoulder',
    'left_elbow':     'right_elbow',
    'right_elbow':    'left_elbow',
    'left_wrist':     'right_wrist',
    'right_wrist':    'left_wrist',
    'left_hip':       'right_hip',
    'right_hip':      'left_hip',
    'left_knee':      'right_knee',
    'right_knee':     'left_knee',
    'left_ankle':     'right_ankle',
    'right_ankle':    'left_ankle',
}

# How much deviation from expected length is tolerated before a penalty kicks
# in.  0.40 = ±40 % of expected length is free; beyond that cost rises linearly.
_LIMB_LENGTH_TOLERANCE = 0.40


def findClosestJoint(keypoint_candidates, skeleton, joint_index,
                     depth_map,
                     PAF_heatmaps=None, paf_index=None,
                     max_distance_threshold=0.35,
                     paf_weight=50.0,
                     depth_weight=0.1,
                     confidence_weight=0.3,
                     # Anatomical heuristics (optional)
                     child_joint_index=None,
                     keypoint_names=None,
                     avg_body_scale=None,
                     limb_length_weight=2.0):
    """
    Choose one candidate joint for a given anchor joint in a skeleton.

    - Distance is in normalized [0..1] coordinate space.
    - Depth penalty is based on depth variation along the segment.
    - PAF reward is based on average |PAF| along the segment for the
      appropriate limb channel (paf_index).
    - Limb-length penalty: penalises candidates whose proposed bone length
      deviates too far from the expected length (estimated from the symmetric
      established limb, or from _LIMB_SCALE_FACTOR × avg_body_scale).
    """
    min_cost = float('inf')
    best_candidate = None

    # Anchor joint (already in skeleton)
    joint_x, joint_y, _ = skeleton[joint_index * 3:joint_index * 3 + 3]

    # If anchor not initialized, nothing to do
    if joint_x == 0 and joint_y == 0:
        return None

    H = depth_map.shape[0] if depth_map is not None else 0
    W = depth_map.shape[1] if depth_map is not None else 0

    # ------------------------------------------------------------------
    # Estimate expected limb length from anatomical heuristics
    # ------------------------------------------------------------------
    child_name   = (keypoint_names[child_joint_index]
                    if child_joint_index is not None and keypoint_names is not None
                       and child_joint_index < len(keypoint_names)
                    else None)
    anchor_name  = (keypoint_names[joint_index]
                    if keypoint_names is not None and joint_index < len(keypoint_names)
                    else None)

    expected_length       = None   # None → penalty not applied
    expected_from_symmetry = False  # True only when derived from a real measured limb

    if child_name is not None and keypoint_names is not None:
        # 1) Symmetry: if the mirrored limb is already established, use its length.
        #    This is a reliable, frame-derived estimate, so we also tighten the
        #    hard distance cap with it.
        sym_child_name  = _SYMMETRIC_LIMB.get(child_name)
        sym_anchor_name = _SYMMETRIC_LIMB.get(anchor_name) if anchor_name else None

        if (sym_child_name and sym_child_name in keypoint_names and
                sym_anchor_name and sym_anchor_name in keypoint_names):
            sc_idx = keypoint_names.index(sym_child_name)
            sa_idx = keypoint_names.index(sym_anchor_name)
            sc_x, sc_y, sc_v = skeleton[sc_idx*3 : sc_idx*3 + 3]
            sa_x, sa_y, sa_v = skeleton[sa_idx*3 : sa_idx*3 + 3]
            if sc_v > 0 and sa_v > 0:
                expected_length        = ((sc_x - sa_x)**2 + (sc_y - sa_y)**2) ** 0.5
                expected_from_symmetry = True

        # 2) Fallback: body scale × anatomical ratio.
        #    avg_body_scale is estimated from whatever joints are already placed;
        #    when only face joints are known the scale is tiny and unreliable for
        #    distant joints like hips/knees.  Use this only as a SOFT penalty —
        #    never tighten the hard distance cap with it.
        if expected_length is None and avg_body_scale is not None and avg_body_scale > 0:
            ratio = _LIMB_SCALE_FACTOR.get(child_name)
            if ratio is not None:
                expected_length = ratio * avg_body_scale

    # Tighten the hard distance cap ONLY when we have a reliable symmetry
    # measurement.  The scale-factor estimate is too noisy (depends on which
    # joints are already placed) to safely restrict the search radius.
    effective_max = max_distance_threshold
    if expected_from_symmetry and expected_length is not None:
        effective_max = min(max_distance_threshold, expected_length * 2.0)

    for candidate in keypoint_candidates:
        candidate_x, candidate_y, candidate_score = candidate

        # Euclidean distance in normalized coords
        dist = ((candidate_x - joint_x) ** 2 +
                (candidate_y - joint_y) ** 2) ** 0.5

        # Skip far-away candidates
        if dist > effective_max:
            continue

        # ----------------- depth penalty (along the line) -----------------
        depth_penalty = 0.0
        if depth_map is not None and W > 1 and H > 1:
            steps = max(3, int(dist * max(W, H)))
            prev_depth = None
            for step in range(steps + 1):
                t = step / float(max(steps, 1))
                xn = joint_x + t * (candidate_x - joint_x)
                yn = joint_y + t * (candidate_y - joint_y)
                px = int(round(xn * (W - 1)))
                py = int(round(yn * (H - 1)))
                if 0 <= px < W and 0 <= py < H:
                    d = float(depth_map[py, px])
                    if prev_depth is not None:
                        depth_penalty += abs(d - prev_depth) / 255.0
                    prev_depth = d

        # ----------------- PAF “goodness” along the limb -----------------
        paf_score = 0.0
        if PAF_heatmaps is not None and paf_index is not None and paf_index >= 0:
            paf_score = sample_paf_along_line(
                PAF_heatmaps, paf_index,
                joint_x, joint_y,
                candidate_x, candidate_y
            )

        # ----------------- Limb-length penalty ---------------------------
        # Penalise deviation beyond the tolerance band.
        # penalty = max(0,  |dist/expected - 1| - tolerance)
        length_penalty = 0.0
        if expected_length is not None and expected_length > 0:
            deviation = abs(dist / expected_length - 1.0)
            length_penalty = max(0.0, deviation - _LIMB_LENGTH_TOLERANCE)

        # Final cost: lower is better
        cost = (dist
                + depth_weight      * depth_penalty
                + limb_length_weight * length_penalty
                - paf_weight        * paf_score
                - confidence_weight * (candidate_score / 240.0))

        if cost < min_cost:
            min_cost = cost
            best_candidate = candidate

    return best_candidate


def calculate_paf_score(paf_heatmaps,paf_index, joint_x, joint_y, candidate_x, candidate_y):
    # This function should access the PAF heatmap based on paf_index
    # and compute the PAF score between the given joints.
    # Placeholder logic:
    paf_score = 0.0
    
    # Assuming `paf_heatmaps` is globally accessible or passed as an argument
    # and it's a list of heatmaps, where each heatmap corresponds to a PAF direction (x, y).
    for i in range(paf_index, paf_index + 2):  # Assuming 2 channels per PAF (x and y directions)
        paf_heatmap = paf_heatmaps[i]
        vec_x = candidate_x - joint_x
        vec_y = candidate_y - joint_y
        magnitude = np.sqrt(vec_x**2 + vec_y**2)
        norm_vec_x = vec_x / magnitude
        norm_vec_y = vec_y / magnitude

        # Sample the PAF heatmap at points along the line
        num_samples = int(magnitude)
        paf_score = 0.0
        for step in range(num_samples):
            intermediate_x = int(joint_x + step * norm_vec_x)
            intermediate_y = int(joint_y + step * norm_vec_y)
            if (0 <= intermediate_x < paf_heatmap.shape[1]) and (0 <= intermediate_y < paf_heatmap.shape[0]):
                paf_x = paf_heatmap[0, intermediate_y, intermediate_x]
                paf_y = paf_heatmap[1, intermediate_y, intermediate_x]
                paf_score += paf_x * norm_vec_x + paf_y * norm_vec_y

        paf_score /= num_samples if num_samples > 0 else 1

    return paf_score



def growSkeletons(skeletons, skeleton_explanation,
                  keypoint_results, keypoint_depths,
                  keypoint_names, keypoint_parents, keypoint_children,
                  depth_map,
                  PAF_heatmaps=None, paf_parents=None):
    newJointExplanations = 0

    for person_index, skeleton in enumerate(skeletons):
        avg_distance = calculateAverageDistance(skeleton) * 2
        if avg_distance == 0:
            avg_distance = 0.35

        for joint_index, explained in enumerate(skeleton_explanation[person_index]):
            if explained:
                # ---------------- Parent ----------------
                parent_name = keypoint_parents[keypoint_names[joint_index]]
                if parent_name != keypoint_names[joint_index]:
                    parent_index = keypoint_names.index(parent_name)
                    if not skeleton_explanation[person_index][parent_index]:
                        paf_index = None
                        if paf_parents is not None and joint_index < len(paf_parents):
                            paf_index = paf_parents[joint_index]

                        new_joint = findClosestJoint(
                            keypoint_results[parent_index],
                            skeleton,
                            joint_index,
                            depth_map,
                            PAF_heatmaps=PAF_heatmaps,
                            paf_index=paf_index,
                            max_distance_threshold=avg_distance,
                            child_joint_index=parent_index,
                            keypoint_names=keypoint_names,
                            avg_body_scale=avg_distance / 2.0,
                        )
                        if new_joint:
                            skeleton[parent_index * 3:parent_index * 3 + 3] = new_joint
                            skeleton_explanation[person_index][parent_index] = True
                            newJointExplanations += 1
                            if new_joint in keypoint_results[parent_index]:
                                keypoint_results[parent_index].remove(new_joint)

                # ---------------- Children ----------------
                for child_name in keypoint_children[keypoint_names[joint_index]]:
                    child_index = keypoint_names.index(child_name)
                    if not skeleton_explanation[person_index][child_index]:
                        paf_index = None
                        if paf_parents is not None and child_index < len(paf_parents):
                            paf_index = paf_parents[child_index]

                        new_joint = findClosestJoint(
                            keypoint_results[child_index],
                            skeleton,
                            joint_index,
                            depth_map,
                            PAF_heatmaps=PAF_heatmaps,
                            paf_index=paf_index,
                            max_distance_threshold=avg_distance,
                            child_joint_index=child_index,
                            keypoint_names=keypoint_names,
                            avg_body_scale=avg_distance / 2.0,
                        )
                        if new_joint:
                            skeleton[child_index * 3:child_index * 3 + 3] = new_joint
                            skeleton_explanation[person_index][child_index] = True
                            newJointExplanations += 1
                            if new_joint in keypoint_results[child_index]:
                                keypoint_results[child_index].remove(new_joint)

    return (newJointExplanations,
            skeletons, skeleton_explanation,
            keypoint_results, keypoint_depths,
            keypoint_names, keypoint_parents, keypoint_children)

def is_skeleton_anatomically_valid(skeleton, keypoint_names, keypoint_parents,
                                   vert_tolerance=0.10,
                                   max_limb_ratio=3.5,
                                   max_pair_dist=None):
    """
    Returns True when the skeleton passes basic anatomical plausibility checks.

    Checks performed (only on joints that are actually filled):

    1. Vertical ordering
       Face joints should sit above shoulders, shoulders above hips,
       hips above knees, knees above ankles.  vert_tolerance (default 10 %
       of image height) allows for perspective tilt and partial occlusion.

    2. Limb-length outlier
       Any individual bone that is more than max_limb_ratio times the
       average established bone length in this skeleton is flagged as an
       implausible cross-person connection.

    3. Symmetric-pair proximity
       Joints that should be on opposite sides of the same body (eyes,
       shoulders, hips, knees, ankles) must not be further apart than the
       per-pair limit supplied in max_pair_dist.
    """

    # Default per-pair max distances (normalised [0..1])
    if max_pair_dist is None:
        max_pair_dist = {
            ('left_eye',      'right_eye'):      0.20,
            ('left_shoulder', 'right_shoulder'): 0.55,
            ('left_hip',      'right_hip'):      0.55,
            ('left_knee',     'right_knee'):     0.70,
            ('left_ankle',    'right_ankle'):    0.80,
        }

    def joint_pos(name):
        """Return (x, y) if the joint is filled, else None."""
        if name not in keypoint_names:
            return None
        idx = keypoint_names.index(name)
        x, y, v = skeleton[idx*3], skeleton[idx*3+1], skeleton[idx*3+2]
        return (x, y) if v > 0 else None

    def mean_y(names):
        ys = [joint_pos(n)[1] for n in names if joint_pos(n) is not None]
        return sum(ys) / len(ys) if ys else None

    # ------------------------------------------------------------------
    # Check 1: Vertical ordering (y increases downward in image coords)
    # ------------------------------------------------------------------
    face_y     = mean_y(['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear'])
    shoulder_y = mean_y(['left_shoulder', 'right_shoulder'])
    hip_y      = mean_y(['left_hip', 'right_hip'])
    knee_y     = mean_y(['left_knee', 'right_knee'])
    ankle_y    = mean_y(['left_ankle', 'right_ankle'])

    ordered_pairs = [
        (face_y,     shoulder_y, 'face above shoulder'),
        (shoulder_y, hip_y,      'shoulder above hip'),
        (hip_y,      knee_y,     'hip above knee'),
        (knee_y,     ankle_y,    'knee above ankle'),
    ]
    for upper_y, lower_y, _ in ordered_pairs:
        if upper_y is not None and lower_y is not None:
            # upper body part must have SMALLER y (higher on screen)
            if upper_y > lower_y + vert_tolerance:
                return False

    # ------------------------------------------------------------------
    # Check 2: Limb-length outlier
    # ------------------------------------------------------------------
    bone_lengths = []
    bones = []
    for jID in range(len(keypoint_names)):
        name = keypoint_names[jID]
        parent_name = keypoint_parents.get(name, name)
        if parent_name == name:
            continue
        if parent_name not in keypoint_names:
            continue
        parentJID = keypoint_names.index(parent_name)
        x1, y1, v1 = skeleton[jID*3],       skeleton[jID*3+1],       skeleton[jID*3+2]
        x2, y2, v2 = skeleton[parentJID*3], skeleton[parentJID*3+1], skeleton[parentJID*3+2]
        if v1 > 0 and v2 > 0:
            bone_len = ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5
            bone_lengths.append(bone_len)
            bones.append((jID, parentJID, bone_len))

    if bone_lengths:
        avg_bone = sum(bone_lengths) / len(bone_lengths)
        if avg_bone > 0:
            for _, _, blen in bones:
                if blen > max_limb_ratio * avg_bone:
                    return False

    # ------------------------------------------------------------------
    # Check 3: Symmetric-pair proximity
    # ------------------------------------------------------------------
    for (left_name, right_name), max_d in max_pair_dist.items():
        lp = joint_pos(left_name)
        rp = joint_pos(right_name)
        if lp is not None and rp is not None:
            d = ((lp[0] - rp[0])**2 + (lp[1] - rp[1])**2) ** 0.5
            if d > max_d:
                return False

    return True


def merge_similar_skeletons(skeletons, keypoint_names, pos_tolerance=0.05):
    """
    If two skeletons are almost identical in most joints, merge them.
    """
    merged = []
    used = [False] * len(skeletons)

    for i in range(len(skeletons)):
        if used[i]:
            continue
        base_skel = skeletons[i]
        for j in range(i+1, len(skeletons)):
            if used[j]:
                continue
            other = skeletons[j]
            same = 0
            total = 0
            for k in range(len(keypoint_names)):
                x1, y1, v1 = base_skel[3*k:3*k+3]
                x2, y2, v2 = other[3*k:3*k+3]
                if v1 != 0 and v2 != 0:
                    d = ((x1-x2)**2 + (y1-y2)**2)**0.5
                    total += 1
                    if d < pos_tolerance:
                        same += 1
            if total > 0 and same / total > 0.75:  # 75% of overlapping joints same
                # merge by taking max score
                new_skel = base_skel[:]
                for k in range(len(keypoint_names)):
                    x1, y1, v1 = new_skel[3*k:3*k+3]
                    x2, y2, v2 = other[3*k:3*k+3]
                    if v2 > v1:
                        new_skel[3*k:3*k+3] = [x2, y2, v2]
                base_skel = new_skel
                used[j] = True
        merged.append(base_skel)
        used[i] = True

    return merged




def resolveJointHierarchy(keypoint_results, keypoint_depths,
                          keypoint_names, keypoint_parents, keypoint_children,
                          depth_map,
                          PAF_heatmaps=None, 
                          paf_parents=None,
                          person_label_map=None,
                          normalized_confidence=True,
                          verbose=False):
    skeletons = []

    if verbose:
        print("\n\n\n")
        print("keypoint_results = ", keypoint_results)
        print("keypoint_names = ",  keypoint_names)
        print("keypoint_parents = ", keypoint_parents)
        print("keypoint_children = ", keypoint_children)

    # No keypoints → no skeletons
    if not keypoint_results:
        return skeletons

    skeleton_explanation = []

    # ------------------------------------------------------------------
    # Determine map size (all maps are 256x256 but we keep it generic)
    # ------------------------------------------------------------------
    H = W = None
    if depth_map is not None:
        H, W = depth_map.shape[:2]
    elif isinstance(PAF_heatmaps, np.ndarray):
        H, W = PAF_heatmaps.shape[:2]
    elif person_label_map is not None:
        H, W = person_label_map.shape[:2]

    # ------------------------------------------------------------------
    # 1) Decide how many skeletons
    #    Prefer segmentation; fall back to old "most popular joint" logic
    # ------------------------------------------------------------------
    numberOfSkeletons = 0
    mostPopularJoint = 0
    use_segmentation = False
    person_ids = []

    if person_label_map is not None:
        # labels: 0 = background, 1..N = persons
        person_ids = np.unique(person_label_map)
        person_ids = [pid for pid in person_ids if pid != 0]
        if len(person_ids) > 0:
            numberOfSkeletons = len(person_ids)
            use_segmentation = True
            if verbose:
                print("Segmentation suggests", numberOfSkeletons,
                      "persons (labels:", person_ids, ")")

    if not use_segmentation:
        # Prefer root joints (self-parented) as seed — they sit at the centre of
        # the hierarchy so growSkeletons can reach all children from them.
        # Fall back to "most popular joint" only when no root has any peaks.
        root_joints = [jID for jID, name in enumerate(keypoint_names)
                       if jID < len(keypoint_results)
                       and keypoint_parents.get(name, '') == name]

        # Count peaks per joint
        joint_peak_counts = [
            sum(1 for x, y, v in keypoint_results[jID]
                if ((x != 0) or (y != 0)) and v != 0)
            if jID < len(keypoint_results) else 0
            for jID in range(len(keypoint_names))
        ]

        # Best root joint = root with most peaks (> 0)
        best_root = -1
        best_root_count = 0
        for jID in root_joints:
            if joint_peak_counts[jID] > best_root_count:
                best_root_count = joint_peak_counts[jID]
                best_root = jID

        if best_root >= 0:
            mostPopularJoint = best_root
            numberOfSkeletons = best_root_count
            if verbose:
                print("Seeding from root joint:",
                      keypoint_names[mostPopularJoint],
                      "with", numberOfSkeletons, "peaks")
        else:
            # Fallback: old heuristic based on "most popular joint"
            for jID, count in enumerate(joint_peak_counts):
                if count > numberOfSkeletons:
                    numberOfSkeletons = count
                    mostPopularJoint = jID
            if verbose:
                print("No root joint peaks — seeding from most popular joint:",
                      keypoint_names[mostPopularJoint])

        if numberOfSkeletons == 0:
            numberOfSkeletons = 1  # at least one slot

        for _ in range(numberOfSkeletons):
            skeleton_explanation.append([False] * len(keypoint_names))
            skeletons.append([0.0] * (len(keypoint_names) * 3))

        if verbose:
            print("Assuming ", len(skeleton_explanation), " skeletons ")
            print("Most popular joint is ", keypoint_names[mostPopularJoint],
                  " ", mostPopularJoint)
            print("Keypoints have ", len(keypoint_results), " elements")
            print("Most popular joint has ",
                  len(keypoint_results[mostPopularJoint]), " elements")

        # Populate seed joint for each skeleton
        for person in range(numberOfSkeletons):
            if len(keypoint_results[mostPopularJoint]) > 0:
                skeleton_explanation[person][mostPopularJoint] = True
                # if fewer candidates than persons, reuse last one
                idx = min(person, len(keypoint_results[mostPopularJoint]) - 1)
                x, y, v = keypoint_results[mostPopularJoint][idx]
                base = mostPopularJoint * 3
                skeletons[person][base + 0] = x
                skeletons[person][base + 1] = y
                if (normalized_confidence):
                    v = v / 256.0
                skeletons[person][base + 2] = v

    else:
        # ------------------------------------------------------------------
        # Segmentation-guided seeding:
        #   - one skeleton per person_id
        #   - per-joint: pick best candidate inside that person blob
        # ------------------------------------------------------------------
        numberOfSkeletons = len(person_ids)
        for _ in range(numberOfSkeletons):
            skeleton_explanation.append([False] * len(keypoint_names))
            skeletons.append([0.0] * (len(keypoint_names) * 3))

        for person_idx, person_id in enumerate(person_ids):
            for jID in range(len(keypoint_names)):
                # Safety: keypoint_results might be longer than keypoint_names
                if jID >= len(keypoint_results):
                    continue

                candidates = keypoint_results[jID]
                if not candidates:
                    continue

                best = assign_best_candidate_for_person(
                    candidates,
                    person_label_map,
                    person_id,
                    W, H
                )
                if best is not None:
                    x, y, v = best
                    base = jID * 3
                    skeletons[person_idx][base + 0] = x
                    skeletons[person_idx][base + 1] = y
                    if (normalized_confidence):
                        v = v / 256.0
                    skeletons[person_idx][base + 2] = v
                    skeleton_explanation[person_idx][jID] = True

        if verbose:
            print("Initialized", numberOfSkeletons,
                  "skeletons from segmentation.")

    # ------------------------------------------------------------------
    # 2) Refinement: grow skeletons using distance + depth + PAFs
    # ------------------------------------------------------------------
    newJointExplanations = 1
    while newJointExplanations != 0:
        (newJointExplanations,
         skeletons, skeleton_explanation,
         keypoint_results, keypoint_depths,
         keypoint_names, keypoint_parents, keypoint_children) = growSkeletons(
            skeletons, skeleton_explanation,
            keypoint_results, keypoint_depths,
            keypoint_names, keypoint_parents, keypoint_children,
            depth_map,
            PAF_heatmaps=PAF_heatmaps,
            paf_parents=paf_parents
        )

    return skeletons






def _nms_peaks(peaks, min_dist=0.10):
    """
    Non-maximum suppression on a list of (x, y, score) peaks (all in
    normalised [0..1] coordinates).  Keeps only peaks that are at least
    min_dist away from every higher-scoring peak already retained.
    Peaks are returned sorted by score descending.
    """
    if len(peaks) <= 1:
        return list(peaks)
    ordered = sorted(peaks, key=lambda p: p[2], reverse=True)
    kept = []
    for p in ordered:
        px, py = p[0], p[1]
        suppressed = False
        for k in kept:
            d = ((px - k[0]) ** 2 + (py - k[1]) ** 2) ** 0.5
            if d < min_dist:
                suppressed = True
                break
        if not suppressed:
            kept.append(p)
    return kept


def find_peak_point_coordinates_from_heatmaps(joint_prediction_heatmaps, threshold,
                                               nms_dist=0.10):
    """
    Extract one peak per blob in each joint heatmap channel.

    nms_dist : minimum normalised distance between two kept peaks of the
               same joint type.  Peaks closer than this to a higher-scoring
               peak are suppressed.  Set to 0 to disable NMS.
    """
    heatmaps = np.array(joint_prediction_heatmaps).astype(np.float32) + 120
    result = list()

    for i in range(heatmaps.shape[2]):
        result.append(list())

    # Iterate through each channel of the heatmap
    for i in range(heatmaps.shape[2]):
        if i < 17:
            channel_heatmap = heatmaps[:, :, i]

            # Threshold the heatmap to obtain binary image
            _, binary = cv2.threshold(channel_heatmap, threshold, 255, cv2.THRESH_BINARY)

            # Find contours in the binary image
            contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            raw_peaks = []
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    score = channel_heatmap[cy, cx]
                    # Centroid can land on a background pixel for thin/ring-shaped
                    # contours.  Fall back to the max-value pixel in the bounding rect.
                    if score <= threshold:
                        bx, by, bw, bh = cv2.boundingRect(contour)
                        roi = channel_heatmap[by:by+bh, bx:bx+bw]
                        local_yx = np.unravel_index(np.argmax(roi), roi.shape)
                        cy, cx = by + local_yx[0], bx + local_yx[1]
                        score = channel_heatmap[cy, cx]
                    if score > threshold:
                        raw_peaks.append((cx / heatmaps.shape[1],
                                          cy / heatmaps.shape[0],
                                          score))

            # Suppress nearby weaker peaks
            result[i] = _nms_peaks(raw_peaks, min_dist=nms_dist) if nms_dist > 0 else raw_peaks

    return result



def grow_skeletons_with_pafs_and_depth(skeletons, PAF_heatmaps, depth_map, keypoint_results, paf_parents):
    # Ensure PAF_heatmaps is a Numpy array
    if not isinstance(PAF_heatmaps, np.ndarray):
        PAF_heatmaps = np.array(PAF_heatmaps)

    # Iterate through each skeleton
    for skeleton_index, skeleton in enumerate(skeletons):
        for joint_index in range(len(skeleton) // 3):
            # Ensure skeleton is initialized properly
            if not skeleton[joint_index * 3: joint_index * 3 + 3]:
                continue  # Skip if joint not initialized

            # Get the parent joint for the current joint
            paf_parent = paf_parents[joint_index]
            
            if paf_parent is None or paf_parent >= len(skeleton) // 3:
                continue

            # Extract joint and parent positions
            joint_data = skeleton[joint_index * 3: joint_index * 3 + 3]
            parent_joint_data = skeleton[paf_parent * 3: paf_parent * 3 + 3]

            if len(parent_joint_data) < 3:
                print(f"Skipping joint index {joint_index}, parent {paf_parent} due to incomplete data.")
                continue

            joint_x, joint_y, joint_visibility = joint_data
            parent_x, parent_y, _ = parent_joint_data

            # Ensure joint_x and joint_y are valid indices (integers)
            joint_x = int(joint_x)
            joint_y = int(joint_y)

            # Check if indices are within the heatmap bounds
            if 0 <= joint_x < PAF_heatmaps.shape[1] and 0 <= joint_y < PAF_heatmaps.shape[0]:
                # Extract the PAF vector (x, y) at the joint's location
                paf_vector_x = PAF_heatmaps[joint_y, joint_x, 0]  # x component of PAF
                paf_vector_y = PAF_heatmaps[joint_y, joint_x, 1]  # y component of PAF
            else:
                paf_vector_x = 0
                paf_vector_y = 0

            # Update the joint position by moving it along the PAF direction
            updated_joint_x = parent_x + paf_vector_x
            updated_joint_y = parent_y + paf_vector_y

            # Incorporate depth from the depth map
            if depth_map is not None and 0 <= updated_joint_x < depth_map.shape[1] and 0 <= updated_joint_y < depth_map.shape[0]:
                joint_depth = depth_map[int(updated_joint_y), int(updated_joint_x)]
            else:
                joint_depth = 1  # Default depth if no depth map is provided

            # Update the skeleton with the new joint position and depth
            skeleton[joint_index * 3: joint_index * 3 + 3] = [updated_joint_x, updated_joint_y, joint_depth]

    return skeletons




def retrieve_keypoint_depth(keypoint_predictions, depthmapIn):
    #print(" len(keypoint_predictions) ",len(keypoint_predictions))
    if not keypoint_predictions:
         return list()

    depthmap = np.array(depthmapIn)
    result = list()
 
    # Iterate through each keypoint
    for i,points in enumerate(keypoint_predictions):
       if points:
         #print(i," -> ",points) 
         x, y, val = points[0]  # Extract x, y coordinates
         xI = min(depthmap.shape[1]-1,max(0,int(x * depthmap.shape[1])))
         yI = min(depthmap.shape[0]-1,max(0,int(y * depthmap.shape[0]))) 
         result.append(depthmap[yI,xI])
       else: 
         result.append(0)
       
    return result




def dump_inputs(keypoint_heatmaps, PAF_heatmaps, depth_map, keypoint_names, keypoint_parents, keypoint_children, paf_parents, folder_path='input_dump'):
    import os

    print("==============================================================================")
    print("==============================================================================")
    print("==============================================================================")
    print("DUMPING inputs to resolveJointHierarchy, deactivate this when running for real")
    print("==============================================================================")
    print("==============================================================================")
    print("==============================================================================")

    # Create a folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save the keypoint heatmaps (assuming these are numpy arrays)
    np.save(os.path.join(folder_path, 'keypoint_heatmaps.npy'), keypoint_heatmaps)
    
    # Save the PAF heatmaps
    np.save(os.path.join(folder_path, 'PAF_heatmaps.npy'), PAF_heatmaps)

    # Save the depth map
    np.save(os.path.join(folder_path, 'depth_map.npy'), depth_map)

    # Save the keypoint names (list of strings)
    with open(os.path.join(folder_path, 'keypoint_names.json'), 'w') as f:
        json.dump(keypoint_names, f)

    # Save the keypoint parents and children (dictionaries)
    with open(os.path.join(folder_path, 'keypoint_parents.json'), 'w') as f:
        json.dump(keypoint_parents, f)

    with open(os.path.join(folder_path, 'keypoint_children.json'), 'w') as f:
        json.dump(keypoint_children, f)

    # Save the paf_parents (dictionary)
    with open(os.path.join(folder_path, 'paf_parents.json'), 'w') as f:
        json.dump(paf_parents, f)

    print(f"Inputs dumped to {folder_path}")


def load_and_run(folder_path='input_dump', verbose=True):
    # Load the keypoint heatmaps
    keypoint_heatmaps = np.load(os.path.join(folder_path, 'keypoint_heatmaps.npy'))

    # Load the PAF heatmaps
    PAF_heatmaps = np.load(os.path.join(folder_path, 'PAF_heatmaps.npy'))

    # Load the depth map
    depth_map = np.load(os.path.join(folder_path, 'depth_map.npy'))

    # Load the keypoint names (list of strings)
    with open(os.path.join(folder_path, 'keypoint_names.json'), 'r') as f:
        keypoint_names = json.load(f)

    # Load the keypoint parents and children (dictionaries)
    with open(os.path.join(folder_path, 'keypoint_parents.json'), 'r') as f:
        keypoint_parents = json.load(f)

    with open(os.path.join(folder_path, 'keypoint_children.json'), 'r') as f:
        keypoint_children = json.load(f)

    # Load the paf_parents (dictionary)
    with open(os.path.join(folder_path, 'paf_parents.json'), 'r') as f:
        paf_parents = json.load(f)

    # Run the function with loaded inputs
    skeletons = resolveJointHierarchyNew(keypoint_heatmaps, PAF_heatmaps, depth_map, keypoint_names, keypoint_parents, keypoint_children, paf_parents, verbose=verbose, dump=False) #Don't redump dumped data

    drawSkeletons(skeletons, keypoint_names, keypoint_parents, image_shape=(480, 640, 3))
    cv2.waitKey(0)
  
    return skeletons


def compute_paf_children(paf_parents, num_joints):
    paf_children = [-1] * num_joints
    for child_jID, paf_id in enumerate(paf_parents):
        if paf_id >= 0:
            paf_children[child_jID] = paf_id
    return paf_children


def compute_paf_children_local(paf_parents_local, keypoint_names, keypoint_children):
    """
    For each joint j, return the local PAF channel index for the limb from
    j's FIRST child (in the tree) that has a valid PAF back to j.
    Returns -1 if no such child exists.

    This gives the filter a second evidence source: even if the parent PAF
    for a joint is weak, we can keep it if the child PAF is active.
    """
    n = len(keypoint_names)
    result = [-1] * n
    for j, jname in enumerate(keypoint_names):
        for child_name in keypoint_children.get(jname, []):
            if child_name not in keypoint_names:
                continue
            child_idx = keypoint_names.index(child_name)
            if child_idx < len(paf_parents_local) and paf_parents_local[child_idx] >= 0:
                result[j] = paf_parents_local[child_idx]
                break  # use the first child that has a PAF
    return result


"""
This function receiveas as input an array of raw heatmaps and should extract a list of unique completed skeletons detected in the image.
All heatmaps/depthmaps etc should have the same dimensions and have values in the range from [-120.0 .. 120.0] 

keypoint_heatmaps[height][width][id] where id is [0..16] encodes 2D joint locations  
PAF_heatmaps[height][width][id] where id is [17..28] encodes 2D PAFs from each joint to its parent (paf_parents provides associations)    
depth_map[height][width] provides depth values for each pixel 
keypoint_names is the list of joint names for debug/printing purposes
keypoint_parents is the list of joint parents for debug/printing purposes
keypoint_parents is a list of lists of joint children for debug/printing purposes
paf_parents is a list of PAF parents for each heatmap 
"""
def resolveJointHierarchyNew(keypoint_heatmaps, PAF_heatmaps, depth_map, keypoint_names, keypoint_parents, keypoint_children, paf_parents, sanity_check=True, person_label_map=None,  verbose=False, dump=False, threshold=0.5, debug=False):

    if dump:
       dump_inputs(keypoint_heatmaps, PAF_heatmaps, depth_map, keypoint_names, keypoint_parents, keypoint_children, paf_parents, folder_path='input_dump')

    skeletons = []

    DBG = debug or verbose

    if DBG:
        print("\n========== resolveJointHierarchyNew DEBUG ==========")
        kp_shape  = keypoint_heatmaps.shape if hasattr(keypoint_heatmaps, 'shape') else f"list len={len(keypoint_heatmaps)}"
        paf_shape = (f"list len={len(PAF_heatmaps)}, each={PAF_heatmaps[0].shape if PAF_heatmaps else 'n/a'}"
                     if isinstance(PAF_heatmaps, list)
                     else (PAF_heatmaps.shape if PAF_heatmaps is not None else "None"))
        print(f"  keypoint_heatmaps : {kp_shape}  dtype={getattr(keypoint_heatmaps,'dtype','?')}")
        print(f"  PAF_heatmaps      : {paf_shape}")
        print(f"  depth_map         : {depth_map.shape if depth_map is not None else 'None'}")
        print(f"  person_label_map  : {person_label_map.shape if person_label_map is not None else 'None'}")
        print(f"  threshold         : {threshold}")
        print(f"  sanity_check      : {sanity_check}")
        kp_arr = keypoint_heatmaps if not isinstance(keypoint_heatmaps, list) else np.stack(keypoint_heatmaps, axis=2)
        for j in range(min(17, kp_arr.shape[2])):
            ch = kp_arr[:, :, j].astype(np.float32)
            print(f"    joint[{j:2d}] {keypoint_names[j] if j < len(keypoint_names) else '?':15s}  min={ch.min():.1f}  max={ch.max():.1f}  mean={ch.mean():.1f}")

    # Convert absolute heatmap PAF indices (17..28) to local channel indices (0..11).
    # Config paf_parents values < paf_start mean "no PAF for this joint" → -1.
    paf_start = 17
    paf_parents_local = [
        (p - paf_start) if (p is not None and p >= paf_start) else -1
        for p in paf_parents
    ] if paf_parents is not None else None

    # Step 1: Extract peak points from the keypoint heatmaps.
    # When sanity_check is on, first suppress peaks that have neither PAF support
    # nor lie inside a person region, then extract peaks from the filtered map.
    if sanity_check and PAF_heatmaps is not None:
        paf_heatmaps_np = (np.stack(PAF_heatmaps, axis=2)
                           if isinstance(PAF_heatmaps, list)
                           else PAF_heatmaps)
        if DBG:
            print(f"\n  PAF heatmaps stacked: {paf_heatmaps_np.shape}  dtype={paf_heatmaps_np.dtype}  min={paf_heatmaps_np.min():.1f}  max={paf_heatmaps_np.max():.1f}")
        paf_children_local = paf_parents_local.copy() if paf_parents_local is not None else None
        filtered_heatmaps = filter_joint_heatmaps_with_segmentation_and_pafs(
            keypoint_heatmaps[:, :, :17],
            paf_heatmaps_np,
            person_label_map,
            paf_parents_local,
            paf_children_local,
            paf_threshold=0.20)
        if DBG:
            for j in range(17):
                ch_raw  = keypoint_heatmaps[:, :, j].astype(np.float32)
                ch_filt = filtered_heatmaps[:, :, j].astype(np.float32)
                n_surviving = int(np.sum(ch_filt > -119))
                print(f"    filt joint[{j:2d}] {keypoint_names[j] if j < len(keypoint_names) else '?':15s}  raw_max={ch_raw.max():.1f}  filt_max={ch_filt.max():.1f}  surviving_px={n_surviving}")
        keypoint_results = find_peak_point_coordinates_from_heatmaps(filtered_heatmaps, threshold=threshold)
    else:
        keypoint_results = find_peak_point_coordinates_from_heatmaps(keypoint_heatmaps, threshold=threshold)

    if DBG:
        print(f"\n  Peak detection (threshold={threshold}):")
        total_peaks = 0
        for j, peaks in enumerate(keypoint_results):
            name = keypoint_names[j] if j < len(keypoint_names) else f'j{j}'
            scores = [f"{p[2]:.0f}" for p in peaks]
            print(f"    joint[{j:2d}] {name:15s}  peaks={len(peaks):2d}  scores=[{', '.join(scores[:6])}{'...' if len(scores)>6 else ''}]")
            total_peaks += len(peaks)
        print(f"  Total peaks across all joints: {total_peaks}")

    # Step 2: Initialize skeletons for the detected people based on keypoints
    keypoint_depths = retrieve_keypoint_depth(keypoint_results, depth_map)

    if DBG and person_label_map is not None:
        unique_ids = np.unique(person_label_map)
        person_ids = [p for p in unique_ids if p != 0]
        print(f"\n  Segmentation: person_ids={person_ids}  (unique labels={list(unique_ids[:10])})")

    # Pass paf_parents_local (0-based channel indices) so findClosestJoint can
    # actually index into the PAF_heatmaps list (which has 12 channels, 0..11).
    skeletons = resolveJointHierarchy(
        keypoint_results, keypoint_depths,
        keypoint_names, keypoint_parents, keypoint_children,
        depth_map,
        PAF_heatmaps=PAF_heatmaps,
        person_label_map=person_label_map,
        paf_parents=paf_parents_local,
        verbose=verbose
    )

    if DBG:
        print(f"\n  resolveJointHierarchy returned {len(skeletons)} skeleton(s)")
        for si, sk in enumerate(skeletons):
            filled = sum(1 for j in range(len(keypoint_names)) if sk[j*3+2] > 0)
            print(f"    skel[{si}]: {filled}/{len(keypoint_names)} joints filled")

    skeletons = merge_similar_skeletons(skeletons, keypoint_names)

    # Drop phantom skeletons that have too few filled joints to be meaningful.
    MIN_JOINTS = 3
    skeletons = [sk for sk in skeletons
                 if sum(1 for j in range(len(keypoint_names)) if sk[j*3+2] > 0) >= MIN_JOINTS]

    # Drop anatomically implausible skeletons (wrong vertical ordering,
    # wildly long bones, or symmetric joints impossibly far apart).
    skeletons = [sk for sk in skeletons
                 if is_skeleton_anatomically_valid(sk, keypoint_names, keypoint_parents)]

    if DBG:
        print(f"  After merge+validity filter (min {MIN_JOINTS} joints): {len(skeletons)} skeleton(s)")
        for si, sk in enumerate(skeletons):
            filled = sum(1 for j in range(len(keypoint_names)) if sk[j*3+2] > 0)
            joints_str = ', '.join(
                f"{keypoint_names[j]}=({sk[j*3]:.2f},{sk[j*3+1]:.2f})"
                for j in range(len(keypoint_names)) if sk[j*3+2] > 0
            )
            print(f"    skel[{si}]: {filled} joints — {joints_str}")
        print("========== END resolveJointHierarchyNew DEBUG ==========\n")

    return skeletons








_OPENPOSE_BONE_COLOR_BGR = {
    # Face — pink / magenta / purple family
    # Values taken from OpenPose COCO COLORS_RENDER (RGB) converted to BGR.
    'left_eye':       (170,   0, 255),  # RGB(255,  0,170) — deep pink
    'right_eye':      (255,   0, 255),  # RGB(255,  0,255) — magenta
    'left_ear':       (255,   0, 170),  # RGB(170,  0,255) — purple-pink
    'right_ear':      (255,   0,  85),  # RGB( 85,  0,255) — blue-purple
    # Left arm — warm spectrum: red → orange → yellow
    'left_shoulder':  (  0,   0, 255),  # RGB(255,  0,  0) — red
    'left_elbow':     (  0,  85, 255),  # RGB(255, 85,  0) — orange
    'left_wrist':     (  0, 170, 255),  # RGB(255,170,  0) — yellow-orange
    # Right arm — cool spectrum: yellow-green → green
    'right_shoulder': (  0, 255, 170),  # RGB(170,255,  0) — yellow-green
    'right_elbow':    (  0, 255,  85),  # RGB( 85,255,  0) — light green
    'right_wrist':    (  0, 255,   0),  # RGB(  0,255,  0) — green
    # Left leg — teal / cyan family
    'left_hip':       ( 85, 255,   0),  # RGB(  0,255, 85) — bright teal
    'left_knee':      (170, 255,   0),  # RGB(  0,255,170) — teal
    'left_ankle':     (200, 200,   0),  # RGB(  0,200,200) — dark teal
    # Right leg — blue family
    'right_hip':      (255, 255,   0),  # RGB(  0,255,255) — cyan
    'right_knee':     (255, 170,   0),  # RGB(  0,170,255) — sky blue
    'right_ankle':    (255,  85,   0),  # RGB(  0, 85,255) — blue
}

def drawSkeletons(skeletons, keypoint_names, keypoint_parents, image_shape=(480, 640, 3)):
    image = np.zeros(image_shape, dtype=np.uint8)

    for keypoints in skeletons:
        # First pass: draw limb lines (thick, behind joints)
        for jID in range(len(keypoint_names)):
            name       = keypoint_names[jID]
            x          = int(keypoints[jID*3+0] * image_shape[1])
            y          = int(keypoints[jID*3+1] * image_shape[0])
            visibility = keypoints[jID*3+2]
            if visibility <= 0:
                continue
            parent_name = keypoint_parents.get(name, name)
            if parent_name == name:
                continue  # root joint, no bone to draw
            parentJID         = keypoint_names.index(parent_name)
            parent_x          = int(keypoints[parentJID*3+0] * image_shape[1])
            parent_y          = int(keypoints[parentJID*3+1] * image_shape[0])
            parent_visibility = keypoints[parentJID*3+2]
            if parent_visibility <= 0:
                continue
            color = _OPENPOSE_BONE_COLOR_BGR.get(name, (200, 200, 200))
            cv2.line(image, (x, y), (parent_x, parent_y), color, 3, cv2.LINE_AA)

        # Second pass: draw joint dots (on top of lines)
        for jID in range(len(keypoint_names)):
            name       = keypoint_names[jID]
            x          = int(keypoints[jID*3+0] * image_shape[1])
            y          = int(keypoints[jID*3+1] * image_shape[0])
            visibility = keypoints[jID*3+2]
            if visibility <= 0:
                continue
            color = _OPENPOSE_BONE_COLOR_BGR.get(name, (200, 200, 200))
            cv2.circle(image, (x, y), 5, color,       -1, cv2.LINE_AA)
            cv2.circle(image, (x, y), 5, (255,255,255), 1, cv2.LINE_AA)

    cv2.imshow('Skeletons', image)
#---------------------------------------------------------------------------------------- 
#============================================================================================
#============================================================================================
# Main Function
#============================================================================================
#============================================================================================
if __name__ == '__main__':
# Specify the path to the trained 2D Pose Estimation model

  skeletons = load_and_run(folder_path='input_dump', verbose=True)
  sys.exit(0)


  keypoint_results =  [[(0.4192708333333333, 0.3020833333333333, 159.04083), (0.6171875, 0.2838541666666667, 141.2194), (0.24479166666666666, 0.2708333333333333, 48.50865), (0.6953125, 0.2786458333333333, 178.62257), (0.5442708333333334, 0.2760416666666667, 197.31694), (0.11979166666666667, 0.234375, 125.39257), (0.2708333333333333, 0.23177083333333334, 187.11096)], [(0.4244791666666667, 0.2994791666666667, 133.68227), (0.2421875, 0.2682291666666667, 43.22541), (0.6223958333333334, 0.2760416666666667, 166.05347), (0.7005208333333334, 0.265625, 148.4746), (0.5546875, 0.265625, 188.59534), (0.12760416666666666, 0.2265625, 142.28362), (0.28125, 0.22395833333333334, 179.24513)], [(0.4140625, 0.2994791666666667, 177.57248), (0.6145833333333334, 0.2786458333333333, 126.27818), (0.6848958333333334, 0.2682291666666667, 83.32413), (0.5364583333333334, 0.2682291666666667, 200.26768), (0.1171875, 0.2265625, 115.697136), (0.265625, 0.22395833333333334, 205.32957)], [(0.4296875, 0.3020833333333333, 47.09539), (0.8046875, 0.2838541666666667, 52.067062), (0.6328125, 0.2786458333333333, 136.86526), (0.7317708333333334, 0.2708333333333333, 203.60585), (0.5755208333333334, 0.2734375, 209.4301), (0.1484375, 0.2265625, 176.34573), (0.3020833333333333, 0.22395833333333334, 204.98811)], [(0.0859375, 0.3359375, 46.993103), (0.40625, 0.2994791666666667, 186.09299), (0.8177083333333334, 0.2786458333333333, 43.10502), (0.6067708333333334, 0.2786458333333333, 54.066452), (0.5286458333333334, 0.2708333333333333, 67.97044)], [(0.6015625, 0.328125, 152.63991), (0.4270833333333333, 0.3203125, 122.23184), (0.7916666666666666, 0.3125, 120.61464), (0.1796875, 0.28125, 165.14168), (0.3307291666666667, 0.2760416666666667, 186.39389)], [(0.5130208333333334, 0.328125, 165.70483), (0.6901041666666666, 0.3255208333333333, 165.68414), (0.3958333333333333, 0.3203125, 140.68881), (0.8307291666666666, 0.3098958333333333, 118.44491), (0.24479166666666666, 0.2890625, 72.42027), (0.10416666666666667, 0.2734375, 158.12773)], [(0.6119791666666666, 0.390625, 145.31255), (0.8072916666666666, 0.3802083333333333, 116.45909), (0.8619791666666666, 0.3567708333333333, 56.746452), (0.3541666666666667, 0.3619791666666667, 149.8564), (0.4375, 0.3489583333333333, 122.73482), (0.19010416666666666, 0.3489583333333333, 134.72284)], [(0.6328125, 0.3984375, 52.127586), (0.671875, 0.390625, 157.29565), (0.5, 0.3854166666666667, 135.04921), (0.3802083333333333, 0.3541666666666667, 156.30101), (0.8515625, 0.3515625, 115.70071), (0.22395833333333334, 0.3489583333333333, 100.81232), (0.0859375, 0.3385416666666667, 196.46118)], [(0.5989583333333334, 0.453125, 139.02786), (0.8463541666666666, 0.4296875, 75.02516), (0.3463541666666667, 0.4192708333333333, 153.04124), (0.8515625, 0.3828125, 40.18486), (0.19270833333333334, 0.4088541666666667, 122.81491), (0.4375, 0.3645833333333333, 100.87214), (0.2708333333333333, 0.2838541666666667, 43.518326)], [(0.8489583333333334, 0.4401041666666667, 63.063007), (0.6536458333333334, 0.4557291666666667, 172.68842), (0.21354166666666666, 0.40625, 69.52523), (0.8515625, 0.3802083333333333, 62.373905), (0.078125, 0.390625, 140.70024), (0.3984375, 0.3697916666666667, 68.664635), (0.2682291666666667, 0.2916666666666667, 51.803497)], [(0.59375, 0.4609375, 57.856064), (0.7604166666666666, 0.453125, 104.55321), (0.1484375, 0.4088541666666667, 42.906784), (0.3046875, 0.4140625, 82.46474), (0.4270833333333333, 0.3697916666666667, 108.73532), (0.96875, 0.359375, 64.48195)], [(0.7083333333333334, 0.453125, 72.64494), (0.5208333333333334, 0.4583333333333333, 81.163345), (0.2526041666666667, 0.4140625, 105.54861), (0.10416666666666667, 0.4088541666666667, 57.807915), (0.4036458333333333, 0.3723958333333333, 79.26636), (0.9609375, 0.3619791666666667, 62.384922)], [(0.7161458333333334, 0.5625, 124.48455), (0.5885416666666666, 0.5651041666666666, 112.03074), (0.13020833333333334, 0.5234375, 99.96415), (0.2994791666666667, 0.5208333333333334, 143.70576), (0.4270833333333333, 0.4140625, 133.75581), (0.9791666666666666, 0.3984375, 47.957527)], [(0.5963541666666666, 0.5703125, 50.468452), (0.7265625, 0.5625, 55.19754), (0.5104166666666666, 0.5625, 142.59222), (0.109375, 0.5260416666666666, 69.28831), (0.23958333333333334, 0.5104166666666666, 200.29053), (0.4114583333333333, 0.4114583333333333, 156.60355), (0.9661458333333334, 0.3958333333333333, 77.7554)], [(0.6848958333333334, 0.6770833333333334, 107.68108), (0.7760416666666666, 0.6692708333333334, 82.541016), (0.5963541666666666, 0.6510416666666666, 150.87589), (0.11979166666666667, 0.6171875, 84.609604), (0.2942708333333333, 0.6197916666666666, 122.23952), (0.3567708333333333, 0.4453125, 44.153313), (0.4166666666666667, 0.4635416666666667, 162.47462), (0.9713541666666666, 0.421875, 70.81378)], [(0.6822916666666666, 0.6796875, 128.42076), (0.7760416666666666, 0.6588541666666666, 115.65214), (0.5963541666666666, 0.6484375, 101.36485), (0.4921875, 0.6614583333333334, 193.99911), (0.11979166666666667, 0.6171875, 117.777725), (0.2734375, 0.6119791666666666, 188.1004), (0.4036458333333333, 0.4635416666666667, 195.92084), (0.9661458333333334, 0.421875, 100.40271)], [], []]
  keypoint_names =  ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'bkg']
  keypoint_parents =  {'nose': 'nose', 'left_eye': 'nose', 'right_eye': 'nose', 'left_ear': 'left_eye', 'right_ear': 'right_eye', 'left_shoulder': 'nose', 'right_shoulder': 'nose', 'left_elbow': 'left_shoulder', 'right_elbow': 'right_shoulder', 'left_wrist': 'left_elbow', 'right_wrist': 'right_elbow', 'left_hip': 'nose', 'right_hip': 'nose', 'left_knee': 'left_hip', 'right_knee': 'right_hip', 'left_ankle': 'left_knee', 'right_ankle': 'right_knee', 'bkg': 'bkg'}
  keypoint_children =  {'nose': ['left_eye', 'right_eye', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'], 'left_eye': ['left_ear'], 'right_eye': ['right_ear'], 'left_ear': [], 'right_ear': [], 'left_shoulder': ['left_elbow'], 'right_shoulder': ['right_elbow'], 'left_elbow': ['left_wrist'], 'right_elbow': ['right_wrist'], 'left_wrist': [], 'right_wrist': [], 'left_hip': ['left_knee'], 'right_hip': ['right_knee'], 'left_knee': ['left_ankle'], 'right_knee': ['right_ankle'], 'left_ankle': [], 'right_ankle': []}

  skeletons = resolveJointHierarchy(keypoint_results, None, keypoint_names, keypoint_parents, keypoint_children, None)
  print("skeletons = ",skeletons)
  drawSkeletons(skeletons, keypoint_names, keypoint_parents, image_shape=(480, 640, 3))
  cv2.waitKey(0)

