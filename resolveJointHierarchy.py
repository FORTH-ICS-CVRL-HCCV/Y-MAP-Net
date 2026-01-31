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

    # 1. Segmentation mask -------------------------------------
    if seg_required and person_label_map is not None:
        seg_mask = (person_label_map > 0)[..., None]   # HxWx1
        filtered *= seg_mask   # zero all joints outside people

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
    filtered *= paf_mask

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


def findClosestJoint(keypoint_candidates, skeleton, joint_index,
                     depth_map,
                     PAF_heatmaps=None, paf_index=None,
                     max_distance_threshold=0.35,
                     paf_weight=50.0):
    """
    Choose one candidate joint for a given anchor joint in a skeleton.

    - Distance is in normalized [0..1] coordinate space.
    - Depth penalty is based on depth variation along the segment.
    - PAF reward is based on average |PAF| along the segment for the
      appropriate limb channel (paf_index).
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

    for candidate in keypoint_candidates:
        candidate_x, candidate_y, _ = candidate

        # Euclidean distance in normalized coords
        distance = ((candidate_x - joint_x) ** 2 +
                    (candidate_y - joint_y) ** 2) ** 0.5

        # Skip far-away candidates
        if distance > max_distance_threshold:
            continue

        # ----------------- depth penalty (along the line) -----------------
        depth_penalty = 0.0
        if depth_map is not None and W > 1 and H > 1:
            steps = max(3, int(distance * max(W, H)))
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
                        depth_penalty += abs(d - prev_depth)
                    prev_depth = d

        # ----------------- PAF “goodness” along the limb -----------------
        paf_score = 0.0
        if PAF_heatmaps is not None and paf_index is not None and paf_index >= 0:
            paf_score = sample_paf_along_line(
                PAF_heatmaps, paf_index,
                joint_x, joint_y,
                candidate_x, candidate_y
            )
            # paf_score is in [0..1], higher is better

        # Final cost: lower is better
        #  - distance and depth_penalty increase cost
        #  - strong PAF decreases cost
        cost = distance + depth_penalty - paf_weight * paf_score

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
                        # PAF channel for joint -> parent
                        paf_index = None
                        if paf_parents is not None and joint_index < len(paf_parents):
                            paf_index = paf_parents[joint_index]

                        new_joint = findClosestJoint(
                            keypoint_results[parent_index],
                            skeleton,
                            joint_index,       # anchor = current joint
                            depth_map,
                            PAF_heatmaps=PAF_heatmaps,
                            paf_index=paf_index,
                            max_distance_threshold=avg_distance
                        )
                        if new_joint:
                            skeleton[parent_index * 3:parent_index * 3 + 3] = new_joint
                            skeleton_explanation[person_index][parent_index] = True
                            newJointExplanations += 1

                # ---------------- Children ----------------
                for child_name in keypoint_children[keypoint_names[joint_index]]:
                    child_index = keypoint_names.index(child_name)
                    if not skeleton_explanation[person_index][child_index]:
                        # PAF channel for child -> parent (our joint)
                        paf_index = None
                        if paf_parents is not None and child_index < len(paf_parents):
                            paf_index = paf_parents[child_index]

                        new_joint = findClosestJoint(
                            keypoint_results[child_index],
                            skeleton,
                            joint_index,       # anchor = current joint
                            depth_map,
                            PAF_heatmaps=PAF_heatmaps,
                            paf_index=paf_index,
                            max_distance_threshold=avg_distance
                        )
                        if new_joint:
                            skeleton[child_index * 3:child_index * 3 + 3] = new_joint
                            skeleton_explanation[person_index][child_index] = True
                            newJointExplanations += 1

    return (newJointExplanations,
            skeletons, skeleton_explanation,
            keypoint_results, keypoint_depths,
            keypoint_names, keypoint_parents, keypoint_children)

def merge_similar_skeletons(skeletons, keypoint_names, pos_tolerance=0.02):
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
        # Fallback: old heuristic based on "most popular joint"
        for jID, joints in enumerate(keypoint_results):
            thisJointCount = 0
            for x, y, v in joints:
                if ((x != 0) or (y != 0)) and (v != 0):
                    thisJointCount += 1

            if thisJointCount > numberOfSkeletons:
                numberOfSkeletons = thisJointCount
                mostPopularJoint = jID

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

        # Populate most popular joint for each skeleton
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






def find_peak_point_coordinates_from_heatmaps(joint_prediction_heatmaps, threshold):
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

            # Extract peak points from contours
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    result[i].append((cx / heatmaps.shape[0], cy / heatmaps.shape[1], channel_heatmap[cy, cx]))

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
            # find parent joint ID by PAF index
            # but simplest is: if parent has this paf, child gets it reversed
            paf_children[child_jID] = paf_id
    return paf_children


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
def resolveJointHierarchyNew(keypoint_heatmaps, PAF_heatmaps, depth_map, keypoint_names, keypoint_parents, keypoint_children, paf_parents, sanity_check=True, person_label_map=None,  verbose=False, dump=False, threshold=0.5):

    if dump:
       dump_inputs(keypoint_heatmaps, PAF_heatmaps, depth_map, keypoint_names, keypoint_parents, keypoint_children, paf_parents, folder_path='input_dump')

    skeletons = [] 
 
    if verbose:
        print("Resolving joint hierarchy with keypoint and PAF heatmaps...")

    if (sanity_check):
      paf_children = compute_paf_children(paf_parents, len(keypoint_names))
      paf_start = 17   # first PAF heatmap index
      paf_parents_local = [
    -1, -1, -1, -1, -1,   # 0–4
    11,  # 5 left_shoulder
    2,   # 6 right_shoulder
    10,  # 7 left_elbow
    1,   # 8 right_elbow
    9,   # 9 left_wrist
    0,   # 10 right_wrist
    8,   # 11 left_hip
    5,   # 12 right_hip
    7,   # 13 left_knee
    4,   # 14 right_knee
    6,   # 15 left_ankle
    3    # 16 right_ankle
]

      paf_children_local = paf_parents_local.copy()
      paf_heatmaps_np = np.stack(PAF_heatmaps, axis=2)


      keypoint_results = filter_joint_heatmaps_with_segmentation_and_pafs(
                                        keypoint_heatmaps[:, :, :17],   # slice channels
                                      paf_heatmaps_np,          # 17..28 heatmaps
                                      person_label_map,      # your segmentation HxW
                                      paf_parents_local,           # mapping jID -> paf
                                      paf_children_local,          # NEW (we generate below)
                                      paf_threshold=0.20) 

    # Step 1: Extract peak points from the keypoint heatmaps
    keypoint_results = find_peak_point_coordinates_from_heatmaps(keypoint_heatmaps, threshold=threshold)
    
    if verbose:
        print(f"Found {len(keypoint_results)} keypoints.")
        print(keypoint_results)

    # Step 2: Initialize skeletons for the detected people based on keypoints
    # This logic assumes each detected keypoint type could correspond to a different person/skeleton.
    keypoint_depths = retrieve_keypoint_depth(keypoint_results, depth_map) #<- Important for this to happen before normalization (castNormalizedCoordinatesToOriginalImage)!

    if verbose:
        print(f"Found {len(keypoint_depths)} keypoint depths.")
        print(keypoint_depths)

    #skeletons = resolveJointHierarchy(keypoint_results, keypoint_depths, keypoint_names, keypoint_parents, keypoint_children,  depth_map, verbose=verbose)
    skeletons = resolveJointHierarchy(
    keypoint_results, keypoint_depths,
    keypoint_names, keypoint_parents, keypoint_children,
    depth_map,
    PAF_heatmaps=PAF_heatmaps,
    person_label_map=person_label_map,
    paf_parents=paf_parents,
    verbose=verbose
)

    if verbose:
        print(f"Returning {len(skeletons)} skeletons first step.")
        print(skeletons)

    # Step 4: Perform growing using PAFs and depth maps
    #skeletons = grow_skeletons_with_pafs_and_depth(skeletons, PAF_heatmaps, depth_map, keypoint_results, paf_parents)
    
    skeletons = merge_similar_skeletons(skeletons, keypoint_names)

    if verbose:
        print(f"Returning {len(skeletons)} skeletons pafs and depth.")
        print(skeletons)

    return skeletons








def drawSkeletons(skeletons, keypoint_names, keypoint_parents, image_shape=(480, 640, 3)):
    # Create a blank image
    image = np.zeros(image_shape, dtype=np.uint8)
    
    for skeletonID,keypoints in enumerate(skeletons):
        # Create a dictionary for easier access to keypoints
        for jID in range(len(keypoint_names)):
               name       = keypoint_names[jID]
               x          = int(keypoints[jID*3+0]*image_shape[1])
               y          = int(keypoints[jID*3+1]*image_shape[0])
               visibility = keypoints[jID*3+2]
               if visibility > 0:  # Only draw visible keypoints and connections
                cv2.circle(image, (int(x), int(y)), 2, (255, 255, 0), 4)
                #cv2.putText(image, "%s - %u "%(keypoint_names[jID],skeletonID), (x,y) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                parent_name = keypoint_parents[name]
                if parent_name != name:  # Avoid drawing line to itself
                    parentJID         = keypoint_names.index(parent_name)
                    parent_x          = keypoints[parentJID*3+0] * image_shape[1]
                    parent_y          = keypoints[parentJID*3+1] * image_shape[0]
                    parent_visibility = keypoints[parentJID*3+2]
                    if parent_visibility > 0:
                        # Draw line between keypoint and its parent
                        cv2.line(image, (int(x), int(y)), (int(parent_x), int(parent_y)), (255, 0, 0), 2)
                        cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
                        cv2.circle(image, (int(parent_x), int(parent_y)), 3, (0, 255, 0), -1)
    
    # Display the image
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

