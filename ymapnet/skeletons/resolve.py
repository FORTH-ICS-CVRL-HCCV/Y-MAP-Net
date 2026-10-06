"""
Author : "Ammar Qammaz"
Copyright : "2024 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"

Bottom-up multi-person skeleton resolution (OpenPose-style part association,
adapted to Y-MAPNet's single-channel PAF ramps, see limbs.py).

    1. joint candidates : NMS peaks per smoothed joint heatmap          (peaks.py)
    2. limb scoring     : every candidate pair of every limb is scored
                          against the expected PAF ramp of the model's
                          encoding ('heatmapPAFEncoding')               (limbs.py)
    3. limb matching    : greedy bipartite matching per limb: hub limbs
                          (parent = nose) first, then the other limbs, then
                          the left<->right shoulder/hip bridges.  The matching
                          priority adds the two joints' heatmap confidence
                          and a bonus when the parent peak already belongs
                          to a person, so weak spurious peaks cannot steal
                          a real person's joint; a pair the assembly refuses
                          does not use up its two peaks.
    4. person assembly  : a connection extends the person owning either end,
                          or joins two partial persons that claim disjoint
                          joint types; upper/lower body fragments that no PAF
                          joins are linked by torso geometry (_linkTorsos)
    5. face joints      : eyes/ears have no PAF; they attach to the nearest
                          compatible nose/eye (or the neck when there is none)
                          within a gate scaled by the person's own size
    6. filtering        : persons with fewer than min_joints joints, or a low
                          summed joint confidence, are dropped

Defaults were tuned on COCO val2017 with benchmarkSkeletons.py (model 288,
1000 images): keypoint AP 0.015 (utils.resolveJointHierarchy) -> 0.20 (PAFs only)
-> 0.24 (+ person_lr_bridge and the left/right joint aggregates) -> 0.243 (+ depth)
-> 0.246 (peak blur sigma 2.5, face gate 1.3) -> 0.252 (+ Hand/Foot/Face masks), against
0.264 for an oracle that groups the same peaks using the ground truth.  Bridges matched
last: +0.005 (300 images).  On perfect DataLoader ground truth (cocoVal.pzpd, child_lit,
hips -> shoulders) the resolver reaches 0.827 with seed_leftovers=True, min_joints=1,
min_person_score=0, against 0.938 for the oracle.  (val2017 is also in the training set:
knowledge/PLAN.md Observation 30.)

Output matches resolveJointHierarchyNew: a list of skeletons, each a flat
[x, y, v] * numberOfJoints list with x, y normalised to [0..1] and v the joint's
heatmap confidence in (0..1] (0 for a missing joint).
"""
import cv2
import numpy as np

from ymapnet.skeletons.peaks import findPeaks
from ymapnet.skeletons.limbs import buildLimbs, prepareChannel, scoreLimbCandidates

# Face joints attached geometrically, in dependency order: (joint, anchors to try in order).
# 'neck' = midpoint of the person's shoulders: the only anchor of a head seen from behind.
_FACE_ATTACH = (
    ('left_eye', ('nose', 'neck')),
    ('right_eye', ('nose', 'neck')),
    ('left_ear', ('left_eye', 'nose', 'neck')),
    ('right_ear', ('right_eye', 'nose', 'neck')),
)

# Bones (of either fragment) that give a person's scale when linking torsos
_SCALE_BONES = (('left_shoulder', 'right_shoulder'), ('left_hip', 'right_hip'), ('left_shoulder', 'left_elbow'),
                ('right_shoulder', 'right_elbow'), ('left_elbow', 'left_wrist'), ('right_elbow', 'right_wrist'),
                ('left_hip', 'left_knee'), ('right_hip', 'right_knee'), ('left_knee', 'left_ankle'),
                ('right_knee', 'right_ankle'))


class _Assembly:

    def __init__(self, numberOfJoints, peaks):
        self.J = numberOfJoints
        self.persons = []  # each: np.array(J) of peak indices (-1 = missing)
        self.owner = [np.full(len(peaks[j]), -1, np.int32) for j in range(numberOfJoints)]

    def new(self):
        self.persons.append(np.full(self.J, -1, np.int32))
        return len(self.persons) - 1

    def set(self, person, joint, peak):
        self.persons[person][joint] = peak
        self.owner[joint][peak] = person

    def connect(self, ja, ia, jb, ib):
        """Join peak ia of joint ja with peak ib of joint jb; False when that contradicts the
        persons they already belong to (the slot is taken, or the two persons overlap)."""
        pa, pb = self.owner[ja][ia], self.owner[jb][ib]
        if pa < 0 and pb < 0:
            p = self.new()
            self.set(p, ja, ia)
            self.set(p, jb, ib)
        elif pa >= 0 and pb < 0:
            if self.persons[pa][jb] >= 0:
                return False
            self.set(pa, jb, ib)
        elif pa < 0 and pb >= 0:
            if self.persons[pb][ja] >= 0:
                return False
            self.set(pb, ja, ia)
        elif pa != pb:
            return self.merge(pa, pb)
        return True

    def merge(self, pa, pb):
        """Move person pb's joints into pa, unless both own the same joint type (two different people)."""
        A, B = self.persons[pa], self.persons[pb]
        if np.any((A >= 0) & (B >= 0)):
            return False
        for j in np.nonzero(B >= 0)[0]:
            self.set(pa, j, B[j])
        B[:] = -1
        return True


def _matchLimb(priority, min_priority, accept):
    """Greedy maximum-priority bipartite matching.  accept(i, k) applies a pair and returns False
    when the assembly refuses it; a refused pair does not use up its two peaks (otherwise a
    collinear limb of a neighbour that cannot be joined would still steal a joint's only
    partner)."""
    if priority.size == 0:
        return
    ii, kk = np.nonzero(priority > min_priority)
    order = np.argsort(-priority[ii, kk], kind='stable')
    usedA, usedB = set(), set()
    for o in order:
        i, k = int(ii[o]), int(kk[o])
        if i in usedA or k in usedB:
            continue
        if accept(i, k):
            usedA.add(i)
            usedB.add(k)


def _weightBySide(peaks, keypoint_names, rightMap, leftMap):
    """Scale each left_*/right_* peak's confidence by how well its location agrees with
    the left/right joint aggregate channels: agree = (own side - other side)/240 in
    [-1, 1], factor 0.5 + 0.5*agree.  This does not change the grouping (tested) but
    makes person scores track skeleton quality: +0.04 COCO AP, fewer junk skeletons."""
    H, W = rightMap.shape
    out = []
    for j, name in enumerate(keypoint_names):
        p = peaks[j]
        side = int(name.startswith('left_')) - int(name.startswith('right_'))
        if side == 0 or len(p) == 0:
            out.append(p)
            continue
        xi = np.clip(np.round(p[:, 0]).astype(int), 0, W - 1)
        yi = np.clip(np.round(p[:, 1]).astype(int), 0, H - 1)
        agree = side * (leftMap[yi, xi] - rightMap[yi, xi]) / 240.0
        q = p.copy()
        q[:, 2] *= np.clip(0.5 + 0.5 * agree, 0.0, 1.0)
        out.append(q)
    return out


# segmentation mask -> joints whose peaks should sit on it (ears lie on the face edge: no gain)
_PART_JOINTS = {'Hand': ('left_wrist', 'right_wrist'), 'Foot': ('left_ankle', 'right_ankle'),
                'Face': ('nose', 'left_eye', 'right_eye')}


def _weightByPartMasks(peaks, idx, masks, floor=0.3, radius=4):
    """Scale joint confidences by the strongest mask value within radius px of the peak:
    factor = floor + (1-floor)*(v+120)/240.  Separates real from spurious peaks (AUC 0.70
    wrists/Hand, 0.86 ankles/Foot, 0.83-0.90 nose+eyes/Face); like _weightBySide it does not
    change the grouping but sharpens person scores: +0.006 COCO AP, fewer junk skeletons."""
    out = list(peaks)
    k = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    for part, joints in _PART_JOINTS.items():
        if part not in masks:
            continue
        m = cv2.dilate(np.ascontiguousarray(masks[part], np.float32), k)
        H, W = m.shape
        for n in joints:
            j = idx.get(n)
            if j is None or len(out[j]) == 0:
                continue
            xi = np.clip(np.round(out[j][:, 0]).astype(int), 0, W - 1)
            yi = np.clip(np.round(out[j][:, 1]).astype(int), 0, H - 1)
            q = out[j].copy()
            q[:, 2] *= floor + (1 - floor) * np.clip((m[yi, xi] + 120.0) / 240.0, 0, 1)
            out[j] = q
    return out


def _depthJumps(depth, childPeaks, parentPeaks, samples=12):
    """NxM largest |depth step| (in units of the 240 range) between consecutive samples
    along each candidate segment."""
    H, W = depth.shape
    a, b = childPeaks[:, None, :2], parentPeaks[None, :, :2]
    t = np.linspace(0.0, 1.0, samples, dtype=np.float32)[None, None, :, None]
    pts = a[:, :, None, :] + t * (b - a)[:, :, None, :]
    xi = np.clip(np.round(pts[..., 0]).astype(int), 0, W - 1)
    yi = np.clip(np.round(pts[..., 1]).astype(int), 0, H - 1)
    return np.abs(np.diff(depth[yi, xi], axis=2)).max(axis=2) / 240.0


def _scaleBones(idx):
    return [(idx[a], idx[b]) for a, b in _SCALE_BONES if a in idx and b in idx]


def _longestBone(person, peaks, bones):
    """Longest present bone of a person (0 if none): a scale that survives foreshortening."""
    return max([float(np.linalg.norm(peaks[a][person[a], :2] - peaks[b][person[b], :2])) for a, b in bones
                if person[a] >= 0 and person[b] >= 0] + [0.0])


def _linkTorsos(asm, peaks, idx, max_ratio, depth=None, depth_weight=0.0):
    """Join upper-body fragments (shoulders, no hips) with lower-body fragments (hips, no shoulders).

    In the layout of serials <= 288 every torso limb runs shoulder/hip -> nose, so when a person's
    nose is not annotated or not detected (back views: ~30% of the COCO val persons) no PAF ties the
    two halves together.  (Models trained with DataLoader PAF_HIPS_TO_SHOULDERS have hip -> shoulder
    PAFs and rarely leave such fragments.)
    A pair is accepted when the torso length (shoulder centre to hip centre) is at most max_ratio
    times the longest bone of the two fragments and, if both have both sides, their left->right
    directions agree; pairs are joined greedily, shortest relative torso first.  (COCO train2017:
    torso / longest bone <= 2.11 for 99% of persons.)"""
    S = [idx[n] for n in ('left_shoulder', 'right_shoulder') if n in idx]
    Hp = [idx[n] for n in ('left_hip', 'right_hip') if n in idx]
    if not S or not Hp:
        return
    bones = _scaleBones(idx)

    def xy(person, j):
        return peaks[j][person[j], :2]

    uppers = [p for p, P in enumerate(asm.persons) if any(P[j] >= 0 for j in S) and all(P[j] < 0 for j in Hp)]
    lowers = [p for p, P in enumerate(asm.persons) if any(P[j] >= 0 for j in Hp) and all(P[j] < 0 for j in S)]
    cands = []
    for u in uppers:
        U = asm.persons[u]
        su = np.mean([xy(U, j) for j in S if U[j] >= 0], axis=0)
        for l in lowers:
            L = asm.persons[l]
            if np.any((U >= 0) & (L >= 0)):
                continue
            ref = max(_longestBone(U, peaks, bones), _longestBone(L, peaks, bones))
            if ref <= 0:
                continue
            hl = np.mean([xy(L, j) for j in Hp if L[j] >= 0], axis=0)
            ratio = float(np.linalg.norm(su - hl)) / ref
            if ratio > max_ratio:
                continue
            if len(S) == 2 and len(Hp) == 2 and all(U[j] >= 0 for j in S) and all(L[j] >= 0 for j in Hp):
                vs, vh = xy(U, S[1]) - xy(U, S[0]), xy(L, Hp[1]) - xy(L, Hp[0])
                # left->right directions agree (cos >= 0.71 for 99% of COCO train2017 persons), but
                # only measurable when neither pair is foreshortened (side views)
                if min(np.linalg.norm(vs), np.linalg.norm(vh)) > 0.3 * ref and np.dot(vs, vh) <= 0:
                    continue
            cost = ratio
            if depth is not None:
                cost += depth_weight * _depthJumps(depth, su[None, :], hl[None, :])[0, 0]
            cands.append((cost, u, l))
    cands.sort()
    used = set()
    for cost, u, l in cands:
        if u in used or l in used:
            continue
        if asm.merge(u, l):
            used.update((u, l))


def _referenceLength(person, peaks, idx, fallback):
    """Person-relative length used to gate the PAF-less face joints."""
    ls, rs, nose = idx.get('left_shoulder'), idx.get('right_shoulder'), idx.get('nose')

    def xy(j):
        return peaks[j][person[j], :2] if (j is not None and person[j] >= 0) else None

    L, R, N = xy(ls), xy(rs), xy(nose)
    if L is not None and R is not None:
        return float(np.linalg.norm(L - R))
    if N is not None and (L is not None or R is not None):
        return float(np.linalg.norm(N - (L if L is not None else R)))
    return fallback


def resolveSkeletons(keypoint_heatmaps, PAF_heatmaps, depth_map, keypoint_names, keypoint_parents, keypoint_children,
                     paf_parents, threshold=50.0, bridge_heatmap=None, min_limb_score=0.15, peak_weight=1.5,
                     owned_bonus=0.5, max_limb_length=0.6, face_gate=1.3, min_joints=3, min_person_score=0.1,
                     side_heatmaps=None, depth_heatmap=None, depth_weight=0.5, part_heatmaps=None,
                     paf_encoding='signed', bridge_encoding='signed', skip_limbs=False, torso_ratio=2.0, min_coverage=0.0, neck_gate=1.0, seed_leftovers=False,
                     **ignored):
    """
    Drop-in replacement for ymapnet.utils.resolveJointHierarchy.resolveJointHierarchyNew
    (same positional arguments; depth_map / keypoint_children / sanity_check /
    person_label_map / verbose / debug are accepted and ignored).

    keypoint_heatmaps : HxWxJ float in [-120..120]
    PAF_heatmaps      : HxWxL array or list of L HxW arrays in [-120..120]
    threshold         : joint peak threshold on the 0..240 scale used by the
                        callers' keypoint_threshold (raw value > threshold-120)
    bridge_heatmap    : optional HxW person_lr_bridge channel ([-120..120]); when
                        given it adds left<->right shoulder and hip limbs
    min_limb_score    : minimum PAF agreement (0..1) for a limb to be accepted
    min_coverage      : minimum fraction of a limb's expected-lit samples that must be lit
                        (see limbs.scoreLimbCandidates)
    peak_weight       : weight of the mean joint confidence in the matching priority
    owned_bonus       : priority bonus for attaching to a parent that already has a person
    max_limb_length   : longest limb considered, as a fraction of max(H, W)
    face_gate         : eyes/ears must lie within face_gate * reference length
                        (shoulder width / neck length) of their anchor
    min_joints        : persons with fewer joints are dropped
    min_person_score  : persons whose mean joint confidence * joints / J is lower are
                        dropped (junk assembled from stray low peaks)
    side_heatmaps     : optional (rightjoints, leftjoints) HxW aggregate channels
                        ([-120..120]); left/right joint confidences are weighted by
                        side agreement
    part_heatmaps     : optional {'Hand': HxW, 'Foot': HxW, 'Face': HxW} segmentation channels
                        ([-120..120]); wrist / ankle / nose+eye confidences are weighted by
                        the mask strength around the peak
    depth_heatmap     : optional HxW depthmap channel ([-120..120]); a limb's matching
                        priority drops by depth_weight * its largest depth jump, since a
                        link between two people usually crosses a depth discontinuity
    torso_ratio       : upper and lower body fragments (no PAF joins them when the nose is
                        missing) are linked when the shoulder-to-hip distance is at most
                        torso_ratio * the longest bone of the two fragments (0 = off)
    neck_gate         : eyes/ears of a person without nose/eyes attach to the neck (shoulder
                        midpoint) within neck_gate * the person's longest bone
    skip_limbs        : also match a joint to its grandparent on its own PAF channel when the
                        person has no parent joint (DataLoader PAF_SKIP_MISSING_PARENT)
    bridge_encoding   : how person_lr_bridge was drawn: 'signed' (default) or 'child_lit'
                        (DataLoader BRIDGE_ENCODING_LIT 1)
    paf_encoding      : how the model's PAFs were drawn, configuration.json
                        cfg.get('heatmapPAFEncoding', 'signed'): 'signed' (serials <= 288)
                        or 'child_lit' (see limbs.py)
    """
    kp = np.asarray(keypoint_heatmaps, dtype=np.float32)
    H, W = kp.shape[:2]
    J = len(keypoint_names)
    idx = {n: i for i, n in enumerate(keypoint_names)}

    peaks = [findPeaks(kp[:, :, j], threshold - 120.0) for j in range(J)]
    if side_heatmaps is not None:
        peaks = _weightBySide(peaks, keypoint_names, side_heatmaps[0], side_heatmaps[1])
    if part_heatmaps:
        peaks = _weightByPartMasks(peaks, idx, part_heatmaps)

    if isinstance(PAF_heatmaps, np.ndarray) and PAF_heatmaps.ndim == 3:
        pafChannels = [PAF_heatmaps[:, :, c] for c in range(PAF_heatmaps.shape[2])]
    else:
        pafChannels = list(PAF_heatmaps) if PAF_heatmaps is not None else []

    limbs = buildLimbs(keypoint_names, keypoint_parents, paf_parents, with_bridge=bridge_heatmap is not None,
                       paf_encoding=paf_encoding, bridge_encoding=bridge_encoding, skip_limbs=skip_limbs)
    sources = {'paf': [prepareChannel(c) for c in pafChannels]}
    if bridge_heatmap is not None:
        sources['bridge'] = [prepareChannel(bridge_heatmap)]

    # hub limbs (parent is a root) first so persons grow outward from the torso; the left<->right
    # bridges last, once each side's chain is built (a bridge is all that joins the two sides of a
    # person without a nose): +0.019 COCO AP on DataLoader ground truth, +0.005 on model 288
    roots = {i for i, n in enumerate(keypoint_names) if keypoint_parents.get(n, n) == n}
    limbs.sort(key=lambda l: 3 if l.source == 'bridge' else (2 if l.missing is not None else
                                                              (0 if l.parent in roots else 1)))

    maxLen = max_limb_length * max(H, W)
    asm = _Assembly(J, peaks)
    for limb in limbs:
        chans = sources[limb.source]
        if limb.channel >= len(chans):
            continue
        s = scoreLimbCandidates(chans[limb.channel], limb.template, peaks[limb.child], peaks[limb.parent], maxLen,
                                min_coverage=min_coverage)
        if s.size == 0:
            continue
        priority = s + peak_weight * 0.5 * (peaks[limb.child][:, 2][:, None] + peaks[limb.parent][:, 2][None, :])
        priority[:, asm.owner[limb.parent] >= 0] += owned_bonus
        if depth_heatmap is not None:
            priority -= depth_weight * _depthJumps(depth_heatmap, peaks[limb.child], peaks[limb.parent])
        priority[s <= min_limb_score] = 0.0
        if limb.missing is None:
            _matchLimb(priority, 0.0, lambda i, k: asm.connect(limb.child, i, limb.parent, k))
        else:  # skip limb: only for persons that really lack the bypassed joint
            def acceptSkip(i, k, limb=limb):
                for j, peak in ((limb.child, i), (limb.parent, k)):
                    o = asm.owner[j][peak]
                    if o >= 0 and asm.persons[o][limb.missing] >= 0:
                        return False
                return asm.connect(limb.child, i, limb.parent, k)
            _matchLimb(priority, 0.0, acceptSkip)

    if torso_ratio > 0:
        _linkTorsos(asm, peaks, idx, torso_ratio, depth_heatmap, depth_weight)

    # noses that no limb claimed still seed (face-only) persons
    if 'nose' in idx:
        n = idx['nose']
        for i in range(len(peaks[n])):
            if asm.owner[n][i] < 0:
                asm.set(asm.new(), n, i)

    # face joints: global greedy by distance, gated by each person's own scale
    fallbackRef = 0.08 * max(H, W)
    bones = _scaleBones(idx)
    for jointName, anchors in _FACE_ATTACH:
        if jointName not in idx:
            continue
        j = idx[jointName]
        cands = []
        for p, person in enumerate(asm.persons):
            if person[j] >= 0:
                continue
            ref = _referenceLength(person, peaks, idx, fallbackRef)
            for a in anchors:
                gate = face_gate * ref
                if a == 'neck':
                    sh = [peaks[idx[n]][person[idx[n]], :2] for n in ('left_shoulder', 'right_shoulder')
                          if n in idx and person[idx[n]] >= 0]
                    anchor = np.mean(sh, axis=0) if sh else None
                    # shoulder width collapses in side views: ear-to-neck is <= 4.4 shoulder widths but
                    # <= 1.0 longest bone for 99% of COCO train2017 persons
                    bone = _longestBone(person, peaks, bones)
                    if bone > 0:
                        gate = neck_gate * bone
                else:
                    anchor = peaks[idx[a]][person[idx[a]], :2] if (a in idx and person[idx[a]] >= 0) else None
                if anchor is not None:
                    d = np.linalg.norm(peaks[j][:, :2] - anchor, axis=1)
                    cands += [(float(d[i]), p, i) for i in np.nonzero(d <= gate)[0]]
                    break
        cands.sort()
        for d, p, i in cands:
            if asm.owner[j][i] < 0 and asm.persons[p][j] < 0:
                asm.set(p, j, i)

    # peaks that no person claimed (e.g. the 1-2 annotated joints of a mostly hidden person)
    if seed_leftovers:
        for j in range(J):
            for i in np.nonzero(asm.owner[j] < 0)[0]:
                asm.set(asm.new(), j, i)

    skeletons = []
    for person in asm.persons:
        present = np.nonzero(person >= 0)[0]
        if len(present) < min_joints:
            continue
        if sum(peaks[j][person[j], 2] for j in present) / J < min_person_score:
            continue
        sk = [0.0] * (J * 3)
        for j in present:
            x, y, v = peaks[j][person[j]]
            sk[j * 3:j * 3 + 3] = [float(x) / W, float(y) / H, float(v)]
        skeletons.append(sk)
    return skeletons
