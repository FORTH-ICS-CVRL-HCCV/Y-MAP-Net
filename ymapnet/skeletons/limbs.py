"""
Author : "Ammar Qammaz"
Copyright : "2024 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"

Limb topology and limb (joint-pair) scoring against Y-MAPNet's single-channel PAFs.

Y-MAPNet's "PAFs" are NOT OpenPose's 2-channel unit vector fields.  Each limb is a
single channel holding a *ramp* drawn along the bone on the -120 background, with
t = 0 at the child joint and t = 1 at the parent joint.  The ramp depends on how the
model's training data was built (datasets/DataLoader/configuration.h,
HeatmapGenerator.c : drawPAFsOnHeatmaps), recorded in configuration.json as
'heatmapPAFEncoding' (missing = 'signed'):

  'signed'    (PAF_ENCODING_CHILD_LIT 0, serials <= 288, drawSignedPAFLine)
              value(t) = -127 + 254 t    if child AND parent are right_* joints
              value(t) = +127 - 254 t    otherwise (FLIP_PAF_GRADIENTS_FOR_LEFT_JOINTS)
  'child_lit' (PAF_ENCODING_CHILD_LIT 1, drawChildLitPAFLine)
              value(t) = PAF_CHILD_VALUE + (PAF_PARENT_VALUE - PAF_CHILD_VALUE) t
                       = +127 -> +64 for every limb

Consequences for decoding:
  * |value| is useless as a limb-presence signal: the background has |v| = 120.
  * In the signed encoding the half of the ramp near -127 is indistinguishable from
    background, so the evidence lives in the lit (+) half.  Samples are weighted by
    how lit the ramp is expected to be there (for child_lit: almost uniformly).
  * The ramp's direction tells the two endpoints apart, so a candidate pair must
    match both the position and the orientation of the drawn line.

The person_lr_bridge instance channel uses the same encoding for the
left_shoulder->right_shoulder and left_hip->right_hip segments (-127 at the left
joint, +127 at the right one, no flip), so those are scored as extra limbs.
"""
import cv2
import numpy as np

BACKGROUND = -120.0

# Expected lit level q = (value - BACKGROUND) / 240 in [0..1] at the (child, parent) ends.
# The signed ramp's -127 end is mapped to q = 0 (it reads as background).
_RISING, _FALLING = (0.0, 1.0), (1.0, 0.0)
_CHILD_LIT = (1.0, (64.0 - BACKGROUND) / 240.0)  # configuration.h PAF_CHILD_VALUE 127, PAF_PARENT_VALUE 64
PAF_ENCODINGS = ('signed', 'child_lit')


class Limb:

    def __init__(self, child, parent, channel, template, source, missing=None):
        self.child = child  # joint index at t = 0
        self.parent = parent  # joint index at t = 1
        self.channel = channel  # channel index within its source stack
        self.template = template  # expected lit level (q at child, q at parent), see _RISING / _CHILD_LIT
        self.source = source  # 'paf' or 'bridge'
        self.missing = missing  # skip limb: the parent joint it bypasses (only valid when neither person has it)


def buildLimbs(keypoint_names, keypoint_parents, paf_parents, paf_start=17, with_bridge=False, paf_encoding='signed',
               bridge_encoding='signed', skip_limbs=False):
    """Derive the limb list from the model configuration.

    paf_parents[j] is the absolute heatmap channel of the PAF joining joint j to
    keypoint_parents[j]; values below paf_start mean "no PAF" (the face joints).
    paf_encoding is the model's configuration.json 'heatmapPAFEncoding'."""
    for enc in (paf_encoding, bridge_encoding):
        if enc not in PAF_ENCODINGS:
            raise ValueError('unknown PAF encoding %r (expected one of %s)' % (enc, PAF_ENCODINGS))
    limbs = []
    for j, name in enumerate(keypoint_names):
        if j >= len(paf_parents) or paf_parents[j] is None or paf_parents[j] < paf_start:
            continue
        parentName = keypoint_parents[name]
        if parentName == name or parentName not in keypoint_names:
            continue
        p = keypoint_names.index(parentName)
        if paf_encoding == 'child_lit':
            template = _CHILD_LIT
        else:
            template = _RISING if (name.startswith('right_') and parentName.startswith('right_')) else _FALLING
        limbs.append(Limb(j, p, paf_parents[j] - paf_start, template, 'paf'))
        # DataLoader PAF_SKIP_MISSING_PARENT: a joint whose parent is not annotated has its PAF drawn to the
        # grandparent in its own channel (wrist -> shoulder without an elbow, ...)
        g = keypoint_names.index(keypoint_parents[keypoint_names[p]])
        if skip_limbs and g != p:
            limbs.append(Limb(j, g, paf_parents[j] - paf_start, template, 'paf', missing=p))

    if with_bridge:  # person_lr_bridge: signed -127 at the left joint, or lit (DataLoader BRIDGE_ENCODING_LIT)
        template = _CHILD_LIT if bridge_encoding == 'child_lit' else _RISING
        for l, r in (('left_shoulder', 'right_shoulder'), ('left_hip', 'right_hip')):
            if l in keypoint_names and r in keypoint_names:
                limbs.append(Limb(keypoint_names.index(l), keypoint_names.index(r), 0, template, 'bridge'))
    return limbs


def prepareChannel(channel, dilate=3):
    """Widen the thin (2px) predicted lines so a sample 1px off the bone still
    reads the ramp.  Grey dilation only raises values, which is harmless because
    low values are treated as 'no evidence' by scoreLimbCandidates."""
    ch = np.ascontiguousarray(channel, dtype=np.float32)
    if dilate > 1:
        ch = cv2.dilate(ch, np.ones((dilate, dilate), np.uint8))
    return ch


def scoreLimbCandidates(channel, template, childPeaks, parentPeaks, max_length_px, samples=12, tolerance=0.7,
                        min_coverage=0.0, lit_level=0.25):
    """
    Score every (child peak, parent peak) pair for one limb.

    channel      : HxW prepared (see prepareChannel) PAF channel
    template     : expected lit level (q at child, q at parent), Limb.template
    childPeaks   : Nx3 (x, y, score) pixel coordinates
    parentPeaks  : Mx3
    returns      : NxM float32 score matrix in [0..1] (0 for pairs beyond max_length_px)

    Per sample at t along the segment the observed value is mapped to
    p = (v+120)/240 and compared with the expected q, linear from template[0] at the
    child to template[1] at the parent (signed: q = t or 1-t);
    agreement a = clip(1 - |p-q|/tolerance, 0, 1) is averaged with weight q, so
    the lit end of the ramp dominates.

    That average alone does not reject a segment over plain background: in the
    signed encoding the dark half "agrees" with background, so such segments score
    0.15-0.35 and pass any useful min_limb_score.  As in OpenPose (>80% of samples
    on the limb), a pair is therefore dropped (score 0) unless at least min_coverage
    of the samples expected to be lit (q >= 0.5) read p >= lit_level.
    """
    n, m = len(childPeaks), len(parentPeaks)
    scores = np.zeros((n, m), np.float32)
    if n == 0 or m == 0:
        return scores
    H, W = channel.shape

    a = childPeaks[:, None, :2]  # N,1,2
    b = parentPeaks[None, :, :2]  # 1,M,2
    length = np.linalg.norm(b - a, axis=2)  # N,M

    t = np.linspace(0.0, 1.0, samples, dtype=np.float32)
    pts = a[:, :, None, :] + t[None, None, :, None] * (b - a)[:, :, None, :]  # N,M,S,2
    xs = np.clip(pts[..., 0], 0, W - 1.001)
    ys = np.clip(pts[..., 1], 0, H - 1.001)
    x0, y0 = xs.astype(np.int32), ys.astype(np.int32)
    fx, fy = xs - x0, ys - y0
    obs = ((channel[y0, x0] * (1 - fx) + channel[y0, x0 + 1] * fx) * (1 - fy) +
           (channel[y0 + 1, x0] * (1 - fx) + channel[y0 + 1, x0 + 1] * fx) * fy)

    p = np.clip((obs - BACKGROUND) / 240.0, 0.0, 1.0)
    q = template[0] + (template[1] - template[0]) * t
    agree = np.clip(1.0 - np.abs(p - q) / tolerance, 0.0, 1.0)
    s = (agree * q).sum(axis=2) / q.sum()
    if min_coverage > 0:
        lit = q >= 0.5
        s[(p[..., lit] >= lit_level).mean(axis=2) < min_coverage] = 0.0

    s[length > max_length_px] = 0.0
    s[length < 1.0] = 0.0
    scores[:] = s
    return scores
