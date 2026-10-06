#!/usr/bin/python3
"""
Author : "Ammar Qammaz"
Copyright : "2024 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"

Synthetic end-to-end check of ymapnet.skeletons.resolveSkeletons.

Heatmaps are rendered exactly the way the C DataLoader encodes ground truth
(Gaussian joints on a -120 background, PAF ramps in either encoding: 'signed' with
the left/central limbs flipped, or 'child_lit'; signed person_lr_bridge lines) for
several people standing close enough that their limbs interleave, then the
resolver must return every person with every joint in the right place.

  python3 -m ymapnet.skeletons.testSkeletons
"""
import sys
import numpy as np

from ymapnet.skeletons import resolveSkeletons

NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow',
         'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
         'right_ankle']
PARENTS = {'nose': 'nose', 'left_eye': 'nose', 'right_eye': 'nose', 'left_ear': 'left_eye', 'right_ear': 'right_eye',
           'left_shoulder': 'nose', 'right_shoulder': 'nose', 'left_elbow': 'left_shoulder',
           'right_elbow': 'right_shoulder', 'left_wrist': 'left_elbow', 'right_wrist': 'right_elbow',
           'left_hip': 'nose', 'right_hip': 'nose', 'left_knee': 'left_hip', 'right_knee': 'right_hip',
           'left_ankle': 'left_knee', 'right_ankle': 'right_knee'}
CHILDREN = {n: [c for c, p in PARENTS.items() if p == n and c != n] for n in NAMES}
PAF_PARENTS = [0, 0, 0, 0, 0, 28, 19, 27, 18, 26, 17, 25, 22, 24, 21, 23, 20]  # ymapnet_model/configuration.json

# a standing person facing the camera (image left = person's right), unit = pixels at scale 1
POSE = {'nose': (0, -52), 'left_eye': (3, -55), 'right_eye': (-3, -55), 'left_ear': (7, -53), 'right_ear': (-7, -53),
        'left_shoulder': (12, -38), 'right_shoulder': (-12, -38), 'left_elbow': (18, -20),
        'right_elbow': (-18, -20), 'left_wrist': (20, -3), 'right_wrist': (-20, -3), 'left_hip': (8, 0),
        'right_hip': (-8, 0), 'left_knee': (9, 22), 'right_knee': (-9, 22), 'left_ankle': (10, 44),
        'right_ankle': (-10, 44)}


def ramp(img, a, b, va, vb, width=2):
    """drawSignedPAFLine / drawChildLitPAFLine: value va at a (child, t=0) to vb at b (parent, t=1)."""
    L = np.hypot(*(b - a))
    for t in np.linspace(0, 1, int(L * 2) + 2):
        x, y = np.round(a + t * (b - a)).astype(int)
        img[max(0, y - width // 2):y + width // 2 + 1, max(0, x - width // 2):x + width // 2 + 1] = va + (vb - va) * t


def render(people, encoding, size=256, sigma=2.5):
    yy, xx = np.mgrid[0:size, 0:size]
    kp = np.full((size, size, 17), -120.0, np.float32)
    paf = np.full((size, size, 12), -120.0, np.float32)
    bridge = np.full((size, size), -120.0, np.float32)
    for joints in people:
        for j, n in enumerate(NAMES):
            x, y = joints[n]
            kp[:, :, j] = np.maximum(kp[:, :, j], -120 + 240 * np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2)))
        for j, n in enumerate(NAMES):
            if PAF_PARENTS[j] >= 17:
                p = PARENTS[n]
                if encoding == 'child_lit':
                    va, vb = 127, 64
                else:
                    va, vb = (-127, 127) if (n.startswith('right_') and p.startswith('right_')) else (127, -127)
                ramp(paf[:, :, PAF_PARENTS[j] - 17], np.array(joints[n], float), np.array(joints[p], float), va, vb)
        for l, r in (('left_shoulder', 'right_shoulder'), ('left_hip', 'right_hip')):
            ramp(bridge, np.array(joints[l], float), np.array(joints[r], float), -127, 127)
    return kp, paf, bridge


def person(cx, cy, s=1.0):
    return {n: (cx + s * dx, cy + s * dy) for n, (dx, dy) in POSE.items()}


def check(people, use_bridge, encoding, tol=1.5):
    kp, paf, bridge = render(people, encoding)
    sks = resolveSkeletons(kp, paf, None, NAMES, PARENTS, CHILDREN, PAF_PARENTS,
                           bridge_heatmap=bridge if use_bridge else None, paf_encoding=encoding)
    if len(sks) != len(people):
        return 'expected %d skeletons, got %d' % (len(people), len(sks))
    for gt in people:
        best = min(sks, key=lambda sk: np.hypot(sk[0] * 256 - gt['nose'][0], sk[1] * 256 - gt['nose'][1]))
        for j, n in enumerate(NAMES):
            x, y, v = best[j * 3:j * 3 + 3]
            if v <= 0 or np.hypot(x * 256 - gt[n][0], y * 256 - gt[n][1]) > tol:
                return 'person at %s: joint %s wrong (%.3f,%.3f,%.2f)' % (gt['nose'], n, x, y, v)
    return None


if __name__ == '__main__':
    scenes = {
        'single person': [person(128, 128)],
        'two side by side, arms interleaving': [person(100, 130), person(138, 130)],
        'three at different scales': [person(60, 120, 0.8), person(128, 140, 1.3), person(200, 110, 0.9)],
        'overlap: one in front, one behind': [person(110, 150, 1.4), person(150, 110, 0.9)],
    }
    failed = 0
    for encoding in ('signed', 'child_lit'):
        for bridge in (False, True):
            for name, people in scenes.items():
                err = check(people, bridge, encoding)
                print('%-4s %-40s %-9s bridge=%-5s %s' % ('OK' if err is None else 'FAIL', name, encoding, bridge,
                                                         err or ''))
                failed += err is not None
    sys.exit(1 if failed else 0)
