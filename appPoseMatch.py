#!/usr/bin/python3

"""
Author : "Ammar Qammaz"
Copyright : "2024 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"

Pose-matching helpers for runYMAPNet.py.

Each call to pose_match_tick() compares the best live skeleton against a
reference pose and returns a PoseMatchResult with a per-joint breakdown and
an overall similarity score in [0, 1].

Usage from runYMAPNet.py main loop
-----------------------------------
    from appPoseMatch import PoseMatcher, draw_pose_match_overlay

    matcher = PoseMatcher(estimator)          # once, after estimator is ready
    matcher.capture_reference()               # press R in the loop, or call here

    result = matcher.tick()                   # every frame
    if result is not None:
        draw_pose_match_overlay(frame, result, estimator)
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Joints used for angle-based comparison (triplets: vertex, child-A, child-B)
# Each angle is vertex-centred: angle between the two limb vectors.
# ---------------------------------------------------------------------------
_ANGLE_TRIPLETS = [
    # name            vertex            arm-A              arm-B
    ("right_elbow",  "right_elbow",   "right_shoulder",  "right_wrist"),
    ("left_elbow",   "left_elbow",    "left_shoulder",   "left_wrist"),
    ("right_knee",   "right_knee",    "right_hip",       "right_ankle"),
    ("left_knee",    "left_knee",     "left_hip",        "left_ankle"),
    ("right_shoulder","right_shoulder","right_elbow",    "right_hip"),
    ("left_shoulder","left_shoulder", "left_elbow",      "left_hip"),
    ("right_hip",    "right_hip",     "right_knee",      "right_shoulder"),
    ("left_hip",     "left_hip",      "left_knee",       "left_shoulder"),
    ("trunk",        "right_shoulder","left_shoulder",   "right_hip"),
]

# Score thresholds (degrees).  Within PERFECT → full credit; beyond BAD → zero.
_PERFECT_DEG = 15.0
_BAD_DEG     = 60.0


# ---------------------------------------------------------------------------
class PoseMatchResult:
    """Returned by PoseMatcher.tick() every frame."""
    def __init__(self, score, joint_scores, ref_angles, live_angles, angle_names):
        self.score        = score          # float [0,1]
        self.joint_scores = joint_scores   # dict name→float [0,1]
        self.ref_angles   = ref_angles     # dict name→degrees
        self.live_angles  = live_angles    # dict name→degrees
        self.angle_names  = angle_names    # ordered list


# ---------------------------------------------------------------------------
def _get_joint_xy(skeleton, keypoint_names, name):
    """Return (x_norm, y_norm) for a named joint, or None if invisible."""
    if name not in keypoint_names:
        return None
    idx = keypoint_names.index(name)
    x, y, vis = skeleton[idx*3], skeleton[idx*3+1], skeleton[idx*3+2]
    if vis <= 0:
        return None
    return float(x), float(y)


def _angle_deg(v_xy, a_xy, b_xy):
    """Angle at vertex v between rays v→a and v→b, in degrees."""
    va = np.array([a_xy[0] - v_xy[0], a_xy[1] - v_xy[1]], dtype=np.float32)
    vb = np.array([b_xy[0] - v_xy[0], b_xy[1] - v_xy[1]], dtype=np.float32)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-6 or nb < 1e-6:
        return None
    cos_a = np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def _extract_angles(skeleton, keypoint_names):
    """Return dict of angle_name→degrees for all triplets that are visible."""
    angles = {}
    for name, v_name, a_name, b_name in _ANGLE_TRIPLETS:
        v = _get_joint_xy(skeleton, keypoint_names, v_name)
        a = _get_joint_xy(skeleton, keypoint_names, a_name)
        b = _get_joint_xy(skeleton, keypoint_names, b_name)
        if v is None or a is None or b is None:
            continue
        deg = _angle_deg(v, a, b)
        if deg is not None:
            angles[name] = deg
    return angles


def _score_angle_diff(diff_deg):
    """Map angle difference in degrees → score in [0, 1]."""
    if diff_deg <= _PERFECT_DEG:
        return 1.0
    if diff_deg >= _BAD_DEG:
        return 0.0
    return 1.0 - (diff_deg - _PERFECT_DEG) / (_BAD_DEG - _PERFECT_DEG)


def _best_skeleton(skeletons):
    """Pick the skeleton with the most visible joints."""
    if not skeletons:
        return None
    return max(skeletons, key=lambda s: sum(1 for i in range(len(s)//3) if s[i*3+2] > 0))


# ---------------------------------------------------------------------------
class PoseMatcher:
    """
    Stateful pose matcher.  Create once after the estimator is initialised.

    Parameters
    ----------
    estimator : YMAPNet instance
    hold_frames : int
        Number of consecutive frames the user must hold the matched pose
        before a "success" flash is triggered.
    """

    def __init__(self, estimator, hold_frames=30):
        self._estimator   = estimator
        self._hold_frames = hold_frames
        self._ref_angles  = None     # dict name→degrees for reference pose
        self._ref_skeleton_snapshot = None
        self._hold_count  = 0
        self._flash_count = 0        # frames remaining for success flash

    # -----------------------------------------------------------------------
    def capture_reference(self):
        """Capture the current best skeleton as the reference pose."""
        e = self._estimator
        skels = getattr(e, 'skeletons', None)
        sk = _best_skeleton(skels) if skels else None
        if sk is None:
            print("[PoseMatch] No skeleton visible — reference NOT updated.")
            return False
        self._ref_angles            = _extract_angles(sk, e.keypoint_names)
        self._ref_skeleton_snapshot = sk[:]   # shallow copy of the flat list
        self._hold_count            = 0
        self._flash_count           = 0
        n = len(self._ref_angles)
        print(f"[PoseMatch] Reference captured ({n} angles).")
        return True

    def has_reference(self):
        return self._ref_angles is not None and len(self._ref_angles) > 0

    # -----------------------------------------------------------------------
    def tick(self):
        """
        Compare current best skeleton to reference.
        Returns PoseMatchResult or None if no reference / no skeleton.
        """
        if not self.has_reference():
            return None

        e     = self._estimator
        skels = getattr(e, 'skeletons', None)
        sk    = _best_skeleton(skels) if skels else None
        if sk is None:
            return None

        live_angles  = _extract_angles(sk, e.keypoint_names)
        joint_scores = {}
        angle_names  = sorted(set(self._ref_angles) & set(live_angles))

        for name in angle_names:
            diff = abs(self._ref_angles[name] - live_angles[name])
            joint_scores[name] = _score_angle_diff(diff)

        if joint_scores:
            score = float(np.mean(list(joint_scores.values())))
        else:
            score = 0.0

        # Hold counter
        if score >= 0.80:
            self._hold_count += 1
        else:
            self._hold_count = 0

        if self._hold_count >= self._hold_frames:
            self._flash_count = 45   # ~1.5 s at 30 fps
            self._hold_count  = 0

        if self._flash_count > 0:
            self._flash_count -= 1

        return PoseMatchResult(score, joint_scores, self._ref_angles,
                               live_angles, angle_names)

    @property
    def success_flash(self):
        """True while the success animation is playing."""
        return self._flash_count > 0

    @property
    def ref_skeleton(self):
        return self._ref_skeleton_snapshot


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

_SCORE_COLORS = [
    (0,   0, 220),   # 0.0 — red
    (0, 165, 255),   # 0.5 — orange
    (0, 220,   0),   # 1.0 — green
]

def _score_to_bgr(score):
    """Interpolate red→orange→green for a score in [0,1]."""
    score = float(np.clip(score, 0.0, 1.0))
    if score < 0.5:
        t = score / 0.5
        c0, c1 = _SCORE_COLORS[0], _SCORE_COLORS[1]
    else:
        t = (score - 0.5) / 0.5
        c0, c1 = _SCORE_COLORS[1], _SCORE_COLORS[2]
    return tuple(int(c0[i] + t * (c1[i] - c0[i])) for i in range(3))


def _draw_skeleton_colored(image, skeleton, keypoint_names, keypoint_parents,
                           joint_scores=None, alpha=1.0):
    """
    Draw a skeleton onto *image* (in-place).
    If joint_scores is provided each limb/joint is tinted by its score colour.
    alpha controls blending when drawing coloured limbs.
    """
    H, W = image.shape[:2]

    def xy(name):
        if name not in keypoint_names:
            return None, None, 0
        idx = keypoint_names.index(name)
        x, y, vis = skeleton[idx*3], skeleton[idx*3+1], skeleton[idx*3+2]
        return int(x * W), int(y * H), vis

    # Limbs
    for jID, name in enumerate(keypoint_names):
        parent_name = keypoint_parents.get(name, name)
        if parent_name == name:
            continue
        x1, y1, v1 = xy(name)
        x2, y2, v2 = xy(parent_name)
        if v1 <= 0 or v2 <= 0:
            continue
        score = joint_scores.get(name, 0.5) if joint_scores else 0.5
        color = _score_to_bgr(score)
        cv2.line(image, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)

    # Joints
    for jID, name in enumerate(keypoint_names):
        x, y, vis = xy(name)
        if vis <= 0:
            continue
        score = joint_scores.get(name, 0.5) if joint_scores else 0.5
        color = _score_to_bgr(score)
        cv2.circle(image, (x, y), 6, color,       -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), 6, (255,255,255), 1, cv2.LINE_AA)


def _draw_reference_skeleton(image, skeleton, keypoint_names, keypoint_parents,
                              tint=(180, 180, 180)):
    """Draw the reference skeleton as a faint grey ghost in the corner."""
    if skeleton is None:
        return
    H, W = image.shape[:2]
    gh, gw = H // 4, W // 4
    ghost = np.zeros((H, W, 3), dtype=np.uint8)

    def xy(name):
        if name not in keypoint_names:
            return None, None, 0
        idx = keypoint_names.index(name)
        x, y, vis = skeleton[idx*3], skeleton[idx*3+1], skeleton[idx*3+2]
        # Scale into top-right corner box
        gx = int(W - gw + x * gw)
        gy = int(y * gh)
        return gx, gy, vis

    for jID, name in enumerate(keypoint_names):
        parent_name = keypoint_parents.get(name, name)
        if parent_name == name:
            continue
        x1, y1, v1 = xy(name)
        x2, y2, v2 = xy(parent_name)
        if v1 <= 0 or v2 <= 0:
            continue
        cv2.line(ghost, (x1,y1), (x2,y2), tint, 2, cv2.LINE_AA)

    for jID, name in enumerate(keypoint_names):
        x, y, vis = xy(name)
        if vis <= 0:
            continue
        cv2.circle(ghost, (x, y), 4, tint, -1, cv2.LINE_AA)

    mask = (ghost > 0).any(axis=2)
    image[mask] = cv2.addWeighted(image, 0.3, ghost, 0.7, 0)[mask]

    # Label the ghost
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "REF", (W - gw + 4, 18), font, 0.55, (0,0,0),   2, cv2.LINE_AA)
    cv2.putText(image, "REF", (W - gw + 4, 18), font, 0.55, tint,      1, cv2.LINE_AA)


def draw_pose_match_overlay(frame, result, estimator, matcher):
    """
    Draw the full pose-match HUD onto *frame* (in-place).

    - Colours each live skeleton limb/joint by match quality (red→green).
    - Shows a per-joint score panel on the left.
    - Shows the reference ghost skeleton in the top-right corner.
    - Flashes green on success.
    """
    H, W = frame.shape[:2]
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.40, W / 1800)
    thickness  = max(1, int(font_scale * 2))
    line_h     = int(font_scale * 30)
    pad        = 8

    # ── 1. Coloured live skeleton ─────────────────────────────────────────────
    skels = getattr(estimator, 'skeletons', None)
    sk    = _best_skeleton(skels) if skels else None
    if sk is not None:
        _draw_skeleton_colored(frame, sk,
                               estimator.keypoint_names,
                               estimator.cfg["keypoint_parents"],
                               joint_scores=result.joint_scores)

    # ── 2. Reference ghost ────────────────────────────────────────────────────
    _draw_reference_skeleton(frame, matcher.ref_skeleton,
                             estimator.keypoint_names,
                             estimator.cfg["keypoint_parents"])

    # ── 3. Left panel: per-joint bars ─────────────────────────────────────────
    panel_lines = [("POSE MATCH", None)] + \
                  [(name, result.joint_scores.get(name)) for name in result.angle_names]
    panel_w = int(W * 0.28)
    panel_h = len(panel_lines) * line_h + pad * 2 + line_h  # +1 for score bar

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, (label, score) in enumerate(panel_lines):
        y = pad + (i + 1) * line_h
        if score is None:
            # Header
            cv2.putText(frame, label, (pad, y), font, font_scale,
                        (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(frame, label, (pad, y), font, font_scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)
        else:
            color = _score_to_bgr(score)
            # Mini bar
            bar_x0 = pad
            bar_x1 = pad + int((panel_w - pad * 2) * score)
            bar_y  = y - line_h // 2
            cv2.rectangle(frame, (bar_x0, bar_y), (panel_w - pad, y - 2),
                          (50, 50, 50), -1)
            cv2.rectangle(frame, (bar_x0, bar_y), (bar_x1, y - 2), color, -1)
            # Label
            ref_deg  = result.ref_angles.get(label, 0.0)
            live_deg = result.live_angles.get(label, 0.0)
            text = f"{label:<16} {live_deg:5.1f}° / {ref_deg:5.1f}°"
            cv2.putText(frame, text, (pad, y), font, font_scale * 0.85,
                        (0, 0, 0), thickness + 1, cv2.LINE_AA)
            cv2.putText(frame, text, (pad, y), font, font_scale * 0.85,
                        color, thickness - 1 if thickness > 1 else 1, cv2.LINE_AA)

    # ── 4. Overall score bar at bottom of panel ───────────────────────────────
    score_y0 = panel_h - line_h
    score_y1 = panel_h - 2
    score_x1 = pad + int((panel_w - pad * 2) * result.score)
    overall_color = _score_to_bgr(result.score)
    cv2.rectangle(frame, (pad, score_y0), (panel_w - pad, score_y1),
                  (50, 50, 50), -1)
    cv2.rectangle(frame, (pad, score_y0), (score_x1, score_y1),
                  overall_color, -1)
    score_pct = f"SCORE  {result.score * 100:.0f}%"
    cv2.putText(frame, score_pct, (pad, score_y1 - 2), font, font_scale,
                (0,0,0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, score_pct, (pad, score_y1 - 2), font, font_scale,
                (255,255,255), thickness, cv2.LINE_AA)

    # ── 5. Success flash ──────────────────────────────────────────────────────
    if matcher.success_flash:
        flash = np.zeros_like(frame)
        flash[:] = (0, 220, 0)
        cv2.addWeighted(frame, 0.65, flash, 0.35, 0, frame)
        msg = "HOLD!"
        (tw, th), _ = cv2.getTextSize(msg, font, font_scale * 3, 4)
        cx, cy = (W - tw) // 2, (H + th) // 2
        cv2.putText(frame, msg, (cx, cy), font, font_scale * 3,
                    (0, 0, 0), 8, cv2.LINE_AA)
        cv2.putText(frame, msg, (cx, cy), font, font_scale * 3,
                    (255, 255, 255), 4, cv2.LINE_AA)

    # ── 6. Hint line ──────────────────────────────────────────────────────────
    hint = "R=capture ref   Q=quit"
    cv2.putText(frame, hint, (pad, H - pad), font, font_scale * 0.85,
                (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, hint, (pad, H - pad), font, font_scale * 0.85,
                (200, 200, 200), thickness - 1 if thickness > 1 else 1, cv2.LINE_AA)
