"""Fullscreen neon spatial-perception view (runYMAPNet.sh --showoff).

Imported from RGBTrack/python/tesla_view.py. The rendering is the same (tinted normal shading,
depth-edge wireframe, glowing category fills/contours/labels, joint peaks, legend, scanlines,
vignette, depth/normals/RGB insets); the difference is where the scene comes from: instead of
running its own TF / GGUF engine it reads the outputs of the YMAPNet estimator that runYMAPNet
already ran on the frame (keypoints_predictions + heatmap_16b), so there is one forward per frame.

YMAPNet sees the centre square of the frame, so the perception layers are square; they fill the
display height, centred, and the HUD / insets use the side margins.

Keys (in addition to runYMAPNet's): h toggle the HUD, f toggle fullscreen.
"""
import os
import re
import threading
import time
from collections import deque
from types import SimpleNamespace

import cv2
import numpy as np

from ymapnet.utils.tools import run_tool

WINDOW = "SPATIAL PERCEPTION"
# BGR neon palette, by YMAPNet category name
PALETTE = {
    "Person": (255, 235, 0),      # electric cyan
    "Face": (0, 200, 255),        # amber
    "Hand": (255, 0, 255),        # magenta
    "Foot": (0, 255, 128),        # lime
    "Tool": (0, 128, 255),        # orange
    "Wall": (255, 110, 0),        # electric blue
    "Floor": (150, 255, 0),       # aquamarine
    "Ceiling": (255, 0, 160),     # violet
    "Door": (0, 255, 255),        # yellow
    "Window": (255, 210, 0),      # azure
    "Vehicle": (0, 64, 255),      # red
    "Animal": (0, 255, 64),       # green
    "Appliance": (190, 0, 255),   # pink
    "Furniture": (255, 0, 96),    # purple
    "Light": (255, 255, 255),
}
# The semantic categories the HUD shows, in argmax order (same list as yd_scene in the C library)
SCENE_CATEGORIES = ["Person", "Face", "Hand", "Foot", "Tool", "Wall", "Floor", "Ceiling", "Door",
                    "Window", "Vehicle", "Animal", "Appliance", "Furniture", "Light"]
CYAN = (255, 220, 0)
DIM = (150, 120, 40)
JOINT_COLOR = (0, 220, 255)      # amber joint points
JOINT_NODE = (255, 255, 255)
JOINT_NAMES = ["NOSE", "L_EYE", "R_EYE", "L_EAR", "R_EAR", "L_SHOULDER", "R_SHOULDER", "L_ELBOW",
               "R_ELBOW", "L_WRIST", "R_WRIST", "L_HIP", "R_HIP", "L_KNEE", "R_KNEE", "L_ANKLE", "R_ANKLE"]

# tesla_view.py's command-line defaults
DEFAULT_OPTIONS = dict(render_width=800, camera_alpha=0.30, normal_alpha=0.75, edge_alpha=0.45,
                       category_alpha=0.16, category_threshold=0.20, min_category_pixels=700,
                       show_hud=True, pip=True, pip_scale=0.16, joints=True, joint_threshold=0.15,
                       fullscreen=True)


def text(image, value, xy, color=CYAN, scale=0.5, thick=1, shadow=True):
    offset = max(1, int(round(scale * 1.2)))
    if shadow:
        cv2.putText(image, value, (xy[0] + offset, xy[1] + offset), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                    thick + offset, cv2.LINE_AA)
    cv2.putText(image, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def normal_layer(normals, depth, camera, args):
    """Dark neon normal shading over a heavily dimmed camera image."""
    n = normals.astype(np.float32)
    valid = np.linalg.norm(n, axis=2) > 0.5
    nmap = np.clip(0.5 + 0.5 * n, 0, 1)
    nmap = np.clip((nmap - 0.5) * 1.45 + 0.5, 0, 1)
    hsv = cv2.cvtColor((nmap * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    # compress the normal hue into the cyan..magenta range and darken the surface; the HUD's
    # brightness comes from the rim, edges and categories, not from a pastel normal map
    hsv[..., 0] = ((80 + hsv[..., 0].astype(np.float32) * 0.55) % 180).astype(np.uint8)
    hsv[..., 1] = 170
    near = np.clip((depth + 1.0) * 0.5, 0, 1)          # +1 = near
    rim = np.clip(1.0 - np.abs(n[..., 2]), 0, 1) ** 3
    hsv[..., 2] = np.clip(28 + 70 * near + 45 * rim, 0, 255).astype(np.uint8)
    surface = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    surface[~valid] = 0
    rim[~valid] = 0
    out = cv2.addWeighted(camera, args.camera_alpha, surface, args.normal_alpha, 0)
    glow = (rim[..., None] * np.array(CYAN, np.float32) * 0.30).astype(np.uint8)
    return cv2.add(out, glow)


def edge_layer(depth, out, args):
    """Crisp cyan wireframe on depth discontinuities, with a soft bloom."""
    d = depth.copy()
    d[d < -1.5] = 0
    d8 = np.clip((d + 1.0) * 127.5, 0, 255).astype(np.uint8)
    # the relative depth is upsampled from a small model grid; smooth the blocks before Canny
    d8 = cv2.medianBlur(d8, 5)
    d8 = cv2.GaussianBlur(d8, (7, 7), 2.0)
    edges = cv2.Canny(d8, 10, 30)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    border = max(2, out.shape[0] // 90)
    edges[:border] = 0
    edges[-border:] = 0
    glow = cv2.GaussianBlur(edges, (0, 0), 2.4)
    edge_rgb = (edges[..., None].astype(np.float32) * np.array(CYAN, np.float32) * 0.30 +
                glow[..., None].astype(np.float32) * np.array(CYAN, np.float32) * 0.50) * args.edge_alpha
    return cv2.add(out, edge_rgb.astype(np.uint8))


def category_layer(out, category, confidence, names, args):
    """Translucent fills, contour glow and label positions for the categories present in the frame."""
    idx = np.round(category).astype(np.int32)
    glow = np.zeros_like(out)
    used, labels = [], []
    kernel = np.ones((5, 5), np.uint8)
    for i, name in enumerate(names):
        mask = (idx == i) & (confidence > args.category_threshold)
        if int(mask.sum()) < args.min_category_pixels:
            continue
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        area = int(mask.sum())
        if area < args.min_category_pixels:
            continue
        color = np.asarray(PALETTE.get(name, (200, 200, 200)), np.float32)
        bgr = tuple(int(v) for v in color)
        # fill
        sel = mask > 0
        out[sel] = (out[sel].astype(np.float32) * (1 - args.category_alpha) + color * args.category_alpha).astype(np.uint8)
        # contour + glow
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, bgr, 1, cv2.LINE_AA)
        cv2.drawContours(glow, contours, -1, bgr, 3, cv2.LINE_AA)
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 900:
            bx, by, bw, bh = cv2.boundingRect(largest)
            labels.append((name, bx, by, bw, bh, bgr))
        used.append((name, area, bgr, float(confidence[mask > 0].mean()), area / float(category.size)))
    used.sort(key=lambda item: item[1], reverse=True)
    labels.sort(key=lambda item: item[3] * item[4], reverse=True)
    if used:
        glow = cv2.GaussianBlur(glow, (0, 0), 3.0)
        out = cv2.add(out, glow)
    return out, used, labels


def draw_category_tags(out, labels, sx, sy, ox=0, oy=0):
    """Crisp category tags at display resolution (the masks themselves are rendered smaller)."""
    u = max(1.0, out.shape[1] / 1280.0)
    scale = 0.38 * u
    thick = max(1, int(round(u)))
    pad = max(2, int(round(4 * u)))
    for name, bx, by, bw, bh, color in labels[:5]:
        x, y = ox + int(bx * sx), oy + int(by * sy)
        label = name.upper()
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.rectangle(out, (x + pad, y + pad), (x + 2 * pad + tw, y + th + 3 * pad), color, -1)
        text(out, label, (x + 2 * pad, y + th + 2 * pad), (0, 0, 0), scale, thick, shadow=False)
    return out


def depth_thumb(depth, size):
    valid = depth > -1.5
    t = np.zeros_like(depth, dtype=np.float32)
    if valid.any():
        lo, hi = np.percentile(depth[valid], [2, 98])
        if hi - lo < 1e-3:
            hi = lo + 1e-3
        t = np.clip((depth - lo) / (hi - lo), 0, 1)
    d8 = (t * 255).astype(np.uint8)
    lut = np.zeros((256, 1, 3), np.uint8)
    v = np.linspace(0, 1, 256, dtype=np.float32)
    lut[:, 0, 0] = np.clip(30 + 225 * v ** 1.2, 0, 255)   # B
    lut[:, 0, 1] = np.clip(20 + 210 * v ** 1.5, 0, 255)   # G
    lut[:, 0, 2] = np.clip(40 + 120 * v ** 2.5, 0, 255)   # R
    img = cv2.applyColorMap(d8, lut)
    img[~valid] = 0
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def normal_thumb(normals, size):
    valid = np.linalg.norm(normals, axis=2) > 0.5
    img = np.clip((normals * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)[..., ::-1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.addWeighted(img, 0.60, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.40, 0)
    img = (img.astype(np.float32) * 0.80).astype(np.uint8)
    img[~valid] = 0
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def draw_pips(out, frame, depth, normals, args, view_right):
    """Depth / normals / RGB square insets: a column in the right margin when it is wide enough
    (view_right = right edge of the perception square), otherwise a row at the bottom right."""
    if not args.pip:
        return out
    h, w = out.shape[:2]
    u = max(1.0, w / 1280.0)
    gap = max(8, int(w * 0.008))
    margin = int(w * 0.02)
    strip_h = max(12, int(round(17 * u)))
    thick = max(1, int(round(u)))
    side_margin = w - view_right
    hud_bottom = int(140 * u)   # keep clear of the FPS / FRAME / SRC block drawn by hud()
    column_pw = min(side_margin - 2 * margin, (h - margin - hud_bottom - 2 * gap) // 3)
    if column_pw >= 120:
        pw = column_pw
        x0, y0 = view_right + (side_margin - pw) // 2, h - margin - (3 * pw + 2 * gap)
        step = (0, pw + gap)
    else:
        pw = max(120, int(w * args.pip_scale))
        x0, y0 = w - margin - (3 * pw + 2 * gap), h - margin - pw
        step = (pw + gap, 0)
    ph = pw
    views = [("DEPTH", depth_thumb(depth, (pw, ph))),
             ("NORMALS", normal_thumb(normals, (pw, ph))),
             ("RGB", cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA))]
    for i, (label, img) in enumerate(views):
        x, y = x0 + i * step[0], y0 + i * step[1]
        out[y:y + ph, x:x + pw] = img
        strip = out[y:y + strip_h, x:x + pw]
        out[y:y + strip_h, x:x + pw] = (strip.astype(np.float32) * 0.35).astype(np.uint8)
        text(out, label, (x + int(6 * u), y + int(13 * u)), (230, 230, 230), 0.34 * u, thick, shadow=False)
        cv2.rectangle(out, (x, y), (x + pw - 1, y + ph - 1), (120, 105, 45), thick)
        cv2.line(out, (x, y), (x + int(12 * u), y), CYAN, thick + 1, cv2.LINE_AA)
        cv2.line(out, (x, y), (x, y + int(12 * u)), CYAN, thick + 1, cv2.LINE_AA)
    return out


def draw_joints(out, joints, joint_confidence, source_size, view_size, offset, args):
    """COCO joint heatmap peaks as prominent amber points (no edges: the peaks are not reliable enough)."""
    if not args.joints:
        return out, 0
    sw, sh = source_size
    vw, vh = view_size
    ox, oy = offset
    sx, sy = vw / sw, vh / sh
    u = max(1.0, out.shape[1] / 1280.0)
    radius = max(4, int(round(5 * u)))
    points = []
    for i in range(len(JOINT_NAMES)):
        x, y = float(joints[i, 0]), float(joints[i, 1])
        if joint_confidence[i] < args.joint_threshold or x < 0 or y < 0 or x >= sw or y >= sh:
            points.append(None)
        else:
            points.append((ox + int(round(x * sx)), oy + int(round(y * sy))))
    count = 0
    for p in points:
        if p is None:
            continue
        count += 1
        cv2.circle(out, p, radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(out, p, radius, JOINT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(out, p, max(1, radius // 2), JOINT_NODE, -1, cv2.LINE_AA)
    return out, count


def hud(out, args, names_used, fps, frames, source_name, width, height, show, joint_count=0):
    h, w = out.shape[:2]
    if show:
        u = max(1.0, w / 1280.0)
        thick = max(1, int(round(u)))
        # corner brackets
        m, arm = int(18 * u), int(34 * u)
        for x, y, dx, dy in ((m, m, 1, 1), (w - m, m, -1, 1), (m, h - m, 1, -1), (w - m, h - m, -1, -1)):
            cv2.line(out, (x, y), (x + dx * arm, y), (120, 100, 40), thick, cv2.LINE_AA)
            cv2.line(out, (x, y), (x, y + dy * arm), (120, 100, 40), thick, cv2.LINE_AA)
        # top title
        text(out, "SPATIAL PERCEPTION", (int(40 * u), int(44 * u)), CYAN, 0.62 * u, thick)
        text(out, "YMAPNET  //  DEPTH NORMALS  +  SEMANTIC FUSION", (int(40 * u), int(64 * u)), DIM, 0.34 * u, 1)
        cv2.line(out, (int(40 * u), int(74 * u)), (w - int(40 * u), int(74 * u)), (60, 48, 18), thick, cv2.LINE_AA)
        text(out, f"FPS {fps:04.1f}" if fps > 0 else "FPS --.-", (w - int(160 * u), int(44 * u)), CYAN, 0.50 * u, 1)
        text(out, f"FRAME {frames:06d}", (w - int(160 * u), int(64 * u)), DIM, 0.34 * u, 1)
        text(out, f"SRC {source_name.upper()}", (w - int(160 * u), int(92 * u)), DIM, 0.32 * u, 1)
        text(out, f"{width}x{height}", (w - int(160 * u), int(108 * u)), DIM, 0.30 * u, 1)
        if joint_count:
            text(out, f"JOINTS {joint_count:02d}/17", (w - int(160 * u), int(124 * u)), DIM, 0.30 * u, 1)
        # legend with a dark glass panel
        if names_used:
            count = min(len(names_used), 5)
            line = int(20 * u)
            lx, ly = int(40 * u), h - int(42 * u) - line * count
            x0, y0 = max(0, lx - int(12 * u)), max(0, ly - int(30 * u))
            x1, y1 = min(w, lx + int(250 * u)), min(h, ly + line * count + int(4 * u))
            roi = out[y0:y1, x0:x1]
            out[y0:y1, x0:x1] = (roi.astype(np.float32) * 0.50).astype(np.uint8)
            text(out, "SEMANTIC LAYERS", (lx, ly - int(12 * u)), DIM, 0.34 * u, 1)
            for i, (name, area, color, conf, fraction) in enumerate(names_used[:5]):
                y = ly + i * line
                cv2.rectangle(out, (lx, y + int(2 * u)), (lx + int(10 * u), y + int(10 * u)),
                              tuple(int(v) for v in color), -1)
                text(out, f"{name.upper():<10s} {100.0 * fraction:5.1f}%  {conf:0.2f}",
                     (lx + int(18 * u), y + int(11 * u)), color, 0.42 * u, 1)
    return out


def vignette(width, height, strength=0.38):
    x = np.linspace(-1, 1, width, dtype=np.float32)[None, :]
    y = np.linspace(-1, 1, height, dtype=np.float32)[:, None]
    r = np.sqrt(x * x + y * y) / np.sqrt(2.0)
    return np.clip(1.0 - strength * r ** 2, 0, 1).astype(np.float32)


def render(frame, small, scene, vig, args, fps, frames, source_name, display_size, render_size):
    """Perception layers at render_size (square), scaled to the display height and centred on a
    display-size canvas; tags, joints, HUD and insets are drawn at display resolution."""
    depth, normals, category, confidence, joints, joint_confidence = scene
    camera = cv2.convertScaleAbs(small, alpha=0.20, beta=6)
    out = normal_layer(normals, depth, camera, args)
    out = edge_layer(depth, out, args)
    out, used, labels = category_layer(out, category, confidence, SCENE_CATEGORIES, args)
    out = (out.astype(np.float32) * vig[..., None]).astype(np.uint8)
    out[::4] = (out[::4] * 0.90).astype(np.uint8)
    dw, dh = display_size
    rw, rh = render_size
    side = min(dw, dh)
    ox, oy = (dw - side) // 2, (dh - side) // 2
    if side != rw:
        out = cv2.resize(out, (side, side), interpolation=cv2.INTER_LINEAR)
        if side <= 2200:
            out = cv2.GaussianBlur(out, (3, 3), 0.8)
    canvas = np.zeros((dh, dw, 3), np.uint8)
    canvas[oy:oy + side, ox:ox + side] = out
    canvas = draw_category_tags(canvas, labels, side / rw, side / rh, ox, oy)
    canvas, joint_count = draw_joints(canvas, joints, joint_confidence, (rw, rh), (side, side), (ox, oy), args)
    canvas = hud(canvas, args, used, fps, frames, source_name, frame.shape[1], frame.shape[0],
                 args.show_hud, joint_count)
    canvas = draw_pips(canvas, frame, depth, normals, args, ox + side)
    return canvas


# ------------------------------------------------------------------------------------------------
# Scene from the estimator's own forward (replaces tesla_view.py's TFSceneEngine.scene)
# ------------------------------------------------------------------------------------------------

def channel_index(names, name):
    """configuration.json channel index for a name (exact, then prefix in either direction)."""
    if name in names:
        return names.index(name)
    low = name.lower()
    for i, candidate in enumerate(names):
        c = candidate.lower()
        if c.startswith(low) or low.startswith(c):
            return i
    return -1


def scene_from_estimator(estimator, out_size):
    """(depth, normals, category, confidence, joints, joint_confidence) at out_size from the raw
    outputs of the last YMAPNet.process(): keypoints_predictions (8-bit heads, +-heatmapActive) and
    heatmap_16b (16-bit depth head, +-32767). Joints are in out_size pixel coordinates."""
    cfg = estimator.cfg
    names = list(cfg.get("heatmaps", []))
    hm = estimator.keypoints_predictions
    grid_h, grid_w = hm.shape[:2]
    hm_scale = float(cfg.get("heatmapActive", 120)) * float(cfg.get("heatmapTanhHeadroom8bit", 1.0))
    out_w, out_h = out_size
    # depth: the 16-bit head, +1 nearest .. -1 farthest (8-bit 'depthmap' channel if there is none)
    hm16 = getattr(estimator, "heatmap_16b", None)
    if hm16 is not None:
        hm16 = np.asarray(hm16, np.float32).reshape(grid_h, grid_w, -1)[..., 0]
        d = hm16 / (32767.0 * float(cfg.get("heatmapTanhHeadroom16bit", 1.0)))
    else:
        c = channel_index(names, "depthmap")
        d = hm[..., c] / hm_scale if c >= 0 else np.zeros((grid_h, grid_w), np.float32)
    depth = cv2.resize(d.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_AREA)
    # normals: unit camera-frame vectors, 0 where the model gave none
    channels = [channel_index(names, name) for name in ("normalX", "normalY", "normalZ")]
    if all(c >= 0 for c in channels):
        n = np.stack([hm[..., c] for c in channels], axis=-1).astype(np.float32) / hm_scale
        length = np.linalg.norm(n, axis=2, keepdims=True)
        invalid = (length <= 1e-3)[..., 0]
        n = np.where(invalid[..., None], 0.0, n / np.where(invalid[..., None], 1.0, length))
    else:
        n = np.zeros((grid_h, grid_w, 3), np.float32)
    # cv2.resize rejects 3-D float arrays on some OpenCV builds; resize the planes separately
    normals = np.stack([cv2.resize(np.ascontiguousarray(n[..., c], np.float32), (out_w, out_h),
                                   interpolation=cv2.INTER_LINEAR) for c in range(3)], axis=-1)
    # category argmax over the scene categories (background = -1), like yd_scene
    best = np.zeros((grid_h, grid_w), np.float32)
    category = np.full((grid_h, grid_w), -1.0, np.float32)
    for i, name in enumerate(SCENE_CATEGORIES):
        c = channel_index(names, name)
        if c < 0:
            continue
        m = hm[..., c]
        better = (m > best) & (m > 0.0)
        category[better] = i
        best = np.maximum(best, m)
    category = cv2.resize(category, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    confidence = cv2.resize(best / hm_scale, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    # COCO joint peaks with a 5x5 weighted centroid
    joints = np.full((17, 2), -1.0, np.float32)
    joint_confidence = np.zeros(17, np.float32)
    sx, sy = out_w / float(grid_w), out_h / float(grid_h)
    for j in range(17):
        m = hm[..., j]
        peak = int(np.argmax(m))
        py, px = divmod(peak, grid_w)
        value = float(m[py, px]) / hm_scale
        if value <= 0.0:
            continue
        y0, y1 = max(0, py - 2), min(grid_h - 1, py + 2)
        x0, x1 = max(0, px - 2), min(grid_w - 1, px + 2)
        window = m[y0:y1 + 1, x0:x1 + 1]
        positive = window > 0.0
        if positive.any():
            ys, xs = np.nonzero(positive)
            weights = window[positive]
            mx = float((weights * (xs + x0)).sum() / weights.sum())
            my = float((weights * (ys + y0)).sum() / weights.sum())
        else:
            mx, my = px, py
        joints[j, 0] = (mx + 0.5) * sx
        joints[j, 1] = (my + 0.5) * sy
        joint_confidence[j] = value
    return depth, normals, category, confidence, joints, joint_confidence


def centre_square(frame):
    """The centre square of the frame: what resize_image_no_borders hands YMAPNet."""
    h, w = frame.shape[:2]
    side = min(h, w)
    x, y = (w - side) // 2, (h - side) // 2
    return frame[y:y + side, x:x + side]


class RenderWorker(threading.Thread):
    """Renders the newest published scene on its own thread: the heavy numpy/OpenCV composition
    overlaps the next forward (which releases the GIL during the native call). Only imshow/waitKey
    run on the main thread. The renderer only reads the published arrays."""

    def __init__(self, args, vig, source_name, display_size, render_size):
        super().__init__(daemon=True)
        self.args = args
        self.vig = vig
        self.source_name = source_name
        self.display_size = display_size
        self.render_size = render_size
        self.condition = threading.Condition()
        self.pending = None       # (frame, small, scene, fps, frames) waiting to be rendered
        self.out = None           # newest rendered frame
        self.error = None
        self.running = True

    def stop(self):
        with self.condition:
            self.running = False
            self.condition.notify_all()

    def submit(self, frame, small, scene, fps, frames):
        with self.condition:
            self.pending = (frame, small, scene, fps, frames)
            self.condition.notify_all()

    def take(self):
        """The newest rendered frame, or None."""
        with self.condition:
            return self.out

    def run(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.pending is not None or not self.running,
                                            timeout=0.5)
                    if self.pending is not None:
                        frame, small, scene, fps, frames = self.pending
                        self.pending = None
                    elif not self.running:
                        break
                    else:
                        continue
                self.out = render(frame, small, scene, self.vig, self.args, fps, frames,
                                  self.source_name, self.display_size, self.render_size)
        except Exception as exc:  # surfaced to the main loop, which re-raises it
            with self.condition:
                self.error = exc
                self.running = False
                self.condition.notify_all()


def monitor_geometries():
    """Connected monitors from xrandr as [(x, y, w, h, name), ...], or [] when unavailable."""
    out = run_tool(["xrandr", "--query"], capture_output=True, timeout=3)
    if out is None:
        return []
    return [(int(mx), int(my), int(mw), int(mh), name) for name, mw, mh, mx, my in
            re.findall(r"^(\S+)\s+connected(?:\s+primary)?\s+(\d+)x(\d+)\+(-?\d+)\+(-?\d+)",
                       out.stdout, re.MULTILINE)]


def resolve_monitor(monitors, selector):
    """'0' / 'DP-1' -> that monitor's (x, y, w, h, name), or None."""
    for i, mon in enumerate(monitors):
        if selector == str(i) or selector == mon[4]:
            return mon
    print(f"[showoff] screen '{selector}' not found; connected monitors: "
          f"{', '.join(f'{mon[4]}[{i}]' for i, mon in enumerate(monitors))}")
    return None


def monitor_for_window(monitors, window):
    """The monitor holding the window's centre. (The Qt backend reports the image rect, not the
    screen, even when fullscreen, so the fullscreen size cannot be read back from the window.)"""
    x, y, w, h = cv2.getWindowImageRect(window)
    cx, cy = x + w // 2, y + h // 2
    for mon in monitors:
        if mon[0] <= cx < mon[0] + mon[2] and mon[1] <= cy < mon[1] + mon[3]:
            return mon
    return monitors[0] if monitors else None


class TeslaView:
    """runYMAPNet --showoff: call show(frame, estimator) after estimator.process(frame), and pass
    each waitKey() value to handle_key(). The window, render size and display canvas are set up on
    the first frame."""

    def __init__(self, source_name="webcam", screen="", shm_stream=None, shm_path=None):
        self.args = SimpleNamespace(**DEFAULT_OPTIONS)
        self.shm_stream = shm_stream  # also publish each rendered frame to this SharedMemoryVideoBuffers stream
        self.shm_path = shm_path or os.environ.get(
            "SHARED_MEMORY_VIDEO_BUFFERS", os.path.expanduser("~/Documents/Programming/SharedMemoryVideoBuffers"))
        self.shm = None
        self.shm_size = None
        self.source_name = source_name
        self.screen = screen      # xrandr monitor index or output name; "" = wherever the window opens
        self.renderer = None
        self.durations = deque(maxlen=30)
        self.last = None

    def _start(self, frame):
        monitors = monitor_geometries()
        target = resolve_monitor(monitors, self.screen) if self.screen else None
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW, frame)
        cv2.waitKey(1)
        if target is not None:
            # move onto the target monitor and give the WM a moment to apply the move before going
            # fullscreen (fullscreen follows the window's monitor)
            cv2.moveWindow(WINDOW, target[0], target[1])
            for _ in range(8):
                cv2.waitKey(10)
        if self.args.fullscreen:
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        for _ in range(8):
            cv2.waitKey(10)
        monitor = target or monitor_for_window(monitors, WINDOW)
        display_size = monitor[2:4] if monitor is not None else (frame.shape[1], frame.shape[0])
        if monitor is not None:
            print(f"[showoff] on monitor {monitor[4]} {monitor[2]}x{monitor[3]}+{monitor[0]}+{monitor[1]}")
        # render at tesla_view's width even when the camera crop is smaller (the heatmaps are
        # resampled to it anyway), but never above the display height the square is shown at
        rw = min(self.args.render_width, min(display_size))
        print(f"[showoff] render {rw}x{rw}; display canvas {display_size[0]}x{display_size[1]}")
        self.renderer = RenderWorker(self.args, vignette(rw, rw), self.source_name, display_size, (rw, rw))
        self.renderer.start()

    def show(self, frame, estimator):
        frame = centre_square(frame)
        if self.renderer is None:
            self._start(frame)
        if self.renderer.error is not None:
            raise self.renderer.error
        now = time.time()
        if self.last is not None:
            self.durations.append(now - self.last)
        self.last = now
        rw, rh = self.renderer.render_size
        small = frame if frame.shape[0] == rh else cv2.resize(
            frame, (rw, rh), interpolation=cv2.INTER_AREA if frame.shape[0] > rh else cv2.INTER_LINEAR)
        scene = scene_from_estimator(estimator, (rw, rh))
        fps = 1.0 / max(1e-6, sum(self.durations) / len(self.durations)) if self.durations else 0.0
        self.renderer.submit(frame, small, scene, fps, estimator.frameNumber)
        out = self.renderer.take()
        cv2.imshow(WINDOW, out if out is not None else frame)
        if out is not None and self.shm_stream:
            self._publish_shm(out)

    def _publish_shm(self, out):
        if self.shm is None:
            import sys
            pydir = os.path.join(self.shm_path, "src", "python")
            if pydir not in sys.path:
                sys.path.insert(0, pydir)
            from SharedMemoryManager import SharedMemoryManager
            h, w = out.shape[:2]
            self.shm = SharedMemoryManager(os.path.join(self.shm_path, "libSharedMemoryVideoBuffers.so"),
                                           descriptor="video_frames.shm", frameName=self.shm_stream,
                                           width=w, height=h, channels=3)
            self.shm_size = (w, h)
            print(f"[showoff] publishing {w}x{h} to shared memory stream '{self.shm_stream}'")
        if (out.shape[1], out.shape[0]) != self.shm_size:
            out = cv2.resize(out, self.shm_size, interpolation=cv2.INTER_AREA)
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
        self.shm.copy_numpy_to_shared_memory(cv2.cvtColor(out[:, :, :3], cv2.COLOR_BGR2RGB))

    def handle_key(self, key):
        if key in (ord("h"), ord("H")):
            self.args.show_hud = not self.args.show_hud
        elif key in (ord("f"), ord("F")):
            self.args.fullscreen = not self.args.fullscreen
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if self.args.fullscreen else cv2.WINDOW_NORMAL)

    def close(self):
        if self.renderer is not None:
            self.renderer.stop()
            self.renderer.join(timeout=10.0)
        self.shm = None
