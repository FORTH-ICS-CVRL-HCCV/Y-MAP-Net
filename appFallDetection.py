#!/usr/bin/python3

"""
Author : "Ammar Qammaz"
Copyright : "2024 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"

Fallen-person detection helpers extracted from runYMAPNet.py.
"""

import cv2
import numpy as np


# =============================================================================
# Fallen-person helpers
# =============================================================================

def _run_fallen_person_detector(estimator):
    """
    Run detect_fallen_person() using the current estimator heatmaps.
    Returns the metrics dict, or None if the required channels are not available.
    """
    from YMAPNet import detect_fallen_person
    e = estimator
    needed = [e.chanNormalX, e.chanNormalY, e.chanNormalZ, e.chanFloor]
    if any(c < 0 or c >= len(e.heatmapsOut) for c in needed):
        return None
    # Build Person Union (max across Person/Face/Hand/Foot channels 39-43)
    human_chans = [i for i in range(39, 43) if i < len(e.heatmapsOut)]
    if not human_chans:
        return None
    union_human = np.max(np.stack([e.heatmapsOut[i] for i in human_chans], axis=0), axis=0)
    # Furniture channel is optional — helps disambiguate fallen vs sleeping on furniture
    chan_furniture = getattr(e, 'chanFurniture', -1)
    furniture_mask = e.heatmapsOut[chan_furniture] if (0 <= chan_furniture < len(e.heatmapsOut)) else None
    metrics = detect_fallen_person(
        person_mask    = union_human,
        normal_x       = e.heatmapsOut[e.chanNormalX],
        normal_y       = e.heatmapsOut[e.chanNormalY],
        normal_z       = e.heatmapsOut[e.chanNormalZ],
        floor_mask     = e.heatmapsOut[e.chanFloor],
        furniture_mask = furniture_mask,
    )
    floor_mask_out = e.heatmapsOut[e.chanFloor]
    return metrics, union_human, floor_mask_out, furniture_mask


def _save_fallen_frame(frame, m, frame_number, union_human=None, floor_mask=None, furniture_mask=None):
    """Overlay fallen-person metrics on a copy of frame and save as JPEG."""
    img = frame.copy()
    H, W = img.shape[:2]

    # ── green overlay for human segmentation pixels ───────────────────────────
    if union_human is not None:
        hm_h, hm_w = union_human.shape[:2]
        mask_full = cv2.resize(union_human, (W, H), interpolation=cv2.INTER_LINEAR)
        kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_full = cv2.erode(mask_full, kernel, iterations=2)
        person_bin = mask_full > 30
        # Black out pixels within 30px of any edge to suppress border artefacts
        border = 10
        person_bin[:border, :]  = False
        person_bin[-border:, :] = False
        person_bin[:, :border]  = False
        person_bin[:, -border:] = False
        green_layer = np.zeros_like(img)
        green_layer[person_bin] = (0, 255, 0)
        img = cv2.addWeighted(img, 1.0, green_layer, 0.4, 0)

    # ── red overlay for floor pixels ──────────────────────────────────────────
    if floor_mask is not None:
        mask_full = cv2.resize(floor_mask, (W, H), interpolation=cv2.INTER_LINEAR)
        floor_bin = mask_full > 30
        red_layer = np.zeros_like(img)
        red_layer[floor_bin] = (0, 0, 255)
        img = cv2.addWeighted(img, 1.0, red_layer, 0.4, 0)

    # ── blue overlay for furniture pixels ─────────────────────────────────────
    if furniture_mask is not None:
        mask_full = cv2.resize(furniture_mask, (W, H), interpolation=cv2.INTER_LINEAR)
        furn_bin = mask_full > 30
        blue_layer = np.zeros_like(img)
        blue_layer[furn_bin] = (255, 0, 0)
        img = cv2.addWeighted(img, 1.0, blue_layer, 0.4, 0)

    lines = [
        "FALLEN PERSON DETECTED",
        f"frame      : {frame_number}",
        f"aspect     : {m['aspect_ratio']:.2f}  (>0.80 = wide)",
        f"angle      : {m['orientation_deg']:.1f} deg from vertical",
        f"floor ovlp : {m['floor_overlap']:.2f}",
        f"furn ovlp  : {m.get('furniture_overlap', 0.0):.2f}",
        f"norm horiz : {m['normal_horiz_frac']:.2f}",
        f"centroid y : {m['centroid_y_frac']:.2f}",
        f"person px  : {m['person_pixel_count']}",
        f"signals    : {', '.join(k for k,v in m['signals'].items() if v)}",
    ]

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, W / 1600)
    thickness  = max(1, int(font_scale * 2))
    line_h     = int(font_scale * 28)
    pad        = 8
    panel_h    = len(lines) * line_h + pad * 2
    panel_w    = int(W * 0.55)

    # Dark translucent background panel
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    for i, text in enumerate(lines):
        color = (0, 0, 255) if i == 0 else (255, 255, 255)
        y = pad + (i + 1) * line_h
        cv2.putText(img, text, (pad, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, text, (pad, y), font, font_scale, color,     thickness,     cv2.LINE_AA)

    # ── bounding box (scale from heatmap space to frame space) ───────────────
    if "bbox_heatmap" in m and m["bbox_heatmap"] is not None:
        hm_w, hm_h = m["heatmap_size"]
        bx0, by0, bx1, by1 = m["bbox_heatmap"]
        sx = W / hm_w
        sy = H / hm_h
        fx0, fy0 = int(bx0 * sx), int(by0 * sy)
        fx1, fy1 = int(bx1 * sx), int(by1 * sy)
        # Thick red box with thin black outline for contrast
        cv2.rectangle(img, (fx0, fy0), (fx1, fy1), (0,   0,   0),   thickness + 3, cv2.LINE_AA)
        cv2.rectangle(img, (fx0, fy0), (fx1, fy1), (0,   0, 255),   thickness + 1, cv2.LINE_AA)
        # Small "FALLEN" label above the box
        lbl = "FALLEN"
        (tw, th), _ = cv2.getTextSize(lbl, font, font_scale, thickness)
        ly = max(fy0 - 6, th + 4)
        cv2.putText(img, lbl, (fx0, ly), font, font_scale, (0, 0, 0),   thickness + 2, cv2.LINE_AA)
        cv2.putText(img, lbl, (fx0, ly), font, font_scale, (0, 0, 255), thickness,     cv2.LINE_AA)

    filename = f"fallen_{frame_number:06d}.jpg"
    cv2.imwrite(filename, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  -> saved {filename}")


def _print_fallen_result(m, frame_number):
    status = "FALLEN" if m["is_fallen"] else "upright"
    fired  = [k for k, v in m["signals"].items() if v]
    print(
        f"[{frame_number:>6}] fallen_person={status:6s} | "
        f"aspect={m['aspect_ratio']:.2f}  "
        f"angle={m['orientation_deg']:.1f}°  "
        f"floor_overlap={m['floor_overlap']:.2f}  "
        f"furn_overlap={m.get('furniture_overlap', 0.0):.2f}  "
        f"normal_horiz={m['normal_horiz_frac']:.2f}  "
        f"centroid_y={m['centroid_y_frac']:.2f}  "
        f"px={m['person_pixel_count']}  "
        f"signals={fired}"
    )
