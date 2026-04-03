#!/usr/bin/python3

"""
Author : "Ammar Qammaz"
Copyright : "2024 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"
"""

# Dependencies:
# tensorflow-2.16.1 needs CUDA 12.3, CUDNN 8.9.6, built with Clang 17.0.6 / Bazel 6.5.0
# python3 -m pip install tf_keras tensorflow==2.16.1 numpy tensorboard opencv-python wget

import glob
import os
import re
import sys
import subprocess
import argparse
from tools import bcolors

try:
    import cv2
    import numpy as np
except Exception as e:
    print(bcolors.WARNING, "Could not import libraries!", bcolors.ENDC)
    print("An exception occurred:", str(e))
    print("Issue:\n source venv/bin/activate\nBefore running this script")
    sys.exit(1)

# =============================================================================
# Argument parser
# =============================================================================

def build_arg_parser():
    p = argparse.ArgumentParser(description="YMAPNet 2D pose estimation runner")
    p.add_argument("--cpu",          action="store_true",
                   help="Force CPU inference")
    p.add_argument("--update",       action="store_true",
                   help="Re-download the model weights")
    p.add_argument("--from",         dest="videoFilePath", default="webcam",
                   metavar="PATH",
                   help="Source: path, webcam, screen, esp, /dev/videoN")
    p.add_argument("--size",         nargs=2, type=int, default=[640, 480],
                   metavar=("W", "H"))
    p.add_argument("--scale",        type=float, default=1.0)
    p.add_argument("--threshold",    type=float, default=84.0, metavar="T")
    p.add_argument("--border",       type=int,   default=0)
    p.add_argument("--crop",         nargs=3, type=int,
                   metavar=("X", "Y", "SIZE"),
                   help="Custom crop: centre X, centre Y, size")
    p.add_argument("--nocrop",       action="store_true",
                   help="Disable centre-crop preprocessing")
    p.add_argument("--blur",         type=int,   default=0, metavar="STRENGTH")
    p.add_argument("--noise",        type=float, default=0.0, metavar="MAG",
                   help="Gaussian noise magnitude [0.0-1.0]")
    p.add_argument("--engine",       default="tensorflow", metavar="ENGINE")
    p.add_argument("--model",        default="2d_pose_estimation", metavar="PATH",
                   help="Model directory or file (default: 2d_pose_estimation)")
    p.add_argument("--save",         action="store_true")
    p.add_argument("--prune",        action="store_true", dest="pruneTokens")
    p.add_argument("--tile",         action="store_true")
    p.add_argument("--illustrate",   action="store_true")
    p.add_argument("--collab",       action="store_true",
                   help="illustrate + save, no display")
    p.add_argument("--headless",     "--novisualization",
                   action="store_true", dest="headless")
    p.add_argument("--profiling",    "--profile", action="store_true")
    p.add_argument("--depth-iterations", type=int, default=10, metavar="N",
                   help="Sobel depth-refinement iterations (0 = disabled)")
    p.add_argument("--no-person-id",  action="store_true",
                   help="Disable per-blob person ID estimation")
    p.add_argument("--no-skeleton",   action="store_true",
                   help="Disable joint-hierarchy skeleton resolution")
    p.add_argument("--fast",          action="store_true",
                   help="Shorthand for --depth-iterations 0 --no-person-id --no-skeleton")
    # Repeatable multi-value options
    p.add_argument("--win",          nargs=3, action="append", default=[],
                   metavar=("X", "Y", "LABEL"),
                   help="Window arrangement entry (repeatable)")
    p.add_argument("--monitor",      nargs=4, action="append", default=[],
                   metavar=("HM", "X", "Y", "LABEL"),
                   help="Heatmap monitor entry (repeatable)")
    p.add_argument("--upload-url",   default="http://ammar.gr/datasets/uploads.php",
                   metavar="URL",    help="Frame upload endpoint")
    p.add_argument("--screen",       nargs=2, type=int, default=None,
                   metavar=("W", "H"),
                   help="Physical display resolution for auto window tiling (auto-detected if omitted)")
    p.add_argument("--fallen-person-detector", action="store_true",
                   dest="fallen_person_detector",
                   help="Enable fallen-person detection using Person/Floor/Normal heatmaps")
    p.add_argument("--no-save-fallen", action="store_false",
                   dest="save_fallen", default=True,
                   help="Disable saving frames when a fallen person is detected (saving is ON by default)")
    p.add_argument("--faces",          action="store_true",
                   help="Dump RGB crops of every detected face (Face segmentation mask) to faces/")
    return p

# =============================================================================
# Video capture factory
# =============================================================================

def getCaptureDeviceFromPath(videoFilePath, videoWidth, videoHeight, videoFramerate=30,
                             model_path="2d_pose_estimation"):
    def _open_camera(index):
        # On Windows the default MSMF backend frequently fails to open cameras;
        # DirectShow (CAP_DSHOW) is far more reliable.
        if sys.platform == 'win32':
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_FPS, videoFramerate)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, videoWidth)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, videoHeight)
        if not cap.isOpened():
            print("ERROR: Could not open camera index %d." % index)
            if sys.platform == 'win32':
                print("       On Windows, also check: Settings -> Privacy & Security -> Camera")
                print("       -> 'Let desktop apps access your camera' must be ON.")
        return cap

    if videoFilePath in ("training_dataset", "validation_dataset"):
        from datasetStream import DatasetStreamer
        key = "TrainingDataset" if videoFilePath == "training_dataset" else "ValidationDataset"
        return DatasetStreamer(key, model_path=model_path)
    elif videoFilePath == "esp":
        from espStream import ESP32CamStreamer
        return ESP32CamStreamer()
    elif videoFilePath == "screen":
        from screenStream import ScreenGrabber
        return ScreenGrabber(region=(0, 0, videoWidth, videoHeight))
    elif videoFilePath == "webcam":
        return _open_camera(0)
    else:
        m = re.fullmatch(r'/dev/video(\d+)', videoFilePath)
        if m:
            return _open_camera(int(m.group(1)))
        # Allow bare integer index on Windows (e.g. --from 1 for second camera)
        if videoFilePath.isdigit():
            return _open_camera(int(videoFilePath))
        from tools import checkIfPathIsDirectory
        if checkIfPathIsDirectory(videoFilePath):
            from folderStream import FolderStreamer
            return FolderStreamer(path=videoFilePath, width=videoWidth, height=videoHeight)
        return cv2.VideoCapture(videoFilePath)

# =============================================================================
# System utilities
# =============================================================================

def save_and_upload_frame(frame, url):
    print("Saving frame")
    cv2.imwrite('frame.jpg', frame)
    print("Uploading frame to", url)
    subprocess.run(["curl", "-F", "file=@frame.jpg", url], check=False)

def prevent_screensaver():
    subprocess.run(
        "xdotool mousemove_relative -- 1 0 && sleep 1 && xdotool mousemove_relative -- -1 0",
        shell=True, check=False,
    )

def disable_screensaver():
    subprocess.run(["xset", "s", "off"], check=False)

def enable_screensaver():
    subprocess.run(["xset", "s", "on"], check=False)

def screenshot(framenumber):
    subprocess.run(["scrot", f"colorFrame_0_{framenumber:05}.png"], check=False)

# =============================================================================
# Image processing helpers
# =============================================================================

def create_ply_file(bgr_image, depth_array, filename, depthScale=1.0):
    height, width = depth_array.shape[:2]
    if bgr_image.shape[:2] != (height, width):
        bgr_image = cv2.resize(bgr_image, (width, height), interpolation=cv2.INTER_LINEAR)

    header = (
        f"ply\nformat ascii 1.0\nelement vertex {height * width}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    zs   = depth_array * depthScale
    r, g, b = bgr_image[:, :, 2], bgr_image[:, :, 1], bgr_image[:, :, 0]
    data = np.stack([xs, -ys, zs, r, g, b], axis=-1).reshape(-1, 6)
    with open(filename, 'w') as ply_file:
        ply_file.write(header)
        np.savetxt(ply_file, data, fmt='%g %g %g %d %d %d')


def extract_centered_rectangle(image):
    """Return the largest centred square crop of the image."""
    img_height, img_width = image.shape[:2]
    side = min(img_height, img_width)
    cx, cy = img_width // 2, img_height // 2
    x0 = max(0, cx - side // 2)
    y0 = max(0, cy - side // 2)
    x1 = min(img_width,  cx + side // 2)
    y1 = min(img_height, cy + side // 2)
    return image[y0:y1, x0:x1]


def custom_crop(image, cX, cY, size):
    """Crop a region of *size* pixels centred on (cX, cY).

    Note: the right/bottom edge extends *size* pixels from the centre while
    the left/top edge extends *size // 2* pixels, making the crop intentionally
    right-/bottom-biased when size is odd.
    """
    img_height, img_width = image.shape[:2]
    x0 = max(0, cX - size // 2)
    y0 = max(0, cY - size // 2)
    x1 = min(img_width,  cX + size)
    y1 = min(img_height, cY + size)
    return image[y0:y1, x0:x1]


def add_horizontal_stripes(image: np.ndarray, stripe_height: int) -> np.ndarray:
    """Paint black bars of *stripe_height* pixels at the top and bottom of *image*."""
    max_stripe_height = image.shape[0] // 2
    if stripe_height > max_stripe_height:
        raise ValueError(
            f"Stripe height cannot exceed half the image height ({max_stripe_height})."
        )
    result = np.copy(image)
    result[:stripe_height, :]  = 0
    result[-stripe_height:, :] = 0
    return result


def apply_blur_to_image(image: np.ndarray, blur_strength: int = 5) -> np.ndarray:
    """Apply a Gaussian blur to *image*. *blur_strength* is the kernel size."""
    if blur_strength % 2 == 0:
        blur_strength += 1
    return cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)


def add_noise_to_image(image: np.ndarray, noise_magnitude: float = 0.1) -> np.ndarray:
    """Add Gaussian noise to *image*.

    Args:
        image: uint8 BGR/RGB image.
        noise_magnitude: standard deviation as a fraction of [0, 255].
                         Expected range: 0.0 (none) to 1.0 (full).
    """
    noise_magnitude = np.clip(noise_magnitude, 0.0, 1.0)
    noise = np.random.normal(0, noise_magnitude * 255, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

# =============================================================================
# Display resolution detection
# =============================================================================

def detect_screen_resolution():
    """Return the total desktop resolution (width, height) as a tuple of ints.

    Tries four methods in order, most-reliable first:

      1. xrandr monitor bounding box — parses every "connected WxH+X+Y" entry
         and computes max(X+W) × max(Y+H).  This correctly spans all monitors
         in an extended-desktop layout even when 'Screen 0: current' only reports
         a single monitor's size (common under Wayland / XWayland / multi-GPU).
      2. xrandr 'Screen 0: current W x H' — the virtual screen headline.
      3. xdpyinfo 'dimensions: WxH pixels'.
      4. tkinter root window geometry.

    Falls back to (3840, 2400) if every method fails.
    """
    # 1. xrandr — bounding box of all active (mode-set) monitors
    try:
        out = subprocess.run(["xrandr"], capture_output=True, text=True, timeout=3)

        # Match lines like:  HDMI-1 connected [primary] 1920x1080+3840+0 (…)
        # Groups:  (width, height, x_offset, y_offset)
        monitors = re.findall(
            r'\bconnected\b(?:\s+primary)?\s+(\d+)x(\d+)\+(\d+)\+(\d+)',
            out.stdout,
        )
        if monitors:
            total_w = max(int(mx) + int(mw) for mw, mh, mx, my in monitors)
            total_h = max(int(my) + int(mh) for mw, mh, mx, my in monitors)
            print(f"Screen resolution computed from {len(monitors)} active "
                  f"xrandr monitor(s): {total_w}x{total_h}")
            return total_w, total_h

        # Fallback within xrandr: 'Screen 0: current W x H'
        m = re.search(r'current\s+(\d+)\s*x\s*(\d+)', out.stdout)
        if m:
            w, h = int(m.group(1)), int(m.group(2))
            print(f"Screen resolution detected via xrandr (Screen 0 current): {w}x{h}")
            return w, h
    except Exception:
        pass

    # 2. xdpyinfo
    try:
        out = subprocess.run(["xdpyinfo"], capture_output=True, text=True, timeout=3)
        m = re.search(r'dimensions:\s+(\d+)x(\d+)', out.stdout)
        if m:
            w, h = int(m.group(1)), int(m.group(2))
            print(f"Screen resolution detected via xdpyinfo: {w}x{h}")
            return w, h
    except Exception:
        pass

    # 3. tkinter
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        print(f"Screen resolution detected via tkinter: {w}x{h}")
        return w, h
    except Exception:
        pass

    fallback = (3840, 2400)
    print(f"Could not detect screen resolution; using fallback {fallback[0]}x{fallback[1]}")
    return fallback


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
    # chanPerson defaults to 39 (from config); allow for models without it
    chan_person = getattr(e, 'chanPerson', 39)
    if chan_person < 0 or chan_person >= len(e.heatmapsOut):
        return None
    # Furniture channel is optional — helps disambiguate fallen vs sleeping on furniture
    chan_furniture = getattr(e, 'chanFurniture', -1)
    furniture_mask = e.heatmapsOut[chan_furniture] if (0 <= chan_furniture < len(e.heatmapsOut)) else None
    return detect_fallen_person(
        person_mask    = e.heatmapsOut[chan_person],
        normal_x       = e.heatmapsOut[e.chanNormalX],
        normal_y       = e.heatmapsOut[e.chanNormalY],
        normal_z       = e.heatmapsOut[e.chanNormalZ],
        floor_mask     = e.heatmapsOut[e.chanFloor],
        furniture_mask = furniture_mask,
    )


def _save_fallen_frame(frame, m, frame_number):
    """Overlay fallen-person metrics on a copy of frame and save as JPEG."""
    img = frame.copy()
    H, W = img.shape[:2]

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


# =============================================================================
# Face-crop helpers
# =============================================================================

# Heatmap channel indices for facial keypoints (fixed by model architecture)
_FACE_KP_CHANNELS = [0, 1, 2, 3, 4]   # nose, left_eye, right_eye, left_ear, right_ear


def _find_kp_peaks(heatmap, threshold=40, blur_k=3):
    """
    Return a list of (x, y, value) peaks in a 2-D heatmap.
    Uses dilate-then-compare to find local maxima above *threshold*.
    """
    if blur_k > 1:
        hm = cv2.GaussianBlur(heatmap.astype(np.float32), (blur_k, blur_k), 0)
    else:
        hm = heatmap.astype(np.float32)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(hm, kernel)
    local_max = (hm == dilated) & (hm >= threshold)
    ys, xs = np.where(local_max)
    return [(int(x), int(y), float(hm[y, x])) for x, y in zip(xs, ys)]


def _cluster_face_keypoints(all_peaks, cluster_radius):
    """
    Greedy single-linkage clustering of (x, y, value) peaks.
    Two peaks belong to the same face if their Euclidean distance ≤ *cluster_radius*.
    Returns a list of clusters; each cluster is a list of (x, y, value).
    """
    clusters = []
    for pt in all_peaks:
        px, py = pt[0], pt[1]
        best_dist, best_idx = float('inf'), -1
        for ci, cl in enumerate(clusters):
            # Compare against the cluster centroid (mean of existing points)
            cx = sum(p[0] for p in cl) / len(cl)
            cy = sum(p[1] for p in cl) / len(cl)
            d  = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
            if d < best_dist:
                best_dist, best_idx = d, ci
        if best_dist <= cluster_radius:
            clusters[best_idx].append(pt)
        else:
            clusters.append([pt])
    return clusters


def _dump_face_crops(estimator, frame, face_crop_counter, output_dir="faces",
                     seg_threshold=64, kp_threshold=40, min_seg_pixels=80,
                     padding=0.5):
    """
    Dump RGB face crops to *output_dir/face_XXXXXX.jpg*.

    Strategy
    --------
    Primary path — keypoint-anchored crops:
      1. Find peaks in the nose / eye / ear heatmaps (channels 0-4).
      2. Cluster nearby peaks across all 5 channels; each cluster = one face.
      3. Derive the crop box from the cluster's point spread, padded by *padding*
         (as a fraction of the box diagonal).  The Face segmentation mask is used
         only to validate that the cluster centre falls in a masked region; it is
         NOT used to define the box shape, so merged blobs (people standing close)
         are split correctly.

    Fallback path — segmentation blobs:
      When no facial keypoints fire at all (e.g. face turned away), fall back to
      connected-component bounding boxes of the Face segmentation mask, as before.
    """
    heatmaps = estimator.heatmapsOut
    n_hm     = len(heatmaps)

    fH, fW = frame.shape[:2]

    # ── Face segmentation mask ───────────────────────────────────────────────
    chan_face = getattr(estimator, 'chanFace', -1)
    face_mask = None
    if 0 <= chan_face < n_hm:
        face_mask = heatmaps[chan_face]          # H_hm × W_hm uint8

    hmH = face_mask.shape[0] if face_mask is not None else fH
    hmW = face_mask.shape[1] if face_mask is not None else fW
    scale_x = fW / hmW
    scale_y = fH / hmH

    def _save_crop(x0_hm, y0_hm, x1_hm, y1_hm):
        """Pad, clip, crop and write one face JPEG."""
        w_hm = max(1, x1_hm - x0_hm)
        h_hm = max(1, y1_hm - y0_hm)
        diag = (w_hm ** 2 + h_hm ** 2) ** 0.5
        pad  = int(diag * padding * 0.5)       # same pad on all sides

        fx0 = max(0,  int((x0_hm - pad) * scale_x))
        fy0 = max(0,  int((y0_hm - pad) * scale_y))
        fx1 = min(fW, int((x1_hm + pad) * scale_x))
        fy1 = min(fH, int((y1_hm + pad) * scale_y))

        crop = frame[fy0:fy1, fx0:fx1]
        if crop.size == 0:
            return
        idx  = face_crop_counter[0]
        face_crop_counter[0] += 1
        cv2.imwrite(os.path.join(output_dir, f"face_{idx:06d}.jpg"),
                    crop, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # ── 1. Collect peaks from all facial keypoint channels ───────────────────
    all_peaks = []
    for ch in _FACE_KP_CHANNELS:
        if ch < n_hm:
            all_peaks.extend(_find_kp_peaks(heatmaps[ch], threshold=kp_threshold))

    # ── 2. Cluster peaks → one cluster per face ──────────────────────────────
    #    Cluster radius: ~15 % of heatmap width covers typical inter-keypoint spread
    cluster_radius = hmW * 0.15
    clusters = _cluster_face_keypoints(all_peaks, cluster_radius) if all_peaks else []

    # ── 3. Keypoint-anchored crops ───────────────────────────────────────────
    # Precompute a binary face mask once (used to extend Y downward)
    bin_face = None
    if face_mask is not None:
        bin_face = (face_mask >= seg_threshold)

    used_kp_path = False
    for cl in clusters:
        # Skip clusters whose centroid isn't in a face-masked region
        if bin_face is not None:
            cx = int(np.clip(sum(p[0] for p in cl) / len(cl), 0, hmW - 1))
            cy = int(np.clip(sum(p[1] for p in cl) / len(cl), 0, hmH - 1))
            if not bin_face[cy, cx]:
                continue

        xs = [p[0] for p in cl]
        ys = [p[1] for p in cl]
        x0_kp, x1_kp = min(xs), max(xs)
        y0_kp, y1_kp = min(ys), max(ys)

        # Keypoints only cover the upper face (eyes/ears/nose); extend the
        # bottom of the box downward using the Face segmentation mask so the
        # chin and jaw are included.  Restrict the column range to the cluster's
        # X span (±cluster_radius) so we don't bleed into a neighbouring face.
        y1_seg = y1_kp
        if bin_face is not None:
            col_lo = max(0,    x0_kp - int(cluster_radius))
            col_hi = min(hmW,  x1_kp + int(cluster_radius))
            col_slice = bin_face[:, col_lo:col_hi]  # (hmH, col_width)
            rows_with_face = np.where(col_slice.any(axis=1))[0]
            if rows_with_face.size:
                y0_kp = min(y0_kp, int(rows_with_face[0]))
                y1_seg = max(y1_kp, int(rows_with_face[-1]))

        _save_crop(x0_kp, y0_kp, x1_kp, y1_seg)
        used_kp_path = True

    # ── 4. Fallback: segmentation blobs (no keypoints detected) ─────────────
    if not used_kp_path and face_mask is not None:
        bin_mask = (face_mask >= seg_threshold).astype(np.uint8) * 255
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            bin_mask, connectivity=8)
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] < min_seg_pixels:
                continue
            x0 = stats[lbl, cv2.CC_STAT_LEFT]
            y0 = stats[lbl, cv2.CC_STAT_TOP]
            x1 = x0 + stats[lbl, cv2.CC_STAT_WIDTH]
            y1 = y0 + stats[lbl, cv2.CC_STAT_HEIGHT]
            _save_crop(x0, y0, x1, y1)


# =============================================================================
# Main routine
# =============================================================================

def main_pose_estimation(args):
    model_path          = args.model
    videoWidth, videoHeight = args.size
    threshold           = int(args.threshold)
    keypoint_threshold  = args.threshold
    cropInputFrame      = not args.nocrop
    customCrop          = args.crop is not None
    customCropX         = args.crop[0] if customCrop else 0
    customCropY         = args.crop[1] if customCrop else 0
    customCropSize      = args.crop[2] if customCrop else 0
    scale               = args.scale
    emulateBorder       = args.border
    noise               = np.clip(args.noise, 0.0, 1.0)
    blur                = min(40, abs(args.blur))
    illustrate          = args.illustrate or args.collab
    save                = args.save or args.collab
    show                = not args.headless and not args.collab
    visualize           = not args.headless

    window_arrangement = [(int(x), int(y), label) for x, y, label in args.win]
    monitor = []
    for hm, x, y, label in args.monitor:
        try:
            hm_spec = int(hm)           # numeric index
        except ValueError:
            hm_spec = hm                # label string — resolved in YMAPNet.__init__
        monitor.append((hm_spec, int(x), int(y), label))
        print(f"Added a monitor @ {x},{y} for {hm}")

    print("Keypoint Threshold :", keypoint_threshold)
    print("Threshold          :", threshold)

    cap = getCaptureDeviceFromPath(args.videoFilePath, videoWidth, videoHeight,
                                   model_path=model_path)

    from YMAPNet import YMAPNet, PoseEstimatorTiler
    estimator = YMAPNet(
        modelPath=model_path,
        threshold=threshold,
        keypoint_threshold=keypoint_threshold,
        engine=args.engine,
        profiling=args.profiling,
        illustrate=illustrate,
        pruneTokens=args.pruneTokens,
        monitor=monitor,
        window_arrangement=window_arrangement,
        screen_w=args.screen[0],
        screen_h=args.screen[1],
        depth_iterations=args.depth_iterations,
        estimate_person_id=not args.no_person_id,
        resolve_skeleton=not args.no_skeleton,
    )
    # noise is [0,1]; add_noise_to_image expects the same range
    estimator.addedNoise = noise

    tiler = PoseEstimatorTiler(
        estimator,
        tile_size=(estimator.cfg['inputWidth'], estimator.cfg['inputHeight']),
        overlap=(0, 0),
    )

    if save and show:
        disable_screensaver()
    if show:
        estimator.setup_threshold_control_window()

    face_crop_counter = [0]   # mutable counter shared across frames

    failedFrames = 0
    try:
     while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            failedFrames += 1
            if failedFrames > 100:
                break
            continue

        failedFrames = 0

        if args.tile:
            tiler.process(frame)
        else:
            if scale != 1.0:
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_AREA)

            if cropInputFrame and estimator.cfg['inputWidth'] == estimator.cfg['inputHeight']:
                if customCrop:
                    frame = custom_crop(frame, customCropX, customCropY, customCropSize)
                else:
                    frame = extract_centered_rectangle(frame)

            if emulateBorder > 0:
                bigBorder = frame.shape[0] * (emulateBorder / estimator.cfg['inputHeight'])
                frame = add_horizontal_stripes(frame, int(bigBorder))

            if blur:
                frame = apply_blur_to_image(frame, blur_strength=blur)

            if estimator.addedNoise != 0.0:
                frame = add_noise_to_image(frame, noise_magnitude=estimator.addedNoise)

            estimator.process(frame)

            if args.fallen_person_detector:
                _fp = _run_fallen_person_detector(estimator)
                if _fp is not None:
                    _print_fallen_result(_fp, estimator.frameNumber)
                    if _fp["is_fallen"] and args.save_fallen:
                        _save_fallen_frame(frame, _fp, estimator.frameNumber)

            if args.faces:
                _dump_face_crops(estimator, frame, face_crop_counter)

        if visualize:
            if show:
                estimator.update_thresholds_from_gui()
            if args.tile:
                frameWithVis = frame.copy()
                tiler.visualize(frameWithVis)
            else:
                frameWithVis = frame.copy()
                estimator.visualize(frameWithVis, show=show, save=save)

            key = cv2.waitKey(1) & 0xFF if show else 255
            if key != 255:
                print("Key Press =", key)
            if key == 81:
                print("Left Arrow")
            elif key == 97:
                print("Save demo screenshot")
                subprocess.run(
                    ["scrot", "-a", "800,10,1157,570",
                     f"scrot{estimator.frameNumber}.png"],
                    check=False,
                )
            elif key in (27, ord('q'), ord('Q')):
                print("Terminating after receiving keyboard request")
                break
            elif key in (ord('u'), ord('U')):
                save_and_upload_frame(frame, args.upload_url)
            elif key in (ord('s'), ord('S')):
                create_ply_file(estimator.imageIn, estimator.depthmap,
                                f"output_{estimator.frameNumber}.ply")

            if save and show:
                screenshot(estimator.frameNumber)

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    print("Average Framerate :", np.average(estimator.keypoints_model.hz), "Hz")
    cap.release()
    if show:
        cv2.destroyAllWindows()

    if save and show:
        enable_screensaver()
        subprocess.run([
            "ffmpeg", "-nostdin", "-framerate", "25",
            "-i", "colorFrame_0_%05d.png",
            "-vf", "scale=-1:720", "-y", "-r", "25",
            "-pix_fmt", "yuv420p", "-threads", "8",
            f"{args.videoFilePath}_lastRun3DHiRes.mp4",
        ], check=False)
        for f in glob.glob("colorFrame*.png"):
            os.remove(f)

    if illustrate:
        subprocess.run([
            "ffmpeg", "-nostdin", "-framerate", "25",
            "-i", "composite_%05d.png",
            "-vf", "scale=-1:720", "-y", "-r", "25",
            "-pix_fmt", "yuv420p", "-threads", "8",
            f"{args.videoFilePath}_illustration.mp4",
        ], check=False)
        for f in glob.glob("composite_*.png"):
            os.remove(f)

# =============================================================================
if __name__ == '__main__':
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.fast:
        args.depth_iterations = 0
        args.no_person_id     = True
        args.no_skeleton      = True

    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    if args.screen is None:
        args.screen = list(detect_screen_resolution())

    if args.update:
        subprocess.run(["rm", "-rf", "2d_pose_estimation/"], check=False)
        if os.path.exists("2d_pose_estimation.zip"):
            os.remove("2d_pose_estimation.zip")
        subprocess.run(["wget", "http://ammar.gr/2d_pose_estimation.zip"], check=True)
        subprocess.run(["unzip", "2d_pose_estimation.zip"], check=True)

    if args.fallen_person_detector:
        for f in glob.glob("fallen_*.jpg"):
            os.remove(f)
        print("Cleaned previous fallen_*.jpg files.")

    if args.faces:
        os.makedirs("faces", exist_ok=True)
        #Gather faces for experiment identifying me :)..
        #for f in glob.glob("faces/face_*.jpg"):
        #    os.remove(f)
        #print("Cleaned previous faces/face_*.jpg files.")

    main_pose_estimation(args)
# =============================================================================
