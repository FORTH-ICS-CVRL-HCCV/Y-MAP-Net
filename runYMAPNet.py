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
import argparse
import time
from tools import bcolors, run_tool, rm_rf, download, unzip, upload_file, warn_unsupported_platform

try:
    import cv2
    import numpy as np
except Exception as e:
    print(bcolors.WARNING, "Could not import libraries!", bcolors.ENDC)
    print("An exception occurred:", str(e))
    print("Issue:\n source venv/bin/activate\nBefore running this script")
    sys.exit(1)

# =============================================================================
# Constants
# =============================================================================
MAX_BLUR_KERNEL               = 40                # capped Gaussian kernel size
MAX_CONSECUTIVE_READ_FAILURES = 100               # give up after this many cap.read() misses
FALLBACK_SCREEN_RES           = (3840, 2400)      # used when every detect method fails
DEMO_SCREENSHOT_REGION        = "800,10,1157,570" # scrot crop window for the 'a' hotkey
DEFAULT_UPLOAD_URL            = "http://ammar.gr/datasets/uploads.php"
DEFAULT_MODEL_URL             = "http://ammar.gr/2d_pose_estimation.zip"
MODEL_ZIP_NAME                = "2d_pose_estimation.zip"


# =============================================================================
# Argument parser
# =============================================================================


def build_arg_parser():
    p = argparse.ArgumentParser(description="YMAPNet 2D pose estimation runner")
    p.add_argument("--cpu", action="store_true", help="Force CPU inference")
    p.add_argument("--update", action="store_true", help="Re-download the model weights")
    p.add_argument("--from", dest="videoFilePath", default="webcam", metavar="PATH",
                   help="Source: path, webcam, screen, esp, /dev/videoN")
    p.add_argument("--size", nargs=2, type=int, default=[640, 480], metavar=("W", "H"))
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=84.0, metavar="T")
    p.add_argument("--border", type=int, default=0)
    p.add_argument("--crop", nargs=3, type=int, metavar=("X", "Y", "SIZE"),
                   help="Custom crop: centre X, centre Y, size")
    p.add_argument("--nocrop", action="store_true", help="Disable centre-crop preprocessing")
    p.add_argument("--blur", type=int, default=0, metavar="STRENGTH")
    p.add_argument("--noise", type=float, default=0.0, metavar="MAG", help="Gaussian noise magnitude [0.0-1.0]")
    p.add_argument("--engine", default="tensorflow", metavar="ENGINE")
    p.add_argument("--model", default="2d_pose_estimation", metavar="PATH",
                   help="Model directory or file (default: 2d_pose_estimation)")
    p.add_argument("--save", action="store_true")
    p.add_argument("--prune", action="store_true", dest="pruneTokens")
    p.add_argument("--tile", action="store_true")
    p.add_argument("--illustrate", action="store_true")
    p.add_argument("--collab", action="store_true", help="illustrate + save, no display")
    p.add_argument("--headless", "--novisualization", action="store_true", dest="headless")
    p.add_argument("--profiling", "--profile", action="store_true")
    p.add_argument("--depth-iterations", type=int, default=15, metavar="N",
                   help="Sobel depth-refinement iterations (0 = disabled)")
    p.add_argument("--no-person-id", action="store_true", help="Disable per-blob person ID estimation")
    p.add_argument("--skeleton", action="store_true",
                   help="Enable joint-hierarchy skeleton resolution (off by default)")
    p.add_argument("--fast", action="store_true",
                   help="Shorthand for --depth-iterations 0 --no-person-id (skeleton already off by default)")
    # Repeatable multi-value options
    p.add_argument("--win", nargs=3, action="append", default=[], metavar=("X", "Y", "LABEL"),
                   help="Window arrangement entry (repeatable)")
    p.add_argument("--monitor", nargs=4, action="append", default=[], metavar=("HM", "X", "Y", "LABEL"),
                   help="Heatmap monitor entry (repeatable)")
    p.add_argument("--upload-url", default=DEFAULT_UPLOAD_URL, metavar="URL",
                   help="Frame upload endpoint")
    p.add_argument("--model-url", default=DEFAULT_MODEL_URL, metavar="URL",
                   help="Model zip download URL used with --update")
    p.add_argument("--screen", nargs=2, type=int, default=None, metavar=("W", "H"),
                   help="Physical display resolution for auto window tiling (auto-detected if omitted)")
    p.add_argument("--fallen-person-detector", action="store_true", dest="fallen_person_detector",
                   help="Enable fallen-person detection using Person/Floor/Normal heatmaps")
    p.add_argument("--no-save-fallen", action="store_false", dest="save_fallen", default=True,
                   help="Disable saving frames when a fallen person is detected (saving is ON by default)")
    p.add_argument("--faces", action="store_true",
                   help="Dump RGB crops of every detected face (Face segmentation mask) to faces/")
    p.add_argument("--pose-match", action="store_true", dest="pose_match",
                   help="Enable real-time pose matching demo (press R to capture reference)")
    p.add_argument(
        "--eco", type=float, default=[3.0], nargs="+", metavar=("THRESHOLD", "MAX_SKIP"),
        help="Skip network run when mean pixel diff of the 256x256 input is below "
        "THRESHOLD (0 = disabled). Try 5.0-15.0 for static scenes. "
        "Optional second value MAX_SKIP forces a network run after that many "
        "consecutive skipped frames (default: 20, e.g. --eco 8.0 30).")
    p.add_argument("--vram", type=int, default=4800, metavar="MB",
                   help="GPU VRAM limit in MB for TensorFlow (default: 4800)")
    return p


# =============================================================================
# Live-stream frame-drop sync
# =============================================================================

def _is_live_source(videoFilePath):
    """Return True for real-time camera sources where frame-drop sync should apply."""
    if videoFilePath in ("webcam", "screen", "esp"):
        return True
    if isinstance(videoFilePath, str) and videoFilePath.isdigit():
        return True
    if isinstance(videoFilePath, str) and re.fullmatch(r'/dev/video\d+', videoFilePath):
        return True
    return False


class LiveStreamSyncer:
    """Drops stale frames from a live capture so processing stays in sync."""

    def __init__(self, cap, nominal_fps: float):
        self.cap = cap
        self.fps = max(nominal_fps, 1.0)
        self._t_last = time.perf_counter()

    def sync_start(self):
        """Reset the drift clock to now. Call once after setup (model load, etc.)
        finishes and right before the steady-state read loop begins, so the
        startup gap isn't counted as accumulated lag on the first read."""
        self._t_last = time.perf_counter()

    def read(self):
        now = time.perf_counter()
        elapsed = now - self._t_last
        n_drop = max(0, int(elapsed * self.fps) - 1)
        for _ in range(n_drop):
            self.cap.grab()
        ret, frame = self.cap.read()
        self._t_last = time.perf_counter()
        if n_drop >= 5:
            print(f"[sync] dropped {n_drop} stale frame(s) "
                  f"(lag {elapsed * 1000:.0f} ms @ {self.fps:.1f} fps)")
        return ret, frame


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
        # Keep the internal buffer small so at most one extra frame queues up.
        # Not all backends honour this, but it limits passive lag where supported.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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
    try:
        upload_file(url, 'frame.jpg')
    except Exception as e:
        print(f"Upload failed: {e}")


def prevent_screensaver():
    run_tool(["xdotool", "mousemove_relative", "--", "1", "0"])
    time.sleep(1)
    run_tool(["xdotool", "mousemove_relative", "--", "-1", "0"])


def disable_screensaver():
    run_tool(["xset", "s", "off"])


def enable_screensaver():
    run_tool(["xset", "s", "on"])


def screenshot(framenumber):
    run_tool(["scrot", f"colorFrame_0_{framenumber:05}.png"])


# =============================================================================
# Image processing helpers
# =============================================================================


def create_ply_file(bgr_image, depth_array, filename, depthScale=1.0):
    height, width = depth_array.shape[:2]
    if bgr_image.shape[:2] != (height, width):
        bgr_image = cv2.resize(bgr_image, (width, height), interpolation=cv2.INTER_LINEAR)

    header = (f"ply\nformat ascii 1.0\nelement vertex {height * width}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    zs = depth_array * depthScale
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
    x1 = min(img_width, cx + side // 2)
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
    x1 = min(img_width, cX + size)
    y1 = min(img_height, cY + size)
    return image[y0:y1, x0:x1]


def add_horizontal_stripes(image: np.ndarray, stripe_height: int) -> np.ndarray:
    """Paint black bars of *stripe_height* pixels at the top and bottom of *image*."""
    max_stripe_height = image.shape[0] // 2
    if stripe_height > max_stripe_height:
        raise ValueError(f"Stripe height cannot exceed half the image height ({max_stripe_height}).")
    result = np.copy(image)
    result[:stripe_height, :] = 0
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

    Falls back to FALLBACK_SCREEN_RES if every method fails.
    """
    # 1. xrandr — bounding box of all active (mode-set) monitors
    try:
        out = run_tool(["xrandr"], capture_output=True, timeout=3)
        if out is not None:
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
        out = run_tool(["xdpyinfo"], capture_output=True, timeout=3)
        if out is not None:
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

    print(f"Could not detect screen resolution; using fallback "
          f"{FALLBACK_SCREEN_RES[0]}x{FALLBACK_SCREEN_RES[1]}")
    return FALLBACK_SCREEN_RES


from appFallDetection import _run_fallen_person_detector, _save_fallen_frame, _print_fallen_result
from appFace import _dump_face_crops
from appPoseMatch import PoseMatcher, draw_pose_match_overlay

# =============================================================================
# Main routine helpers
# =============================================================================


def parse_monitor_args(monitor_args):
    """Convert --monitor entries to (hm_spec, x, y, label) tuples.

    The heatmap selector is either a numeric index or a label string;
    label resolution happens inside YMAPNet.__init__.
    """
    monitors = []
    for hm, x, y, label in monitor_args:
        try:
            hm_spec = int(hm)  # numeric index
        except ValueError:
            hm_spec = hm  # label string — resolved in YMAPNet.__init__
        monitors.append((hm_spec, int(x), int(y), label))
        print(f"Added a monitor @ {x},{y} for {hm}")
    return monitors


def build_estimator(args):
    """Construct the YMAPNet estimator and its tile wrapper from CLI args."""
    from YMAPNet import YMAPNet, PoseEstimatorTiler

    illustrate = args.illustrate or args.collab
    show = not args.headless and not args.collab
    window_arrangement = [(int(x), int(y), label) for x, y, label in args.win]
    monitor = parse_monitor_args(args.monitor)

    estimator = YMAPNet(
        modelPath=args.model,
        threshold=int(args.threshold),
        keypoint_threshold=args.threshold,
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
        resolve_skeleton=args.skeleton,
        vram_limit=args.vram,
        compileModel=False,  # skip optimizer state loading — not needed for inference
        show=show,
        addedNoise=float(np.clip(args.noise, 0.0, 1.0)))

    tiler = PoseEstimatorTiler(
        estimator,
        tile_size=(estimator.cfg['inputWidth'], estimator.cfg['inputHeight']),
        overlap=(0, 0),
    )
    return estimator, tiler


def preprocess_frame(frame, args, estimator):
    """Apply scale, crop, border, blur, and noise pre-processing to a frame."""
    if args.scale != 1.0:
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (int(w * args.scale), int(h * args.scale)),
                           interpolation=cv2.INTER_AREA)

    cropInputFrame = not args.nocrop
    if cropInputFrame and estimator.cfg['inputWidth'] == estimator.cfg['inputHeight']:
        if args.crop is not None:
            frame = custom_crop(frame, args.crop[0], args.crop[1], args.crop[2])
        else:
            frame = extract_centered_rectangle(frame)

    if args.border > 0:
        bigBorder = frame.shape[0] * (args.border / estimator.cfg['inputHeight'])
        frame = add_horizontal_stripes(frame, int(bigBorder))

    blur = min(MAX_BLUR_KERNEL, abs(args.blur))
    if blur:
        frame = apply_blur_to_image(frame, blur_strength=blur)

    if estimator.addedNoise != 0.0:
        frame = add_noise_to_image(frame, noise_magnitude=estimator.addedNoise)

    return frame


# OpenCV waitKey() & 0xFF return values. 255 means no key was pressed in the
# wait window. The asymmetry below (LEFT_ARROW/SCREENSHOT vs the rest) is
# intentional: only the action keys accept both cases.
KEY_NONE        = 255
KEY_LEFT_ARROW  = {81}
KEY_SCREENSHOT  = {ord('a')}
KEY_QUIT        = {27, ord('q'), ord('Q')}
KEY_RECORD      = {ord('r'), ord('R')}
KEY_UPLOAD      = {ord('u'), ord('U')}
KEY_SAVE_PLY    = {ord('s'), ord('S')}


def handle_keypress(key, args, estimator, frame, pose_matcher):
    """Process a single OpenCV keypress. Returns True if the loop should exit."""
    if key == KEY_NONE:
        return False
    print("Key Press =", key)
    if key in KEY_LEFT_ARROW:
        print("Left Arrow")
    elif key in KEY_SCREENSHOT:
        print("Save demo screenshot")
        run_tool(["scrot", "-a", DEMO_SCREENSHOT_REGION,
                  f"scrot{estimator.frameNumber}.png"])
    elif key in KEY_QUIT:
        print("Terminating after receiving keyboard request")
        return True
    elif key in KEY_RECORD:
        if pose_matcher is not None:
            pose_matcher.capture_reference()
    elif key in KEY_UPLOAD:
        save_and_upload_frame(frame, args.upload_url)
    elif key in KEY_SAVE_PLY:
        create_ply_file(estimator.imageIn, estimator.depthmap,
                        f"output_{estimator.frameNumber}.ply")
    return False


def encode_video(input_pattern, output, cleanup_glob, fps=25):
    """Run ffmpeg over a numbered PNG sequence and remove the source frames."""
    run_tool([
        "ffmpeg", "-nostdin",
        "-framerate", str(fps),
        "-i", input_pattern,
        "-vf", "scale=-1:720",
        "-y", "-r", str(fps),
        "-pix_fmt", "yuv420p",
        "-threads", "8",
        output,
    ])
    for f in glob.glob(cleanup_glob):
        os.remove(f)


# =============================================================================
# Main routine
# =============================================================================


def main_pose_estimation(args):
    videoWidth, videoHeight = args.size
    illustrate = args.illustrate or args.collab
    save = args.save or args.collab
    show = not args.headless and not args.collab
    visualize = not args.headless

    print("Keypoint Threshold :", args.threshold)
    print("Threshold          :", int(args.threshold))

    cap = getCaptureDeviceFromPath(args.videoFilePath, videoWidth, videoHeight,
                                   model_path=args.model)

    is_live = _is_live_source(args.videoFilePath)
    if is_live:
        nominal_fps = cap.get(cv2.CAP_PROP_FPS) if hasattr(cap, 'get') else 30.0
        nominal_fps = nominal_fps if nominal_fps > 0 else 30.0
        _syncer = LiveStreamSyncer(cap, nominal_fps)
        print(f"[sync] Live source — frame-drop sync enabled at {nominal_fps:.1f} fps")

    estimator, tiler = build_estimator(args)

    if save and show:
        disable_screensaver()
    if show:
        estimator.setup_threshold_control_window()

    face_crop_counter = [0]  # mutable counter shared across frames
    pose_matcher = PoseMatcher(estimator) if args.pose_match else None

    eco_threshold = args.eco[0]
    eco_max_skip = int(args.eco[1]) if len(args.eco) > 1 else 20

    if is_live:
        _syncer.sync_start()

    failedFrames = 0
    try:
        while True:
            ret, frame = _syncer.read() if is_live else cap.read()
            if not ret:
                print("Failed to capture frame")
                failedFrames += 1
                if failedFrames > MAX_CONSECUTIVE_READ_FAILURES:
                    break
                continue

            failedFrames = 0

            if args.tile:
                tiler.process(frame)
            else:
                frame = preprocess_frame(frame, args, estimator)
                estimator.process(frame, static_frame_threshold=eco_threshold,
                                  eco_max_skip=eco_max_skip)

                if args.fallen_person_detector:
                    _fp_result = _run_fallen_person_detector(estimator)
                    if _fp_result is not None:
                        _fp, _union_human, _floor_mask, _furniture_mask = _fp_result
                        _print_fallen_result(_fp, estimator.frameNumber)
                        if _fp["is_fallen"] and args.save_fallen:
                            _save_fallen_frame(frame, _fp, estimator.frameNumber, _union_human, _floor_mask,
                                               _furniture_mask)

                if args.faces:
                    _dump_face_crops(estimator, frame, face_crop_counter)

                if pose_matcher is not None:
                    pm_result = pose_matcher.tick()
                    if pm_result is not None:
                        draw_pose_match_overlay(frame, pm_result, estimator, pose_matcher)

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
                if handle_keypress(key, args, estimator, frame, pose_matcher):
                    break

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
        encode_video("colorFrame_0_%05d.png",
                     f"{args.videoFilePath}_lastRun3DHiRes.mp4",
                     "colorFrame*.png")

    if illustrate:
        encode_video("composite_%05d.png",
                     f"{args.videoFilePath}_illustration.mp4",
                     "composite_*.png")


# =============================================================================
if __name__ == '__main__':
    parser = build_arg_parser()
    args = parser.parse_args()

    warn_unsupported_platform()

    if args.fast:
        args.depth_iterations = 0
        args.no_person_id = True
        args.skeleton = False  # --fast forces skeleton off even if --skeleton was passed

    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    if args.screen is None:
        args.screen = list(detect_screen_resolution())

    if args.update:
        rm_rf("2d_pose_estimation")
        rm_rf(MODEL_ZIP_NAME)
        print(f"Downloading {args.model_url} -> {MODEL_ZIP_NAME}")
        download(args.model_url, MODEL_ZIP_NAME)
        print(f"Extracting {MODEL_ZIP_NAME}")
        unzip(MODEL_ZIP_NAME)

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
