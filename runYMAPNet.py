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
    return p

# =============================================================================
# Video capture factory
# =============================================================================

def getCaptureDeviceFromPath(videoFilePath, videoWidth, videoHeight, videoFramerate=30):
    def _open_v4l2(index):
        cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_FPS, videoFramerate)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, videoWidth)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, videoHeight)
        return cap

    if videoFilePath == "esp":
        from espStream import ESP32CamStreamer
        return ESP32CamStreamer()
    elif videoFilePath == "screen":
        from screenStream import ScreenGrabber
        return ScreenGrabber(region=(0, 0, videoWidth, videoHeight))
    elif videoFilePath == "webcam":
        return _open_v4l2(0)
    else:
        m = re.fullmatch(r'/dev/video(\d+)', videoFilePath)
        if m:
            return _open_v4l2(int(m.group(1)))
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
# Main routine
# =============================================================================

def main_pose_estimation(args):
    model_path          = '2d_pose_estimation'
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

    cap = getCaptureDeviceFromPath(args.videoFilePath, videoWidth, videoHeight)

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

    main_pose_estimation(args)
# =============================================================================
