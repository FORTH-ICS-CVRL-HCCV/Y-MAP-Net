#!/usr/bin/python3

"""
Author : "Ammar Qammaz"
Copyright : "2024 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"
"""

#Dependencies should be :
#tensorflow-2.15.0 needs CUDA 12.2, CUDNN 8.9.4 and is built with Clang 16.0.0 and Bazel 6.1.0 
#python3 -m pip install tensorflow==2.15.0 numpy tensorboard opencv-python wget

import os
import sys
import json
from tools import bcolors
#----------------------------------------------------------------------------------------
try:
 import cv2
 import numpy as np
 from numpy.linalg import norm
 #import keras
 #import tensorflow as tf
 from createJSONConfiguration import loadJSONConfiguration
 from imageProcessing import castNormalizedCoordinatesToOriginalImage,castNormalizedBBoxCoordinatesToOriginalImage,resize_image_no_borders,resize_image_with_borders
 from resolveJointHierarchy import resolveJointHierarchyNew,drawSkeletons
 
 from calculateNormalsFromDepthmap import compute_normals, rgb_to_grayscale, apply_bilateral_filter, improve_depthmap, integrate_normals
except Exception as e:
 print(bcolors.WARNING,"Could not import libraries!",bcolors.ENDC)
 print("An exception occurred:", str(e))
 print("Issue:\n source venv/bin/activate")
 print("Before running this script")
 sys.exit(1)

#----------------------------------------------------------------------------------------
def loadJSON(json_file_path):
  import json
  data = dict()
  # Read the JSON file and load its contents into a dictionary
  with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)
  return data

def get_key_from_index(d, index):
    """
    Given a dictionary and an index, returns the key name corresponding to the index.
    
    Parameters:
    d (dict): The dictionary with keys as names and values as indices.
    index (int): The index for which the corresponding key name is to be returned.
    
    Returns:
    str: The key name corresponding to the given index. If the index is not found, returns 'Index not found'.
    """
    for key, value in d.items():
        if value == index:
            return key
    return 'Index not found'
#----------------------------------------------------------------------------------------
def preprocess_image(frame, target_size=(128, 128), add_borders=False):
    # Resize the frame to the target size and normalize pixel values
    image = frame.astype(np.uint8)
    if (add_borders):
          print("Going to ",target_size," also adding borders")
          image, keypointXMultiplier, keypointYMultiplier, keypointXOffset, keypointYOffset  = resize_image_with_borders(image,target_size)
          keypointXMultiplier = 1.0
          keypointYMultiplier = 1.0
          keypointXOffset     = 0
          keypointYOffset     = 0
          print("Recovered image size ",image.shape," ")
          print("keypointXYMultiplier ",keypointXMultiplier,",",keypointYMultiplier)
          print("keypointXYOffset ",keypointXOffset,",",keypointYOffset)
    else:
          #print("Going to ",target_size," without borders")
          image, keypointXMultiplier, keypointYMultiplier, keypointXOffset, keypointYOffset  = resize_image_no_borders(image,target_size)
          #print("Recovered image size ",image.shape," ")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    return image_rgb, keypointXMultiplier, keypointYMultiplier, keypointXOffset, keypointYOffset
#----------------------------------------------------------------------------------------
def apply_bilateral_filter(normals):
    # Apply bilateral filter to smooth the image while preserving edges
    smoothed_normals = cv2.bilateralFilter(normals, d=3, sigmaColor=75, sigmaSpace=75)
    return smoothed_normals
#----------------------------------------------------------------------------------------
def subpixel_refinement(channel_heatmap, max_loc, neighborhood_size):
    # Extract the neighborhood around the peak
    y, x = int(max_loc[1]), int(max_loc[0])
    neighborhood = channel_heatmap[
                                    max(0, y - neighborhood_size):min(channel_heatmap.shape[0], y + neighborhood_size + 1),
                                    max(0, x - neighborhood_size):min(channel_heatmap.shape[1], x + neighborhood_size + 1)
                                  ]

    # Calculate the centroid of the neighborhood
    y_offset, x_offset = np.unravel_index(np.argmax(neighborhood), neighborhood.shape)
    refined_loc        = (x - neighborhood_size + x_offset, y - neighborhood_size + y_offset)
    
    return refined_loc
#----------------------------------------------------------------------------------------
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
#----------------------------------------------------------------------------------------
def flip_heatmap_values(channel_heatmap):
    # Convert the heatmap to float32 to avoid overflow during flipping
    flipped_heatmap = channel_heatmap.copy().astype(np.float32)
    
    # Flip the heatmap values
    flipped_heatmap = flipped_heatmap * -1
    
    # Cast the flipped heatmap back to signed char
    flipped_heatmap = flipped_heatmap.astype(np.int8)
    
    return flipped_heatmap
#----------------------------------------------------------------------------------------
def find_peak_pointsSINGLE(keypoint_predictions, threshold):
    heatmaps = np.array(keypoint_predictions).astype(np.float32) + 120
    result = list()
 
    for i in range(heatmaps.shape[2]):
        result.append(list())

    # Iterate through each channel of the heatmap
    for i in range(heatmaps.shape[2]):
        if i<17:
           channel_heatmap = heatmaps[:, :, i]

           #if (i%2==0):
           #   channel_heatmap = flip_heatmap_values(channel_heatmap)           

           # Find the maximum value and its location in the channel heatmap
           _, max_val, _, max_loc = cv2.minMaxLoc(channel_heatmap)

           # Append the peak point to the list
           if (max_val>threshold):
             refined_loc = subpixel_refinement(channel_heatmap,max_loc,3)
             result[i].append((refined_loc[0]/heatmaps.shape[0], refined_loc[1]/heatmaps.shape[1], max_val))

    return result
#----------------------------------------------------------------------------------------
def find_peak_points(keypoint_predictions, threshold):
    heatmaps = np.array(keypoint_predictions).astype(np.float32) + 120
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
#----------------------------------------------------------------------------------------
def find_peak_points_from_convertIO(heatmap_outputs, threshold):
    result = []

    for i, heatmap in enumerate(heatmap_outputs):
        result.append([])

        if i < 17:
            # Threshold the heatmap
            _, binary = cv2.threshold(heatmap, threshold, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Normalize coordinates to [0,1] and include confidence
                    norm_x = cx / heatmap.shape[1]
                    norm_y = cy / heatmap.shape[0]
                    confidence = heatmap[cy, cx]
                    result[i].append((norm_x, norm_y, confidence))

    return result
#----------------------------------------------------------------------------------------
def detect_blobs(heatmap, threshold=120, minblob = 3):
    image = np.array(heatmap).astype(np.float32)
    image = (255 + image).astype(np.uint8) 

    # Threshold the image
    _ , thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV) #

    kernel = np.ones((5, 5), np.uint8) 
    thresh = cv2.erode(thresh, kernel, iterations=1) 

    #Active = 255 , Deactive = 0 
    thresh = 255 - thresh

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create list to store blob information
    blobs = []

    heatmapWidth  = heatmap.shape[1]
    heatmapHeight = heatmap.shape[0]

    # Loop through contours and extract blob information
    for contour in contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        x = float(x / heatmapWidth)
        y = float(y / heatmapHeight)
        w = float(w / heatmapWidth)
        h = float(h / heatmapHeight) 
        
        # Add the blob information to the list
        #if (w>minblob) and (h>minblob):
        blobs.append( (x,y,w,h) ) 

    for blob in blobs:
                 x,y,w,h = blob
                 x = int(x * heatmapWidth)
                 y = int(y * heatmapHeight)
                 w = int(w * heatmapWidth)
                 h = int(h * heatmapHeight) 
                 cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,255),3) 

    cv2.imshow('Blobs', image)
    cv2.imshow('Blobs Thresholded', thresh)

    return blobs
#----------------------------------------------------------------------------------------
def segment_and_label_persons(heatmap, threshold=180, min_area=1050, debug=False):
    """
    Segments blobs from a heatmap in range [0, 255], filters small ones, and returns unique-labeled regions.
    
    Args:
        heatmap (np.ndarray): 2D np.uint8 or float heatmap image.
        threshold (int): Minimum intensity to consider a pixel part of a person blob.
        min_area (int): Minimum pixel area to keep a blob.
        debug (bool): If True, shows the binary mask and prints stats.
    
    Returns:
        labeled_map (np.ndarray): 2D array with each blob labeled with a unique int.
        bounding_boxes (list): List of (x, y, w, h) bounding boxes for each blob.
    """
    if debug:
        print("Heatmap range: min =", heatmap.min(), "max =", heatmap.max())

    # Apply threshold
    binary_mask = (heatmap >= threshold).astype(np.uint8)

    if debug:
        debug_mask = binary_mask * 255
        cv2.imshow("Binary Mask", debug_mask)
        cv2.waitKey(1)

    # Label connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    labeled_map = np.zeros_like(labels, dtype=np.int32)
    bounding_boxes = []

    current_label = 1
    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            labeled_map[labels == i] = current_label
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            bounding_boxes.append((x, y, w, h))
            current_label += 1

    if debug:
        print(f"Found {current_label - 1} valid blobs (threshold={threshold}, min_area={min_area})")

    return labeled_map, bounding_boxes
#----------------------------------------------------------------------------------------
def draw_floor_normal_axes(image, floor_normal_result, scale=60, margin=14):
    """
    Draw an XYZ axis indicator showing the floor plane normal on *image* (in-place).

    A small 3D-axis widget is rendered in the bottom-right corner:
      • Red   arrow  →  X axis reference  (+X right in camera space)
      • Green arrow  →  Y axis reference  (+Y up   in camera space)
      • Blue  arrow  →  Z axis reference  (+Z into scene, shown foreshortened)
      • White arrow  →  actual floor normal vector projected onto the same axes

    The normal components also appear as a text legend beside the widget.

    Parameters
    ----------
    image : np.ndarray  [H, W, 3]  BGR uint8 — modified in-place.
    floor_normal_result : dict returned by compute_floor_plane_normal().
    scale  : int  — arrow length in pixels for a unit-length vector.
    margin : int  — gap from the image edge to the widget origin.
    """
    if floor_normal_result is None or floor_normal_result.get("normal") is None:
        return image

    H, W = image.shape[:2]
    nx, ny, nz = floor_normal_result["normal"]
    tilt        = floor_normal_result["tilt_deg"]
    conf        = floor_normal_result["confidence"]
    floor_px    = floor_normal_result.get("floor_pixels",   floor_normal_result.get("floor_pixel_count", 0))
    ceil_px     = floor_normal_result.get("ceiling_pixels", 0)
    wall_px     = floor_normal_result.get("wall_pixels",    0)
    sources     = floor_normal_result.get("sources", ["floor"])

    # ── widget origin (bottom-right corner) ──────────────────────────────────
    ox = W - margin - scale - 5
    oy = H - margin - scale - 5

    # ── 3-D → 2-D projection (simple cabinet/oblique: X right, Y up, Z diagonal) ──
    # Each unit 3-D vector maps to a 2-D displacement in pixel space.
    def proj(vx, vy, vz):
        px = int( vx * scale - vz * scale * 0.35)
        py = int(-vy * scale + vz * scale * 0.35)   # Y up = negative pixel-Y
        return px, py

    axes = [
        ((1, 0, 0), (0,   0, 255), "X"),   # red
        ((0, 1, 0), (0, 255,   0), "Y"),   # green
        ((0, 0, 1), (255,  0,   0), "B"),  # blue  (OpenCV BGR: blue = (255,0,0))
    ]

    font       = cv2.FONT_HERSHEY_SIMPLEX
    fscale     = max(0.38, W / 1800)
    thick      = max(1, int(fscale * 2))
    tip_radius = max(3, int(scale * 0.08))

    # Dark background panel for the widget
    pad = 6
    x0p, y0p = ox - scale - pad, oy - scale - pad
    x1p, y1p = ox + scale + pad, oy + scale + pad + int(scale * 0.4)
    cv2.rectangle(image, (max(0, x0p), max(0, y0p)),
                         (min(W-1, x1p), min(H-1, y1p)), (20, 20, 20), -1)

    # Reference axes
    for (ax, ay, az), color, lbl in axes:
        dx, dy = proj(ax, ay, az)
        tip = (ox + dx, oy + dy)
        cv2.arrowedLine(image, (ox, oy), tip, (0, 0, 0), thick + 2, cv2.LINE_AA, tipLength=0.25)
        cv2.arrowedLine(image, (ox, oy), tip, color,     thick,     cv2.LINE_AA, tipLength=0.25)

    # Floor normal vector (white with black outline)
    ndx, ndy = proj(nx, ny, nz)
    ntip = (ox + ndx, oy + ndy)
    cv2.arrowedLine(image, (ox, oy), ntip, (0,   0,   0), thick + 3, cv2.LINE_AA, tipLength=0.30)
    cv2.arrowedLine(image, (ox, oy), ntip, (255, 255, 255), thick + 1, cv2.LINE_AA, tipLength=0.30)

    # Origin dot
    cv2.circle(image, (ox, oy), tip_radius, (200, 200, 200), -1, cv2.LINE_AA)

    # Text legend to the left of the widget
    src_label = "+".join(sources) if sources else "none"
    lines = [
        (f"scene normal",           (200, 200, 200)),
        (f"src: {src_label}",       (160, 160, 160)),
        (f"X {nx:+.2f}",            (0,   0, 255)),
        (f"Y {ny:+.2f}",            (0, 255,   0)),
        (f"Z {nz:+.2f}",            (255, 50,  50)),
        (f"tilt {tilt:.1f}\u00b0",  (200, 200, 200)),
        (f"fl:{floor_px} cl:{ceil_px} wl:{wall_px}", (130, 130, 130)),
    ]
    lh   = int(fscale * 22)+20
    tx   = max(4, x0p - int(fscale * 115))
    ty0  = y0p + lh
    for i, (txt, col) in enumerate(lines):
        y = ty0 + i * lh
        cv2.putText(image, txt, (tx,  y), font, fscale, (0, 0, 0), thick + 1, cv2.LINE_AA)
        cv2.putText(image, txt, (tx,  y), font, fscale, col,        thick,     cv2.LINE_AA)

    return image
#----------------------------------------------------------------------------------------
def compute_floor_plane_normal(normal_x, normal_y, normal_z, floor_mask,
                                ceiling_mask=None, wall_mask=None,
                                floor_threshold=30, ceiling_threshold=30, wall_threshold=30,
                                min_pixels=50):
    """
    Estimate the scene gravity / world-up normal vector by fusing surface normals
    from all available planar surfaces: floor, ceiling and walls.

    Sign convention (all contributions are aligned to point in the +Y / world-up direction
    before blending):
      • Floor   normals are used as-is   — they already point upward (+Y).
      • Ceiling normals are sign-flipped  — they point downward (-Y), so negating them
                                            aligns them with the floor contribution.
      • Wall    normals are used as-is   — they are roughly horizontal so their Y
                                            component is small, but they still constrain
                                            the X/Z orientation of the scene frame.

    All three sources are weighted by their pixel count before averaging, so a large
    visible floor dominates over a thin strip of ceiling, etc.

    The normal maps are assumed to be stored in [0, 255] uint8/float range and are
    remapped to [-1, 1] before averaging.  The result is a unit vector.

    Parameters
    ----------
    normal_x, normal_y, normal_z : np.ndarray [H, W]
        Surface normal component heatmaps (heatmapsOut[chanNormalX/Y/Z]).
    floor_mask   : np.ndarray [H, W]   — Floor   segmentation heatmap (required).
    ceiling_mask : np.ndarray [H, W]   — Ceiling segmentation heatmap (optional).
    wall_mask    : np.ndarray [H, W]   — Wall    segmentation heatmap (optional).
    floor_threshold / ceiling_threshold / wall_threshold : int
        Minimum heatmap value to count a pixel as that surface type.
    min_pixels : int
        Minimum total contributing pixels across all sources.

    Returns
    -------
    dict with keys:
        normal          : (nx, ny, nz) unit-length tuple, or None if insufficient data
        confidence      : float in [0, 1] — total contributing pixels / image area
        floor_pixels    : int
        ceiling_pixels  : int
        wall_pixels     : int
        tilt_deg        : float — angle between the estimated normal and world-up (0,1,0);
                          0° = perfectly level camera
        sources         : list[str] — which surfaces contributed ('floor','ceiling','wall')
    """
    H, W = floor_mask.shape

    # ── binary masks ─────────────────────────────────────────────────────────
    floor_bin   = (floor_mask > floor_threshold)
    ceil_bin    = (ceiling_mask  > ceiling_threshold) if ceiling_mask  is not None else np.zeros((H, W), dtype=bool)
    wall_bin    = (wall_mask     > wall_threshold)    if wall_mask     is not None else np.zeros((H, W), dtype=bool)

    floor_px   = int(floor_bin.sum())
    ceil_px    = int(ceil_bin.sum())
    wall_px    = int(wall_bin.sum())
    total_px   = floor_px + ceil_px + wall_px

    if total_px < min_pixels:
        return dict(normal=None, confidence=0.0,
                    floor_pixels=floor_px, ceiling_pixels=ceil_px, wall_pixels=wall_px,
                    tilt_deg=0.0, sources=[])

    # ── remap [0, 255] → [-1, 1] ──────────────────────────────────────────────
    nx = normal_x.astype(np.float32) / 127.5 - 1.0
    ny = normal_y.astype(np.float32) / 127.5 - 1.0
    nz = normal_z.astype(np.float32) / 127.5 - 1.0

    # ── weighted sum of normals (pixel-count weights) ─────────────────────────
    sum_nx = sum_ny = sum_nz = 0.0
    sources = []

    if floor_px > 0:
        # Floor normals point upward — use as-is
        sum_nx += float(nx[floor_bin].sum())
        sum_ny += float(ny[floor_bin].sum())
        sum_nz += float(nz[floor_bin].sum())
        sources.append("floor")

    if ceil_px > 0:
        # Ceiling normals point downward — negate to align with world-up
        sum_nx -= float(nx[ceil_bin].sum())
        sum_ny -= float(ny[ceil_bin].sum())
        sum_nz -= float(nz[ceil_bin].sum())
        sources.append("ceiling")

    if wall_px > 0:
        # Wall normals are horizontal — they constrain X/Z but barely affect Y;
        # included to improve scene-frame orientation when floor/ceiling are scarce.
        sum_nx += float(nx[wall_bin].sum())
        sum_ny += float(ny[wall_bin].sum())
        sum_nz += float(nz[wall_bin].sum())
        sources.append("wall")

    avg_nx = sum_nx / total_px
    avg_ny = sum_ny / total_px
    avg_nz = sum_nz / total_px

    # ── normalise to unit length ───────────────────────────────────────────────
    length = (avg_nx**2 + avg_ny**2 + avg_nz**2) ** 0.5
    if length < 1e-6:
        return dict(normal=None, confidence=0.0,
                    floor_pixels=floor_px, ceiling_pixels=ceil_px, wall_pixels=wall_px,
                    tilt_deg=0.0, sources=sources)

    unit = (avg_nx / length, avg_ny / length, avg_nz / length)

    # ── tilt: angle between estimated normal and world-up (0, 1, 0) ──────────
    dot_up   = abs(unit[1])
    tilt_deg = float(np.degrees(np.arccos(np.clip(dot_up, 0.0, 1.0))))

    confidence = total_px / float(H * W)

    return dict(
        normal         = unit,
        confidence     = round(confidence, 4),
        floor_pixels   = floor_px,
        ceiling_pixels = ceil_px,
        wall_pixels    = wall_px,
        tilt_deg       = round(tilt_deg, 2),
        sources        = sources,
    )
#----------------------------------------------------------------------------------------
def detect_fallen_person(person_mask, normal_x, normal_y, normal_z, floor_mask,
                          furniture_mask=None,
                          person_threshold=30,
                          floor_threshold=30,
                          furniture_threshold=30,
                          aspect_ratio_threshold=0.80,
                          orientation_angle_threshold=35.0,
                          floor_overlap_threshold=0.20,
                          normal_horiz_threshold=0.45,
                          furniture_overlap_threshold=0.25,
                          min_bbox_area=400):
    """
    Heuristically determine whether the visible person has fallen.

    Inputs (all 2-D numpy arrays of the same spatial resolution):
        person_mask    -- Person segmentation heatmap  (heatmapsOut[chanPerson], uint8/float)
        normal_x/y/z   -- Surface normal components    (heatmapsOut[chanNormalX/Y/Z], 0-255)
        floor_mask     -- Floor segmentation heatmap   (heatmapsOut[chanFloor], uint8/float)
        furniture_mask -- Furniture segmentation heatmap (heatmapsOut[chanFurniture], optional)
                          When provided, high person/furniture overlap raises the
                          'on_furniture' signal which can veto the fallen verdict
                          (person is likely sleeping/resting on furniture, not fallen).

    Thresholds (all keyword-overridable):
        person_threshold        : min heatmap value to count as a person pixel
        floor_threshold         : min heatmap value to count as a floor pixel
        furniture_threshold     : min heatmap value to count as a furniture pixel
        aspect_ratio_threshold  : bbox width/height above which the person is considered
                                  horizontal  (1.0 = square, <1 = tall = standing)
        orientation_angle_threshold : degrees from vertical of the blob's principal axis
                                  above which the person is considered fallen  (0=upright, 90=flat)
        floor_overlap_threshold : fraction of person pixels that must also be floor pixels
                                  for the floor-contact signal to fire
        normal_horiz_threshold  : fraction of person pixels whose surface normal is more
                                  horizontal than vertical (high = person lying flat)
        furniture_overlap_threshold : fraction of person pixels overlapping furniture above
                                  which the person is considered on furniture (not fallen)
        min_bbox_area           : minimum bounding-box area in heatmap pixels below which
                                  the detection is ignored (person too small / distant)

    Returns a dict with:
        is_fallen           bool   -- overall verdict (majority vote of sub-signals)
        aspect_ratio        float  -- bbox W/H  (>1 wide, <1 tall)
        orientation_deg     float  -- degrees of long axis from vertical (90 = fully horizontal)
        floor_overlap       float  -- fraction of person pixels on floor
        normal_horiz_frac   float  -- fraction of person pixels with horizontal surface normal
        furniture_overlap   float  -- fraction of person pixels on furniture (0 if no furniture mask)
        centroid_y_frac     float  -- person centroid Y / image height  (1 = bottom of frame)
        person_pixel_count  int    -- number of person pixels above threshold
        signals             dict   -- individual boolean sub-signals that voted
    """
    H, W = person_mask.shape

    # ── binary masks ──────────────────────────────────────────────────────────
    p_bin  = (person_mask > person_threshold).astype(np.uint8)
    fl_bin = (floor_mask  > floor_threshold ).astype(np.uint8)
    fu_bin = (furniture_mask > furniture_threshold).astype(np.uint8) if furniture_mask is not None else None

    person_pixel_count = int(p_bin.sum())
    if person_pixel_count == 0:
        return dict(is_fallen=False, aspect_ratio=0.0, orientation_deg=0.0,
                    floor_overlap=0.0, normal_horiz_frac=0.0, furniture_overlap=0.0,
                    centroid_y_frac=0.0, person_pixel_count=0, signals={})

    # ── 1. bounding-box aspect ratio ──────────────────────────────────────────
    ys, xs = np.where(p_bin)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bbox_w = float(x1 - x0 + 1)
    bbox_h = float(y1 - y0 + 1)
    bbox_area = bbox_w * bbox_h
    if bbox_area < min_bbox_area:
        return dict(is_fallen=False, aspect_ratio=round(bbox_w / max(bbox_h, 1.0), 3),
                    orientation_deg=0.0, floor_overlap=0.0, normal_horiz_frac=0.0,
                    furniture_overlap=0.0, centroid_y_frac=0.0,
                    person_pixel_count=person_pixel_count,
                    bbox_heatmap=(x0, y0, x1, y1), heatmap_size=(W, H), signals={})
    aspect_ratio = bbox_w / max(bbox_h, 1.0)

    # ── 2. principal-axis orientation via image moments ───────────────────────
    #    mu20, mu02, mu11 give the covariance ellipse; theta is the long-axis angle.
    M = cv2.moments(p_bin)
    mu20 = M["mu20"]; mu02 = M["mu02"]; mu11 = M["mu11"]
    # angle of principal axis from vertical, in [0, 90]
    if (mu20 - mu02) == 0 and mu11 == 0:
        orientation_deg = 0.0
    else:
        theta_rad = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)  # from horizontal
        orientation_deg = abs(np.degrees(theta_rad))            # 0=vertical, 90=horizontal

    # centroid
    cx = M["m10"] / max(M["m00"], 1)
    cy = M["m01"] / max(M["m00"], 1)
    centroid_y_frac = cy / H

    # ── 3. floor overlap ──────────────────────────────────────────────────────
    floor_overlap = float((p_bin & fl_bin).sum()) / person_pixel_count

    # ── 3b. furniture overlap ─────────────────────────────────────────────────
    #    High overlap means the person is likely on a bed/sofa (sleeping), not fallen.
    if fu_bin is not None:
        furniture_overlap = float((p_bin & fu_bin).sum()) / person_pixel_count
    else:
        furniture_overlap = 0.0

    # ── 4. normal orientation within person region ────────────────────────────
    #    Normals are stored 0-255; map to [-1, 1].
    nx = normal_x.astype(np.float32) / 127.5 - 1.0
    ny = normal_y.astype(np.float32) / 127.5 - 1.0
    nz = normal_z.astype(np.float32) / 127.5 - 1.0

    # Horizontal normal: |ny| dominates (world-up direction).
    # For a standing person the body faces the camera (nz dominant);
    # for a fallen person the body faces up like the floor (ny dominant).
    p_idx = p_bin.astype(bool)
    nx_p = nx[p_idx]; ny_p = ny[p_idx]; nz_p = nz[p_idx]
    abs_ny = np.abs(ny_p)
    abs_nz = np.abs(nz_p)
    normal_horiz_frac = float((abs_ny > abs_nz).sum()) / person_pixel_count

    # ── 5. majority vote ──────────────────────────────────────────────────────
    on_furniture = furniture_overlap > furniture_overlap_threshold
    signals = {
        "wide_bbox":       aspect_ratio      > aspect_ratio_threshold,
        "horizontal_axis": orientation_deg   > orientation_angle_threshold,
        "on_floor":        floor_overlap     > floor_overlap_threshold,
        "flat_normals":    normal_horiz_frac > normal_horiz_threshold,
        "on_furniture":    on_furniture,   # informational — used as veto below
    }
    positive_votes = sum(v for k, v in signals.items() if k != "on_furniture")
    # A person mostly overlapping furniture is more likely sleeping than fallen:
    # require 3 of 4 positive signals to override the furniture veto, otherwise 2 suffice.
    required_votes = 3 if on_furniture else 2
    is_fallen = positive_votes >= required_votes

    return dict(
        is_fallen          = is_fallen,
        aspect_ratio       = round(aspect_ratio,       3),
        orientation_deg    = round(orientation_deg,    1),
        floor_overlap      = round(floor_overlap,      3),
        normal_horiz_frac  = round(normal_horiz_frac,  3),
        furniture_overlap  = round(furniture_overlap,  3),
        centroid_y_frac    = round(centroid_y_frac,    3),
        person_pixel_count = person_pixel_count,
        signals            = signals,
        # Bounding box in heatmap pixel space + heatmap dimensions for caller scaling
        bbox_heatmap       = (x0, y0, x1, y1),
        heatmap_size       = (W, H),
    )
#----------------------------------------------------------------------------------------
def reseed_ranomizer():
  import time
  t = int( time.time() * 1000.0 )
  np.random.seed( ((t & 0xff000000) >> 24) + ((t & 0x00ff0000) >>  8) + ((t & 0x0000ff00) <<  8) + ((t & 0x000000ff) << 24)   )
#----------------------------------------------------------------------------------------
def get_color_for_label(label_id):
    """Generate a consistent color for a label using a hash."""
    np.random.seed(label_id)  # Seed with label ID
    result = tuple(np.random.randint(0, 255, size=3).tolist())

    reseed_ranomizer()
    return result
#----------------------------------------------------------------------------------------
def draw_labeled_blobs(labeled_map, bounding_boxes):
    """
    Draws labeled blobs with consistent unique colors and overlays bounding boxes.

    Args:
        labeled_map (np.ndarray): Array with blob labels (> 0).
        bounding_boxes (list of (x, y, w, h)): Bounding boxes for each blob.

    Returns:
        np.ndarray: Color visualization image.
    """
    h, w = labeled_map.shape
    output_image = np.zeros((h, w, 3), dtype=np.uint8)

    for label_id in range(1, np.max(labeled_map) + 1):
        color = get_color_for_label(label_id)
        mask = (labeled_map == label_id)
        output_image[mask] = color

    # Draw bounding boxes
    for i, (x, y, w_box, h_box) in enumerate(bounding_boxes):
        color = get_color_for_label(i + 1)
        cv2.rectangle(output_image, (x, y), (x + w_box, y + h_box), color, 2)
        area = w_box * h_box
        cv2.putText(output_image, f'ID {i+1} / area {area}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return output_image
#----------------------------------------------------------------------------------------
def convertIO(cfg,frame,heatmaps,heatmap_threshold=0.0,smoothDepth=False):
    i=0

    depthmap = None
    #Scale back window
    rgb_uint8_image = frame

    # Convert to uint8 type for display
    rgb_uint8_image = np.uint8(rgb_uint8_image)

    # Display the result
    bgr_uint8_image = cv2.cvtColor(rgb_uint8_image, cv2.COLOR_BGR2RGB) 
 
    heatmap_outputs = []
    #print("convertIO received : dtype=",heatmaps.dtype, " shape=",heatmaps.shape) 
    #print("Iterating over dimension ",heatmaps.shape[2] ) 
    #print("heatmaps = ",heatmaps)

    for i in range(0,heatmaps.shape[2]):
        #---------------------------------------------------------------
        heatmap         =  heatmaps[:,:,i]
        #---------------------------------------------------------------
        resized_heatmapF = np.array(heatmap,dtype=np.float32)
        #Filter too strong NN reponses :
        resized_heatmapF[resized_heatmapF > cfg['heatmapActive']]      = cfg['heatmapActive']
        resized_heatmapF[resized_heatmapF < cfg['heatmapDeactivated']] = cfg['heatmapDeactivated']
        #Undo int8 = -120..120 and convert to bytes
        #resized_heatmapF = np.float32( (1.0 + (resized_heatmapF/cfg['heatmapActive'])) * 170.0 )
        resized_heatmapF = resized_heatmapF + abs(cfg['heatmapDeactivated'])
        #resized_heatmapF = np.float32( (1.0 + (resized_heatmapF)) * 170.0 )
        #resized_heatmapF[resized_heatmapF > 255.0] = 255.0 #<- Why is this needed ?
  
        if (heatmap_threshold!=0.0):
           if i<17: 
             # Create a boolean mask for values above the heatmap_threshold
             resized_heatmapF[resized_heatmapF <= heatmap_threshold] = 0.0
        #---------------------------------------------------------------
        #Cast back to uint8
        resized_heatmap  = np.array(resized_heatmapF,dtype=np.uint8)


        if (i==heatmaps.shape[2]-1): 
            #print("Depthmap is ",i) 
            if (smoothDepth):
               resized_heatmap = apply_bilateral_filter(apply_bilateral_filter(resized_heatmap))
            depthmap = resized_heatmap.copy()

        heatmap_outputs.append(resized_heatmap)

    return bgr_uint8_image, depthmap, heatmap_outputs
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
def visualize_normals(cfg, heatmapsOut):
    if ('heatmapAddNormals' in cfg) and (cfg['heatmapAddNormals']):
       normalX          =  heatmapsOut[30] #18
       normalY          =  heatmapsOut[31] #19
       normalZ          =  heatmapsOut[32] #20
        
       normalXYZ = cv2.merge([normalX, normalY, normalZ])
       cv2.imshow('Combined Normals', normalXYZ)
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
def calculateRelativeValue(y,h,value,minimum,maximum):
    if (maximum==minimum):
       return int(y + (h/2)) 
    #-------------------------------------------------
    #TODO IMPROVE THIS!
    vRange = (maximum - minimum)
    return int( y + (h/2) - ( value / vRange ) * (h/2) )
#----------------------------------------------------------------------------------------
def drawSinglePlotValueList(valueListRAW,color,itemName,image,x,y,w,h,minimumValue=None,maximumValue=None):
    import cv2
    import numpy as np

 
    #Make sure to only display last items of list that fit
    margin=10 
    #--------------------------------------------
    if (len(valueListRAW) > w+margin ):
       itemsToRemove = len(valueListRAW) - w
       valueList = valueListRAW[itemsToRemove:]
    else:
       valueList = valueListRAW
    #--------------------------------------------

    #Auto Scale Y Axis if there is no minimum/maximum
    #--------------------------------------------
    if minimumValue is None:
        minimumValue = min(valueList)
    if maximumValue is None:
        maximumValue = max(valueList)
    #--------------------------------------------

    
    if (minimumValue==maximumValue):
      color = (40,40,40) #Dead plot

    listMaxValue = np.max(valueList)
    if (listMaxValue>maximumValue):
          maximumValue=listMaxValue*2 #Adapt to maximum

    #---------------------------------------------------------------------------------------
    cv2.line(image, pt1=(x,y+h), pt2=(x+w,y+h), color=color, thickness=1) #X-Axis
    cv2.line(image, pt1=(x,y),   pt2=(x,y+h),   color=color, thickness=1)            #Y-Axis
    #---------------------------------------------------------------------------------------

    font = cv2.FONT_HERSHEY_SIMPLEX 
    org = (x,y) 
    fontScale = 0.3 
    tColor = (123,123,123)
    thickness = 1
    message =  '%s' % (itemName) 
    image = cv2.putText(image, message , (x-1,y-1), font, fontScale, (0,0,0) , thickness, cv2.LINE_AA)
    image = cv2.putText(image, message , org, font, fontScale, color, thickness, cv2.LINE_AA)
    message =  'Max %0.2f ' % (maximumValue) 
    org = (x,y+10) 
    image = cv2.putText(image, message , (x-1,y+10-1), font, fontScale, (0,0,0) , thickness, cv2.LINE_AA)
    image = cv2.putText(image, message , org, font, fontScale, color, thickness, cv2.LINE_AA)
    message =  'Min %0.2f ' % (minimumValue) 
    org = (x,y+h+10) 
    image = cv2.putText(image, message , (x-1,y+h+10-1), font, fontScale, (0,0,0) , thickness, cv2.LINE_AA)
    image = cv2.putText(image, message , org, font, fontScale, color, thickness, cv2.LINE_AA)
    
    if (len(valueList)>2):
      for frameID in range(1,len(valueList)):
            #-------------------------------------------------------------------------------------
            previousValue = calculateRelativeValue(y,h,valueList[frameID-1],minimumValue,maximumValue)
            nextValue     = calculateRelativeValue(y,h,valueList[frameID],minimumValue,maximumValue)
            #-------------------------------------------------------------------------------------
            jointPointPrev = (int(x+ frameID-1),      previousValue )
            jointPointNext = (int(x+ frameID),        nextValue )
            if (itemName=="hip_yrotation"):  
                color=(0,0,255) 
            
            cv2.line(image, pt1=jointPointPrev, pt2=jointPointNext, color=color, thickness=1)  

    org = (int(x+len(valueList)), calculateRelativeValue(y,h,valueList[len(valueList)-1],minimumValue,maximumValue) ) 
    message =  '%0.2f' % (valueList[len(valueList)-1]) 
    image = cv2.putText(image, message , org, font, fontScale, (0,0,0), thickness, cv2.LINE_AA)

    org = (1+int(x+len(valueList)), 1+calculateRelativeValue(y,h,valueList[len(valueList)-1],minimumValue,maximumValue) ) 
    image = cv2.putText(image, message , org, font, fontScale, color, thickness, cv2.LINE_AA)
# Default heatmap name list used when cfg["heatmaps"] is absent (legacy configs).
DEFAULT_HEATMAP_NAMES = [
    "nose",               #0
    "left_eye",           #1
    "right_eye",          #2
    "left_ear",           #3
    "right_ear",          #4
    "left_shoulder",      #5
    "right_shoulder",     #6
    "left_elbow",         #7
    "right_elbow",        #8
    "left_wrist",         #9
    "right_wrist",        #10
    "left_hip",           #11
    "right_hip",          #12
    "left_knee",          #13
    "right_knee",         #14
    "left_ankle",         #15
    "right_ankle",        #16
    #------------------------
    "rhand->relbow",      #17
    "relbow->rshoulder",  #18
    "rshoulder->head",    #19
    "rfoot->rknee",       #20
    "rknee->rhip",        #21
    "rhip->head",         #22
    "lfoot->lknee",       #23
    "lknee->lhip",        #24
    "lhip->head",         #25
    "lhand->lelbow",      #26
    "lelbow->lshoulder",  #27
    "lshoulder->head",    #28
    #------------------------
    "depthmap",           #29
    "normalX",            #30
    "normalY",            #31
    "normalZ",            #32
    "text",               #33
    "Person",             #34
    "Vehicle",            #35
    "Animal",             #36
    "Object",             #37
    "Furniture",          #38
    "Appliance",          #39
    "Material",           #40
    "Obstacle",           #41
    "Building",           #42
    "Nature",             #43
    "depthmap>128",       #44
    "Denosing R",         #45
    "Denosing G",         #46
    "Denosing B",         #47
    "left/right pattern", #48
]

#----------------------------------------------------------------------------------------
def label_heatmap(cfg, heatmapID, keypoint_names=None, instanceLabels=None):
        heatmapNames = cfg.get('heatmaps', DEFAULT_HEATMAP_NAMES)
        title = "#%u" % heatmapID
        if heatmapID < len(heatmapNames):
            title = heatmapNames[heatmapID]
        return title
#----------------------------------------------------------------------------------------
def retrieveHeatmapIndex(heatmapNames, name):
    """
    Returns the index of `name` in heatmapNames, or -1 if not found.
    """
    try:
        return heatmapNames.index(name)
    except ValueError:
        return -1


#----------------------------------------------------------------------------------------
def filter_depthmap_by_heatmap(depthmap, heatmap, threshold):
    # Create a mask where the heatmap values are above the threshold
    mask = heatmap > threshold
    
    # Initialize a new array with the same shape as depthmap, filled with zeros or any other default value
    result = np.zeros_like(depthmap)
    
    # Apply the mask to the depthmap and copy the values to the result array
    result[mask] = depthmap[mask]
    
    return result
#----------------------------------------------------------------------------------------
def _display_regions(screen_w, screen_h):
    """Return (processed_x, heatmap_x) — the x boundaries between the three layout regions.

    Region 0  [0 .. processed_x)          Primary outputs  (Overlay, Person IDs, …)
    Region 1  [processed_x .. heatmap_x)  Processed maps   (Depth, Normals, Unions, …)
    Region 2  [heatmap_x .. screen_w)     Per-channel heatmaps

    For ultra-wide displays (aspect > 3, e.g. three 1080 p monitors) each region
    gets exactly one monitor's worth of width.  For normal / 4K displays the two
    named regions are merged into the left half and heatmaps take the right half.
    """
    aspect = screen_w / max(screen_h, 1)
    if aspect > 3.0:
        third = screen_w // 3
        # Ultra-wide (3+ monitors): primary+processed share monitors 1+2, heatmaps get
        # monitors 2+3.  Monitor 2 is intentionally shared so both categories have room.
        return 2 * third, third       # proc_w=2*third, hm_x=third
    else:
        half = screen_w // 2
        return half, half             # merged: processed_x == heatmap_x → single named region


def arrange_windows(window_specs, screen_w=3840, screen_h=2400,
                    offset_x=0, offset_y=10, margin=6, titlebar_h=30, 
                    border_w=120, border_h=12):
    """Compute (x, y) positions for named windows using left-to-right shelf packing.

    Args:
        window_specs: iterable of (name, width, height) in display-priority order.
                      width/height should be the *content* dimensions (image.shape).
        screen_w:     usable width  of the target region in pixels.
        screen_h:     usable height of the target region in pixels.
        offset_x:     horizontal origin of the region on the physical display.
        offset_y:     vertical   origin of the region on the physical display.
        margin:       gap between windows in pixels.
        titlebar_h:   height of the OS window title bar in pixels (typically
                      28-35 px on most Linux desktop environments).  This is
                      added to the content height when computing row advances so
                      that consecutive rows do not overlap each other.
        border_w:     total horizontal WM decoration width in pixels (left border +
                      right border).  This is added to the content width when
                      computing column advances so that consecutive windows in the
                      same row do not overlap each other on the X axis.
        border_h:     total vertical WM border height below the title bar in pixels
                      (top border + bottom border, excluding the title bar itself).
                      This is added to the content height when computing row advances
                      so that consecutive rows do not overlap each other on the Y axis.

    Returns:
        dict  {name: (x, y)}  with absolute screen coordinates.
    """
    layout  = {}
    x       = offset_x + margin
    y       = offset_y + margin
    row_h   = 0                    # tallest on-screen frame height in current row
    for name, w, h in window_specs:
        frame_w = w + border_w            # actual on-screen width  including WM borders
        frame_h = h + titlebar_h + border_h  # actual on-screen height including title bar + borders
        # Wrap to a new row when this window would exceed the region's right edge
        if x + frame_w + margin > offset_x + screen_w:
            x     = offset_x + margin
            y    += row_h + margin
            row_h = 0
        # Skip (don't crash) if the window no longer fits vertically
        if y + frame_h + margin > offset_y + screen_h:
            continue
        layout[name] = (x, y)
        x    += frame_w + margin
        row_h = max(row_h, frame_h)
    return layout


def apply_window_layout(layout):
    """Call cv2.moveWindow for every entry in a layout dict, then waitKey(1)."""
    for name, (x, y) in layout.items():
        try:
            cv2.moveWindow(name, x, y)
        except Exception:
            pass  # window may not exist yet (common on Windows before first imshow)
    try:
        cv2.waitKey(1)
    except Exception:
        pass


#----------------------------------------------------------------------------------------
def visualize_heatmaps(cfg, instanceLabels, imageIn, frameNumber, heatmapsOut, keypoint_names,
                        threshold=0.0, drawJoints=False, drawPAFs=False,
                        resizeHighHeatmaps=True, showClassHeatmaps=True,
                        showDepthInsteadOfMask=False,
                        screen_w=3840, screen_h=2400):
    i=0
    hm_specs = []  # (name, w, h) collected on frame 1 for auto-layout

    #imageIn, depthmap, heatmapsOut = convertIO(cfg,frame,heatmaps,threshold)

    #cv2.imshow('RGB Input', imageIn)
    #if (frameNumber==1):
    #   cv2.moveWindow("RGB Input", wnd_x, wnd_y)
    #   wnd_y+=imageIn.shape[0]

    if (len(heatmapsOut)<29):
         print("Can't visualize with only (",len(heatmapsOut)," heatmaps)..")
         return None


    depthmap = heatmapsOut[29] #17

    #print("Heatmaps returned by NN ",len(heatmapsOut))
    for i in range(0,len(heatmapsOut)):
        heatmap          =  heatmapsOut[i] 

        #Re-enable this if we have many segmentations
        if (i>33) and (showDepthInsteadOfMask): #20
           hmCP    = heatmap
           heatmap = filter_depthmap_by_heatmap(depthmap,hmCP,30)

        if (resizeHighHeatmaps):
          if i>33: #20
           original_height, original_width = heatmap.shape[:2] 
           new_width  = int(original_width  * 0.75)
           new_height = int(original_height * 0.75)
           heatmap = cv2.resize(heatmap, (new_width, new_height))
          elif i>=29: #17
           #original_height, original_width = heatmap.shape[:2] 
           #new_width  = int(original_width  * 0.75) #1.25
           #new_height = int(original_height * 0.75) #1.25
           #heatmap = cv2.resize(heatmap, (new_width, new_height))
           pass
 
        #Get Window Title
        title = label_heatmap(cfg,i,keypoint_names,instanceLabels)

        if (drawPAFs and i>=17 and i<=28) or  (drawJoints and i<=17) or ((i>=29) and ((showClassHeatmaps) or (i<=33)) ): # 17 / 20
         cv2.imshow(title, heatmap)
         if (frameNumber==1):
             hm_specs.append((title, heatmap.shape[1], heatmap.shape[0]))

        i=i+1

    # On the first frame place all heatmap windows in the rightmost region
    if frameNumber == 1 and hm_specs:
        _, hm_x = _display_regions(screen_w, screen_h)
        layout  = arrange_windows(hm_specs,
                                  screen_w=screen_w - hm_x,
                                  screen_h=screen_h,
                                  offset_x=hm_x)
        apply_window_layout(layout)
#----------------------------------------------------------------------------------------
def countHits(heatmap): 
    return np.count_nonzero(heatmap)
#----------------------------------------------------------------------------------------
def visualization(frame, frameNumber, keypoint_results, keypoint_depths, keypoint_names):  #, blobs
    # Create a copy of the original frame to draw keypoints on
    visualized_image = frame.copy()

    frameWidth  = frame.shape[1]
    frameHeight = frame.shape[0]

    # Iterate through each keypoint
    for i, keypoint_name in enumerate(keypoint_names):
      if keypoint_name!="bkg": #Ignore Background
        # Get the peak points for the current keypoint
        peak_points = keypoint_results[i]

        # Draw circles for each peak point
        for peak_point in peak_points:
            x, y, _ = peak_point  # Extract x, y coordinates
            cv2.circle(visualized_image, ( int(x*frameWidth),int(y*frameHeight)), radius=5, color=(0, 255, 0), thickness=-1)  # Draw a filled circle

            # Add text with keypoint name
            text_position = (int(10+x*frameWidth),int(10+y*frameHeight))  # Position text slightly away from the circle
            depth = 255
            if keypoint_depths:
                depth = int(keypoint_depths[i]+125)
            cv2.putText(visualized_image, keypoint_name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, depth, 0), 1)

    #for blob in blobs:
    #             x,y,w,h = blob
    #             x = int(x * frameWidth)
    #             y = int(y * frameHeight)
    #             w = int(w * frameWidth)
    #             h = int(h * frameHeight) 
    #             cv2.rectangle(visualized_image,(x,y),(x+w,y+h),(0,255,0),3)


    return visualized_image
#----------------------------------------------------------------------------------------
def visualize_monitors(frame, hm, monitors):   
    frameWidth  = frame.shape[1]
    frameHeight = frame.shape[0]

    heatmapWidth  = hm.shape[1]
    heatmapHeight = hm.shape[0]

    # Iterate through each keypoint
    for i, entry in enumerate(monitors):

        hmTarget = entry[0]
        x        = entry[1] / heatmapWidth
        y        = entry[2] / heatmapHeight
        label    = entry[3]
 
        cv2.circle(frame, ( int(x*frameWidth),int(y*frameHeight)), radius=5, color=(255, 255, 0), thickness=-1)  # Draw a filled circle


        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        text_width = cv2.getTextSize(label, font, font_scale, font_thickness)[0]


        # Add text with keypoint name
        text_position = (int(int(x*frameWidth)-text_width[0]//2),int(20+y*frameHeight))  # Position text slightly away from the circle 
        cv2.putText(frame, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
 
    return frame
#----------------------------------------------------------------------------------------
def checkIfFileExists(filename):
    return os.path.isfile(filename)
#----------------------------------------------------------------------------------------
def read_json_files(file_path):
    """
    Read JSON files from given file path.
    """
    import json
    with open(file_path, 'r') as file:
            data = json.load(file) 
            return data
    return None
#----------------------------------------------------------------------------------------
def decodeSingleOneHotDescriptionToString(vocabulary, onehot):
    if (vocabulary is None) or (onehot is None):
          return None
    # Invert the vocabulary to get a mapping from index to word
    index_to_word = {int(k): v for k, v in vocabulary.items()}
    
    # Initialize an empty list to store the words
    description_words = []
    
    # Loop through each vector in the one-hot encoding
    for vector in onehot:
        # Find the index of the maximum value (which indicates the presence of a word)
        max_index = np.argmax(vector)
        
        # Map the index to word and add it to the description words list
        word = index_to_word[max_index]
        description_words.append(word)
    
    # Join all the words to form the final description string
    description = ' '.join(description_words)
    
    return description
#----------------------------------------------------------------------------------------
"""
def decodeOneHotDescriptionToString(vocabulary, onehot):
    if (vocabulary is None) or (onehot is None):
        return None
    
    # Invert the vocabulary to get a mapping from index to word
    index_to_word = {int(k): v for k, v in vocabulary.items()}
    
    # Initialize an empty list to store the words
    description_words = []
    
    # Loop through each vector in the one-hot encoding
    for vector in onehot:
        # Find the indices of the top 5 values
        top_indices = np.argsort(vector)[-5:][::-1]
        
        # Map the indices to words and their corresponding values
        top_words = [(index_to_word[idx], vector[idx]) for idx in top_indices]
        
        # Format the top words and values as desired
        formatted_words = ','.join([f'{word}:{value:.4f}' for word, value in top_words])
        description_words.append(f'({formatted_words})')
    
    # Join all the formatted strings to form the final description string
    description = ' '.join(description_words)
    
    return description
"""


#----------------------------------------------------------------------------------------
def decodeClassesDescriptionToString(vocabulary, classscores):
    if (vocabulary is None) or (classscores is None):
        return None
    # Ensure vocabulary and classscores have the same length
    if len(vocabulary) != len(classscores):
        raise ValueError("Length of vocabulary and classscores must be the same")
    
    # Create a list of (score, token) tuples
    activations = [(classscores[i], vocabulary[str(i)]) for i in range(len(classscores))]
    
    # Sort activations based on scores in descending order
    activations.sort(reverse=True, key=lambda x: x[0])
    
    # Take top 10 activations
    top10 = activations[:10]
    
    # Format the top 10 activations into a string
    result = "\n".join(f"{score}: {token}" for score, token in top10)
    
    return result
#----------------------------------------------------------------------------------------
def decodeMultiHotDescriptionToString(vocabulary, multihot, threshold=0.29):
    if (vocabulary is None) or (multihot is None):
        return None
    
    # Invert the vocabulary to get a mapping from index to word
    index_to_word = {int(k): v for k, v in vocabulary.items()}
    
    # Initialize an empty list to store the words
    description_words = []
    
    # Loop through each vector in the one-hot encoding
    for vector in multihot:
        # Find the indices of the top 7 values
        top_indices = np.argsort(vector)[-7:][::-1]
        
        # Map the indices to words and their corresponding values
        top_words = []
        for idx in top_indices:
            if (threshold<vector[idx]):
               top_words.append((index_to_word[idx], vector[idx]))          

        #top_words = [(index_to_word[idx], vector[idx]) for idx in top_indices]
        
        # Format the top words and values as desired
        for word, value in top_words:
          description_words.append("%s:%0.2f" % (word,value))
        #formatted_words = ','.join([f'{word}:{value:.4f}' for word, value in top_words])
        #description_words.append(f'({formatted_words})')
    
    # Join all the formatted strings to form the final description string
    description = ' '.join(description_words)
    
    return description.strip() #remove whitespaces
#----------------------------------------------------------------------------------------
# Compute cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    # Ensure both vectors are 1D numpy arrays of the same shape
    vec1 = np.asarray(vec1).flatten()
    vec2 = np.asarray(vec2).flatten()
    
    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return -1  # Return the lowest similarity for zero vectors
    
    # Compute cosine similarity as a scalar
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
#----------------------------------------------------------------------------------------
def mean_squared_error(vec1, vec2):
    # Ensure both vectors are 1D numpy arrays of the same shape
    vec1 = np.asarray(vec1).flatten()
    vec2 = np.asarray(vec2).flatten()
    
    if vec1.shape != vec2.shape:
        raise ValueError("Vectors must have the same shape (",vec1.shape()," vs ",vec2.shape(),").")
    
    # Compute MSE as the average of squared differences
    return np.mean((vec1 - vec2) ** 2)
#----------------------------------------------------------------------------------------
def recover_index(labelNeedle, glove_embeddings):
  for idx, labelHaystack in glove_embeddings.items():
    if labelHaystack == labelNeedle:
         return idx
#----------------------------------------------------------------------------------------
def vector_to_sentence_close(vectors, glove_embeddings,useMSE=False, D=300):
  closest_words = []
  used_indices = set()  # To track already used indices
  empty_vector  = np.full(D,0)

  for number in range(8):
   vector = vectors[number]
   # Flatten the input vector to ensure it matches the embedding dimensions
   vector = np.asarray(vector).flatten()
   zero_similarity = mean_squared_error(vector, empty_vector)
   #print("Zero sim ",zero_similarity)
   if (zero_similarity>0.003):
    
    # Iterate over all words in the GloVe embedding index
    #print("Check vector ",vector)
    #print("Against ",len(glove_embeddings.items())," vectors")
    glove_keys = list(glove_embeddings.keys())

    if (useMSE):
      bestScore = 100000
      bestIndex = 0
      for i,embedding_vector in enumerate(glove_embeddings.values()):
        #print("Cosine similarity of ",vector," against ",embedding_vector)
        similarity = mean_squared_error(vector, embedding_vector)
        if (bestScore>similarity) and i not in used_indices:
            #print("Key ",glove_keys[i]," is better than ",glove_keys[bestIndex])
            #print("Beats it by  ",bestScore-similarity)
            bestScore = similarity
            bestIndex = i
            used_indices.add(bestIndex)  # Mark this index as used
      if (bestScore!=100000):
        thisWord = glove_keys[bestIndex]
        if (thisWord == "man") or (thisWord =="woman"):
             thisWord = "person"
        closest_words.append("%s(%0.2f)" % (thisWord,bestScore) )
    else:
      bestScore = -1
      bestIndex = 0
      for i, embedding_vector in enumerate(glove_embeddings.values()):
        # Calculate cosine similarity
        similarity = cosine_similarity(vector, embedding_vector)
        
        # Update if this similarity is better than the current best
        if similarity > bestScore and i not in used_indices:
            bestScore = similarity
            bestIndex = i
            used_indices.add(bestIndex)  # Mark this index as used
      if (bestScore!=-1):
        thisWord = glove_keys[bestIndex]
        if (thisWord == "man") or (thisWord =="woman"):
              thisWord = "person"
        closest_words.append("%s(%0.2f)" % (thisWord,bestScore) )
    
  return closest_words
#----------------------------------------------------------------------------------------
def visualize_activity(activity, labels, threshold, image_size=(500, 500), font_scale=0.4, thickness=1):
    """
    Creates an OpenCV visualization for the given activity sums.
    
    Parameters:
    - activity: List of activity sums (tf.Tensor objects or numpy values).
    - labels: List of labels corresponding to each activity sum.
    - threshold: Threshold value to determine the color (green/red).
    - image_size: Size of the output image (height, width).
    - font_scale: Scale of the font used for the labels.
    - thickness: Thickness of the rectangle and text.
    
    Returns:
    - image: The OpenCV image with visualized activity.
    """
    # Create a blank image
    image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    
    # Define initial start position for drawing labels
    start_y = 5
    start_x = 5
    margin  = 5
    rect_height = 20

    # Maximum height before moving to the next column
    max_height = image_size[0] - margin
    max_width = image_size[1]  # Available width for text

    for i, val in enumerate(activity): 

        # Determine color based on the threshold
        color = (0, 255, 0) if val > threshold else (0, 0, 255)  # Green if above threshold, Red otherwise
        
        # Define text for the label
        val = val - threshold
        text = f"{labels[i]}: {val:.2f}"
        
        # Calculate text size
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        
        # Check if we need to move to the next column
        if start_y + rect_height > max_height:
            start_x += max_width // 3  # Move to the right by a third of the width
            start_y = margin  # Reset y position

        # Adjust max_width based on the current start_x position
        if start_x + text_size[0] + 20 > image_size[1]:
            break  # Stop adding labels if there is no more horizontal space

        # Define rectangle coordinates
        top_left = (start_x, start_y)
        bottom_right = (start_x + text_size[0] + 10, start_y + rect_height)

        # Draw the rectangle
        cv2.rectangle(image, top_left, bottom_right, color, thickness=cv2.FILLED)
        
        # Draw the label text inside the rectangle
        text_position = (top_left[0] + 5, top_left[1] + rect_height - 5)
        cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        text_position = (top_left[0] + 7, top_left[1] + rect_height - 7)
        cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # Update the starting y-coordinate for the next label
        start_y = bottom_right[1] + margin

    cv2.imshow('Activity',image)
    return image
#----------------------------------------------------------------------------------------
def find_bounding_boxes(image, area_threshold=0):
    # Threshold the image to get a binary image
    _, binary = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # List to hold bounding boxes
    bounding_boxes = []
    
    # Iterate through contours and get bounding boxes
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        # Apply the area threshold
        if area > area_threshold:
            bounding_boxes.append((x, y, w, h))
    
    return bounding_boxes
#----------------------------------------------------------------------------------------
def resize_images(images, target_size):
    """
    Resizes a list of images to the specified target size.

    Parameters:
    - images (list): List of images (as numpy arrays).
    - target_size (tuple): Desired size (width, height) for each image.

    Returns:
    - resized_images (list): List of resized images.
    """
    return [cv2.resize(img, target_size) for img in images]


#----------------------------------------------------------------------------------------
def thresholded_sum_of_heatmaps(a, b, c, threshold=120/3):
    s = a + b + c
    return np.where(s > threshold, 120, -120)
#----------------------------------------------------------------------------------------
class YMAPNet:
    def __init__(self, modelPath, threshold=30, keypoint_threshold=50.0, engine="tensorflow", profiling = False, illustrate=False,
                       pruneTokens=False, monitor=list(), window_arrangement=list(),
                       screen_w=3840, screen_h=2400, depth_iterations=10,
                       estimate_person_id=True, resolve_skeleton=True):
        self.model_path     = '2d_pose_estimation'
        self.cfg            = loadJSONConfiguration("%s/configuration.json" % self.model_path)
        self.serial         = self.cfg["serial"]
        #self.instanceLabels = loadJSON("datasets/detectron2/labels.json")
        self.instanceLabels    = loadJSON("2d_pose_estimation/vocabulary.json")
        # Specify the names of keypoints (change accordingly based on your model's keypoint order)
        self.keypoint_names = self.cfg['keypoint_names']
        print("Reported Keypoints : ",self.keypoint_names) 
        #---------------------------------------------------------------------
        from NNExecutor import NNExecutor
        self.keypoints_model    = NNExecutor(
                                             engine=engine,
                                             profiling=profiling,
                                             inputWidth     = self.cfg['inputWidth'],
                                             inputHeight    = self.cfg['inputHeight'],
                                             targetWidth    = self.cfg['outputWidth'],
                                             targetHeight   = self.cfg['outputHeight'],
                                             outputChannels = self.cfg['outputChannels'],
                                             pruneTokens    = pruneTokens
                                            )
        #---------------------------------------------------------------------
        """
        from NNConverter import saveNNModel
        verySmallWeights = 0
        veryBigWeights = 0
        for layer in self.keypoints_model.model.model.layers:
         print(f"Layer: {layer.name}")
         weights = layer.get_weights()
         if weights:  # Some layers may not have weights
          for i, weight in enumerate(weights):
            print(f"  Weights {i}: shape {weight.shape}")
            #print(weight)  # Print or manipulate weight as needed
            for index in np.ndindex(weight.shape):
                 if (weight[index]<0.00001):
                        weight[index] = 0 
                        verySmallWeights += 1
                 elif (weight[index]>10000):
                        veryBigWeights += 1
        print(f" Very big weights {veryBigWeights} / Very small weights {verySmallWeights}")
        saveNNModel("2d_pose_estimation_cut",model,formats=["keras"])
        """
        #---------------------------------------------------------------------
        self.input_size         = self.keypoints_model.model.input_size
        self.output_size        = self.keypoints_model.model.output_size
        self.numberOfHeatmaps   = self.keypoints_model.model.numberOfHeatmaps
        self.keypoint_threshold = keypoint_threshold
        self.heatmap_threshold  = threshold
        self.onehot_description = None
        self.frameNumber        = 0
        self.frameRateList      = list()
        self.depthmap           = None
        self.activity           = None
        self.activityThreshold  = -17554720.0
        self.labeled_map        = None 
        self.bounding_boxes     = None
        self.bounding_boxes_threshold = 180
        self.bounding_boxes_minarea   = 1000
        self.drawJoints         = 0
        self.drawPAFs           = 0
        self.stop               = 0
        self.saveFrame          = 0
        #---------------------------------------------------------------------
        self.illustrate         = illustrate
        self.addedNoise         = 0.0
        #---------------------------------------------------------------------
        self.skeletons          = None
        self.vocabulary         = None
        self.possible_glove_embeddings = None 
        self.description        = None
        self.heatmap_16b        = None
        self.D                  = 300
        #---------------------------------------------------------------------
        self.window_arrangement = window_arrangement
        self.screen_w           = screen_w
        self.screen_h           = screen_h
        self.depth_iterations   = depth_iterations
        self.estimate_person_id = estimate_person_id
        self.resolve_skeleton   = resolve_skeleton
        #---------------------------------------------------------------------
        self.keypointXMultiplier = 1.0 
        self.keypointYMultiplier = 1.0
        self.keypointXOffset     = 0
        self.keypointYOffset     = 0
        #---------------------------------------------------------------------
        # Resolve string heatmap labels to indices using cfg["heatmaps"],
        # falling back to DEFAULT_HEATMAP_NAMES for legacy configs without the key.
        heatmap_labels = self.cfg.get('heatmaps', DEFAULT_HEATMAP_NAMES)
        resolved_monitor = []
        for hm, x, y, lbl in monitor:
            if isinstance(hm, str):
                if hm in heatmap_labels:
                    hm = heatmap_labels.index(hm)
                else:
                    print(f"Warning: heatmap label '{hm}' not found in cfg['heatmaps'], skipping monitor entry")
                    continue
            resolved_monitor.append((hm, x, y, lbl))
        self.monitor            = resolved_monitor
        self.monitorValues      = list()
        if (len(self.monitor)>0):
                for i in range(len(self.monitor)):
                    self.monitorValues.append(list())
        #---------------------------------------------------------------------
        if ("outputTokens" in self.cfg) and (self.cfg["outputTokens"]):
             self.vocabulary = read_json_files("2d_pose_estimation/vocabulary.json")
             self.vocabulary["0"] = " "
             self.vocabulary["1"] = " "

             self.possible_glove_embeddings = dict()
             with open("2d_pose_estimation/embeddings_6B_D%u.json" % self.D, 'r') as json_file:
               self.possible_glove_embeddings = json.load(json_file)
        #---------------------------------------------------------------------
        self.labels = list()
        for i in range(self.cfg['outputChannels']):
                     self.labels.append(label_heatmap(self.cfg,i,self.keypoint_names,self.instanceLabels))
        #---------------------------------------------------------------------
        
        self.denoised = None

        #Legacy heatmap IDs
        self.chanDepth     = 29
        self.chanNormalX   = 30
        self.chanNormalY   = 31
        self.chanNormalZ   = 32
        self.chanText      = 33
        self.chanVehicle   = 35
        self.chanAnimal    = 36
        self.chanObject    = 37
        self.chanFurniture = 38
        self.chanAppliance = 39
        self.chanFloor     = -1   # set from config below
        self.chanCeiling   = -1
        self.chanWall      = -1

        # "Programmable" heatmap IDs
        if "heatmaps" in self.cfg:
 
          #Fix typo..
          if "Denosing R" in self.cfg['heatmaps']:
            self.chanDenoiseR = retrieveHeatmapIndex(self.cfg['heatmaps'],"Denosing R")
          else:
            self.chanDenoiseR = retrieveHeatmapIndex(self.cfg['heatmaps'],"Denoising R")

          if "Denosing G" in self.cfg['heatmaps']:
            self.chanDenoiseG = retrieveHeatmapIndex(self.cfg['heatmaps'],"Denosing G")
          else:
            self.chanDenoiseG = retrieveHeatmapIndex(self.cfg['heatmaps'],"Denoising G")

          if "Denosing B" in self.cfg['heatmaps']:
            self.chanDenoiseB = retrieveHeatmapIndex(self.cfg['heatmaps'],"Denosing B")
          else:
            self.chanDenoiseB = retrieveHeatmapIndex(self.cfg['heatmaps'],"Denoising B")

          self.chanDepth    = retrieveHeatmapIndex(self.cfg['heatmaps'],"depthmap")
          self.chanNormalX  = retrieveHeatmapIndex(self.cfg['heatmaps'],"normalX")
          self.chanNormalY  = retrieveHeatmapIndex(self.cfg['heatmaps'],"normalY")
          self.chanNormalZ  = retrieveHeatmapIndex(self.cfg['heatmaps'],"normalZ")
          self.chanText     = retrieveHeatmapIndex(self.cfg['heatmaps'],"Text")
          self.chanVehicle  = retrieveHeatmapIndex(self.cfg['heatmaps'],"Vehicle")
          self.chanAnimal   = retrieveHeatmapIndex(self.cfg['heatmaps'],"Animal")
          self.chanObject   = retrieveHeatmapIndex(self.cfg['heatmaps'],"Object")
          self.chanFurniture= retrieveHeatmapIndex(self.cfg['heatmaps'],"Furniture")
          self.chanAppliance= retrieveHeatmapIndex(self.cfg['heatmaps'],"Appliance")
          self.chanFloor    = retrieveHeatmapIndex(self.cfg['heatmaps'],"Floor")
          self.chanCeiling  = retrieveHeatmapIndex(self.cfg['heatmaps'],"Ceiling")
          self.chanWall     = retrieveHeatmapIndex(self.cfg['heatmaps'],"Wall")
          self.chanPerson   = retrieveHeatmapIndex(self.cfg['heatmaps'],"Person")
        #--------------------------------------------------------------------- 



    def process_tiled(self, frame, borders=False, overlap=3):
     tile_size = 256
     h, w, _ = frame.shape
     stride = tile_size - overlap  # Step size with overlap
     tiles = []
     positions = []
    
 
     # Tile the image with overlap
     for y in range(0, h, stride):
        for x in range(0, w, stride):
            tile = frame[y:y+tile_size, x:x+tile_size]
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded_tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded_tile
            tiles.append(tile)
            positions.append((x, y))
    
     tiles = np.array(tiles)  # Convert to batch
    
     # Preprocess images for the network
     processed_tiles = []
     for tile in tiles:
        self.input_image, keypointXMultiplier, keypointYMultiplier, keypointXOffset, keypointYOffset = preprocess_image(tile, target_size=self.input_size, add_borders=borders)
        if self.cfg['RGBImageEncoding'] == 'rgb8':
            from imageProcessing import mix_rgb_to_8bit
            processed_tiles.append(mix_rgb_to_8bit(self.input_image))
        elif self.cfg['RGBImageEncoding'] == 'rgb16':
            from imageProcessing import mix_rgb_to_16bit
            processed_tiles.append(mix_rgb_to_16bit(self.input_image))
        else:
            processed_tiles.append(self.input_image)
    
     processed_tiles = np.array(processed_tiles)  # Convert to batch
    
     # Predict using batch inference
     batch_predictions = self.keypoints_model.predict_multi(processed_tiles)

     heatmap_outputs = batch_predictions
     token_outputs   = self.keypoints_model.multihot()

     print("frame: ",frame.shape)
     print("heatmap_outputs: ",heatmap_outputs.shape)
     print("token_outputs: ",token_outputs.shape)
     numberOfTiledFrames = heatmap_outputs.shape[0]
 
     # Reconstruct heatmap composite
     composite_heatmap = np.zeros((h, w, heatmap_outputs.shape[3]), dtype=np.float32)
     count_map = np.zeros((h, w, 1), dtype=np.float32)  # To avoid double counting in overlaps
    
     for i, (x, y) in enumerate(positions):
        #print("Tile Number ",i," x=",x," y=",y)
        h_tile, w_tile, _ = heatmap_outputs[i].shape  # Get actual tile dimensions
        h_tile = min(tile_size, h - y)
        w_tile = min(tile_size, w - x)
        print("Tile Number ",i," x=",x," y=",y, "h_tile=",h_tile, "w_tile=",h_tile )

        composite_heatmap[y:y+h_tile, x:x+w_tile] += heatmap_outputs[i][:h_tile, :w_tile]
        count_map[y:y+h_tile, x:x+w_tile] += 1

        #composite_heatmap[y:y+tile_size, x:x+tile_size] += heatmap_outputs[i]
        #count_map[y:y+tile_size, x:x+tile_size] += 1
    
     composite_heatmap /= np.maximum(count_map, 1)  # Normalize overlapping areas
    
     # Merge token outputs
     all_tokens = np.mean(token_outputs, axis=0)  # Assuming a mean aggregation for tokens
    
     self.imageIn, self.depthmap, self.heatmapsOut = convertIO(self.cfg, self.input_image, composite_heatmap, self.heatmap_threshold)
    
     # Process keypoints
     #keypoint_results = find_peak_points(composite_heatmap, threshold=self.keypoint_threshold)
     keypoint_results = find_peak_points_from_convertIO(self.heatmapsOut, threshold=self.keypoint_threshold)
            
    
     if self.cfg['heatmapAddDepthmap']:
        self.keypoint_depths = retrieve_keypoint_depth(keypoint_results, self.depthmap)
    
     # Convert tokens to description
     self.multihot_labels = self.keypoints_model.multihot()
     from TokenEstimator import decodeOneHotDescriptionToString
     self.description = decodeOneHotDescriptionToString(self.vocabulary, self.multihot_labels, K = 21)


     if (self.description):
               print("\n\nDescription : ",self.description)
    

     self.keypoint_in_nn_coordinate_image = keypoint_results
     self.keypoint_results = castNormalizedCoordinatesToOriginalImage(frame,self.input_image,self.output_size, keypoint_results, keypointXMultiplier, keypointYMultiplier, keypointXOffset, keypointYOffset)

     print("Final Description:", self.description)
     self.frameNumber = self.frameNumber + 1
    
     return composite_heatmap, self.description

    def setup_threshold_control_window(self):
      cv2.namedWindow("Threshold Controls")

      def nothing(x):
        pass

      # Create trackbars
      cv2.createTrackbar("Stop", "Threshold Controls",  int(self.stop), 1, nothing)
      cv2.createTrackbar("Save", "Threshold Controls",  int(self.saveFrame), 1, nothing) 
      cv2.createTrackbar("Draw All Joints", "Threshold Controls",  int(self.drawJoints), 1, nothing) 
      cv2.createTrackbar("Draw All PAFs", "Threshold Controls",  int(self.drawPAFs), 1, nothing) 
      cv2.createTrackbar("Keypoint Threshold", "Threshold Controls", int(self.keypoint_threshold), 255, nothing)
      cv2.createTrackbar("Heatmap Threshold", "Threshold Controls",  int(self.heatmap_threshold), 240, nothing)
      cv2.setTrackbarMin("Heatmap Threshold", "Threshold Controls", 0)
      cv2.createTrackbar("Person BBox Threshold", "Threshold Controls", int(self.bounding_boxes_threshold), 256, nothing)
      cv2.createTrackbar("Person BBox Min Area", "Threshold Controls",  int(self.bounding_boxes_minarea), 10000, nothing)
      cv2.createTrackbar("Synthetic Noise", "Threshold Controls", int(self.addedNoise), 256, nothing)

    def update_thresholds_from_gui(self):
      self.keypoint_threshold       = cv2.getTrackbarPos("Keypoint Threshold", "Threshold Controls")
      self.heatmap_threshold        = cv2.getTrackbarPos("Heatmap Threshold",  "Threshold Controls")
      self.stop                     = cv2.getTrackbarPos("Stop", "Threshold Controls")
      self.saveFrame                = cv2.getTrackbarPos("Save", "Threshold Controls")
      self.drawJoints               = cv2.getTrackbarPos("Draw All Joints", "Threshold Controls")
      self.drawPAFs                 = cv2.getTrackbarPos("Draw All PAFs", "Threshold Controls")
      self.bounding_boxes_threshold = cv2.getTrackbarPos("Person BBox Threshold", "Threshold Controls")
      self.bounding_boxes_minarea   = cv2.getTrackbarPos("Person BBox Min Area",  "Threshold Controls")
      self.addedNoise               = cv2.getTrackbarPos("Synthetic Noise",  "Threshold Controls") / 256
      if (self.stop):
              sys.exit(0)


    def process(self,frame,borders=False):
            self.input_image, keypointXMultiplier, keypointYMultiplier, keypointXOffset, keypointYOffset = preprocess_image(frame,target_size=self.input_size,add_borders=borders)

            self.keypointXMultiplier = keypointXMultiplier 
            self.keypointYMultiplier = keypointYMultiplier
            self.keypointXOffset     = keypointXOffset
            self.keypointYOffset     = keypointYOffset

            #if (borders==True):
            #   frame = self.input_image

            if (self.cfg['RGBImageEncoding']=='rgb8'):
                from imageProcessing import mix_rgb_to_8bit
                self.imageGivenToNN =  mix_rgb_to_8bit(self.input_image)
            elif (self.cfg['RGBImageEncoding']=='rgb16'):
                from imageProcessing import mix_rgb_to_16bit
                self.imageGivenToNN =  mix_rgb_to_16bit(self.input_image)
            else:
                self.imageGivenToNN = self.input_image

            #cv2.imshow('Debug Image', imageGivenToNN)
            #cv2.waitKey(0)

            # Make predictions using the model 
            self.keypoints_predictions = self.keypoints_model.predict(self.imageGivenToNN)
            self.heatmap_16b           = self.keypoints_model.heatmaps_16b()
            self.onehot_description    = self.keypoints_model.description()
            self.activity              = self.keypoints_model.activity()

            #     for i,val in enumerate(self.activity):
            #         print("#",label_heatmap(self.cfg,i,self.keypoint_names,self.instanceLabels)," = ",val)
            

            if ("outputTokens" in self.cfg) and (self.cfg["outputTokens"]):
             self.description = " "

             
             self.multihot_labels = self.keypoints_model.multihot()

             #Dynamically switch description subsystem based on what is available (GloVe or multihot)
             if (self.multihot_labels is None):
               self.glove = self.keypoints_model.description()
               if (self.glove is not None):
                 print("Recovered Glove shape ",self.glove.shape)
                 words = []
                 words = vector_to_sentence_close(self.glove, self.possible_glove_embeddings, D=self.D) 
                 #for i in range(7):
                  #word = vector_to_sentence_close(self.glove[i], self.possible_glove_embeddings, D=self.D)
                 for i,word in enumerate(words):
                  print(i," - ",word)
                  self.description = self.description + " " + word
             else:
               self.multihot_labels = self.keypoints_model.multihot()

               #print("Multi hot Glove : ",self.multihot_labels)
               #print("Multi hot Glove shape ",self.multihot_labels.shape)
               #print("Vocabulary Keys ",len(self.vocabulary.keys()))
               #self.description = decodeMultiHotDescriptionToString(self.vocabulary, self.multihot_labels)
               from TokenEstimator import decodeOneHotDescriptionToString
               self.description = decodeOneHotDescriptionToString(self.vocabulary, self.multihot_labels)


             if (self.description):
               print("\n\nDescription : ",self.description)



            self.imageIn, self.depthmap, self.heatmapsOut = convertIO(self.cfg,self.input_image,self.keypoints_predictions,self.heatmap_threshold)

            
            if "heatmaps" in self.cfg:
              self.denoised = self.process_denoising(self.imageIn, self.heatmapsOut[self.chanDenoiseR], self.heatmapsOut[self.chanDenoiseG], self.heatmapsOut[self.chanDenoiseB])

            """
            vClose = countHits(self.heatmapsOut[44])
            print("Very Close ", vClose) 
            if ( vClose >39000):
                os.system("pgrep -x aplay > /dev/null || aplay beep.wav&")
            """
 
            #keypoint_results = find_peak_points(self.keypoints_predictions, threshold=self.keypoint_threshold) #self.keypoints_predictions instead of self.heatmapsOut, maybe change this in the future
            keypoint_results = find_peak_points_from_convertIO(self.heatmapsOut, threshold=self.keypoint_threshold)
            keypoint_depths  = list()
            if (self.cfg['heatmapAddDepthmap']):
                self.keypoint_depths  = retrieve_keypoint_depth(keypoint_results, self.depthmap) #<- Important for this to happen before normalization (castNormalizedCoordinatesToOriginalImage)!

 
            self.person_union = np.max(self.heatmapsOut[33:36], axis=0)
            if self.estimate_person_id:
                self.labeled_map, self.bounding_boxes = segment_and_label_persons(self.person_union,threshold=self.bounding_boxes_threshold, min_area=self.bounding_boxes_minarea)
            """
            #Disabled for performance
            """
            if self.resolve_skeleton and ("keypoint_children" in self.cfg):
              # Convert uint8 heatmaps back to float [-120..120] to match resolveJointHierarchyNew expectations
              _kp_hm  = np.stack(self.heatmapsOut[:17], axis=2).astype(np.float32) - 120.0
              _paf_hm = [self.heatmapsOut[17 + j].astype(np.float32) - 120.0 for j in range(min(12, len(self.heatmapsOut) - 17))]
              _debug_once = (self.frameNumber <= 1)   # print debug only on first frame
              self.skeletons = resolveJointHierarchyNew(_kp_hm,
                                                        _paf_hm,
                                                        self.depthmap,
                                                        self.cfg["keypoint_names"],
                                                        self.cfg["keypoint_parents"],
                                                        self.cfg["keypoint_children"],
                                                        self.cfg["paf_parents"],
                                                        person_label_map=None,  # labeled_map coords don't align with joint heatmaps
                                                        threshold=self.keypoint_threshold,
                                                        debug=_debug_once
                                                       )


            self.keypoint_in_nn_coordinate_image = keypoint_results
            self.keypoint_results = castNormalizedCoordinatesToOriginalImage(frame,self.input_image,self.output_size, keypoint_results, keypointXMultiplier, keypointYMultiplier, keypointXOffset, keypointYOffset)

            #blobs = detect_blobs(self.keypoints_predictions[:,:,-1])
            #self.blobs = castNormalizedBBoxCoordinatesToOriginalImage(frame,self.input_image,self.output_size, blobs, keypointXMultiplier, keypointYMultiplier, keypointXOffset, keypointYOffset)


            if (len(self.monitor)>0):
                 for i,entry in enumerate(self.monitor): 
                      hmTarget = entry[0]
                      x        = entry[1]
                      y        = entry[2]
                      tL       = entry[3]
                      value    = self.heatmapsOut[hmTarget][y,x]
                      self.monitorValues[i].append(value)
                      if (len(self.monitorValues[i])>200): 
                         self.monitorValues[i].pop(0) #Keep history on limits



            self.frameRateList.append(self.keypoints_model.hz)
            self.frameNumber = self.frameNumber + 1

    def plot_depthmap_with_matplotlib(self,depthmap,reverse=False):
        import matplotlib.pyplot as plt
        if (reverse):
          depthmap = 1.0 / (0.001 + depthmap)  # Assuming index 29 corresponds to the depth map
        plt.figure()
        plt.imshow(depthmap, cmap='viridis')  # You can use 'gray' or 'plasma' if preferred
        plt.title("Depthmap")
        plt.colorbar()
        plt.axis('off')
        plt.show()
        plt.savefig("test.png", bbox_inches='tight', dpi=150)

    def process_denoising(self, rgb, noiseR, noiseG, noiseB):
     """
     Denoise an RGB image by subtracting detected noise from each channel.

     Args:
        rgb (np.ndarray): Input RGB image of shape (256, 256, 3), dtype=uint8.
        noiseR (float or np.ndarray): Noise estimate for Red channel.
        noiseG (float or np.ndarray): Noise estimate for Green channel.
        noiseB (float or np.ndarray): Noise estimate for Blue channel.

     Returns:
        np.ndarray: Denoised RGB image (uint8).
     """
     # Split channels
     R, G, B = cv2.split(rgb)

     # Subtract noise for each channel
     R = cv2.subtract(np.float32(noiseR) * 2.0,R.astype(np.float32))
     G = cv2.subtract(np.float32(noiseG) * 2.0,G.astype(np.float32))
     B = cv2.subtract(np.float32(noiseB) * 2.0,B.astype(np.float32))

     # Clip values to valid range [0, 255] and convert back to uint8
     R = np.clip(255-R, 0, 255).astype(np.uint8)
     G = np.clip(255-G, 0, 255).astype(np.uint8)
     B = np.clip(255-B, 0, 255).astype(np.uint8)

     # Merge channels back
     denoised = cv2.merge([R, G, B])

     return denoised
    #----------------------------------------------------------------------------------------
    def encodeSkeletonAsDict(self, i, keypoint_names=None, denormalize=True, flipX=False):
     from imageProcessing import castNormalizedXYCoordinatesToOriginalImage
     sk = self.skeletons[i]
     out = {}

     if (keypoint_names is None):
          keypoint_names = self.keypoint_names

     for kp_index, name in enumerate(keypoint_names):
        base = kp_index * 3
        x    = sk[base + 0]
        y    = sk[base + 1]
        v    = sk[base + 2]

        # If denormalizing, convert normalized → original-image coordinates
        if denormalize:
            if None in (self.input_image.shape[1], self.input_image.shape[0], self.keypointXMultiplier, self.keypointYMultiplier, self.keypointXOffset, self.keypointYOffset):
                raise ValueError("Missing normalization parameters for denormalization step.")
            #print("self.input_image.shape = ",self.input_image.shape)
            #print("keypointOffset = ",self.keypointXOffset,",",self.keypointYOffset)
            #print("keypointMultiplier = ",self.keypointXMultiplier,",",self.keypointYMultiplier)
            x, y = castNormalizedXYCoordinatesToOriginalImage(
                                                              self.input_image.shape[1], self.input_image.shape[0],
                                                              self.keypointXMultiplier, self.keypointYMultiplier,
                                                              self.keypointXOffset, self.keypointYOffset,
                                                              x, y
                                                             )

        if (flipX):
             x = 1.0 - x
        out[f"2dx_{name}"] = x
        out[f"2dy_{name}"] = y
        out[f"visible_{name}"] = v
     return out
    #----------------------------------------------------------------------------------------

    def visualize(self,frame,show=True,save=False):
            #if not self.activity is None:
                  #visualize_activity(self.activity, self.labels, threshold=self.activityThreshold)
            #      pass

            # Visualize the heatmaps on the source image
            if show:
               visualize_heatmaps(self.cfg, self.instanceLabels, self.imageIn, self.frameNumber, self.heatmapsOut, self.keypoint_names, threshold=self.heatmap_threshold, drawJoints=self.drawJoints, drawPAFs=self.drawPAFs, screen_w=self.screen_w, screen_h=self.screen_h)
               if self.denoised is not None:
                 cv2.imshow('Denoised RGB', self.denoised)
                 cv2.imshow('Input RGB',    self.imageIn) #If showing denoised then also show input

            #depthmap         =  self.heatmapsOut[17]
            #normals = compute_normals(self.depthmap)
            #if show:
            #  cv2.imshow('Normal', normals)
            #smooth = apply_bilateral_filter(apply_bilateral_filter(normals))
            #if show:
            #  cv2.imshow('Smooth', smooth)

            if (len(self.heatmapsOut)<44):
                 print("Can't visualize with only (",len(self.heatmapsOut)," heatmaps)..")
                 return None


            selected_joints  = self.heatmapsOut[0:17]  # Slicing to get heatmaps 0 through 16
            union_joints     = np.max(selected_joints, axis=0)
            if show:
              cv2.imshow('Joint Heatmap Union', union_joints)

            selected_pafs    = self.heatmapsOut[17:29]  # Slicing to get heatmaps 0 through 16
            union_pafs       = np.max(selected_pafs, axis=0)
            if show:
              cv2.imshow('PAFs Union', union_pafs)

            selected_segms   = self.heatmapsOut[33:61]  # Slicing to get heatmaps
            union_segms      = np.max(selected_segms, axis=0)
            not_segmented = (union_segms <= 0.0).astype(np.float32)
            if show:
              cv2.imshow('Class segmentation Union', union_segms)
              cv2.imshow('Unsegmented', not_segmented)

            improved_depth   = self.heatmapsOut[self.chanDepth] #17
            if ('heatmapAddNormals' in self.cfg) and (self.cfg['heatmapAddNormals']):
              if show:
                 visualize_normals(self.cfg, self.heatmapsOut) 
              depthmap         =  self.heatmapsOut[self.chanDepth] #17
              normalX          =  self.heatmapsOut[self.chanNormalX] #18
              normalY          =  self.heatmapsOut[self.chanNormalY] #19
              normalZ          =  self.heatmapsOut[self.chanNormalZ] #20
              if self.depth_iterations > 0:
                improved_depth = improve_depthmap(depthmap, normalX, normalY, normalZ, learning_rate = 0.02, iterations = self.depth_iterations)
              else:
                improved_depth = depthmap
              improved_depth   = apply_bilateral_filter(improved_depth)
              #improved_depth = integrate_normals(normalX, normalY, normalZ, initial_depth=depthmap, iterations=5000)
              if show:
                 cv2.imshow('Improved Depth', improved_depth)
                 #print("Improved Depth 8 bit MIN: ",np.min(improved_depth))
                 #print("Improved Depth 8 bit MAX: ",np.max(improved_depth))

            #if(self.saveFrame): 
            #  self.plot_depthmap_with_matplotlib(improved_depth,reverse=True)
            #   self.plot_depthmap_with_matplotlib(self.heatmapsOut[self.chanDepth])

            if (self.heatmap_16b is not None):
                 #print("Depth 16 bit MIN: ",np.min(self.heatmap_16b))
                 #print("Depth 16 bit MAX: ",np.max(self.heatmap_16b))
                 #self.heatmap_16b = self.heatmap_16b / 32767
                 self.heatmap_16b = 0.5 + ( self.heatmap_16b / 65534 )
                 if show:
                    cv2.imshow('Depth 16-bit', self.heatmap_16b)



            if (self.saveFrame):
               self.saveFrame = 0;
               for i in range(len(self.heatmapsOut)):
                  cv2.imwrite("hm%u.png" % i , self.heatmapsOut[i])


              #Normals..
              #self.depthmap = improved_depth * 255.0
              #normalsImp = compute_normals(self.depthmap)
              #if show:
              #  cv2.imshow('Normals from improved depthmap', normalsImp)

            if not self.skeletons is None:
               if show:
                 drawSkeletons(self.skeletons, self.keypoint_names, self.cfg["keypoint_parents"], image_shape=(480, 640, 3))


            if (not self.labeled_map is None) and (not self.bounding_boxes is None):
               labelsVisualiation = draw_labeled_blobs(self.labeled_map, self.bounding_boxes)
               if show:
                 cv2.imshow('Person IDs', labelsVisualiation)

            if (self.description):
             if show:
               textWindow = np.full( (50,1800,3), 0, dtype = np.uint8)
               cv2.putText(textWindow, "%s" % self.description, (8,30),  cv2.FONT_HERSHEY_SIMPLEX, 1.3, (123, 123, 123), 1)
               cv2.imshow('Description', textWindow)
            


            #Do main visualization
            cv2.putText(frame, "NN @ %0.2f Hz" % self.keypoints_model.hz, (8,23),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (123, 123, 123), 1)
            cv2.putText(frame, "NN @ %0.2f Hz" % self.keypoints_model.hz, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
            visRGB = visualization(frame, self.frameNumber, self.keypoint_results,self.keypoint_depths, self.keypoint_names) #, self.blobs

            if self.chanFloor >= 0 and len(self.heatmapsOut) > max(self.chanFloor, self.chanNormalX, self.chanNormalY, self.chanNormalZ):
                _n_hm = len(self.heatmapsOut)
                _ceiling_mask = self.heatmapsOut[self.chanCeiling] if (self.chanCeiling >= 0 and self.chanCeiling < _n_hm) else None
                _wall_mask    = self.heatmapsOut[self.chanWall]    if (self.chanWall    >= 0 and self.chanWall    < _n_hm) else None
                _floor_plane = compute_floor_plane_normal(
                    self.heatmapsOut[self.chanNormalX],
                    self.heatmapsOut[self.chanNormalY],
                    self.heatmapsOut[self.chanNormalZ],
                    self.heatmapsOut[self.chanFloor],
                    ceiling_mask = _ceiling_mask,
                    wall_mask    = _wall_mask,
                )
                draw_floor_normal_axes(visRGB, _floor_plane)


            if (self.illustrate):
                 normal_output =  np.stack((normalX, normalY, normalZ), axis=-1)
                 #normalized_frame = cv2.normalize(improved_depth, None, 0, 1, cv2.NORM_MINMAX)
                 improved_depth = (improved_depth.clip(min=0) * 255).astype(np.uint8)
                 images_resized = (256,256)
                 imagesF = [visRGB, normal_output, depthmap, improved_depth, union_pafs, union_joints, union_segms, self.person_union,
                           self.heatmapsOut[self.chanText], self.heatmapsOut[self.chanVehicle], self.heatmapsOut[self.chanAnimal], 
                           self.heatmapsOut[self.chanObject],self.heatmapsOut[self.chanFurniture], self.heatmapsOut[self.chanAppliance]]  
                 labels = ['RGB+Pose', 'Normals', 'Depth', 'Depth Improved', 'PAF Union',  'Joint Heatmap Union', 'Class Union', 'Person',
                            'Text',  'Vehicle', 'Animal', 
                            'Object','Furniture', 'Appliance'
                          ] #,  'Obstacle' ,'Material', 'Nature' , 'Building' 

                 
                 from illustrate import compose_visualization                                                                             #self.keypoints_model.hz
                 images_resized = (230,230)
                 images = resize_images(imagesF,images_resized)
                 illu = compose_visualization(images, labels, description=self.description, bg_size=(1920, 1080), img_size=images_resized, frameRate=self.frameRateList, relative_border_size = 20 / self.cfg['inputHeight'])
                 if show:
                   cv2.imshow('Illustration',illu) 

            if show:
              if (len(self.monitor)>0):
                 for i,entry in enumerate(self.monitor):
                      tL       = entry[3]
                      tPlot = np.full((255,255,3),0, dtype = np.uint8) 
                      drawSinglePlotValueList(self.monitorValues[i],(0,255,0),tL,tPlot,10,10,220,220, -120, 120)
                      cv2.imshow(tL,tPlot) 
                 visRGB = visualize_monitors(visRGB, depthmap, self.monitor)
                 monVis = visualize_monitors(self.imageGivenToNN, depthmap, self.monitor)
                 monVis = cv2.cvtColor(monVis, cv2.COLOR_BGR2RGB) 
                 if show:
                   cv2.imshow('Monitor Visualization', monVis) 



            if show:
              cv2.imshow('Overlay', visRGB) 

            if (save):
              normal_output =  np.stack((normalX, normalY, normalZ), axis=-1)
              cv2.imwrite("normals.png", normal_output)
              cv2.imwrite("overlay.png", visRGB)
              cv2.imwrite("depth.png", depthmap)
              cv2.imwrite("improved_depth.png", improved_depth)
              cv2.imwrite("joint_heatmap_union.png", union_joints)
              cv2.imwrite("pafs_union.png", union_pafs)
              cv2.imwrite("class_union.png", union_segms)
              for hm in range(43+1):
                cv2.imwrite("hm%02u.png"%hm, self.heatmapsOut[hm])


            if show:
             if (self.frameNumber==1):
                # Layout strategy:
                #   Monitor 1  [0 .. screen_w//3)       Fixed positions (MONITOR1_FIXED below)
                #   Monitors 2+3 [screen_w//3 .. end)   Overflow named windows + per-channel
                #                                       heatmaps (heatmaps placed by
                #                                       visualize_heatmaps, not here)

                normals_on  = ('heatmapAddNormals' in self.cfg) and self.cfg['heatmapAddNormals']
                mon1_w      = self.screen_w // 3   # width of monitor 1 in the 3-monitor fixed layout

                # ── helpers ──────────────────────────────────────────────────
                primary_specs   = []
                processed_specs = []

                def _pri(name, img):
                    if img is not None and img.ndim >= 2:
                        primary_specs.append((name, img.shape[1], img.shape[0]))

                def _pro(name, img):
                    if img is not None and img.ndim >= 2:
                        processed_specs.append((name, img.shape[1], img.shape[0]))

                # ── primary outputs (Monitor 1) ───────────────────────────────
                _pri("Overlay",           visRGB)
                if (self.labeled_map is not None) and (self.bounding_boxes is not None):
                    _pri("Person IDs",    labelsVisualiation)
                primary_specs.append(("Skeletons", 640, 480))
                if self.description:
                    primary_specs.append(("Description", min(1800, mon1_w - 12), 50))
                primary_specs.append(("Threshold Controls", 400, 150))
                if self.denoised is not None:
                    _pri("Denoised RGB", self.denoised)
                    _pri("Input RGB",    self.imageIn)
                if len(self.monitor) > 0:
                    _pri("Monitor Visualization", monVis)

                # ── processed maps (Monitor 2) ────────────────────────────────
                _pro("Depthmap",                self.heatmapsOut[self.chanDepth] if len(self.heatmapsOut) > self.chanDepth else None)
                _pro("Joint Heatmap Union",     union_joints)
                _pro("PAFs Union",              union_pafs)
                _pro("Class segmentation Union",union_segms)
                _pro("Unsegmented",             not_segmented)
                if normals_on:
                    _pro("Improved Depth",      improved_depth)
                    _pro("Combined Normals",    self.heatmapsOut[self.chanNormalX])
                    _pro("Normals X",           self.heatmapsOut[self.chanNormalX])
                    _pro("Normals Y",           self.heatmapsOut[self.chanNormalY])
                    _pro("Normals Z",           self.heatmapsOut[self.chanNormalZ])
                if self.heatmap_16b is not None:
                    _pro("Depth 16-bit",        self.heatmap_16b)
                if hasattr(self, 'chanText') and len(self.heatmapsOut) > self.chanText:
                    _pro("Text",                self.heatmapsOut[self.chanText])

                # ── pack and apply ────────────────────────────────────────────
                is_compact = self.cfg.get('outputChannels', 73) < 45

                if is_compact:
                    # Fewer output channels — all named windows fit on monitor 1.
                    # Auto-pack everything rather than using the 3-monitor fixed grid.
                    layout = arrange_windows(
                        primary_specs + processed_specs,
                        screen_w=mon1_w,
                        screen_h=self.screen_h,
                        offset_x=0,
                    )
                else:
                    # Monitor 1: fixed layout optimised for 1920×1080.
                    # Any named window absent from this dict is auto-placed on monitors 2+3.
                    MONITOR1_FIXED = {
                        "Overlay":                  (1,    1),
                        "Description":              (0,    930),
                        "Person IDs":               (0,    650),
                        "Threshold Controls":       (0,    650),
                        "Depthmap":                 (480,  0),
                        "Improved Depth":           (480,  350),
                        "Combined Normals":         (480,  650),
                        "Class segmentation Union": (820,  0),
                        "Joint Heatmap Union":      (830,  350),
                        "PAFs Union":               (830,  650),
                        "Normals X":                (1180, 0),
                        "Normals Y":                (1180, 350),
                        "Normals Z":                (1180, 650),
                        "Very Close":               (1550, 0),
                        "Person":                   (1550, 270),
                        "Text":                     (1550, 530),
                    }
                    layout = dict(MONITOR1_FIXED)   # copy; only entries whose windows exist matter

                    # Overflow: windows not in the fixed layout go to monitors 2+3
                    placed = set(MONITOR1_FIXED)
                    overflow = [(n, w, h) for n, w, h in primary_specs + processed_specs
                                if n not in placed]
                    if overflow:
                        third = self.screen_w // 3
                        layout.update(arrange_windows(
                            overflow,
                            screen_w=self.screen_w - third,
                            screen_h=self.screen_h,
                            offset_x=third,
                        ))

                apply_window_layout(layout)

                # User-specified overrides are applied last and take precedence
                if self.window_arrangement:
                    for command in self.window_arrangement:
                        try:
                            cv2.moveWindow(command[2], int(command[0]), int(command[1]))
                        except Exception:
                            pass  # window may not exist yet (common on Windows before first imshow)
                    try:
                        cv2.waitKey(1)
                    except Exception:
                        pass
 
#----------------------------------------------------------------------------------------
class PoseEstimatorTiler:
    def __init__(self, pose_estimator, tile_size=(256, 256), overlap=(0, 0)):
        """
        Initializes the PoseEstimatorTiler.
        
        Args:
            pose_estimator: An instance of PoseEstimator2D to process each tile.
            tile_size: A tuple (width, height) specifying the size of each tile.
            overlap: A tuple (x_overlap, y_overlap) specifying the overlap between tiles in pixels.
        """
        self.pose_estimator = pose_estimator
        self.tile_width, self.tile_height = tile_size
        self.x_overlap, self.y_overlap = overlap
        self.stitched_image = None

    def _split_image_into_tiles(self, image):
        """
        Splits the image into tiles with optional overlapping areas.

        Args:
            image: Input image as a NumPy array.

        Returns:
            tiles: List of image tiles.
            coords: List of top-left corner coordinates of each tile.
        """
        h, w, c = image.shape
        tiles = []
        coords = []

        for y in range(0, h - self.tile_height + 1, self.tile_height - self.y_overlap):
            for x in range(0, w - self.tile_width + 1, self.tile_width - self.x_overlap):
                tile = image[y:y + self.tile_height, x:x + self.tile_width]
                tiles.append(tile)
                coords.append((x, y))

        # Handle any remaining areas (rightmost and bottommost edges)
        if h % (self.tile_height - self.y_overlap) != 0:
            for x in range(0, w - self.tile_width + 1, self.tile_width - self.x_overlap):
                tile = image[-self.tile_height:, x:x + self.tile_width]
                tiles.append(tile)
                coords.append((x, h - self.tile_height))
        if w % (self.tile_width - self.x_overlap) != 0:
            for y in range(0, h - self.tile_height + 1, self.tile_height - self.y_overlap):
                tile = image[y:y + self.tile_height, -self.tile_width:]
                tiles.append(tile)
                coords.append((w - self.tile_width, y))

        return tiles, coords

    def _stitch_tiles(self, tiles, coords, image_shape):
        """
        Stitches tiles back into a single image.

        Args:
            tiles: List of processed tiles.
            coords: List of top-left corner coordinates of each tile.
            image_shape: Original image shape (height, width, channels).

        Returns:
            result: Reconstructed image with the same dimensions as the input image.
        """
        result = np.zeros(image_shape, dtype=np.uint8)

        for tile, (x, y) in zip(tiles, coords):
            h, w, c = tile.shape
            result[y:y + h, x:x + w] = tile

        return result

    def process(self, image):
        """
        Processes a larger image by splitting it into tiles, running PoseEstimator2D, 
        and stitching the results back together.

        Args:
            image: Input image as a NumPy array.

        Returns:
            stitched_image: Resultant image with the same dimensions as the input image.
        """
        # Split image into tiles
        tiles, coords = self._split_image_into_tiles(image)

        # Process each tile
        processed_tiles = [self.pose_estimator.process(tile) for tile in tiles]

        # Stitch tiles back into a single image
        self.stitched_image = self._stitch_tiles(processed_tiles, coords, image.shape)

        return self.stitched_image

    def visualize(self,frame,save=False):
        #print("No visualization")
        cv2.imshow('Overlay', self.stitched_image) 
#----------------------------------------------------------------------------------------
if __name__ == '__main__':
    from run2DPoseEstimator import main_pose_estimation
    main_pose_estimation()

