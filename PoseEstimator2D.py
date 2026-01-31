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
#----------------------------------------------------------------------------------------
def label_heatmap(cfg,heatmapID,keypoint_names,instanceLabels):

        heatmapNames = [
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
        "left/right pattern"  #48
        ]

        if ('heatmaps' in cfg):
             heatmapNames = cfg['heatmaps']           

        title = "#%u" % heatmapID

        if (heatmapID<len(heatmapNames)):
            title = heatmapNames[heatmapID]
        return title


        """
        if (heatmapID<len(keypoint_names)):
            title = keypoint_names[heatmapID]

        elif (heatmapID==17): 
             title = "rhand->relbow"
        elif (heatmapID==18): 
             title = "relbow->rshoulder"
        elif (heatmapID==19): 
             title = "rshoulder->head"
        elif (heatmapID==20): 
             title = "rfoot->rknee"
        elif (heatmapID==21): 
             title = "rknee->rhip"
        elif (heatmapID==22): 
             title = "rhip->head"
        elif (heatmapID==23): 
             title = "lfoot->lknee"
        elif (heatmapID==24): 
             title = "lknee->lhip"
        elif (heatmapID==25): 
             title = "lhip->head"
        elif (heatmapID==26): 
             title = "lhand->lelbow"
        elif (heatmapID==27): 
             title = "lelbow->lshoulder"
        elif (heatmapID==28): 
             title = "lshoulder->head"
        elif (heatmapID==29): 
             title = "Depthmap"
        elif (heatmapID==30): 
             title = "Normals X"
        elif (heatmapID==31): 
             title = "Normals Y"
        elif (heatmapID==32): 
             title = "Normals Z"
        elif (heatmapID==33): 
             title = "Text"
        elif (heatmapID==34): 
             title = "Person" 
        elif (heatmapID==35): 
             title = "Vehicle" 
        elif (heatmapID==36): 
             title = "Animal" 
        elif (heatmapID==37): 
             title = "Object" 
        elif (heatmapID==38): 
             title = "Furniture" 
        elif (heatmapID==39): 
             title = "Appliance" 
        elif (heatmapID==40): 
             title = "Material"  
        elif (heatmapID==41): 
             title = "Obstacle"  
        elif (heatmapID==42): 
             title = "Building"  
        elif (heatmapID==43): 
             title = "Nature" 
        elif (heatmapID==44): 
             title = "Very Close" 
        elif (heatmapID==45): 
             title = "NoiseR" 
        elif (heatmapID==46): 
             title = "NoiseG" 
        elif (heatmapID==47): 
             title = "NoiseB" 
        elif (heatmapID==45): 
             title = "Empty / Unlabeled" 
        #if (heatmapID>22):
        #     title = " %s "% get_key_from_index(instanceLabels,heatmapID-22)
        return title
        """
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
def visualize_heatmaps(cfg, instanceLabels , imageIn, frameNumber, heatmapsOut, keypoint_names, threshold=0.0, drawJoints=False, drawPAFs=False, resizeHighHeatmaps=True, showClassHeatmaps=True, showDepthInsteadOfMask=False):
    i=0 
    wnd_x = 1210
    wnd_y = 0
 
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
 
        stepX=heatmap.shape[1]+140 
        stepY=heatmap.shape[0]+80

        #Get Window Title
        title = label_heatmap(cfg,i,keypoint_names,instanceLabels) 

        if (drawPAFs and i>=17 and i<=28) or  (drawJoints and i<=17) or ((i>=29) and ((showClassHeatmaps) or (i<=33)) ): # 17 / 20 
         cv2.imshow(title, heatmap)
         if (frameNumber==1):
          cv2.moveWindow(title, wnd_x, wnd_y) 
          wnd_y+=stepY

          #if i>=17:
          #    wnd_y+=stepY 

          if ( wnd_y > 900 ):
            wnd_x+=stepX
            wnd_y = 25

        i=i+1
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
class PoseEstimator2D:
    def __init__(self, modelPath, threshold=30, keypoint_threshold=50.0, engine="tensorflow", profiling = False, illustrate=False, monitor=list(), window_arrangement=list()):
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
                                             outputChannels = self.cfg['outputChannels']
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
        #---------------------------------------------------------------------
        self.keypointXMultiplier = 1.0 
        self.keypointYMultiplier = 1.0
        self.keypointXOffset     = 0
        self.keypointYOffset     = 0
        #---------------------------------------------------------------------
        self.monitor            = monitor
        self.monitorValues      = list()
        if (len(monitor)>0):
                for i in range(len(monitor)):
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

        # "Programmable" heatmap IDs
        if "heatmaps" in self.cfg:
          self.chanDenoiseR = retrieveHeatmapIndex(self.cfg['heatmaps'],"Denosing R")
          self.chanDenoiseG = retrieveHeatmapIndex(self.cfg['heatmaps'],"Denosing G")
          self.chanDenoiseB = retrieveHeatmapIndex(self.cfg['heatmaps'],"Denosing B")
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
            self.labeled_map, self.bounding_boxes = segment_and_label_persons(self.person_union,threshold=self.bounding_boxes_threshold, min_area=self.bounding_boxes_minarea)
            """
            #Disabled for performance
            """
            if ("keypoint_children" in self.cfg):
              self.skeletons = resolveJointHierarchyNew(self.keypoints_predictions,
                                                        #self.heatmapsOut[0:16],
                                                        self.heatmapsOut[17:29], # 17..28 inclusive
                                                        self.depthmap,
                                                        self.cfg["keypoint_names"],
                                                        self.cfg["keypoint_parents"],
                                                        self.cfg["keypoint_children"],
                                                        self.cfg["paf_parents"], 
                                                        person_label_map=self.labeled_map,        # <<< NEW
                                                        threshold=self.keypoint_threshold
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
               visualize_heatmaps(self.cfg, self.instanceLabels, self.imageIn, self.frameNumber, self.heatmapsOut, self.keypoint_names, threshold=self.heatmap_threshold, drawJoints=self.drawJoints, drawPAFs=self.drawPAFs )
               cv2.imshow('Denoised RGB', self.denoised)
               cv2.imshow('Input RGB',    self.imageIn)

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
              improved_depth   = improve_depthmap(depthmap, normalX, normalY, normalZ, learning_rate = 0.02, iterations = 20) # learning_rate = 0.01, iterations = 35
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
                #black = np.zeros((1080,1920,3))
                #cv2.imshow('BG',black)
                #cv2.waitKey(1) 
                #cv2.moveWindow("BG",0,-10)
                #Make a three screen demo layout
                top = 20
                #Fit basic stuff in a 1920x1080 screen
                cv2.moveWindow("Description",0,930) 
                cv2.moveWindow("Depthmap",480,top) 
                cv2.moveWindow("Improved Depth",480,350) 
                cv2.moveWindow("Combined Normals",480,650) 
                cv2.moveWindow("Class segmentation Union",820,top)
                cv2.moveWindow("Unsegmented",5409,350)
                cv2.moveWindow("Joint Heatmap Union",830,350)
                cv2.waitKey(1)
                cv2.moveWindow("PAFs Union",830,650)
                cv2.moveWindow("Normals X",1180,top)
                cv2.moveWindow("Normals Y",1180,350)
                cv2.moveWindow("Normals Z",1180,650)
                cv2.moveWindow("Very Close",1550,top) 
                cv2.waitKey(1)
                #cv2.moveWindow("Close",1550,270) 
                #cv2.moveWindow("Person",230,710) 
                #cv2.moveWindow("Text",-40,570)
  
                cv2.moveWindow("Person IDs",0,600)  
                cv2.moveWindow("Threshold Controls",5350,650) 
                cv2.waitKey(1)

                #cv2.moveWindow("Person",1550,270)  #Far
                cv2.moveWindow("Text",450,top)  #Very Far
                cv2.moveWindow("Overlay",1,top)
                cv2.waitKey(1)


                cv2.moveWindow("Skeletons",4646,350) 
                cv2.moveWindow("Depth 16-bit",4650,top)
                cv2.moveWindow("Denoised RGB",5025,top)
                cv2.moveWindow("Input RGB",5409,top)
                cv2.waitKey(1)
                #==========================================
                if (len(self.window_arrangement)>0):
                    for command in self.window_arrangement:
                           x = int(command[0])
                           y = int(command[1])
                           window_title = command[2]
                           cv2.moveWindow(window_title,x,y)
                    cv2.waitKey(1)
 
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

