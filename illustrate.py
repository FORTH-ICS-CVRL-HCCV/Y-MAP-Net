import os
import cv2
import numpy as np

illustrationFrames = 0
#bkg = cv2.imread("illustration/composite_bkg.png")
logo = cv2.imread("illustration/logo.png")


# Open background video
background_video_path = "illustration/DISABLEDbg5.mp4"
if os.path.exists(background_video_path):
     background_video = cv2.VideoCapture(background_video_path)
     if not background_video.isOpened():
        raise FileNotFoundError(f"Unable to open background video: {bg_video_path}")
else:
     background_video = None

os.system("rm composite_*.png")



def get_next_background(video,bg_size):
    """
    Retrieves the next frame from the video. If the video ends, it restarts.
    """
    ret, frame = video.read()
    if not ret:
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop back to the first frame
        ret, frame = video.read()
    return cv2.resize(frame, bg_size)


def convert_to_three_channel(image):
    """
    Converts a single-channel grayscale image to a 3-channel image 
    by replicating the grayscale values across each channel. 
    If the image is already 3-channel, it returns it unchanged.

    Parameters:
    - image: np.ndarray, the input image array

    Returns:
    - np.ndarray, the 3-channel version of the input image
    """
    # Check if the image is single-channel
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        # Convert to 3-channel by stacking the same data on each channel
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image

def load_images_and_labels(filenames):
    """
    Loads images from disk and assigns labels based on filenames.
    
    Returns:
    - images (list): List of loaded images.
    - labels (list): List of corresponding labels for each image.
    """
    # Filepaths for images

    
    labelMap =  {'overlay':'RGB+Pose',
                 'normals':'Normals',
                 'depth':'Depth',
                 'improved_depth':'Depth Improved',
                 'pafs_union':'PAFs',
                 'class_union':'Class Union',
                 'joint_heatmap_union':'Joint Heatmap Union',
                 'hm33':'Text',
                 'hm34':'Person',
                 'hm35':'Vehicle',
                 'hm36':'Animal',
                 'hm37':'Object',
                 'hm38':'Furniture',
                 'hm39':'Appliance',
                 'hm40':'Material',
                 'hm41':'Obstacle',
                 'hm42':'Building',
                 'hm43':'Nature'
                 }

    # Load images and labels
    images = []
    labels = []
    for filename in filenames:
        img = cv2.imread(filename)
        if img is not None:
            images.append(img)
            labelFromFilename = filename.split('.')[0]
            if labelFromFilename in labelMap:
               label = labelMap[labelFromFilename]
            else:
               label = labelFromFilename
            
            labels.append(label)  # Label as filename without extension
        else:
            print(f"Warning: {filename} could not be loaded.")
    

    return images, labels


def calculateRelativeValueIllustration(y,h,value,minimum,maximum):
    if (maximum==minimum):
       return int(y + (h/2)) 
    #-------------------------------------------------
    #TODO IMPROVE THIS!
    vRange = (maximum - minimum)
    return int( y + (h/2) - ( value / vRange ) * (h/2) )

def drawSinglePlotValueListIllustration(valueListRAW,color,itemName,image,x,y,w,h,minimumValue=None,maximumValue=None):
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
            previousValue = calculateRelativeValueIllustration(y,h,valueList[frameID-1],minimumValue,maximumValue)
            nextValue     = calculateRelativeValueIllustration(y,h,valueList[frameID],minimumValue,maximumValue)
            #-------------------------------------------------------------------------------------
            jointPointPrev = (int(x+ frameID-1),      previousValue )
            jointPointNext = (int(x+ frameID),        nextValue )
            if (itemName=="hip_yrotation"):  
                color=(0,0,255) 
            
            cv2.line(image, pt1=jointPointPrev, pt2=jointPointNext, color=color, thickness=1)  

    org = (int(x+len(valueList)), calculateRelativeValueIllustration(y,h,valueList[len(valueList)-1],minimumValue,maximumValue) ) 
    message =  '%0.2f' % (valueList[len(valueList)-1]) 
    image = cv2.putText(image, message , org, font, fontScale, (0,0,0), thickness, cv2.LINE_AA)

    org = (1+int(x+len(valueList)), 1+calculateRelativeValueIllustration(y,h,valueList[len(valueList)-1],minimumValue,maximumValue) ) 
    image = cv2.putText(image, message , org, font, fontScale, color, thickness, cv2.LINE_AA)


#python3 run2DPoseEstimator.py --illustrate --from /media/ammar/MAGICIAN16TB/Magician/tofas.mp4 --crop 89 50 480 450
def compose_visualization(images, labels, description=None, bg_size=(1920, 1080), img_size=(256, 256), 
                          text_color=(255, 255, 255), margin=20, save=True, frameRate=None, 
                          relative_border_size=0.1, simpleBackground=None, corner_radius=20):  
    """
    Creates a composited image with visualizations placed inside rounded rectangles.

    Parameters:
    - images (list): List of images to place.
    - labels (list): Corresponding labels.
    - bg_size (tuple): Size of background canvas.
    - img_size (tuple): Each visualization size.
    - text_color (tuple): Color for label text.
    - margin (int): Padding between visualizations.
    - save (bool): Whether to save the image.
    - frameRate (float or list): Optional framerate to annotate.
    - relative_border_size (float): Crop border fraction.
    - simpleBackground (int or None): Background fill value.
    - corner_radius (int): Rounded corner radius.

    Returns:
    - composite (np.array): Final composited image.
    """

    global background_video
    if background_video is None:
        #If no background video was found set it to (255,255,255) white / (0,0,0) black
        simpleBackground=255

    if simpleBackground is None:
        composite = get_next_background(background_video, (bg_size[0], bg_size[1]))
        composite = cv2.GaussianBlur(composite, (5, 5), 1) #Smooth background to make it less prevalent
        composite = cv2.GaussianBlur(composite, (5, 5), 1) #Smooth background to make it less prevalent
    else:
        composite = np.full((bg_size[1], bg_size[0], 3), simpleBackground, dtype=np.uint8)

    global logo
    if (logo is not None):
        # Create a mask where logo is not black
        mask = np.any(logo != [0, 0, 0], axis=-1)  # shape: (h, w), dtype: bool
        # Overlay the logo onto the composite using the mask
        composite[mask] = logo[mask]


    cols = bg_size[0] // (img_size[0] + margin)
    rows = (len(images) + cols - 1) // cols
    max_rows = bg_size[1] // (img_size[1] + img_size[1] // 3 + margin)
    rows = min(rows, max_rows)

    start_x = (bg_size[0] - (cols * img_size[0] + (cols - 1) * margin)) // 2
    start_y = (bg_size[1] - (rows * img_size[1] + (rows - 1) * (img_size[1] // 3) + rows * margin)) // 2

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.6
    font_thickness = 1

    for i, (img, label) in enumerate(zip(images, labels)):
        border_size = int(img.shape[0] * relative_border_size)
        if border_size > 0 and img.shape[0] > 2 * border_size:
            img = img[border_size:-border_size, :]

        row = i // cols
        col = i % cols

        x = start_x + col * (img_size[0] + margin)
        y = start_y + row * (img_size[1] + img_size[1] // 3 + margin)

        img_resized = convert_to_three_channel(cv2.resize(img, img_size))

        box_w = img_size[0]
        box_h = img_size[1] + 30  # Extra for label text
        box = np.zeros((box_h, box_w, 3), dtype=np.uint8)
        overlay = box.copy()

        # Draw rounded rectangle background
        mask = np.zeros_like(box, dtype=np.uint8)
        cv2.rectangle(mask, (corner_radius, 0), (box_w - corner_radius, box_h), (255, 255, 255), -1)
        cv2.rectangle(mask, (0, corner_radius), (box_w, box_h - corner_radius), (255, 255, 255), -1)

        # Draw corner circles
        cv2.circle(mask, (corner_radius, corner_radius), corner_radius, (255, 255, 255), -1)
        cv2.circle(mask, (box_w - corner_radius, corner_radius), corner_radius, (255, 255, 255), -1)
        cv2.circle(mask, (corner_radius, box_h - corner_radius), corner_radius, (255, 255, 255), -1)
        cv2.circle(mask, (box_w - corner_radius, box_h - corner_radius), corner_radius, (255, 255, 255), -1)

        # Fill rounded rectangle with semi-transparent background
        overlay[mask[:, :, 0] == 255] = (30, 30, 30)  # dark gray

        # Apply slight blur to simulate soft edge
        overlay = cv2.GaussianBlur(overlay, (5, 5), 1)

        # Composite the image and label
        overlay[0:img_size[1], 0:img_size[0]] = img_resized

        # Add label centered at the bottom
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        text_x = (img_size[0] - text_size[0]) // 2
        text_y = img_size[1] + text_size[1] + 5

        cv2.putText(overlay, label, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)


        if y + box_h <= bg_size[1] and x + box_w <= bg_size[0]:
            region = composite[y:y + box_h, x:x + box_w]
            mask_bool = mask[:, :, 0] > 0
            region[mask_bool] = overlay[mask_bool]

        # Paste the overlay into the composite
        #if y + box_h <= bg_size[1] and x + box_w <= bg_size[0]:
        #    composite[y:y + box_h, x:x + box_w] = overlay
        #else:
        #    print(f"Overlay box at index {i} exceeds background bounds.")

    if frameRate is not None:
        if type(frameRate) is list:
          drawSinglePlotValueListIllustration(frameRate,(0,200,0),"Neural Network Framerate (Hz)",composite,1650,50,200,100,minimumValue=0,maximumValue=25)
        else:
          fr_text = "Framerate: %0.2f Hz" % frameRate
          text_size = cv2.getTextSize(fr_text, font, font_scale, font_thickness)[0]
          fx = bg_size[0] - text_size[0] - 40
          fy = text_size[1] + 50
          cv2.putText(composite, fr_text, (fx+2, fy+2), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
          cv2.putText(composite, fr_text, (fx, fy), font, font_scale, (0,0,0), font_thickness, cv2.LINE_AA)

    if description is not None:
        desc_text = f"Caption: {description}"
        text_size = cv2.getTextSize(desc_text, font, font_scale, font_thickness)[0]
        dx = 420 #(bg_size[0] - text_size[0]) // 2 #Not auto centering makes it less dizzying
        dy = bg_size[1] - 100
        cv2.putText(composite, desc_text, (dx+2, dy+2), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.putText(composite, desc_text, (dx, dy), font, font_scale, (0,0,0) , font_thickness, cv2.LINE_AA)

    if save:
        global illustrationFrames
        cv2.imwrite("composite_%05u.png" % illustrationFrames, composite)
        illustrationFrames += 1

    return composite



if __name__ == '__main__':
  filenames = [
        "overlay.png", "normals.png", "depth.png",  "improved_depth.png",  
        "joint_heatmap_union.png", "pafs_union.png", "class_union.png",  
        "hm33.png", "hm34.png", "hm35.png", "hm36.png", "hm37.png", "hm38.png",
        "hm39.png", "hm40.png", "hm41.png", "hm42.png", "hm43.png"]

  # Load images and labels
  images, labels = load_images_and_labels(filenames)

  # Generate composite visualization
  composite_image = compose_visualization(images, labels, save=True)

  # Show the result
  cv2.imshow("Composite Visualization", composite_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

