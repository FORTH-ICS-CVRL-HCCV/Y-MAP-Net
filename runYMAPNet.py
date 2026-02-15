#!/usr/bin/python3

"""
Author : "Ammar Qammaz"
Copyright : "2024 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"
"""

#Dependencies should be :
#tensorflow-2.16.1 needs CUDA 12.3, CUDNN 8.9.6 and is built with Clang 17.0.6 Bazel 6.5.0
#python3 -m pip install tf_keras tensorflow==2.16.1 numpy tensorboard opencv-python wget

import os
import sys
from tools import bcolors
#----------------------------------------------------------------------------------------
try:
 import cv2
 import numpy as np
except Exception as e:
 print(bcolors.WARNING,"Could not import libraries!",bcolors.ENDC)
 print("An exception occurred:", str(e))
 print("Issue:\n source venv/bin/activate")
 print("Before running this script")
 sys.exit(1)
#----------------------------------------------------------------------------------------
useGPU = True
if (len(sys.argv)>1):
       #print('Argument List:', str(sys.argv))
       for i in range(0, len(sys.argv)): 
           if (sys.argv[i]=="--cpu"):
             useGPU = False
           if (sys.argv[i]=="--update"):
              os.system("rm -rf 2d_pose_estimation/")
              os.system("rm 2d_pose_estimation.zip")
              os.system("wget http://ammar.gr/2d_pose_estimation.zip && unzip 2d_pose_estimation.zip")

# Set CUDA_VISIBLE_DEVICES to an empty string to force TensorFlow to use the CPU
if (not useGPU):
     os.environ['CUDA_VISIBLE_DEVICES'] = '' #<- Force CPU
#----------------------------------------------------------------------------------------
def getCaptureDeviceFromPath(videoFilePath,videoWidth,videoHeight,videoFramerate=30):
  #------------------------------------------
  if (videoFilePath=="esp"):
     from espStream import ESP32CamStreamer
     cap = ESP32CamStreamer()
  if (videoFilePath=="screen"):
     from screenStream import ScreenGrabber
     cap =  ScreenGrabber(region=(0,0,videoWidth,videoHeight))
  elif (videoFilePath=="webcam"):
     cap = cv2.VideoCapture(0)
     cap.set(cv2.CAP_PROP_FPS,videoFramerate)
     cap.set(cv2.CAP_PROP_FRAME_WIDTH, videoWidth)
     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, videoHeight)
  elif (videoFilePath=="/dev/video0"):
     cap = cv2.VideoCapture(0)
     cap.set(cv2.CAP_PROP_FPS,videoFramerate)
     cap.set(cv2.CAP_PROP_FRAME_WIDTH, videoWidth)
     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, videoHeight)
  elif (videoFilePath=="/dev/video1"):
     cap = cv2.VideoCapture(1)
     cap.set(cv2.CAP_PROP_FPS,videoFramerate)
     cap.set(cv2.CAP_PROP_FRAME_WIDTH, videoWidth)
     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, videoHeight)
  elif (videoFilePath=="/dev/video2"):
     cap = cv2.VideoCapture(2)
     cap.set(cv2.CAP_PROP_FPS,videoFramerate)
     cap.set(cv2.CAP_PROP_FRAME_WIDTH, videoWidth)
     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, videoHeight)
  else:
     from tools import checkIfPathIsDirectory
     if (checkIfPathIsDirectory(videoFilePath) and (not "/dev/" in videoFilePath) ):
        from folderStream import FolderStreamer
        cap = FolderStreamer(path=videoFilePath,width=videoWidth,height=videoHeight)
     else:
        cap = cv2.VideoCapture(videoFilePath)
  return cap 
#----------------------------------------------------------------------------------------

# Function to save frame to disk and upload it to web server
def save_and_upload_frame(frame):
    # Save frame to disk
    print("Saving frame")
    cv2.imwrite('frame.jpg', frame)
    print("Uploading frame")    
    # Upload frame to web server using curl
    os.system("curl -F file=@frame.jpg http://ammar.gr/datasets/uploads.php")

def prevent_screensaver(): 
    #os.system("xdotool key Shift_L")  # Simulate pressing the left Shift key 
    os.system("xdotool mousemove_relative -- 1 0 && sleep 1 && xdotool mousemove_relative -- -1 0")  # Move mouse 1 pixel right

def disable_screensaver():
    os.system("xset s off")  # Disable screensaver
    #os.system("xset -dpms")  # Disable Display Power Management Signaling

def enable_screensaver():
    os.system("xset s on")  # Enable screensaver
    #os.system("xset +dpms")  # Enable DPMS

def screenshot(framenumber):
    # Format the frame number with leading zeros (5 digits)
    filename = f"colorFrame_0_{framenumber:05}.png"    
    # Call the scrot command to take a screenshot and save it with the given filename
    os.system("scrot %s" % filename)

def create_ply_file(bgr_image, depth_array, filename, depthScale=1.0):
    # Validate inputs
    #height = len(bgr_image)
    #width  = len(bgr_image[0])
    
    height = len(depth_array)
    width  = len(depth_array[0])
    
    # Check if the dimensions of bgr_image and depth_array match
    if len(bgr_image) != height or any(len(row) != width for row in bgr_image):
        # Resize the bgr_image to match the size of depth_array
        bgr_image = cv2.resize(bgr_image, (width, height), interpolation=cv2.INTER_LINEAR)

    # Header for the PLY file
    header = f"""ply
format ascii 1.0
element vertex {height * width}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(filename, 'w') as ply_file:
        # Write header
        ply_file.write(header)
        
        # Write vertex data
        for y in range(height):
            for x in range(width):
                z = depth_array[y][x] * depthScale
                b, g, r = bgr_image[y][x]
                yR = -y
                ply_file.write(f"{x} {yR} {z} {r} {g} {b}\n")

def extract_centered_rectangle(image):
    """
    Extract a centered square region from the input image with the side length equal to
    the minimum dimension of the original image.

    :param image: Input image (numpy array)
    :return: Cropped image with the centered square
    """
    # Get the dimensions of the input image
    img_height, img_width = image.shape[:2]
    
    # Determine the side length of the square as the minimum dimension of the image
    side_length = min(img_height, img_width)
    
    # Calculate the center of the image
    center_x, center_y = img_width // 2, img_height // 2
    
    # Calculate the top-left corner of the rectangle
    top_left_x = max(0, center_x - side_length // 2)
    top_left_y = max(0, center_y - side_length // 2)
    
    # Calculate the bottom-right corner of the rectangle
    bottom_right_x = min(img_width, center_x + side_length // 2)
    bottom_right_y = min(img_height, center_y + side_length // 2)
    
    # Extract the region of interest (ROI)
    cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    
    return cropped_image


def custom_crop(image,cX,cY,size):
    """
    Extract a centered square region from the input image with the side length equal to
    the minimum dimension of the original image.

    :param image: Input image (numpy array)
    :return: Cropped image with the centered square
    """
    # Get the dimensions of the input image
    img_height, img_width = image.shape[:2]
     
    # Calculate the center of the image
    center_x, center_y = cX, cY
    
    # Calculate the top-left corner of the rectangle
    top_left_x = max(0, center_x - size // 2)
    top_left_y = max(0, center_y - size // 2)
    
    # Calculate the bottom-right corner of the rectangle
    bottom_right_x = min(img_width,  center_x + size)
    bottom_right_y = min(img_height, center_y + size)
    
    # Extract the region of interest (ROI)
    cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    
    return cropped_image
#----------------------------------------------------------------------------------------
def add_horizontal_stripes(image, stripe_height):
    """
    Adds black horizontal stripes at the top and bottom of the image.
    
    Parameters:
    - image: np.ndarray
        The input image (H x W x C for color or H x W for grayscale).
    - stripe_height: int
        The height of the black stripes to add at the top and bottom.
    
    Returns:
    - np.ndarray
        The image with added black stripes.
    """
    # Ensure the stripe height is not larger than half the image height
    max_stripe_height = image.shape[0] // 2
    if stripe_height > max_stripe_height:
        raise ValueError(f"Stripe height cannot exceed half the image height ({max_stripe_height}).")

    # Create a black stripe with the same width and number of channels as the image
    stripe = np.zeros((stripe_height, image.shape[1], image.shape[2] if len(image.shape) > 2 else 1), dtype=image.dtype)
    
    # Add the stripes to the image
    image_with_stripes = np.copy(image)
    image_with_stripes[:stripe_height, :] = stripe  # Top stripe
    image_with_stripes[-stripe_height:, :] = stripe  # Bottom stripe
    
    return image_with_stripes
#----------------------------------------------------------------------------------------
def apply_blur_to_image(image: np.ndarray, blur_strength: int = 5) -> np.ndarray:
    """
    Applies a blur filter to an RGB image.
    
    Parameters:
        image (np.ndarray): Input RGB image as a NumPy array (dtype: uint8).
        blur_strength (int): Kernel size for the blur filter. Must be a positive odd number.
    
    Returns:
        np.ndarray: Blurred RGB image.
    """
    # Ensure the blur strength is a positive odd number
    if blur_strength % 2 == 0:
        blur_strength += 1  # Make it odd
    
    # Apply a Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
    
    return blurred_image
#----------------------------------------------------------------------------------------
def add_noise_to_image(image: np.ndarray, noise_magnitude: float = 0.1) -> np.ndarray:
    """
    Adds Gaussian noise to an RGB image with a fixed magnitude.
    
    Parameters:
        image (np.ndarray): Input RGB image as a NumPy array (dtype: uint8).
        noise_magnitude (float): Magnitude of noise as a fraction of the color range [0-255].
                                  Typical range: 0.0 (no noise) to 1.0 (high noise).
    
    Returns:
        np.ndarray: Noisy RGB image.
    """
    # Ensure the noise magnitude is within a valid range
    noise_magnitude = np.clip(noise_magnitude, 0.0, 1.0)
    
    # Generate Gaussian noise with mean 0 and standard deviation proportional to noise_magnitude
    noise = np.random.normal(loc=0, scale=noise_magnitude * 255, size=image.shape).astype(np.float32)
    
    # Add noise to the image
    noisy_image = image.astype(np.float32) + noise
    
    # Clip the result to ensure pixel values remain valid
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image
#----------------------------------------------------------------------------------------
def main_pose_estimation():
    model_path           = '2d_pose_estimation'
    videoFilePath        = "webcam" 
    videoWidth           = 640
    videoHeight          = 480
    threshold            = 84
    keypoint_threshold   = 60.0
    cropInputFrame       = True #<- Disable to fix handling of points
    customCrop           = False
    customCropX          = 0.0
    customCropY          = 0.0
    customCropSize       = 0.0
    scale                = 1.0
    illustrate           = False
    emulateBorder        = 0
    # Run the webcam keypoints detection

    profiling = False
    visualize = True
    show      = True
    save      = False
    tile      = False
    noise     = 0.0
    pruneTokens = False
    blur      = 0
    engine    = "tensorflow"
    monitor   = list()
    window_arrangement = list()

    if (len(sys.argv)>1):
       #print('Argument List:', str(sys.argv))
       for i in range(0, len(sys.argv)):
           if (sys.argv[i]=="--win"):
              x  = int(sys.argv[i+1])
              y  = int(sys.argv[i+2])
              label = sys.argv[i+3]
              window_arrangement.append((x,y,label))
           if (sys.argv[i]=="--monitor"):
              hm = int(sys.argv[i+1])
              x  = int(sys.argv[i+2])
              y  = int(sys.argv[i+3])
              label = sys.argv[i+4]
              monitor.append((hm,x,y,label))
              #from PoseEstimator2D import label_heatmap
              print("Added a monitor @ %u,%u for %u " % (x,y,hm)) 
           if (sys.argv[i]=="--threshold"):
              keypoint_threshold = float(sys.argv[i+1])
              threshold          = int(sys.argv[i+1])  
           if (sys.argv[i]=="--border"):
              emulateBorder = int(sys.argv[i+1])
           if (sys.argv[i]=="--crop"):
              cropInputFrame = True
              customCrop = True
              customCropX          = int(sys.argv[i+1])
              customCropY          = int(sys.argv[i+2])
              customCropSize       = int(sys.argv[i+3])
           if (sys.argv[i]=="--nocrop"):
              cropInputFrame=False
              customCrop = False 
           if (sys.argv[i]=="--save"):
              save=True
           if (sys.argv[i]=="--prune"):
              pruneTokens=True
           if (sys.argv[i]=="--tile"):
              tile=True
           if (sys.argv[i]=="--blur"):
              blur=min(40,abs(int(sys.argv[i+1])))
           if (sys.argv[i]=="--noise"):
              noise=min(1.0,abs(float(sys.argv[i+1])))
           if (sys.argv[i]=="--collab"):
              illustrate= True
              save      = True
              show      = False
              visualize = True
           if (sys.argv[i]=="--illustrate"):
              illustrate=True
           if (sys.argv[i]=="--from"):
              videoFilePath=sys.argv[i+1]
           if (sys.argv[i]=="--headless") or (sys.argv[i]=="--novisualization"):
              visualize = False
           if (sys.argv[i]=="--scale"):
              scale = float(sys.argv[i+1])
           if (sys.argv[i]=="--size"):
              videoWidth  = int(sys.argv[i+1])
              videoHeight = int(sys.argv[i+2])
           if (sys.argv[i]=="--profiling") or (sys.argv[i]=="--profile"):
               profiling = True
               print("Profiling Activated")
           if (sys.argv[i]=="--engine"):
               engine=sys.argv[i+1]
               print("Set Engine to ",engine)
    

    print("Keypoint Threshold : ",keypoint_threshold)
    print("Threshold : ",threshold)

    cap = getCaptureDeviceFromPath(videoFilePath,videoWidth,videoHeight)

    from PoseEstimator2D import PoseEstimator2D, PoseEstimatorTiler
    estimator = PoseEstimator2D(modelPath=model_path, threshold=threshold, keypoint_threshold=keypoint_threshold, engine=engine, profiling=profiling, illustrate=illustrate, 
                                pruneTokens=pruneTokens, monitor=monitor, window_arrangement=window_arrangement )
    estimator.addedNoise = noise * 255.0

    tiler     = PoseEstimatorTiler(estimator, tile_size=(estimator.cfg['inputWidth'],estimator.cfg['inputHeight']), overlap=(0, 0) )

    if (save) and (show):
        disable_screensaver()

    if (show):
           estimator.setup_threshold_control_window()

    failedFrames = 0
    while True:
        #-------------------------------------------------------------------------------------------------------------------------------------------------
        # Capture a single frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            failedFrames=failedFrames+1
            if (failedFrames>100):
               break
        else:
            failedFrames = 0

            if (tile):
             tiler.process(frame)
             #if (customCrop):
             #     frame = custom_crop(frame,customCropX,customCropY,customCropSize)
             #estimator.process_tiled(frame)
            else:
             if (scale!=1.0):
               original_height, original_width = frame.shape[:2]
               # Calculate new dimensions based on the scale
               new_width  = int(original_width * scale)
               new_height = int(original_height * scale)
               # Resize the image using the new dimensions
               frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

             # Preprocess the frame for the model if we have selected this behavior and the input image should be a rectangle
             if (cropInputFrame) and (estimator.cfg['inputWidth'] == estimator.cfg['inputHeight']):
              if (customCrop):
                  frame = custom_crop(frame,customCropX,customCropY,customCropSize)
              else:
                  frame = extract_centered_rectangle(frame)

             if (emulateBorder>0):
                bigBorder = frame.shape[0] * (emulateBorder / estimator.cfg['inputHeight']) 
                frame = add_horizontal_stripes(frame,int(bigBorder))

             if (blur!=0):
                frame = apply_blur_to_image(frame,blur_strength=blur) 

             if (estimator.addedNoise !=0.0):
                frame = add_noise_to_image(frame,noise_magnitude=estimator.addedNoise)

             #Extract results
             estimator.process(frame)



             #print("Skeletons :",estimator.skeletons)
             #print("Skeneleton Dict :",estimator.encodeSkeletonAsDict(0, denormalize=True))


            #estimator.heatmapsOut has 45 heatmaps

            #Visualize
            if (visualize):
              
              if (show):
               estimator.update_thresholds_from_gui()
              if (tile):
               frameWithVis = frame.copy()
               tiler.visualize(frameWithVis)
               #estimator.visualize(frameWithVis)
              else:
               frameWithVis = frame.copy()
               estimator.visualize(frameWithVis,show=show,save=save)
             
              # Check for the escape key or 'q' key press
              key = 255
              if show:
                key = cv2.waitKey(1) & 0xFF
              if (key!=255):
                print("Key Press = ",key) 
              #---------------------------
              if key == 81:
                 print("Left Arrow") 
              elif key == 97:
                 print("Save demo screenshot") 
                 os.system("scrot -a 800,10,1157,570 scrot%u.png" % estimator.frameNumber )
              elif key == 27 or key == ord('q') or key == ord('Q'):
                print("Terminating after receiving keyboard request")
                break
              elif key == ord('u') or key == ord('U'):
              # If 'U' key is pressed, save and upload the frame
                save_and_upload_frame(frame)
              elif key == ord('s') or key == ord('S'):
              # If 'S' key is pressed, save a 3D PLY file
                create_ply_file(estimator.imageIn, estimator.depthmap, "output_%u.ply" % estimator.frameNumber)
              #--------------------------------------

              if (save and show):
                screenshot(estimator.frameNumber)



        #-------------------------------------------------------------------------------------------------------------------------------------------------
    print("Average Framerate : ",np.average(estimator.keypoints_model.hz)," Hz")
    # Release the webcam and close all OpenCV windows
    cap.release()
    if show:
      cv2.destroyAllWindows()

    if (save) and (show):
        enable_screensaver()
        os.system("ffmpeg -nostdin -framerate 25 -i colorFrame_0_%%05d.png -vf scale=-1:720 -y -r 25 -pix_fmt yuv420p -threads 8 %s_lastRun3DHiRes.mp4" % videoFilePath)
        os.system("rm colorFrame*.png")

    if (illustrate):
        os.system("ffmpeg -nostdin -framerate 25 -i composite_%%05d.png -vf scale=-1:720 -y -r 25 -pix_fmt yuv420p -threads 8 %s_illustration.mp4" % videoFilePath)
        os.system("rm composite_*.png")


#---------------------------------------------------------------------------------------- 
#============================================================================================
#============================================================================================
# Main Function
#============================================================================================
#============================================================================================
if __name__ == '__main__':
   main_pose_estimation()
#============================================================================================
#============================================================================================
