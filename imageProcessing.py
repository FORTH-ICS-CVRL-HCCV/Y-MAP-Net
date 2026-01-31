
"""
Author : "Ammar Qammaz"
Copyright : "2024 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"
"""
import cv2
import numpy as np
#from numba import njit
#-------------------------------------------------------------------------------
def mix_rgb_to_8bit(rgb_image):
    packed_image = np.zeros((rgb_image.shape[0], rgb_image.shape[1], 2), dtype=np.uint8)
    rgb_imageH = rgb_image / 3
    packed_image[:,:,0] =  (rgb_imageH[:,:,0] + rgb_imageH[:,:,1]) + rgb_imageH[:,:,2]  
    return packed_image
#-------------------------------------------------------------------------------
def unmix_8bit_to_rgb(packed_image):
    unpacked_image = np.zeros((packed_image.shape[0], packed_image.shape[1], 3), dtype=np.uint8)
    unpacked_image[:,:,0] =   packed_image[:,:,0]  
    unpacked_image[:,:,1] =   packed_image[:,:,0]   
    unpacked_image[:,:,2] =   packed_image[:,:,0]
    return unpacked_image
#-------------------------------------------------------------------------------
def mix_rgb_to_16bit(rgb_image):
    packed_image = np.zeros((rgb_image.shape[0], rgb_image.shape[1], 2), dtype=np.uint8)
    rgb_imageH = rgb_image / 2
    packed_image[:,:,0] =  (rgb_imageH[:,:,0] + rgb_imageH[:,:,1])  / 2
    packed_image[:,:,1] =  (rgb_imageH[:,:,1] + rgb_imageH[:,:,2])  / 2
    return packed_image
#-------------------------------------------------------------------------------
def unmix_16bit_to_rgb(packed_image):
    unpacked_image = np.zeros((packed_image.shape[0], packed_image.shape[1], 3), dtype=np.uint8)
    unpacked_image[:,:,0] =  packed_image[:,:,0] * 2  
    unpacked_image[:,:,1] =  (packed_image[:,:,0] + packed_image[:,:,1])   
    unpacked_image[:,:,2] =  packed_image[:,:,1] * 2
    return unpacked_image
#-------------------------------------------------------------------------------
def rgb_to_yuv(rgb_image):
    yuv_image = np.zeros_like((rgb_image.shape[0], rgb_image.shape[1], 3), dtype=np.uint8)
    # Conversion matrix for RGB to YUV

    # RGB to YUV conversion
    yuv_image[:,:,0] = np.clip(0.299 * rgb_image[:,:,0]  + 0.587 * rgb_image[:,:,1] + 0.114 * rgb_image[:,:,2], 0, 255)
    yuv_image[:,:,1] = np.clip(-0.147 * rgb_image[:,:,0] - 0.289 * rgb_image[:,:,1] + 0.436 * rgb_image[:,:,2] + 128, 0, 255)
    yuv_image[:,:,2] = np.clip(0.615 * rgb_image[:,:,0]  - 0.515 * rgb_image[:,:,1] - 0.100 * rgb_image[:,:,2] + 128, 0, 255)
    
    return yuv_image
#-------------------------------------------------------------------------------
def yuv_to_rgb(yuv_image):
    rgb_image = np.zeros_like(yuv_image, dtype=np.uint8)
    # Conversion matrix for YUV to RGB
    rgb_matrix = np.array([[1, 0, 1.13983],
                            [1, -0.39465, -0.58060],
                            [1, 2.03211, 0]])
    yuv_image = yuv_image.astype(np.float32) - 128
    rgb_image[:,:,0]  = np.clip(np.dot(yuv_image, rgb_matrix.T), 0, 255) # R channel
    rgb_image[:,:,1:] = np.clip((np.dot(yuv_image, rgb_matrix.T)), 0, 255) # G and B channels
    return rgb_image
#-------------------------------------------------------------------------------
def castNormalizedXYCoordinatesToOriginalImage(shrinkedImageWidth,shrinkedImageHeight,keypointXMultiplier, keypointYMultiplier, keypointXOffset, keypointYOffset,x,y):
    widthWithoutCropping   = shrinkedImageWidth  + (2*keypointYOffset)  #<- This is correct, but looks wrong is the mistake somewhere else ?
    heightWithoutCropping  = shrinkedImageHeight + (2*keypointXOffset)  #<- This is correct, but looks wrong is the mistake somewhere else ?
    # Go to shrinkedImage x,y coordinates
    x_shrinked = x * shrinkedImageWidth   # x coordinate in shrinkedImage space
    y_shrinked = y * shrinkedImageHeight  # y coordinate in shrinkedImage space
    # Undo cropping
    x_uncropped = x_shrinked + keypointXOffset
    y_uncropped = y_shrinked + keypointYOffset
    # Undo scaling
    x_original = x_uncropped / heightWithoutCropping
    y_original = y_uncropped / widthWithoutCropping
    # Store output
    return  x_original,y_original
#-------------------------------------------------------------------------------
def castNormalizedCoordinatesToOriginalImage(originalImage, shrinkedImage, heatmapSize, keypoints, keypointXMultiplier, keypointYMultiplier, keypointXOffset, keypointYOffset):
    #print("Original image : ", originalImage.shape[1], "x", originalImage.shape[0])
    #print("Shrinked image : ", shrinkedImage.shape[1], "x", shrinkedImage.shape[0])
    #print("Heatmap size   : ", heatmapSize)
    for i, name in enumerate(keypoints):
        for j in range(0, len(keypoints[i]) ):
            peak_points = keypoints[i][j]
            y = peak_points[0]
            x = peak_points[1]
            v = peak_points[2]
             
            x_original,y_original = castNormalizedXYCoordinatesToOriginalImage(shrinkedImage.shape[1],shrinkedImage.shape[0],keypointXMultiplier, keypointYMultiplier, keypointXOffset, keypointYOffset,x,y)

            keypoints[i][j] = (y_original, x_original, v)
    return keypoints
#-------------------------------------------------------------------------------
def castNormalizedBBoxCoordinatesToOriginalImage(originalImage, shrinkedImage, heatmapSize, bbox, keypointXMultiplier, keypointYMultiplier, keypointXOffset, keypointYOffset):
    #print("Original image : ", originalImage.shape[1], "x", originalImage.shape[0])
    #print("Shrinked image : ", shrinkedImage.shape[1], "x", shrinkedImage.shape[0])
    #print("Heatmap size   : ", heatmapSize)
    for i in range(len(bbox)):
            x,y,w,h = bbox[i]
            
            x1,y1=castNormalizedXYCoordinatesToOriginalImage(shrinkedImage.shape[1],shrinkedImage.shape[0],keypointXMultiplier, keypointYMultiplier, keypointXOffset, keypointYOffset,x,y)
            x2,y2=castNormalizedXYCoordinatesToOriginalImage(shrinkedImage.shape[1],shrinkedImage.shape[0],keypointXMultiplier, keypointYMultiplier, keypointXOffset, keypointYOffset,(x+w),(y+h))
 
            bbox[i] = x1,y1,(x2-x1),(y2-y1)
            
    return bbox
#-------------------------------------------------------------------------------
def resize_image_no_borders(image, target_size=(300, 300)):
    try:
        #There needs to be a flip here..
        target_size = tuple(reversed(target_size))

        # Get the original image size
        originalWidth  = image.shape[1]
        originalHeight = image.shape[0]
        # Notice that we use a different convention than OpenCV
        newWidth  = target_size[0]
        newHeight = target_size[1]

        # Calculate the aspect ratios of the original and target sizes
        aspect_ratio_original = originalWidth / originalHeight
        aspect_ratio_target   = newWidth / newHeight

        keypointXOffset = 0
        keypointYOffset = 0

        # Determine the resizing factor and size for maintaining the aspect ratio
        if aspect_ratio_original > aspect_ratio_target:
            # Resize based on width
            newWidth      = int(newHeight * aspect_ratio_original)
            #print("A New Width/Height",(newWidth,newHeight))
            resized_image = cv2.resize(image, (newWidth, newHeight), interpolation=cv2.INTER_CUBIC)
            # Crop the image to fit the target size
            crop_start      = (newWidth - target_size[0]) // 2
            keypointXOffset = crop_start#(target_size[0] - newWidth)  // 2
            cropped_image = resized_image[:, crop_start:crop_start + target_size[0]]
        else:
            # Resize based on height
            newHeight     = int(newWidth / aspect_ratio_original)
            #print("B New Width/Height",(newWidth,newHeight))
            resized_image = cv2.resize(image, (newWidth, newHeight), interpolation=cv2.INTER_CUBIC)
            # Crop the image to fit the target size
            crop_start      = (newHeight - target_size[1]) // 2
            keypointYOffset = crop_start#(target_size[1] - newHeight) // 2
            cropped_image = resized_image[crop_start:crop_start + target_size[1], :]

        # Calculate the position to paste the cropped image onto the new image (no border)
        keypointXMultiplier = target_size[0] / originalWidth
        keypointYMultiplier = target_size[1] / originalHeight

        return cropped_image, keypointXMultiplier, keypointYMultiplier, keypointXOffset, keypointYOffset

    except Exception as e:
        print(f"Error resizing image: {e}")
        return image, 0.0, 0.0, 0.0, 0.0
#-------------------------------------------------------------------------------
def resize_image_with_borders(image, target_size=(300, 300)):
    try:
        #There needs to be a flip here..
        target_size = tuple(reversed(target_size))

        # Get the original image size
        originalWidth  = image.shape[1]
        originalHeight = image.shape[0]
        #Notice that we use a different convention than OpenCV
        newWidth       = target_size[0]
        newHeight      = target_size[1]

        # Calculate the aspect ratios of the original and target sizes
        aspect_ratio_original = originalWidth  / originalHeight
        aspect_ratio_target   = newWidth / newHeight

        # Determine the resizing factor and size for maintaining the aspect ratio
        if aspect_ratio_original > aspect_ratio_target:
            newHeight = int(newWidth / aspect_ratio_original)
        else:
            newWidth  = int(newHeight * aspect_ratio_original)

        # Resize the image while maintaining the aspect ratio
        resized_image = cv2.resize(image, (newWidth, newHeight), interpolation=cv2.INTER_CUBIC)

        # Create a new image with a black background
        new_image = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

        # Calculate the position to paste the resized image onto the new image
        x_offset = (target_size[0] - newWidth)  // 2
        y_offset = (target_size[1] - newHeight) // 2

        # Paste the resized image onto the new image
        new_image[y_offset:y_offset + newHeight, x_offset:x_offset + newWidth] = resized_image

        keypointXMultiplier =  newWidth  / originalWidth
        keypointYMultiplier =  newHeight / originalHeight
        keypointXOffset     =  x_offset
        keypointYOffset     =  y_offset

        return new_image, keypointXMultiplier, keypointYMultiplier, keypointXOffset, keypointYOffset

    except Exception as e:
        print(f"Error resizing image: {e}")
        return image, 0.0, 0.0, 0.0, 0.0
#-------------------------------------------------------------------------------
def add_gaussian_noise_to_rgb(image, std_dev=25):
    # Ensure the input image is in int8
    if image.dtype != np.int8:
        raise ValueError("Image must be int8 encoded.")

    # Generate Gaussian noise
    noise = np.random.normal(0, std_dev, image.shape)

    # Add the noise to the image
    noisy_image = image.astype(np.float32) + noise

    # Clip and convert back to int8 to avoid overflow issues
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image
#-------------------------------------------------------------------------------
def rotateImage(image, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    return rotated_image
#-------------------------------------------------------------------------------
def flipImage(image, flip_x=False, flip_y=False):
    if flip_x:
        image = cv2.flip(image, 1)
    if flip_y:
        image = cv2.flip(image, 0)
    return image
#-------------------------------------------------------------------------------
def adjust_brightness_contrast(image, brightness_factor=1.0, contrast_factor=1.0):
    image = image * contrast_factor + brightness_factor
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)
#-------------------------------------------------------------------------------



#============================================================================================
#============================================================================================
# Main Function
#============================================================================================
#============================================================================================
if __name__ == '__main__':

  # Open webcam
  cap = cv2.VideoCapture(0)

  while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Display original and unpacked images
    cv2.imshow('Original', frame)


    # Convert frame to RGB
    frame_mix = mix_rgb_to_16bit(frame) 
    frame_unmix = unmix_16bit_to_rgb(frame_mix)
    cv2.imshow('unmix', frame_unmix)

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #packed_image = pack_rgb_to_16bit(frame_rgb)
    #unpacked_image = unpack_16bit_to_rgb(packed_image)
    #cv2.imshow('Unpacked', cv2.cvtColor(unpacked_image, cv2.COLOR_RGB2BGR))


    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  # When everything done, release the capture
  cap.release()
  cv2.destroyAllWindows()

