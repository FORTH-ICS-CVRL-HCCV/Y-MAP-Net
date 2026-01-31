
"""
Author : "Ammar Qammaz"
Copyright : "2024 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"
"""

import sys
import cv2
import numpy as np

def load_image(image_path):
    # Open the image file using OpenCV
    img = cv2.imread(image_path,-1)
    if img is None:
        print(f"Error: Unable to open image file {image_path}")
        sys.exit(1)
    return img

def combine_channels(color_image):
    if color_image.ndim != 3 or color_image.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel color image (height, width, 3)")
    
    # Define the weights for the RGB channels
    weights = np.array([0.3333, 0.3333, 0.3333])
    
    # Apply the weights to the color image to get the grayscale image
    grayscale_image = np.dot(color_image[..., :3], weights)
    
    return grayscale_image

def rgb_to_grayscale(img_data):
    # Convert the image to grayscale
    grayscale = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
    return grayscale

def compute_normals(grayscale):
    # Compute gradients in x and y directions
    dx = cv2.Sobel(grayscale, cv2.CV_32F, 1, 0, ksize=3)  # cv2.CV_64F
    dy = cv2.Sobel(grayscale, cv2.CV_32F, 0, 1, ksize=3)  # cv2.CV_64F
    dz = np.ones_like(grayscale, dtype=np.float32)        # np.float64

    # Stack gradients to form normal vectors
    normals = np.stack((dx, dy, dz), axis=-1)

    # Normalize the normal vectors
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / (norm + 1e-8)  # Add a small epsilon to avoid division by zero

    # Normalize to [0, 255] for RGB representation
    normals = (normals + 1) / 2 * 255
    normals = normals.astype(np.uint8)

    return normals

def apply_bilateral_filter(normals):
    # Apply bilateral filter to smooth the image while preserving edges
    smoothed_normals = cv2.bilateralFilter(normals, d=9, sigmaColor=75, sigmaSpace=75)
    return smoothed_normals

def print_image_stats(img,label):
    minimum = np.min(img)
    maximum = np.max(img)
    mean    = np.mean(img)
    std     = np.std(img)
    print("--------------------")
    print("",label)
    print("Minimum : ",minimum)
    print("Maximum : ",maximum)
    print("Mean    : ",mean)
    print("StD     : ",std)
    print("--------------------")

def improve_depthmapSimple(depthmap_int8, normalX_int8, normalY_int8, normalZ_int8, learning_rate, iterations):

    improved_depthmap = depthmap_int8.astype(np.float32) / 255.0
    normalX = normalX_int8.astype(np.float32) / 255.0
    normalY = normalY_int8.astype(np.float32) / 255.0
    normalZ = normalZ_int8.astype(np.float32) / 255.0

    #print_image_stats(depthmap_int8,"depthmap_int8")
    #print_image_stats(normalX_int8,"normalX_int8")
    #print_image_stats(normalY_int8,"normalY_int8")
    #print_image_stats(normalZ_int8,"normalZ_int8")

    #print_image_stats(improved_depthmap,"improved_depthmap")
    #print_image_stats(normalX,"normalX")
    #print_image_stats(normalY,"normalY")
    #print_image_stats(normalZ,"normalZ") 
 
    # Iteratively update the depthmap 
    for i in range(iterations):
        #if (i % 5 ==0):
        #  cv2.imshow('Improved Depth %u' % i, improved_depthmap)

        # Compute the gradients of the current improved depthmap
        gradX = cv2.Sobel(improved_depthmap, cv2.CV_32F, 1, 0, ksize=3)
        gradY = cv2.Sobel(improved_depthmap, cv2.CV_32F, 0, 1, ksize=3)
        
        # Normalize the depthmap gradients to unit vectors for comparison
        magnitude = np.sqrt(gradX**2 + gradY**2 + 1)
        gradX_normalized = gradX / magnitude
        gradY_normalized = gradY / magnitude
        gradZ_normalized = 1 / magnitude

        # Calculate the difference between the normalized computed normals and the given normals
        deltaX = normalX - gradX_normalized
        deltaY = normalY - gradY_normalized
        deltaZ = normalZ - gradZ_normalized

        # Update the depthmap by minimizing the normal difference
        residual = learning_rate * (deltaX * gradX + deltaY * gradY + deltaZ)

        #print("Residual StD iteration ",i," : ",np.std(residual))
        #print_image_stats(residual,"residual %u"% i)
        improved_depthmap = improved_depthmap + residual
        #print_image_stats(improved_depthmap,"improved_depthmap %u"% i)
        
    return improved_depthmap.astype(np.float32)

def improve_depthmap(depthmap_int8, normalX_int8, normalY_int8, normalZ_int8, learning_rate, iterations, depth_threshold=20):
    # Convert the inputs to float32 and normalize to [0, 1]
    improved_depthmap = depthmap_int8.astype(np.float32) / 255.0
    #improved_depthmap = np.full((256,256),0.5,dtype=np.float32) <- Try to disregard depth altogether

    normalX = normalX_int8.astype(np.float32) / 255.0
    normalY = normalY_int8.astype(np.float32) / 255.0
    normalZ = normalZ_int8.astype(np.float32) / 255.0

    # Create a mask to identify pixels above the depth threshold
    #mask = improved_depthmap >= (depth_threshold / 255.0)
    
    # Iteratively update the depthmap
    for i in range(iterations):
        # Compute the gradients of the current improved depthmap
        gradX = cv2.Sobel(improved_depthmap, cv2.CV_32F, 1, 0, ksize=3)
        gradY = cv2.Sobel(improved_depthmap, cv2.CV_32F, 0, 1, ksize=3)
        
        # Normalize the depthmap gradients to unit vectors for comparison
        magnitude        = np.sqrt(gradX**2 + gradY**2 + 1)
        gradX_normalized = gradX / magnitude
        gradY_normalized = gradY / magnitude
        gradZ_normalized = 1 / magnitude

        # Calculate the difference between the normalized computed normals and the given normals
        deltaX = normalX - gradX_normalized
        deltaY = normalY - gradY_normalized
        deltaZ = normalZ - gradZ_normalized

        # Update the depthmap by minimizing the normal difference only where mask is True
        residual          = learning_rate * (deltaX * gradX + deltaY * gradY + deltaZ)
        improved_depthmap = improved_depthmap + residual
        #improved_depthmap[mask] = improved_depthmap[mask] + residual[mask]

    # Set the pixels outside the mask (below depth threshold) to zero
    #improved_depthmap[~mask] = 0

    return improved_depthmap.astype(np.float32)

def integrate_normals(normalX_int8, normalY_int8, normalZ_int8, initial_depth=None, iterations=500):
    Normal = np.stack((normalX_int8, normalY_int8, normalZ_int8), axis=-1)

    # Dimensions of the normal map
    h, w, _ = Normal.shape
    
    if initial_depth is None:
        depth = np.zeros((h, w))
    else:
        depth = initial_depth.copy()
    
    # Integration loop
    for _ in range(iterations):
        # Compute gradients from normals
        dzdx = -Normal[:,:,0] / Normal[:,:,2]
        dzdy = -Normal[:,:,1] / Normal[:,:,2]
        
        # Depth update estimates from horizontal and vertical neighbors
        depth_x = np.roll(depth, -1, axis=1) - dzdx
        depth_y = np.roll(depth, -1, axis=0) - dzdy
        
        # Average updates
        depth_update = (np.roll(depth_x, 1, axis=1) + depth_x +
                        np.roll(depth_y, 1, axis=0) + depth_y) / 4
        
        # Update depth
        depth = depth_update

    return depth.astype(np.float32)

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_depthmap.png> <output_normals.png>")
        sys.exit(1)

    input_path = sys.argv[1]

    # Load the image
    img_data = load_image(input_path)
    # Convert to grayscale
    print("img_data", img_data.shape)
    if (len(img_data.shape)>2):
       img_data = rgb_to_grayscale(img_data)
    # Compute surface normals
    normals = compute_normals(img_data)

    smoothed_normals = apply_bilateral_filter(apply_bilateral_filter(normals))

    # Save the normals image
    cv2.imwrite("normX.png", normals[:,:,0])
    cv2.imwrite("normY.png", normals[:,:,1])
    cv2.imwrite("normZ.png", normals[:,:,2])
    cv2.imwrite("normals.png", normals)
    cv2.imwrite("smoothed.png", smoothed_normals)

    # Display the result in a window
    cv2.imshow('Surface Normals', normals)
    cv2.imshow('Smoothed Normals', smoothed_normals)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
