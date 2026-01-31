import os
from datetime import datetime
import sys
import gradio as gr
import numpy as np
import cv2

#----------------------------------------------------------------------
#                  Gradio UI Translation Layer
#----------------------------------------------------------------------
def file_exists(path):
    return os.path.exists(path)

#If a file greek exists, trigger translation
GREEK_MENU=False
if (file_exists("greek")):
  GREEK_MENU=True

menuT = dict()
#----------------------------------------------------------------------
menuT["RGB To Pose, Depth, Normals, Class and Token Estimation"]="Μετατροπή έγχρωμων εικόνων σε Πόζα, Βάθος, Κάθετα Διανύσματα, Κλάσεις και Περιγραφή "
menuT["Description"]="Περιγραφή"
menuT["Image"]="Εικόνα Εισόδου"
menuT["RGB and 2D Pose"]="Εικόνα Εισόδου με δισδιάστατη πόζα"
menuT["Normals"]="Κάθετα Διανύσματα"
menuT["Depth"]="Βάθος"
menuT["Improved Depth"]="Βελτιωμένο Βάθος"
menuT["Persons"]="Παρουσία Ανθρώπων"
menuT["Joints"]="Ανθρώπινες Αρθρώσεις"
menuT["PAFs"]="Σύνδεση μεταξύ ανθρώπινων αρθρώσεων"
menuT["Segmentation Union"]="Συνδυασμός όλων των κλάσεων"
menuT["Animal"]="Ζώο"
menuT["Objects"]="Αντικέιμενο"
menuT["Furniture"]="Έπιπλα"
menuT["Appliance"]="Συσκευές"
menuT["Material"]="Υλικά"
menuT["Obstacle"]="Εμπόδιο"
menuT["Building"]="Κτήριο"
menuT["Nature"]="Φύση"
menuT["Text Detection"]="Εντοπισμός κειμένου"
menuT["Vehicle"]="Όχημα"
menuT["Input Image"]="Εικόνα εισόδου"
menuT["Heatmap Joint Threshold"]="Όριο σε απόκριση αρθρώσεων"
menuT["Keypoint Threshold"]="Όριο σε σημεία κλειδιά"
menuT["Crop Image"]="Περικοπή εικόνας"

menuT["Output"]="Έξοδος"
menuT["Result"]="Αποτέλεσμα"
menuT["Process"]="Υποβολή"
#---------------------------------------------------------------------- 

def t(inputString):
 global GREEK_MENU
 if (GREEK_MENU):
  global menuT
  return menuT[inputString]
 return inputString
#---------------------------------------------------------------------- 


from PoseEstimator2D import PoseEstimator2D, visualization
from TokenEstimator import TokenEstimator2D
sys.path.append('datasets/')
from calculateNormalsFromDepthmap import improve_depthmap

port="8080"
server_name="0.0.0.0"

useStandaloneTokenEstimator = True


# Initialize the model
model_path = '2d_pose_estimation'

threshold            = 20
keypoint_threshold   = 40.0
estimator_pose = PoseEstimator2D(modelPath=model_path, threshold=threshold, keypoint_threshold=keypoint_threshold)
 
estimator = TokenEstimator2D(modelPath=model_path)
#from TokenEstimator import read_json_files 
#estimator.vocabulary = read_json_files("2d_pose_estimation/vocabulary.json")


if (GREEK_MENU):
    from PoseEstimator2D import read_json_files
    print("Forcing greek vocabulary")
    estimator_pose.vocabulary = read_json_files("2d_pose_estimation/vocabulary_el.json")
    estimator.vocabulary = read_json_files("2d_pose_estimation/vocabulary_el.json")


# Visual theme
visual_theme = gr.themes.Default()  # Default, Soft or Monochrome

# Constants
MAX_OUTPUT_TOKENS = 2048
MAX_IMAGE_SIZE = (1120, 1120)



def log_image(frame, directory="server_log"):
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Generate a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the filename
    filename = f"{timestamp}.png"  # You can change the extension as needed
    
    # Create the full path
    filepath = os.path.join(directory, filename)
    
    # Save the image
    cv2.imwrite(filepath, frame)
    print(f"Image saved at: {filepath}")
 
def encode_depth_map(depth, grayscale=False):

    """
    Encodes a one-channel depth map into a color image.
    
    Parameters:
    - depth (numpy.ndarray): A 2D array representing the depth map.
    - grayscale (bool): If True, returns a grayscale image. If False, applies a colormap.
    
    Returns:
    - numpy.ndarray: A 3-channel image of the encoded depth map.
    """
    # Normalize depth values to range [0, 255]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)

    if grayscale:
        # Repeat the single channel into 3 channels (grayscale)
        #depth_color = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        return depth
    else:
        import matplotlib.pyplot as plt
        # Apply a colormap (e.g., 'plasma') and convert to BGR format for OpenCV
        cmap = plt.get_cmap('plasma')  # You can experiment with other colormaps
        depth_color = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)
        depth_color = depth_color[:, :, ::-1]  # Convert to BGR format

    return depth_color


def threshold_image(image, threshold=20, keypoint_threshold=30.0,preserve_above=True):
    """
    Apply thresholding to a single-channel (grayscale) image.
    
    Parameters:
    - image (numpy.ndarray): 2D NumPy array representing the grayscale image.
    - threshold (int or float): Threshold value. Pixels below or equal to this value are set to 0.
    - preserve_above (bool): If True, preserves the original values above the threshold. 
      If False, sets values above the threshold to 255.
      
    Returns:
    - numpy.ndarray: The thresholded image.
    """
    # Ensure the image is a NumPy array
    image = np.array(image, dtype=np.uint8)
    
    if preserve_above:
        # Set pixels below or equal to the threshold to 0, keep others as they are
        thresholded_image = np.where(image > threshold, image, 0)
    else:
        # Set pixels above the threshold to 255, below or equal to 0
        thresholded_image = np.where(image > threshold, 255, 0)
    
    return thresholded_image.astype(np.uint8)

# Function to process the image and generate a description
def describe_image(image, crop_button, threshold, keypoint_threshold, history):
    log_image(image)

    # Initialize cleaned_output variable
    cleaned_output = "" 
    # Preprocess the input image as per the model's requirements
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #Y-NET
    #estimator_pose.visualize = False
    estimator_pose.threshold          = threshold 
    estimator_pose.keypoint_threshold = keypoint_threshold
    estimator_pose.process(frame,borders = not crop_button)

   
    global useStandaloneTokenEstimator
    #Standalone
    if useStandaloneTokenEstimator:
      estimator.threshold          = threshold 
      estimator.keypoint_threshold = keypoint_threshold
      #frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      caption = estimator.process(frame, visualization=False)  # Process the frame
      print("Using standalone captioner : ",caption)
    else:
      caption = estimator_pose.description
    #==========================================

    input_image = estimator_pose.input_image
    
    input_image = visualization(input_image, estimator_pose.frameNumber, estimator_pose.keypoint_in_nn_coordinate_image, estimator_pose.keypoint_depths, estimator_pose.keypoint_names) #, self.blobs

    visBGR = visualization(frame, estimator_pose.frameNumber, estimator_pose.keypoint_results,estimator_pose.keypoint_depths, estimator_pose.keypoint_names)  
    visRGB = cv2.cvtColor(visBGR, cv2.COLOR_RGB2BGR)

    depthmap         =  estimator_pose.heatmapsOut[29] #17
    depthmap_col     =  encode_depth_map(estimator_pose.heatmapsOut[29], grayscale=True) #17
    #depth_output     =  np.stack((depthmap, depthmap, depthmap), axis=-1)

    selected_joints  = estimator_pose.heatmapsOut[0:17]  # Slicing to get heatmaps 0 through 16
    union_joints     = np.max(selected_joints, axis=0)

    selected_pafs    = estimator_pose.heatmapsOut[17:29]  # Slicing to get heatmaps 0 through 16
    union_pafs       = np.max(selected_pafs, axis=0)

    selected_segms   = estimator_pose.heatmapsOut[34:]  # Slicing to get heatmaps 0 through 16
    union_segms      = np.max(selected_segms, axis=0)

    normalX          =  estimator_pose.heatmapsOut[30] #18
    normalY          =  estimator_pose.heatmapsOut[31] #19
    normalZ          =  estimator_pose.heatmapsOut[32] #20
    normal_output    =  np.stack((normalX, normalY, normalZ), axis=-1)

    text             =  estimator_pose.heatmapsOut[33]  
    #unlabeled        =  estimator_pose.heatmapsOut[34]
    
    #print("Estimator heatmaps out : ",len(estimator_pose.heatmapsOut))  
    person           =  threshold_image(estimator_pose.heatmapsOut[33+1])
    vehicle          =  threshold_image(estimator_pose.heatmapsOut[33+2])
    animal           =  threshold_image(estimator_pose.heatmapsOut[33+3])
    oobject          =  threshold_image(estimator_pose.heatmapsOut[33+4])
    furniture        =  threshold_image(estimator_pose.heatmapsOut[33+5])
    appliance        =  threshold_image(estimator_pose.heatmapsOut[33+6])
    material         =  threshold_image(estimator_pose.heatmapsOut[33+7])
    obstacle         =  threshold_image(estimator_pose.heatmapsOut[33+8])
    building         =  threshold_image(estimator_pose.heatmapsOut[33+9])
    nature           =  threshold_image(estimator_pose.heatmapsOut[33+10])


    improved_depth = improve_depthmap(depthmap, normalX, normalY, normalZ, learning_rate = 0.01, iterations = 35) 
    improved_depth = encode_depth_map(improved_depth, grayscale=True)
     
    # Append the new conversation to the history
    history=[]
    history.append((t("Result"),caption))

    return history, input_image, visRGB, union_joints, union_pafs, union_segms, normal_output, depthmap_col, improved_depth, text ,  person, vehicle, animal, oobject, furniture, appliance, material, obstacle, building, nature

# Function to clear the chat history
def clear_chat():
    return []

# Gradio Interface
def gradio_interface():
    with gr.Blocks(visual_theme) as demo:
        gr.HTML(
        """
    <h1 style='text-align: center'>
    %s
    </h1>
    """ % t("RGB To Pose, Depth, Normals, Class and Token Estimation") )
        with gr.Row():
            # Left column with image and parameter inputs
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label=t("Image"), 
                    type="numpy", 
                    image_mode="RGB", 
                    height=512,  # Set the height
                    width=512   # Set the width
                )

                # Parameter sliders
                threshold          = gr.Slider(label=t("Heatmap Joint Threshold"), minimum=0.0, maximum=255.0, value=20.0, step=0.1, interactive=True)
                keypoint_threshold = gr.Slider(label=t("Keypoint Threshold"), minimum=0.0, maximum=255.0, value=30.0, step=0.1, interactive=True)

                crop_button = gr.Checkbox(label=t("Crop Image"), value=True)

                generate_button = gr.Button(t("Process")) 

            # Right column with the interface
            with gr.Column(scale=2):
                chat_history = gr.Chatbot(label=t("Description"), height=100)
                rgb_output  = gr.Image(label=t("RGB and 2D Pose"), height=512, width=512)  
                with gr.Row():
                   normal_output         = gr.Image(label=t("Normals"), height=256, width=256)
                   depth_output          = gr.Image(label=t("Depth"),  height=256, width=256)
                with gr.Row():
                   improved_depth_output = gr.Image(label=t("Improved Depth"), height=256, width=256)
                   person                = gr.Image(label=t("Persons"),  height=256, width=256)
                with gr.Row():
                   union_joints          = gr.Image(label=t("Joints"),   height=256, width=256)
                   union_pafs            = gr.Image(label=t("PAFs"),   height=256, width=256)
                with gr.Row():
                   union_segms           = gr.Image(label=t("Segmentation Union"),   height=256, width=256)
                   animal                = gr.Image(label=t("Animal"),   height=256, width=256)
                with gr.Row():
                   oobject               = gr.Image(label=t("Objects"),  height=256, width=256)
                   furniture             = gr.Image(label=t("Furniture"),height=256, width=256)
                with gr.Row():
                   appliance             = gr.Image(label=t("Appliance"),height=256, width=256)
                   material              = gr.Image(label=t("Material"), height=256, width=256)
                with gr.Row():
                   obstacle              = gr.Image(label=t("Obstacle"), height=256, width=256)
                   building              = gr.Image(label=t("Building"), height=256, width=256)
                with gr.Row():
                   nature                = gr.Image(label=t("Nature"),   height=256, width=256)
                   text_segment_output   = gr.Image(label=t("Text Detection"), height=256, width=256)
                with gr.Row():
                   vehicle               = gr.Image(label=t("Vehicle"),  height=256, width=256)
                   input_image           = gr.Image(label=t("Input Image"),  height=256, width=256)
 
            # Define the action for the generate button
            generate_button.click(
                    fn       = describe_image, 
                    inputs   = [image_input, crop_button, threshold, keypoint_threshold, chat_history],
                    outputs  = [chat_history, input_image, rgb_output, union_joints, union_pafs, union_segms, normal_output, depth_output, improved_depth_output, text_segment_output, person, vehicle, animal, oobject, furniture, appliance, material, obstacle, building, nature],
                    api_name = "predict"
                   )

    return demo

# Launch the interface
demo = gradio_interface()
demo.launch(favicon_path="doc/favicon.ico",server_name=server_name, server_port=int(port))

