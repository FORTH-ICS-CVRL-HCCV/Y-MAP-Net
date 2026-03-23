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
menuT["Face"]="Πρόσωπο"
menuT["Hand"]="Χέρι"
menuT["Foot"]="Πόδι"
menuT["Joints"]="Ανθρώπινες Αρθρώσεις"
menuT["PAFs"]="Σύνδεση μεταξύ ανθρώπινων αρθρώσεων"
menuT["Segmentation Union"]="Συνδυασμός όλων των κλάσεων"
menuT["Animal"]="Ζώο"
menuT["Robot"]="Ρομπότ"
menuT["Objects"]="Αντικέιμενο"
menuT["Box"]="Κουτί"
menuT["Tool"]="Εργαλείο"
menuT["Instrument"]="Όργανο"
menuT["Furniture"]="Έπιπλα"
menuT["Appliance"]="Συσκευές"
menuT["Conveyor"]="Ιμάντας"
menuT["Chair"]="Καρέκλα"
menuT["Table"]="Τραπέζι"
menuT["Bed"]="Κρεβάτι"
menuT["Light"]="Φωτιστικό"
menuT["Floor"]="Δάπεδο"
menuT["Ceiling"]="Οροφή"
menuT["Wall"]="Τοίχος"
menuT["Door"]="Πόρτα"
menuT["Window"]="Παράθυρο"
menuT["Building"]="Κτήριο"
menuT["Nature"]="Φύση"
menuT["Plant"]="Φυτό"
menuT["Road"]="Δρόμος"
menuT["Dirt"]="Χώμα"
menuT["Sidewalk"]="Πεζοδρόμιο"
menuT["Mountain"]="Βουνό"
menuT["Sky"]="Ουρανός"
menuT["Food"]="Φαγητό"
menuT["Fruit"]="Φρούτο"
menuT["Water"]="Νερό"
menuT["Cup"]="Κούπα"
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
  return menuT.get(inputString, inputString)
 return inputString
#----------------------------------------------------------------------


from YMAPNet import YMAPNet, visualization, retrieveHeatmapIndex
from TokenEstimator import TokenEstimator2D
sys.path.append('datasets/')
from calculateNormalsFromDepthmap import improve_depthmap

port="8080"
server_name="0.0.0.0"

useStandaloneTokenEstimator  = True
TOKEN_ESTIMATOR_SKIP_FRAMES  = 1   # run token estimator once every N frames
_token_frame_counter         = 0
_token_caption_cache         = ""


# Initialize the model
model_path = '2d_pose_estimation'

threshold            = 20
keypoint_threshold   = 40.0
estimator_pose = YMAPNet(modelPath=model_path, threshold=threshold, keypoint_threshold=keypoint_threshold)

estimator = TokenEstimator2D(modelPath=model_path)


if (GREEK_MENU):
    from YMAPNet import read_json_files
    print("Forcing greek vocabulary")
    estimator_pose.vocabulary = read_json_files("2d_pose_estimation/vocabulary_el.json")
    estimator.vocabulary = read_json_files("2d_pose_estimation/vocabulary_el.json")


# Visual theme
visual_theme = gr.themes.Default()  # Default, Soft or Monochrome

# Constants
MAX_OUTPUT_TOKENS = 2048
MAX_IMAGE_SIZE = (1120, 1120)



def log_image(frame, directory="server_log"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"{timestamp}.png"
    filepath  = os.path.join(directory, filename)
    cv2.imwrite(filepath, frame)
    print(f"Image saved at: {filepath}")

def encode_depth_map(depth, grayscale=False):
    """Encodes a one-channel depth map into a color or grayscale image."""
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255.0
    depth = depth.astype(np.uint8)
    if grayscale:
        return depth
    import matplotlib.pyplot as plt
    cmap        = plt.get_cmap('plasma')
    depth_color = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)
    depth_color = depth_color[:, :, ::-1]
    return depth_color


def threshold_image(image, threshold=20, preserve_above=True):
    """Apply thresholding to a single-channel image."""
    if image is None:
        return None
    image = np.array(image, dtype=np.uint8)
    if preserve_above:
        return np.where(image > threshold, image, 0).astype(np.uint8)
    return np.where(image > threshold, 255, 0).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# All channel names in the order they appear in the config heatmaps list.
# This list drives both describe_image and the Gradio output components.
# ──────────────────────────────────────────────────────────────────────────────
SEG_CHANNELS = [
    # (config_name,        ui_label,          apply_threshold)
    ("Person",             "Persons",          True),
    ("Face",               "Face",             True),
    ("Hand",               "Hand",             True),
    ("Foot",               "Foot",             True),
    ("Vehicle",            "Vehicle",          True),
    ("Animal",             "Animal",           True),
    ("Robot",              "Robot",            True),
    ("Label / Text",       "Text Detection",   True),
    ("Box",                "Box",              True),
    ("Tool",               "Tool",             True),
    ("Instrument",         "Instrument",       True),
    ("Appliance / Electronics", "Appliance",   True),
    ("Conveyor",           "Conveyor",         True),
    ("Chair",              "Chair",            True),
    ("Table",              "Table",            True),
    ("Bed",                "Bed",              True),
    ("Furniture",          "Furniture",        True),
    ("Light",              "Light",            True),
    ("Floor",              "Floor",            True),
    ("Ceiling",            "Ceiling",          True),
    ("Wall",               "Wall",             True),
    ("Door",               "Door",             True),
    ("Window",             "Window",           True),
    ("Plant / Vegetation", "Plant",            True),
    ("Road",               "Road",             True),
    ("Dirt",               "Dirt",             True),
    ("Sidewalk",           "Sidewalk",         True),
    ("Building",           "Building",         True),
    ("Mountain",           "Mountain",         True),
    ("Sky",                "Sky",              True),
    ("Food",               "Food",             True),
    ("Fruit",              "Fruit",            True),
    ("Water",              "Water",            True),
    ("Cup",                "Cup",              True),
]


def describe_image(image, crop_button, threshold, keypoint_threshold, history):
    log_image(image)

    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    estimator_pose.threshold          = threshold
    estimator_pose.keypoint_threshold = keypoint_threshold
    estimator_pose.process(frame, borders=not crop_button)

    global useStandaloneTokenEstimator, _token_frame_counter, _token_caption_cache
    if useStandaloneTokenEstimator:
        if _token_frame_counter % TOKEN_ESTIMATOR_SKIP_FRAMES == 0:
            estimator.threshold          = threshold
            estimator.keypoint_threshold = keypoint_threshold
            _token_caption_cache = estimator.process(frame, visualization=False)
            print("Using standalone captioner:", _token_caption_cache)
        _token_frame_counter += 1
        caption = _token_caption_cache
    else:
        caption = estimator_pose.description

    # ── channel lookup helpers ────────────────────────────────────────────────
    heatmaps      = estimator_pose.heatmapsOut
    heatmap_names = estimator_pose.cfg.get('heatmaps', [])

    def _raw(name, fallback_idx=-1):
        idx = retrieveHeatmapIndex(heatmap_names, name)
        if idx < 0:
            idx = fallback_idx
        if idx < 0 or idx >= len(heatmaps):
            return None
        return heatmaps[idx]

    def _thr(name, fallback_idx=-1):
        return threshold_image(_raw(name, fallback_idx))

    # ── fixed outputs ─────────────────────────────────────────────────────────
    input_image = estimator_pose.input_image
    input_image = visualization(input_image, estimator_pose.frameNumber,
                                estimator_pose.keypoint_in_nn_coordinate_image,
                                estimator_pose.keypoint_depths,
                                estimator_pose.keypoint_names)

    visBGR = visualization(frame, estimator_pose.frameNumber,
                           estimator_pose.keypoint_results,
                           estimator_pose.keypoint_depths,
                           estimator_pose.keypoint_names)
    visRGB = cv2.cvtColor(visBGR, cv2.COLOR_RGB2BGR)

    depthmap     = _raw('depthmap', 29)
    depthmap_col = encode_depth_map(depthmap, grayscale=True) if depthmap is not None else None

    normalX = _raw('normalX', 30)
    normalY = _raw('normalY', 31)
    normalZ = _raw('normalZ', 32)
    normal_output = np.stack((normalX, normalY, normalZ), axis=-1) \
                    if (normalX is not None and normalY is not None and normalZ is not None) else None

    improved_depth = None
    if depthmap is not None and normalX is not None:
        improved_depth = improve_depthmap(depthmap, normalX, normalY, normalZ,
                                          learning_rate=0.01, iterations=10)
        improved_depth = encode_depth_map(improved_depth, grayscale=True)

    # Joints / PAFs / segmentation unions
    selected_joints = heatmaps[0:17]
    union_joints    = np.max(selected_joints, axis=0)

    selected_pafs   = heatmaps[17:29]
    union_pafs      = np.max(selected_pafs, axis=0)

    # Segmentation union starts at first Person channel
    first_seg_idx = retrieveHeatmapIndex(heatmap_names, 'Person')
    if first_seg_idx < 0:
        first_seg_idx = 34
    selected_segms = heatmaps[first_seg_idx:]
    union_segms    = np.max(selected_segms, axis=0) if len(selected_segms) else None

    # ── per-class segmentation outputs (driven by SEG_CHANNELS table) ─────────
    seg_outputs = []
    for cfg_name, _, apply_thr in SEG_CHANNELS:
        if apply_thr:
            seg_outputs.append(_thr(cfg_name))
        else:
            seg_outputs.append(_raw(cfg_name))

    history = [(t("Result"), caption)]

    return (history, input_image, visRGB,
            union_joints, union_pafs, union_segms,
            normal_output, depthmap_col, improved_depth,
            *seg_outputs)


# Function to clear the chat history
def clear_chat():
    return []


# Gradio Interface
def gradio_interface():
    with gr.Blocks(visual_theme) as demo:
        gr.HTML("""
    <h1 style='text-align: center'>%s</h1>
    """ % t("RGB To Pose, Depth, Normals, Class and Token Estimation"))

        with gr.Row():
            # ── Left column: inputs ──────────────────────────────────────────
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label=t("Image"),
                    type="numpy",
                    image_mode="RGB",
                    height=512,
                    width=512,
                )
                threshold_sl          = gr.Slider(label=t("Heatmap Joint Threshold"), minimum=0.0, maximum=255.0, value=20.0,  step=0.1, interactive=True)
                keypoint_threshold_sl = gr.Slider(label=t("Keypoint Threshold"),      minimum=0.0, maximum=255.0, value=30.0,  step=0.1, interactive=True)
                crop_button           = gr.Checkbox(label=t("Crop Image"), value=True)
                generate_button       = gr.Button(t("Process"))

            # ── Right column: outputs ────────────────────────────────────────
            with gr.Column(scale=2):
                chat_history  = gr.Chatbot(label=t("Description"), height=100)
                rgb_output    = gr.Image(label=t("RGB and 2D Pose"), height=512, width=512)

                # Depth & normals
                with gr.Row():
                    normal_output         = gr.Image(label=t("Normals"),        height=256, width=256)
                    depth_output          = gr.Image(label=t("Depth"),          height=256, width=256)
                with gr.Row():
                    improved_depth_output = gr.Image(label=t("Improved Depth"), height=256, width=256)
                    input_image_out       = gr.Image(label=t("Input Image"),    height=256, width=256)

                # Union outputs
                with gr.Row():
                    union_joints  = gr.Image(label=t("Joints"),            height=256, width=256)
                    union_pafs    = gr.Image(label=t("PAFs"),              height=256, width=256)
                with gr.Row():
                    union_segms   = gr.Image(label=t("Segmentation Union"), height=256, width=256)

                # Per-class segmentation outputs — two per row
                seg_components = []
                for i in range(0, len(SEG_CHANNELS), 2):
                    with gr.Row():
                        _, lbl_a, _ = SEG_CHANNELS[i]
                        comp_a = gr.Image(label=t(lbl_a), height=256, width=256)
                        seg_components.append(comp_a)
                        if i + 1 < len(SEG_CHANNELS):
                            _, lbl_b, _ = SEG_CHANNELS[i + 1]
                            comp_b = gr.Image(label=t(lbl_b), height=256, width=256)
                            seg_components.append(comp_b)

        # ── fixed output list (matches describe_image return order) ──────────
        fixed_outputs = [
            chat_history,
            input_image_out,
            rgb_output,
            union_joints,
            union_pafs,
            union_segms,
            normal_output,
            depth_output,
            improved_depth_output,
        ]

        generate_button.click(
            fn       = describe_image,
            inputs   = [image_input, crop_button, threshold_sl, keypoint_threshold_sl, chat_history],
            outputs  = fixed_outputs + seg_components,
            api_name = "predict",
        )

    return demo


# Launch the interface
demo = gradio_interface()
demo.launch(favicon_path="doc/favicon.ico", server_name=server_name, server_port=int(port))
