# Shared output schema for gradioServer.py and gradioClient.py.
# Edit this file to add/remove/rename outputs — both sides update automatically.

# Fixed outputs returned before the per-class segmentation channels.
# Order must match the return statement in describe_image().
FIXED_OUTPUT_KEYS = [
    "history",  # result[0]  Chatbot / description
    "input_image",  # result[1]  NN input image with joints
    "rgb_output",  # result[2]  RGB frame with 2-D pose overlay
    "union_joints",  # result[3]  Joint heatmap union
    "union_pafs",  # result[4]  PAF union
    "union_segms",  # result[5]  Segmentation union
    "normal_output",  # result[6]  Combined XYZ normals
    "depth_output",  # result[7]  Depth map (colourised)
    "improved_depth",  # result[8]  Improved depth map
]

# Per-class segmentation channels.
# (config_name, output_key, ui_label, apply_threshold)
# config_name   — must match the string in cfg['heatmaps']
# output_key    — used as the dict key in the client's saved-image map
# ui_label      — label shown in the Gradio UI
# apply_threshold — whether to apply threshold_image() before returning
SEG_CHANNELS = [
    ("Person", "person", "Persons", True),
    ("Face", "face", "Face", True),
    ("Hand", "hand", "Hand", True),
    ("Foot", "foot", "Foot", True),
    ("Vehicle", "vehicle", "Vehicle", True),
    ("Animal", "animal", "Animal", True),
    ("Robot", "robot", "Robot", True),
    ("Label / Text", "text", "Text Detection", True),
    ("Box", "box", "Box", True),
    ("Tool", "tool", "Tool", True),
    ("Instrument", "instrument", "Instrument", True),
    ("Appliance / Electronics", "appliance", "Appliance", True),
    ("Conveyor", "conveyor", "Conveyor", True),
    ("Chair", "chair", "Chair", True),
    ("Table", "table", "Table", True),
    ("Bed", "bed", "Bed", True),
    ("Furniture", "furniture", "Furniture", True),
    ("Light", "light", "Light", True),
    ("Floor", "floor", "Floor", True),
    ("Ceiling", "ceiling", "Ceiling", True),
    ("Wall", "wall", "Wall", True),
    ("Door", "door", "Door", True),
    ("Window", "window", "Window", True),
    ("Plant / Vegetation", "plant", "Plant", True),
    ("Road", "road", "Road", True),
    ("Dirt", "dirt", "Dirt", True),
    ("Sidewalk", "sidewalk", "Sidewalk", True),
    ("Building", "building", "Building", True),
    ("Mountain", "mountain", "Mountain", True),
    ("Sky", "sky", "Sky", True),
    ("Food", "food", "Food", True),
    ("Fruit", "fruit", "Fruit", True),
    ("Water", "water", "Water", True),
    ("Cup", "cup", "Cup", True),
]

# ── Helpers ───────────────────────────────────────────────────────────────────


def output_index(key: str) -> int:
    """Return the positional index of *key* in the server's return tuple."""
    if key in FIXED_OUTPUT_KEYS:
        return FIXED_OUTPUT_KEYS.index(key)
    for i, (_, out_key, _, _) in enumerate(SEG_CHANNELS):
        if out_key == key:
            return len(FIXED_OUTPUT_KEYS) + i
    raise KeyError(f"Unknown output key: {key!r}")


def all_output_keys() -> list:
    """Ordered list of every output key the server produces."""
    return FIXED_OUTPUT_KEYS + [out_key for _, out_key, _, _ in SEG_CHANNELS]
