import json
import time
import sys
import os
from gradio_client import Client, handle_file
import cv2  # For saving image files
import numpy as np

#python3 gradioClient.py datasets/coco/cache/coco/val2017/000000412362.jpg 
#python3 gradioClient.py --directory datasets/coco/cache/coco/val2017/ && python3 encode_outputs_as_html.py 

#Hello world
# Replace with the actual server URL if different
ip = "127.0.0.1"
port = "8080"

# Define the user prompt (caption)
user_prompt = "Thoroughly and carefully describe this image."

files = []
output_file = "output.json"
image_output_dir = "output_images"

# Ensure the output directory exists
if not os.path.exists(image_output_dir):
    os.makedirs(image_output_dir)

# Hyperparameters
threshold = 0.2
startAt = 0

# Parse command line arguments
argumentStart = 1
if len(sys.argv) > 1:
    for i in range(0, len(sys.argv)):
        if sys.argv[i] == "--ip":
            ip = sys.argv[i+1]
            argumentStart += 2
        if sys.argv[i] == "--directory":
            directory = sys.argv[i+1]
            argumentStart += 2
            # Populate files with image (.jpg, .png) contents of directory
            if os.path.isdir(directory):
                directoryList = os.listdir(directory)
                directoryList.sort() 
                for file in directoryList:
                    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                        files.append(os.path.join(directory, file))
            else:
                print(f"Error: Directory '{directory}' does not exist.")
                sys.exit(1)
        elif sys.argv[i] == "--start":
            startAt = int(sys.argv[i+1])
            argumentStart += 2
        elif sys.argv[i] == "--port":
            port = sys.argv[i+1]
            argumentStart += 2
        elif sys.argv[i] == "--threshold":
            threshold = float(sys.argv[i+1])
            argumentStart += 2
        elif sys.argv[i] in ("--output", "-o"):
            output_file = sys.argv[i+1]
            argumentStart += 2

results = dict()
results["prompt"] = user_prompt

for i in range(argumentStart, len(sys.argv)):
    files.append(sys.argv[i])

# Make sure the list is sorted
files.sort()

if (len(files)==0):
   print("No input files to process")
   sys.exit(0)

# Initialize the Gradio client with the server URL
client = Client(f"http://{ip}:{port}")
# client.view_api()


# Possibly start at specific index
for i in range(startAt, len(files)):
    # Grab next image path
    image_path = files[i]

    # Count start time
    start = time.time()

    # Make query to LLM
    try:
        # Send the image file path and the prompt to the Gradio app for processing
        result = client.predict(
            image=handle_file(image_path),   # Provide the file path directly
            threshold=0.2,
            history=[],
            api_name="/predict"
        )
    except Exception as e:
        print("Failed to complete job, please restart using --start", i)
        output_file = f"partial_until_{i}_{output_file}"
        break

    # Calculate elapsed time
    seconds = time.time() - start
    remaining = (len(files) - i) * seconds
    hz = 1 / (seconds + 0.0001)


    print(result)

    # Output the result
    response = result[0][0][1]
    print(f"Processing {1+i}/{len(files)} | {hz:.2f} Hz / remaining {remaining/60:.2f} minutes")
    print("Image:", image_path, "\nResponse:", response)

    # Store each path as the key pointing to each description
    results[image_path] = response


    # Save each returned image
    output_image_paths = { 
        "rgb_raw": image_path,
        "rgb_input": result[1],
        "rgb_output": result[1+1], 
        "joints_output": result[2+1],
        "pafs_output": result[3+1],
        "segms_output": result[4+1],
        "normal_output": result[2+3+1],
        "depth_output": result[3+3+1],
        "depth_improved_output": result[3+3+1+1],
        "text_segment_output": result[4+3+1+1],
        "person": result[5+3+1+1],
        "vehicle": result[6+3+1+1],
        "animal": result[7+3+1+1],
        "object": result[8+3+1+1],
        "furniture": result[9+3+1+1],
        "appliance": result[10+3+1+1],
        "material": result[11+3+1+1],
        "obstacle": result[12+3+1+1],
        "building": result[13+3+1+1],
        "nature": result[14+3+1+1]
    }

    image_path = "%05u" % i

    # Save the response to a text file in the output directory
    text_output_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_response.txt"
    text_output_filepath = os.path.join(image_output_dir, text_output_filename)
    with open(text_output_filepath, "w") as text_file:
        text_file.write(response)
        print(f"Saved response text at {text_output_filepath}")

    #i=0
    for label, output_image_path in output_image_paths.items():
        # Save the image using OpenCV
        output_image = cv2.imread(output_image_path)
        if output_image is not None:  # Confirm image loaded successfully
            output_image_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_{label}.png"
            output_image_filepath = os.path.join(image_output_dir, output_image_filename)
            cv2.imwrite(output_image_filepath, output_image)
            print(f"Saved {label} at {output_image_filepath}")
            #i=i+1

print(f"\n\n\nStoring results in JSON file {output_file}")
with open(output_file, "w") as outfile:
    json.dump(results, outfile, indent=4)

