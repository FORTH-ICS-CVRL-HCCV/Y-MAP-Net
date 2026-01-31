import numpy as np
import os
import sys
import cv2
import json

serialNumber="180"
deleteAfterUse=False

#-----------------------------------------------

if (len(sys.argv)>1):
       #print('Argument List:', str(sys.argv))
       for i in range(0, len(sys.argv)):
           if (sys.argv[i]=="--serial"):
              serialNumber = sys.argv[i+1]
           if (sys.argv[i]=="--deleteAfterUse"):
              deleteAfterUse = True

sourceDirectory = "cvpr_coco_gt/"
targetDirectory = "cvpr_ynet_%s/" % serialNumber

#-----------------------------------------------
if (not os.path.isdir(sourceDirectory)):
   print("Source directory %s does not exist" % sourceDirectory) 
   sys.exit(1)
if (not os.path.isdir(targetDirectory)):
   print("Target directory %s does not exist" % targetDirectory) 
   sys.exit(1)
#-----------------------------------------------

numberOfSamples = 4991

#-----------------------------------------------
rangeName     = ["Everything", "Joints", "PAFs", "Depth", "Normals", "Text", "Class Segmentation"]
heatmapRanges = [(0,43), (0,16), (17,28), (29,29), (30,32), (33,33),  (33,43)] 

rangeName     = ["Joints", "PAFs", "Depth", "Normals", "Text", "Class Segmentation"]
heatmapRanges = [(0,16), (17,28), (29,29), (30,32), (33,33),  (33,43)] 
# Constants for SSIM calculation
C1 = 0.01 ** 2
C2 = 0.03 ** 2

# Define MSE function
def compute_mse(image1, image2):
    return np.mean((image1 - image2) ** 2)

# Define SSIM function
def compute_ssim(image1, image2):
    # Means
    mu1 = np.mean(image1)
    mu2 = np.mean(image2)

    # Variances and covariance
    sigma1_sq = np.var(image1)
    sigma2_sq = np.var(image2)
    sigma12 = np.mean((image1 - mu1) * (image2 - mu2))

    # SSIM calculation
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim

def compute_confusion_metrics(y_true, y_pred, threshold=127):
    y_true_bin = (y_true > threshold).astype(np.uint8)
    y_pred_bin = (y_pred > threshold).astype(np.uint8)

    TP = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
    TN = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
    FP = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
    FN = np.sum((y_true_bin == 1) & (y_pred_bin == 0))
    return TP, TN, FP, FN

# Define HDM function
def compute_hdm_old(y_true, y_pred, threshold=0.1):
    thresholdBytes = threshold * 255.0
    # Calculate the absolute difference and binarize based on threshold
    diffs = np.abs(y_true - y_pred)
    correct_pixels = np.sum(diffs <= thresholdBytes)
    total_pixels = y_true.size
    return correct_pixels / total_pixels

# Define HDM function
def compute_hdm(y_true, y_pred, threshold=0.1):
    diff = np.abs(y_true.astype(np.float32) - y_pred.astype(np.float32))
    return np.mean(diff <= threshold * 255.0)
# Dictionary to store SSIM, MSE, and HDM metrics




def show_debug_sample(source, target, scores=None, window_name="Debug View"):
    """
    Displays the source, target, and prediction side-by-side for debugging.
    
    Parameters:
    - source: np.ndarray, the input image
    - target: np.ndarray, ground truth (H x W or H x W x C)
    - prediction: np.ndarray, predicted output (H x W or H x W x C)
    - scores: dict, optional. e.g., {"SSIM": 0.27, "MSE": 1.2, "HDM0.1": 0.99}
    """
    def to_vis(img):
        # Normalize and convert to 3-channel 8-bit for visualization
        if len(img.shape) == 3 and img.shape[2] > 3:
            img = img[..., :3]  # Trim to 3 if too many channels
        elif len(img.shape) == 2:
            img = np.stack([img]*3, axis=-1)

        img = img.astype(np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        img = (img * 255).astype(np.uint8)
        return img

    src_vis = to_vis(source)
    tgt_vis = to_vis(target)
    #pred_vis = to_vis(prediction)

    # Resize all to the same height if needed
    H = max(src_vis.shape[0], tgt_vis.shape[0])
    src_vis = cv2.resize(src_vis, (src_vis.shape[1], H))
    tgt_vis = cv2.resize(tgt_vis, (tgt_vis.shape[1], H))
    #pred_vis = cv2.resize(pred_vis, (pred_vis.shape[1], H))

    combined = np.hstack([src_vis, tgt_vis])

    # Add text overlay
    text_lines = [
       # f"Heatmaps: {source.shape[-1] if source.ndim == 3 else 1}"
    ]
    if scores:
        for k, v in scores.items():
            text_lines.append(f"{k}: {v:.4f}")

    for i, line in enumerate(text_lines):
        y = 25 + i * 25
        cv2.putText(combined, line, (10, y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 255, 0),
                    thickness=2)

    cv2.imshow(window_name, combined)
    print(f"[DEBUG] Showing sample: Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)




#metrics = {f'Test group {rangeName[testNumber]}': {'SSIM': [], 'MSE': [], 'HDM0.01': [], 'HDM0.1': [], 'HDM0.3': [], 'HDM0.5': [], 'HDM0.8': []} for testNumber in range(len(heatmapRanges))}
metrics = {
    f'Test group {rangeName[testNumber]}': {
        'SSIM': [], 'MSE': [], 'HDM0.01': [], 'HDM0.1': [], 'HDM0.3': [], 'HDM0.5': [], 'HDM0.8': [],
        'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0
    } for testNumber in range(len(heatmapRanges))
}

debugEverything = False
debugHm = 2

for testNumber, rangeStartStop in enumerate(heatmapRanges):
    start = rangeStartStop[0]
    stop  = rangeStartStop[1] + 1
    print("Test group ", rangeName[testNumber]," ( start ",start," stop ",stop,") ")
    #------------------------------------------
    ssim_values  = []
    mse_values   = []
    hdm001_values= []
    hdm01_values = []
    hdm03_values = []
    hdm05_values = []
    hdm08_values = []
    #------------------------------------------
    for hm in range(start, stop):
       print(" Processing Heatmap ",hm-start," / ",stop-start,end="  \r")
       total_TP, total_TN, total_FP, total_FN = 0, 0, 0, 0
       for sampleID in range(numberOfSamples):
            #print("Processing sample%05u_C8bit_hm%u.png" % (sampleID, hm))

            sourcePath = "%s/sample%05u_C8bit_hm%u.png" % (sourceDirectory, sampleID, hm)
            targetPath = "%s/sample%05u_C8bit_hm%u.png" % (targetDirectory, sampleID, hm)
            
            # Load 1 channel png file
            src_image = cv2.imread(sourcePath, cv2.IMREAD_GRAYSCALE)
            trg_image = cv2.imread(targetPath, cv2.IMREAD_GRAYSCALE)

            #Each image has values 0..255
            #You can uncomment below to make sure opencv does not pull a trick on us
            if (debugEverything) and (hm==debugHm):
              print("Src Min:",np.min(src_image))
              print("Src Max:",np.max(src_image))
              print("Trg Min:",np.min(trg_image))
              print("Trg Max:",np.max(trg_image))
            
            # Ensure both images are loaded correctly

            if src_image is not None and trg_image is not None:
                # Calculate SSIM, MSE, and HDM for the current heatmap
                ssim_value     = compute_ssim(src_image, trg_image)
                mse_value      = compute_mse(src_image, trg_image)
                hdm_001_value  = compute_hdm(src_image, trg_image, threshold=0.01)
                hdm_01_value   = compute_hdm(src_image, trg_image, threshold=0.1)
                hdm_03_value   = compute_hdm(src_image, trg_image, threshold=0.3)
                hdm_05_value   = compute_hdm(src_image, trg_image, threshold=0.5)
                hdm_08_value   = compute_hdm(src_image, trg_image, threshold=0.8)
                
                TP, TN, FP, FN = compute_confusion_metrics(src_image, trg_image)
                total_TP += TP
                total_TN += TN
                total_FP += FP
                total_FN += FN


                if (debugEverything) and (hm==debugHm):
                  vis_scores = {
                  "hm #": int(hm),
                  "SSIM": ssim_value,
                  "MSE": mse_value,
                  "HDM0.1": hdm_01_value, }
                  show_debug_sample(src_image, trg_image, vis_scores)

                ssim_values.append(ssim_value)
                mse_values.append(mse_value)
                hdm001_values.append(hdm_001_value)
                hdm01_values.append(hdm_01_value)
                hdm03_values.append(hdm_03_value)
                hdm05_values.append(hdm_05_value)
                hdm08_values.append(hdm_08_value)

            else:
                print(f"Error loading images {sourcePath} or {targetPath} for sampleID={sampleID}, hm={hm}")

    # Store average SSIM, MSE, and HDM for the current sample in this test group
    if ssim_values and mse_values and hdm01_values  and hdm03_values  and hdm05_values and hdm08_values:
            #---------------------------------------------------------------------------
            avg_ssim   = np.mean(ssim_values)
            avg_mse    = np.mean(mse_values)
            avg_hdm001 = np.mean(hdm001_values)
            avg_hdm01  = np.mean(hdm01_values)
            avg_hdm03  = np.mean(hdm03_values)
            avg_hdm05  = np.mean(hdm05_values)
            avg_hdm08  = np.mean(hdm08_values)
            #---------------------------------------------------------------------------
            metrics[f'Test group %s'%rangeName[testNumber]]['SSIM'].append(avg_ssim)
            metrics[f'Test group %s'%rangeName[testNumber]]['MSE'].append(avg_mse)
            metrics[f'Test group %s'%rangeName[testNumber]]['HDM0.01'].append(avg_hdm001)
            metrics[f'Test group %s'%rangeName[testNumber]]['HDM0.1'].append(avg_hdm01)
            metrics[f'Test group %s'%rangeName[testNumber]]['HDM0.3'].append(avg_hdm03)
            metrics[f'Test group %s'%rangeName[testNumber]]['HDM0.5'].append(avg_hdm05)
            metrics[f'Test group %s'%rangeName[testNumber]]['HDM0.8'].append(avg_hdm08)
            metrics[f'Test group %s'%rangeName[testNumber]]['TP'] += total_TP
            metrics[f'Test group %s'%rangeName[testNumber]]['TN'] += total_TN
            metrics[f'Test group %s'%rangeName[testNumber]]['FP'] += total_FP
            metrics[f'Test group %s'%rangeName[testNumber]]['FN'] += total_FN

    print("Count of SSIM values used to average for "  ,rangeName[testNumber]," calculation = ",len(ssim_values))
    print("Count of MSE values used to average for "   ,rangeName[testNumber]," calculation = ",len(mse_values))
    print("Count of HDM0.01 values used to average for ",rangeName[testNumber]," calculation = ",len(hdm001_values))
    print("Count of HDM0.1 values used to average for ",rangeName[testNumber]," calculation = ",len(hdm01_values))
    print("Count of HDM0.3 values used to average for ",rangeName[testNumber]," calculation = ",len(hdm03_values))
    print("Count of HDM0.5 values used to average for ",rangeName[testNumber]," calculation = ",len(hdm05_values))
    print("Count of HDM0.8 values used to average for ",rangeName[testNumber]," calculation = ",len(hdm08_values)) 

# Calculate and display summary statistics for each test group
for test_group, values in metrics.items():
    print(f"\nSummary for ",test_group," :")

    for metric_name, metric_values in values.items():
      if metric_name in ["TP", "TN", "FP", "FN"]:
        print(f"{metric_name}: {metric_values}")
      elif metric_values:
        avg_value = np.mean(metric_values)
        #min_value = np.min(metric_values)
        #max_value = np.max(metric_values)
        #avg_value = np.mean(metric_values)
        print(f"{metric_name} - Avg: {avg_value:.4f}")
      else:
        print(f"{metric_name} - No data available")
 

#TypeError: Object of type int64 is not JSON serializable
#outputJSONFilename = "evaluation_results_%s.json" % serialNumber
#with open(outputJSONFilename, "w") as f:
#    json.dump(metrics, f, indent=4)

print("Source was ",sourceDirectory)
print("Target was ",targetDirectory)

if (deleteAfterUse):
   os.system("rm -rf %s" % targetDirectory)
#print("Results stored in ",outputJSONFilename)


