import sys
import numpy as np

def generate_heatmap(gradientSize,heatmapSize,thisActive = 127,heatmapDeactivated = -128):
    target_size = (heatmapSize, heatmapSize)  # Define your target size here
    rangeMin = heatmapDeactivated  # Define your range min value here
    rangeMax = thisActive  # Define your range max value here
    gradientStep = 1  # Define your gradient step value here
    
    heatmaps = np.full((target_size[1], target_size[0], 1),heatmapDeactivated, dtype=np.int8)

    y=int(target_size[1]/2)
    x=int(target_size[0]/2)
    heatmaps[y, x, 0] = thisActive
            
    # Add gradient around the active value
    if gradientSize > 0:
                thisJointGradientSize = gradientSize
                for yy in range(max(0, y - thisJointGradientSize), min(target_size[1], y + thisJointGradientSize)):
                    for xx in range(max(0, x - thisJointGradientSize), min(target_size[0], x + thisJointGradientSize)):
                        if yy != y or xx != x:
                            distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
                            intensity = int(thisActive - ((thisActive - heatmapDeactivated) / thisJointGradientSize) * distance)
                            #intensity = intensity - gradientStep  # Add a gradient step to prioritize peak

                            newValue = np.int8(np.clip(intensity, rangeMin, rangeMax))
                            heatmaps[yy, xx, 0] = newValue

    return heatmaps[:,:,0]

def generate_c_code_aligned2D(matrix):
    max_val_length = max(len(str(val)) for row in matrix for val in row)
    code = "const int heatmap_size_%u = %u; \n"%(gradientSize,heatmapSize)
    code += "//generated using python3 createCHeatmapPrecalcs.py %u %u\n"%(gradientSize,heatmapSize)
    code += "const signed char heatmap_%u_%ux%u "%(gradientSize,heatmapSize,heatmapSize)
    code += " [{}][{}] = {{\n".format(len(matrix), len(matrix[0]))
    for row in matrix:
        code += "    {"
        for val in row:
            val_str = str(val)
            spaces_needed = max_val_length - len(val_str) + 1
            code += " " * spaces_needed + val_str + ","
        code = code[:-1]  # Remove the trailing comma
        code += "},\n"
    code += "};\n"
    return code

def generate_c_code_aligned(matrix,gradientSize,heatmapSize,thisActive,heatmapDeactivated):
    max_val_length = max(len(str(val)) for row in matrix for val in row)
    code = "const int heatmap_size_%u = %u; \n"%(gradientSize,heatmapSize)
    code += "//generated using python3 createCHeatmapPrecalcs.py %u %u\n"%(gradientSize,heatmapSize)
    code += "const signed char heatmap_%u_%ux%u[%u] = {\n"%(gradientSize,heatmapSize,heatmapSize,heatmapSize*heatmapSize)
    for row in matrix:
        code += "    "
        for val in row:
            if (val==thisActive):
               val_str="MAXV"
            elif (val==heatmapDeactivated):
               val_str="MINV"
            else:
               val_str = str(val)
            spaces_needed = max_val_length - len(val_str) + 1
            code += " " * spaces_needed + val_str + ","
        code += "\n"
    code = code[:-2]  # Remove the trailing comma
    code += "\n};\n"
    return code


def generateAllUntil(limit,  thisActive = 120,  heatmapDeactivated = -120):
 for i in range(2,limit):
  gradientSize = int(i)
  heatmapSize  = 2*int(i)
  heatmap = generate_heatmap(gradientSize,heatmapSize,thisActive = thisActive,heatmapDeactivated = heatmapDeactivated)
  print("\n\n\n\n")
  c_code = generate_c_code_aligned(heatmap,gradientSize,heatmapSize,thisActive,heatmapDeactivated)
  print(c_code)


#============================================================================================
if __name__ == '__main__':
  thisActive = 120
  heatmapDeactivated = -120

  if len(sys.argv) != 3  :
        print("\n\nNo arguments given")
        print("Correct usage: python3 createCHeatmapPrecalcs.py gradientSize HeatmapSize")
        generateAllUntil(13)
        sys.exit(1)

  gradientSize = int(sys.argv[1])
  heatmapSize  = int(sys.argv[2])
  heatmap = generate_heatmap(gradientSize,heatmapSize,thisActive = thisActive,heatmapDeactivated = heatmapDeactivated)
  print(heatmap)
  print("\n\n\n\n")
  c_code = generate_c_code_aligned(heatmap,gradientSize,heatmapSize,thisActive = thisActive,heatmapDeactivated = heatmapDeactivated)
  print(c_code)
