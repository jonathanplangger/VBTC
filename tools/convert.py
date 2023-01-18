# --- Convert Rellis-3D Image names to a useful format ---#
# ---------------------------------------------------------
# This approach converts the name of the Rellis3D images into a format that can be easily employed 

#NOTE: This program must be placed in the sequence directory of Rellis (0000X) before use
# Make sure to create a backup prior to utilizing this tools

import os 

#
img_path = "./pylon_camera_node/"

for file in os.scandir(img_path):
    try: #rename all files to format: "frameXXXXXX.jpg 
        os.rename(img_path + file.name, img_path + file.name[:11] + ".jpg")
    except: 
        pass
