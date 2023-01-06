import os 
import cv2

label_dir = '../../datasets/Rellis-3D/00000/pylon_camera_node_label_id/'
fileName = 'frame000000-1581624652_750.png'

filePath = label_dir + fileName


assert os.path.exists(filePath)

# open the label file for examination
label = cv2.imread(filePath)



print(label[1100,10])



