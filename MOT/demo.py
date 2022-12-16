# this file is employed to implement and demo the results from the use of the blendmask model 
import os, json, cv2, random
import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

#Used for file handling 
import json
import time 

# KITTI dataset
seq_dir = "../../datasets/odom_KITTI/sequences/10/"
img_dir = seq_dir + "image_0/" # generalized directory path

# code taken from the demo website for detectron2
cfg = get_cfg()

# Select the model being implemented
cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml")) # model configuration file
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml") # Model weights

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

# obtain the predictor based on the configuration
predictor = DefaultPredictor(cfg)

# Function: onImage
# Employs the instance segmentation model on the image being input into this function 
# img_name : string -> name of the image being displayed 
# predictor: DefaultPredictor -> predictor being employed in the segmentation 
def onImage(img_name, predictor): 
    # read the image 
    im = cv2.imread(img_dir + img_name)
    # obtain predictions from the model 
    outputs = predictor(im)

    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("Segmentation Test", out.get_image())
    cv2.waitKey(0) == 27 #keep window open after execution (27 is the ascii character for ESC)


# ----- For a video stream (KITTI sequence) -------- #

# obtain the frame times from the times.txt file
with open(seq_dir + "times.txt") as f: 
    lines = f.readlines()  

# open the image files for a video stream
cap = cv2.VideoCapture(img_dir + "%06d.png")

# change in time between frames
oldTime = 0.0

# display frames while there are images left in the directory
while(cap.isOpened()): 
    ret, frame = cap.read()

    # obtain a prediction for the frame
    outputs  = predictor(frame)
    v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # display the image
    cv2.imshow("image", out.get_image())
    
    # break out of the loop when q is pressed
    if cv2.waitKey(1) == ord('q'):
        break
    
    # obtain the time difference between the two frames
    newTime = float(lines.pop(0))
    delta = newTime - oldTime
    oldTime = newTime

    # use the measured delay when displaying
    # time.sleep(delta)






