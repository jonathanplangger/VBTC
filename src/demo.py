# This file is employed to demo panoptic segmentation model
import os, json, cv2, random
import time
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

import  pred

# Filepath for the given dataset (relative to the location of demo.py)
# img_dir = "../../datasets/odom_KITTI/sequences/10/image_0/"
img_dir = "../../datasets/Rellis-3D/00000/pylon_camera_node/"

# code taken from the demo website for detectron2
cfg = get_cfg()

# Select the model being implemented
cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml")) # model configuration file
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml") # Model weights

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

# obtain the predictor based on the configuration
predictor = DefaultPredictor(cfg)

vid  = pred.VideoPredictor(predictor, img_dir, cfg, display=False)
vid.predict()








