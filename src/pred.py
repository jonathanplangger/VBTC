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

# ----- For a video stream (KITTI sequence) -------- #
class VideoPredictor(object):
    def __init__(self, pred, img_dir, cfg, display = False, dataset = "kitti"):
        """
            Args:
                pred (DefaultPredictor): Predictor trained for use in segmentation\n
                img_dir (str): Path to the targeted sequence directory\n
                cfg (?): Configuration file from detectron2\n
                dataset: selects what dataset type is being used. Options are: "kitti", "rellis"
        """
        self.img_dir = img_dir
        self.pred = pred
        self.cfg = cfg
        self.disp = display
        self.dataset = dataset

    def getImageFilePath(self, dataset="kitti"): 
        """
            Args: 
                dataset(str): selects the dataset to be used: Ex. "kitti", "rellis"
        """    
        if(dataset == "kitti"): 
            return "image_0/"
        elif(dataset == "rellis"): 
            return


    def displaySeg(self, img, seg, seg_info): 
        """
            Args:
                img(Image->Tensor): Image file being displayed 
                seg(list): List of masks representing each segmentation mask
                seg_info(numpy ndarray): Array containing information regarding each class present during segmentation
        """
            # Visualize the segmentation onto a viewer 
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_panoptic_seg_predictions((seg).to("cpu"),seg_info)
        
        # display the image
        cv2.imshow("image", out.get_image())
        
        # exit out of the program if the display is turned off 
        if cv2.waitKey(1) == ord('q'):
            exit("Exiting out of program...")


    def predict(self):
        # should update the code later to use directly the class variables rather than creating a new variable
        img_dir = self.img_dir
        predictor = self.pred

        # create a new directory if it does not already exist
        try: 
            os.mkdir("./results")
        except:
            pass
        
        # open the image files for a video stream
        # cap = cv2.VideoCapture(img_dir + "%06d.jpg") # KITTI dataset
        cap = cv2.VideoCapture(img_dir + "frame%06d.jpg")

        # change in time between frames
        oldTime = 0.0
        frame_count = 0 # obtain the count of the frame
        frame_time = []

        # display frames while there are images left in the directory
        while(cap.isOpened()):
            # obtain an image file 
            ret, frame = cap.read()

            if frame is not None:

                # increment the quantity of frames present
                frame_count = frame_count + 1

                # start time
                start_time = time.time()
                # Panoptic segmentation
                outputs, segments_info  = predictor(frame)["panoptic_seg"]
                
                # total execution time
                exec_time = round(time.time() - start_time,6)
                # print("Frame execution time: " + str(exec_time) + " s")
                frame_time.append(exec_time) # add the time to the array 

                # display the image 
                if self.disp:
                    self.displaySeg(frame, outputs, segments_info)


            else: 
                break

        # log the output file
        log  = open('results/log.txt', 'w')
        print(frame_time, file=log)
