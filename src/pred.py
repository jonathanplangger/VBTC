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
    def __init__(self, predictor, img_dir, cfg, display = False, dataset = "kitti", save=False):
        """
            Args:
                pred (DefaultPredictor): Predictor trained for use in segmentation\n
                img_dir (str): Path to the targeted sequence directory\n
                cfg (?): Configuration file from detectron2\n
                dataset: selects what dataset type is being used. Options are: "kitti", "rellis"\n
                save: segmentation resulting images are saved within the results/ folder \n
        """
        self.img_dir = img_dir
        self.predictor = predictor
        self.cfg = cfg
        self.disp = display
        self.dataset = dataset
        self.save = save

    def __createResultsDir(self):
        """
            Creates the directory structure required to output the results for the program.
            No args required
        """
        # create a new directory if it does not already exist
        try: 
            os.mkdir("./results")
        except:
            pass

        # Create a new directory for the predictions
        try: 
            os.mkdir("./results/pred")
        except:
            pass

        # Create a new directory for the segmentation information 
        try: 
            os.mkdir("./results/seginfo")
        except:
            pass

    def __printProgress(self, total:int, cur:int): 
        """
            Print the current progress of the program onto the console 
            Args: 
                total (int): Total amount of files to be processed 
                cur (int): current file being processed
        """
        print('\r', "Progress: {}/{}".format(cur, total), end='')

    def getImageFilePath(self, dataset="kitti"): 
        """
            Obtain the file path based on the different datasets
            Args: 
                dataset(str): selects the dataset to be used: Ex. "kitti", "rellis"
        """    
        if(dataset == "kitti"): 
            return "image_0/"
        elif(dataset == "rellis"): 
            return

    def displaySeg(self, img, seg, seg_info, wait:int=1): 
        """
            Display the segmentation information onto images. \n
            Args:\n
                img(Image->Tensor): Image file being displayed \n
                seg(List->Tensor): List of masks representing each segmentation mask\n
                seg_info(numpy ndarray): Array containing information regarding each class present during segmentation\n
                wait (int): Amount of time (in ms) opencv will wait for a key input before displaying the next frame (default = 1ms)\n
        """
            # Visualize the segmentation onto a viewer 
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_panoptic_seg_predictions((seg).to("cpu"),seg_info)
        
        # display the image
        cv2.imshow("image", out.get_image())
        
        # exit out of the program if the display is turned off 
        if cv2.waitKey(wait) == ord('q'):
            exit("Exiting out of program...")


    def displaySingleImg(self, frame_num:int=0):
        """
            Allows for the selective display of a single image. Press 'q' to exit opencv window\n
            Args: \n
                frame_num: Number of frame image to be displayed
        """    
        im  = cv2.imread(self.img_dir+"frame{:06d}.jpg".format(frame_num))
        seg = torch.load("results/pred/{:06d}.pt".format(frame_num))
        with open("results/seginfo/{:06d}.json".format(frame_num), 'r') as j: 
            seg_info = json.loads(j.read())

        # display the image
        self.displaySeg(im, seg, seg_info, wait=-1)


    def predict(self):
        # should update the code later to use directly the class variables rather than creating a new variabl
        predictor = self.predictor

        # create the directory structure
        self.__createResultsDir()
        
        # open the image files for a video stream
        # cap = cv2.VideoCapture(img_dir + "%06d.jpg") # KITTI dataset
        cap = cv2.VideoCapture(self.img_dir + "frame%06d.jpg")

        # obtain the total amounts of frames within the img directory
        frame_total = len(os.listdir(self.img_dir))

        # change in time between frames
        oldTime = 0.0
        frame_count = 0 # obtain the count of the frame
        frame_time = []
        preds = [] # list that stores the obtained predictions 
        seginfos = [] # list which stores segmentatio information for each instance 

        # display frames while there are images left in the directory
        while(cap.isOpened()):
            # obtain an image file 
            ret, frame = cap.read()

            # obtain the current progress for the process.
            # NOTE: This can slow down the operation of the program significantly 
            # self.__printProgress(frame_total, frame_count)

            if frame is not None:

                # start time
                start_time = time.time()
                # Panoptic segmentation
                outputs, segments_info  = predictor(frame)["panoptic_seg"]
                # total execution time
                exec_time = round(time.time() - start_time,6)
                # print("Frame execution time: " + str(exec_time) + " s")
                frame_time.append(exec_time) # add the time to the array 

                # save the outputs to files 
                if self.save:
                    # Save result predictions
                    torch.save(outputs, "results/pred/{:06d}.pt".format(frame_count))

                    # Save segmentation information to JSON
                    with open("results/seginfo/{:06d}.json".format(frame_count), 'w') as f: 
                        json.dump(segments_info,f)

                # display the image 
                if self.disp:
                    self.displaySeg(frame, outputs, segments_info)


                # increment the count for the frames 
                frame_count = frame_count + 1
            else: 
                break

        # log the output file
        log  = open('results/time_log.txt', 'w')
        print(frame_time, file=log)
        log.close()


        


