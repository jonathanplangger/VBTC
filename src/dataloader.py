# This file provides the ability of registering and configuring the loading the data from rellis
import os, json, cv2
import time
import torch, detectron2
import numpy as np

from detectron2.data import DatasetCatalog
from detectron2.data import build_detection_train_loader, DatasetMapper
from detectron2.config import CfgNode

# directory path for the entire sequence


# ------------- Dataloader for the new dataset ------------- #
class DataLoader(object):

    def __init__(self):
        """
            Dataloader provides an easy interface for loading data for training and testing of new models.
        """
    
        # only configured for the rellis dataset as of right now, would be good to add some configuration for multiple datasets
        DatasetCatalog.register("rellis", self.__reg_rellis)
        data: list[dict] = DatasetCatalog.get("rellis")
        # store the obtained metadata as a parameter
        self.metadata = data


    # --------------------------- Database Registrations --------------------------------------#
    def __reg_rellis(data):
        """
            Provides the dataloader with the required format used in detectron2 for  the Rellis 3D Dataset.\n 
            Current iteration employs filepath tied directly to the current file structure of the VM. This may need to be updated in future versions\n
            Parameters: \n
            data: Required by the datasetcatalog for some reason, for now it is not used in the code below.
            -----------------------------------------\n
            Returns: List[dict] - Metadata regarding each data file 
        """
        path = "../../datasets/Rellis-3D/"

        meta_files = []

        train_lst = open(path + "train.lst", "r")
        
        for line in train_lst.readlines():
            # obtain the image file name as well as the associated segmentation mask
            [img_name, seg_name] = line.split(' ')
            img_id = img_name.split("frame")[1][0:6]
            # Create the new dictionary
            meta = dict(
                file_name = img_name,
                height="1200",
                width="1920", 
                image_id=img_id, 
                sem_seg_file_name= seg_name
            )
            # add the file to the list
            meta_files.append(meta)
        
    

        return meta_files

    


# --------- Testing the class above, REMOVE later ------------ #
loader = DataLoader()






