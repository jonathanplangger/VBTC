# This file provides the ability of registering and configuring the loading the data from rellis
import os, json, cv2
import time
import torch
import numpy as np

# directory path for the entire sequence


# ------------- Dataloader for the new dataset ------------- #
class DataLoader(object):

    def __init__(self, path=""):
        """
            Dataloader provides an easy interface for loading data for training and testing of new models.\n
            ----------------------------\n
            Parameters:\n
            path (str) = File path to the dataset being loaded\n

        """
        self.path = path
        # only configured for the rellis dataset as of right now, would be good to add some configuration for multiple datasets
        data = self.__reg_rellis()
        # store the obtained metadata as a parameter
        self.metadata = data
        self.num_classes = 20 # TODO update to depend on the dataset


    # --------------------------- Database Registrations --------------------------------------#
    def __reg_rellis(self):
        """
            Provides the dataloader with the required format used in detectron2 for  the Rellis 3D Dataset.\n 
            Current iteration employs filepath tied directly to the current file structure of the VM. This may need to be updated in future versions\n
            Parameters: \n
            -----------------------------------------\n
            Returns: List[dict] - Metadata regarding each data file 
        """
        path = self.path

        meta_files = []

        train_lst = open(path + "train.lst", "r")
        
        for line in train_lst.readlines():
            # obtain the image file name as well as the associated segmentation mask
            [img_name, seg_name] = line.split(' ')
            seg_name = seg_name[:-1] # remove the eol character
            img_id = img_name.split("frame")[1][0:6]
            # Create the new dictionary
            meta = dict(
                file_name = self.path + img_name, # path for image file
                height="1200",
                width="1920", 
                image_id=img_id, 
                sem_seg_file_name= self.path + seg_name # paht for segmentation map
            )
            # add the file to the list
            meta_files.append(meta)

        return meta_files

    


# # --------- Testing the class above, REMOVE later ------------ #
# rellis_path = "../../datasets/Rellis-3D/" #path ot the dataset directory

# loader = DataLoader(rellis_path)
 

# # print(loader.metadata[0]["file_name"])





