# This file is employed to train a new model using the provided dataset 
import os, json, cv2
import time
import torch, detectron2
import numpy as np

from detectron2.data import DatasetCatalog

# directory path for the entire sequence
seq_dir = "../../datasets/Rellis-3D/"

# --------------------------- Database Registrations --------------------------------------#
def reg_rellis(path: str):
    """
        Provides the dataloader with the required format used in detectron2 for the Rellis 3D Dataset.\n 
        Current iteration employs filepath tied directly to the current file structure of the VM. This may need to be updated in future versions\n
        Parameters: \n
        path (str) = Path to the sequence directory (00000, 00001, etc) which contains the images and segmentations
        -----------------------------------------\n
        Returns: List[dict]
    """
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

    
# ------------------------------------------ #
# Main Program Execution


# test out the current dictionary created...
print(reg_rellis(seq_dir), sep="\n\n\n")





