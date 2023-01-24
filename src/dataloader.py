# This file provides the ability of registering and configuring the loading the data from rellis
import os, json, cv2
import time
import tensorflow as tf 
import numpy as np
import random

# ------------- Dataloader for the new dataset ------------- #
class DataLoader(object):
    """
        Provides the dataloading capabilities for outsides datasets.\n
        --------------------\n
        Parameters: 
        path (str) = Directory path to the data\n
        metadata (List[dict]) = List of dictionnary elements containing information regarding image files\n
        num_classes = quantity of differentiable classes within the dataset\n
    """
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
        self.size = len(self.metadata) # n# of elements in the entire dataset
        self.height = int(self.metadata[0]["height"])
        self.width = int(self.metadata[0]["width"])


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

    def randomizeOrder(self):
        """
            Randomizes the order of the metadata object. This will shuffle the elements in the dict in a random order
            \n This will update the object metadata parameter and is NON reversable. Only use for training.
        """    
        random.shuffle(self.metadata) # shuffle the current order of the list

    def load_frame(self, img_path, mask_path): 
        """
            Loads the given image alongside its segmentation mask. \n
            ---------------------------\n
            parameters:\n
            img_path(str) = Path to the image 
            mask_path(str) = Path to the mask file 
            transform = transformation function to be applied to the incoming data
        """
        img = cv2.imread(img_path)
        print(img)
        mask = cv2.imread(mask_path)
        return img, mask


    def load_batch(self, idx, batch_size:int):
        """
            Load a batch of size "batch_size". Returns the images, annotation maps, and the newly updated index value
        """

        images = np.empty((batch_size, self.height, self.width, 3))
        annMap = np.empty((batch_size, self.height, self.width, 3))

        # load image and mask files
        for i in range(batch_size): 
            images[i], annMap[i] = self.load_frame(self.metadata[i]["file_name"], self.metadata[i]["sem_seg_file_name"])

        # update the index value 
        idx += batch_size

        return images, annMap, idx









# # --------- Testing the class above, REMOVE later ------------ #
# rellis_path = "../../datasets/Rellis-3D/" #path ot the dataset directory

# loader = DataLoader(rellis_path)
# print(loader.metadata[0]["file_name"])
# images, annMap, idx = loader.load_batch(0, 3)

