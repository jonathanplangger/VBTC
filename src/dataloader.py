# This file provides the ability of registering and configuring the loading the data from rellis
import os, json, cv2
import time
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
    def __init__(self, path="../../datasets/Rellis-3D/"):
        """
            Dataloader provides an easy interface for loading data for training and testing of new models.\n
            ----------------------------\n
            Parameters:\n
            path (str) = File path to the dataset being loaded\n

        """
        self.path = path
        # only configured for the rellis dataset as of right now, would be good to add some configuration for multiple datasets
        self.train_meta, self.test_meta = self.__reg_rellis() # register training and test dataset
        self.num_classes = 35 # TODO update to depend on the dataset
        self.size = [len(self.train_meta), len(self.test_meta)] # n# of elements in the entire dataset
        self.height = int(self.train_meta[0]["height"])
        self.width = int(self.train_meta[0]["width"])


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

        train_meta = []

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
            train_meta.append(meta)

        # holds metadata for the testing set 
        test_meta = []

        test_lst = open(path + "test.lst", "r")

        for line in test_lst.readlines():
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
            test_meta.append(meta)

        return train_meta, test_meta

    def randomizeOrder(self):
        """
            Randomizes the order of the metadata object. This will shuffle the elements in the dict in a random order
            \n This will update the object metadata parameter and is NON reversable. Only use for training.
        """    
        random.shuffle(self.train_meta) # shuffle the current order of the list
        random.shuffle(self.test_meta) # shuffle the testing set 

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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert the color code to use RGB
        mask = cv2.imread(mask_path)
        return img, mask


    def load_batch(self, idx:int=None, batch_size:int = 1, isTraining=True):
        """
            Load a batch of size "batch_size". Returns the images, annotation maps, and the newly updated index value
            ----------------\n
            Parameters: \n 
            idx (int): the index of the image in the list, this function iterates this index and returns the value later \n
            batch_size (int): size of batch being retrieved by the program \n
            isTraining (bool): batch returns training set if TRUE
            --------\n
            Returns: (images, annMap, idx)
            - images: list of images in the given batch 
            - annMap: list of annotation maps in the given batch
            - idx: new index of the iamge on the list.
        """

        # select which kind of data is used for the images
        if isTraining: 
            metadata = self.train_meta
        else: 
            metadata = self.test_meta


        #initialize the index
        if idx == None: 
            idx = 1

        images = np.empty((batch_size, self.height, self.width, 3))
        annMap = np.empty((batch_size, self.height, self.width, 3))

        # load image and mask files
        for i in range(batch_size): 
            images[i], annMap[i] = self.load_frame(metadata[i]["file_name"], metadata[i]["sem_seg_file_name"])

        # update the index value 
        idx += batch_size

        return images, annMap, idx









# # --------- Testing the class above, REMOVE later ------------ #
# if __name__ == "__main__": 
#     rellis_path = "../../datasets/Rellis-3D/" #path ot the dataset directory
#     loader = DataLoader(rellis_path)
#     images, ann, idx = loader.load_batch(0, 3)
#     print(images.shape)
#     print(ann.shape)