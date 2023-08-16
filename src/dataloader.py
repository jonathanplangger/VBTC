# This file provides the ability of registering and configuring the loading the data from rellis
import os, json, cv2
import time
import numpy as np
import random
import patchify
import albumentations as album 


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
    def __init__(self, path="../../datasets/Rellis-3D/", setType = "train", preprocessing=None):
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
        self.setType = setType # sets the data type (train,test,val) loaded by the dataloader
        self.preprocessing = preprocessing # set the preprocessing function employed on the image inputs to the model


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


    def load_batch(self, idx:int=None, batch_size:int = 1, resize = None):
        """
            Load a batch of size "batch_size". Returns the images, annotation maps, and the newly updated index value
            ----------------\n
            Parameters: \n 
            idx (int): the index of the image in the list, this function iterates this index and returns the value later \n
            batch_size (int): size of batch being retrieved by the program \n
            --------\n
            Returns: (images, annMap, idx)
            - images: list of images in the given batch 
            - annMap: list of annotation maps in the given batch
            - idx: new index of the iamge on the list.
            - resize (h,w): new size to be applied to ONLY the image. Annotation will be left alone. 
        """

        if self.setType == "train": 
            metadata = self.train_meta
        elif self.setType == "test":
            metadata = self.test_meta 
        elif self.setType == "val": 
            metadata = None # TODO - > Update this set type to work
        else: # incorrect version used
            raise Exception("Invalid set type. Please select either train/test/val.")

        #initialize the index
        if idx == None: 
            idx = 1

        # If no new size is requested, employ the original dimensions
        if resize == None: 
            resize = (self.height, self.width)
            
        orig_images = np.empty((batch_size, self.height, self.width, 3)) # stores the original images
        images = np.empty((batch_size, resize[0], resize[1], 3)) # stores the re-sized images
        annMap = np.empty((batch_size, self.height, self.width, 3))

        # load image and mask files
        for i in range(batch_size): 
            orig_images[i], annMap[i] = self.load_frame(metadata[i + idx]["file_name"], metadata[i + idx]["sem_seg_file_name"])

        if resize == None: 
            images = orig_images  # keep the images the same size
        else: 
            for i, img in enumerate(orig_images): # re-size the images to the desired size 
                # re-size the image to the desired sizes
                images[i] = cv2.resize(img, (resize[1], resize[0]), interpolation=cv2.INTER_LINEAR)

        # update the index value 
        idx += batch_size

        # pre-process the image input if defined in the class
        if self.preprocessing:
            self.preprocessing = self.get_preprocessing() # unwrap the preprocessing function 
            sample = self.preprocessing(image = images, mask = annMap)
            images, annMap = sample['image'], sample['mask']

        return orig_images, images, annMap, idx

    def getPatches(self, img: np.ndarray, patch_size, stride=[1,1], padding=[0,0]): 
        """
            Converts the image/annotation into smaller configurable patches. 
            ---------------------------------\n
            Parameters: \n
            img (numpy_ndarray): (B,H,W,C) image in question being patched \n
            patch_size: [h,w] dimensions of the patch\n
            stride: [h,w] stride size between patches \n
            padding: [h,w] padding around the image prior to patching\n
            -----------------------------------\n
            Returns: (patch)
            - patch: patches retrieved from the source image. 
        """
        # Break down the patching parameters
        ph, pw = padding
        sh, sw = stride 
        h,w = patch_size

        # get the size of the input image 
        img_size = img.shape[1:2]

        patch = patchify.patchify(img[0], (h,w,3), step=w)
        print(patch.shape)

        import matplotlib.pyplot as plt 
        # plt.imshow(img[0].astype('int'))
        # plt.show()

        fig, axs = plt.subplots(patch.shape[0], patch.shape[1])


        for row, r in enumerate(patch): 
            for col, c in enumerate(r): 
                axs[row,col].margins(2,2)
                axs[row,col].axis('off')
                axs[row,col].imshow(c[0].astype('int'))
                


        plt.show()


        pass

    def map_labels (self, label, inverse = False):
        # Code below obtained from Rellis implementation in HRNet
        # Class 1 (Dirt) is omitted due to how sparse it is in the dataset (see Rellis-3D paper)
        label_mapping = {0: 0,
                        1: 0,
                        3: 1,
                        4: 2,
                        5: 3,
                        6: 4,
                        7: 5,
                        8: 6,
                        9: 7,
                        10: 8,
                        12: 9,
                        15: 10,
                        17: 11,
                        18: 12,
                        19: 13,
                        23: 14,
                        27: 15,
                        31: 16,
                        33: 17,
                        34: 18}
        
        temp = label.clone().detach() # store the old version of the label
        if inverse: # if (0->18), convert to (0->34)
            for v, k in label_mapping.items():
                label[temp == k] = v
        else: # if (0->34), convert to (0->18)
            for k, v in label_mapping.items():
                label[temp == k] = v
        return label
    
    # Complete the pre-processing step for the image
    def get_preprocessing(self): 
        _transform = []
        if self.preprocessing: 
            _transform.append(album.Lambda(image = self.preprocessing))
    
        return album.Compose(_transform)

        
# --------- Testing the class above, REMOVE later ------------ #
if __name__ == "__main__": 
    rellis_path = "../../datasets/Rellis-3D/" #path ot the dataset directory

    from segmentation_models_pytorch.encoders import get_preprocessing_fn

    processing = get_preprocessing_fn('resnet101', 'imagenet')


    loader = DataLoader(rellis_path, preprocessing = processing)
    orig_images, images, ann, idx = loader.load_batch(0, 3)
    print(images.shape)
    print(ann.shape)

    # 240 X 240 seems like the best bet
    loader.getPatches(orig_images, patch_size=[600,480])