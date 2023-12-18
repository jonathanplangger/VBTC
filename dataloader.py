# This file provides the ability of registering and configuring the loading the data from rellis
import os, json, cv2
import time
import numpy as np
import random
import patchify
import albumentations as album 
import torch
import yaml


def get_dataloader( cfg, setType = "train"): 
    """_summary_

    :param cfg: Configuration file object. Contains the configuration information for the project
    :type cfg: argparse.Namespace
    :param setType: Type of action to be taken (train, test). Default setting = "train" 
    :type setType: str, optional
    :return: Dataloader object providing the necessary functions for the loading of data
    :rtype: Dataloader
    """    
    # Get the name of the database being examined. 
    db_name = cfg.DB.DB_NAME
    
    # Return the correct dataloader based on the configuration options 
    if db_name == "rellis": # Rellis-3D dataset
        # Overwrite the default configuration to use the desired one 
        cfg.DB = cfg.DB.RELLIS
        return Rellis(cfg.DB.PATH, setType=setType) # Using the old version of the Rellis-3D config (no need to fix what aint broke)
    elif db_name == "rugd":  # RUGD dataset
        cfg.DB = cfg.DB.RUGD  # overwrite
        return RUGD(cfg, setType)
    else: 
        exit("DB_NAME not properly configured, please review options and update the configuration file. ")

class DataLoader(object): 
    def __init__(self, cfg, setType = "train"): 
        self.cfg = cfg
        self.setType = setType
    
    def setup_data(self): 
        self.size = [len(self.train_meta), len(self.test_meta)] # n# of elements in the entire dataset
        self.height = int(self.train_meta[0]["height"])
        self.width = int(self.train_meta[0]["width"])

    def __reg_db(self): 
        """__reg_db() is implemented to register the desired database. This must be overwritten for each dataset to function properly
        """
        exit("No implementation for __reg_db() is provided for this dataset. Please update before continuing...")
    
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
        annMap = np.empty((batch_size, self.height, self.width, 3))

        # load image and mask files
        for i in range(batch_size): 
            orig_images[i], annMap[i] = self.load_frame(metadata[i + idx]["file_name"], metadata[i + idx]["sem_seg_file_name"])

        # update the index value 
        idx += batch_size

        # pre-process the image input if defined in the class
        if self.preprocessing:
            sample = self.preprocessing(image = orig_images, mask = annMap)
            images, annMap = sample['image'], sample['mask']
        else: # set the images to simply be the original images
            images = orig_images

        if resize != None:
            resized_img = np.empty((batch_size, resize[0], resize[1], 3)) # create a temporary container variable
            for i, img in enumerate(images): # re-size the images to the desired size 
                # re-size the image to the desired sizes
                resized_img[i] = cv2.resize(img, (resize[1], resize[0]), interpolation=cv2.INTER_LINEAR)
            images = resized_img # override the images with the newly resized images

        # Convert the numpy ndarray into useful tensor form
        images = ((torch.from_numpy(images)).to(torch.float32).permute(0,3,1,2))
        annMap = (torch.from_numpy(annMap)).to(torch.float32).permute(0,3,1,2)[:,0,:,:]

        # Re-map the annotation labels to match the other scale
        if self.remap == True: 
            annMap = self.map_labels(annMap)

        # Normalize the input image 
        if self.input_norm: 
            images = self.norm(images) 
        else: # perform a simple division to reduce the overall distribution
            images = images/255.0

        return orig_images, images, annMap, idx

    def randomizeOrder(self):
        """
            Randomizes the order of the metadata object. This will shuffle the elements in the dict in a random order
            \n This will update the object metadata parameter and is NON reversable. Only use for training.
        """    
        random.shuffle(self.train_meta) # shuffle the current order of the list
        random.shuffle(self.test_meta) # shuffle the testing set 

    def get_colors(self, remap_labels = False): 
        exit("No specific implementation provided for the get_colors function. Please update the code")
        
# ------------- Dataloader for the new dataset ------------- #
class Rellis(DataLoader):
    """
        Provides the dataloading capabilities for outsides datasets.\n
        --------------------\n
        Parameters: 
        path (str) = Directory path to the data\n
        metadata (List[dict]) = List of dictionnary elements containing information regarding image files\n
        num_classes = quantity of differentiable classes within the dataset\n
    """
    def __init__(self, path="../datasets/Rellis-3D/", setType = "train", preprocessing=None, remap = False, input_norm = False):
        """
            Dataloader provides an easy interface for loading data for training and testing of new models.\n
            ----------------------------\n
            Parameters:\n
            path (str) = File path to the dataset being loaded\n
            setType (str) = Type of operation for the dataloader, operation will vary depending on this configuration, can use either 'train" or "eval"\n
            preprocessing = preprocessing function (implement to assure compatibility with other models )\n
            remap (bool)=  selects whether the annotation files must be remapped to a new range of value (to limit the class label range)\n
            input_norm(bool) = True -> Normalization of the input will occur. 
        """
        self.path = path
        self.label_mapping = False # updated during database registration
        # only configured for the rellis dataset as of right now, would be good to add some configuration for multiple datasets
        self.train_meta, self.test_meta, self.class_labels = self.__reg_rellis() # register training and test dataset
        self.size = [len(self.train_meta), len(self.test_meta)] # n# of elements in the entire dataset
        self.height = int(self.train_meta[0]["height"])
        self.width = int(self.train_meta[0]["width"])
        self.setType = setType # sets the data type (train,test,val) loaded by the dataloader
        self.preprocessing = preprocessing # set the preprocessing function employed on the image inputs to the model
        self.remap = remap
        self.input_norm = input_norm

        # Make sure the correct number of classes are configured -> this value is employed by the modelhandler itself
        if self.remap == True: 
            self.num_classes = 19
        else: 
            self.num_classes = 35

        # Batch normalization for the input image (hence c = 3)
        self.norm = torch.nn.BatchNorm2d(3)

        # Retrieve the preprocessing function 
        if self.preprocessing: 
            self.preprocessing = self.get_preprocessing()



    # --------------------------- Database Registrations --------------------------------------#
    def __reg_rellis(self):
        """
            Provides the dataloader with the required format used in detectron2 for  the Rellis 3D Dataset.\n 
            Current iteration employs filepath tied directly to the current file structure of the VM. This may need to be updated in future versions\n
            Parameters: \n
            -----------------------------------------\n
            Returns: List[dict] - Metadata regarding each data file 
        """

        self.label_mapping = {0: 0,
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

        class_labels = {
            0: "void",
            1: "dirt",
            3: "grass",
            4: "tree",
            5: "pole",
            6: "water",
            7: "sky",
            8: "vehicle",
            9: "object",
            10: "asphalt",
            12: "building",
            15: "log",
            17: "person",
            18: "fence",
            19: "bush",
            23: "concrete",
            27: "barrier",
            31: "puddle",
            33: "mud",
            34: "rubble",
        }


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

        return train_meta, test_meta, class_labels

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
        """
            Converts the mapping labels on annotation files / prediction masks from 0->34 to 0->19 
            -----------\n
            Params: \n
            label (int tensor): labels to be converted from one scale to another \n
            inverse (bool): sets the direction that the conversion is being made, inverse = True converts 0->19 to 0->34\n
        """
        return map_labels(label, inverse)
    
    # Complete the pre-processing step for the image
    def get_preprocessing(self): 
        _transform = []
        if self.preprocessing: 
            _transform.append(album.Lambda(image = self.preprocessing))
    
        return album.Compose(_transform)
    

    def map_results(self, results):
        """
            Converts the 0->19 results into 0->34 tensor(Using the original numbering scheme) 
        """
        results = torch.cat((results, torch.zeros(35-19).cuda()))
        temp = torch.zeros(35) # empty array for assigning values 
        for k,v in self.label_mapping.items():
            temp[k] = results[v]

        return temp 

    def get_colors(self, remap_labels=False): 
        # open the ontology file for rellis and obtain the colours for them
        # ---- TODO -- This needs to be handled by the dataloader and NOT the eval.py
        with open("Rellis_3D_ontology/ontology.yaml", "r") as stream: 
            try: 
                ont = yaml.safe_load(stream)
            except yaml.YAMLError as exc: 
                print(exc)
                exit()

        # add all the colours to a list object 
        colors = []
        for i in range(35): 
            try: 
                val = tuple(ont[1][i])
                colors.append(val)
            except: # if the dict element does not exist
                colors.append((0,0,0)) # assign black colour to the unused masks

        if remap_labels: 
            temp = colors
            colors = [0]*len(self.label_mapping)
            for k,v in self.label_mapping.items(): 
                colors[v+1] = temp[k]
            colors[0] = (0,0,0)

        return colors

class RUGD(DataLoader): 

    def __init__(self, cfg, setType="train"): 
        # Base implementation
        super().__init__(cfg, setType=setType)
        # Complete the registration of the dataset -> obtain the list of images in each set
        self.train_meta, self.test_meta, self.class_labels = self.__reg_db()
        super().setup_data() # complete the same steps as the DataLoader class

    def __reg_db(self): 
        """
            Provides the dataloader with the required format used in detectron2 for  the Rellis 3D Dataset.\n 
            Current iteration employs filepath tied directly to the current file structure of the VM. This may need to be updated in future versions\n
            Parameters: \n
            -----------------------------------------\n
            Returns: List[dict] - Metadata regarding each data file 
        """
        class_labels = {
            0:"void",
            1: "dirt",
            2: "sand",
            3: "grass",
            4: "tree",
            5: "pole",
            6: "water",
            7: "sky",
            8: "vehicle",
            9: "container/generic-object",
            10: "asphalt",
            11: "gravel",
            12: "building",
            13: "mulch", 
            14: "rock-bed", 
            15: "log",
            16: "bicycle",
            17: "person",
            18: "fence",
            19: "bush",
            20: "sign", 
            21: "rock", 
            22: "bridge",
            23: "concrete",
            24: "picnic-table"
        }

        train_meta, test_meta = [], []

        train_lst = open(self.cfg.DB.PATH + "train.lst", "r")
        test_lst = open(self.cfg.DB.PATH + "test.lst", "r") 
        lsts = [train_lst, test_lst] # 
        
        for i, lst in enumerate(lsts): 

            for line in lst.readlines():
                # obtain the image file name as well as the associated segmentation mask
                [img_name, seg_name] = line.split(' ')
                seg_name = seg_name[:-1] # remove the eol character
                img_id = img_name.split("/")[-1] # get the image file name from this
                # Create the new dictionary
                meta = dict(
                    file_name = self.cfg.DB.PATH + img_name, # path for image file
                    height=self.cfg.DB.IMG_SIZE.HEIGHT,
                    width=self.cfg.DB.IMG_SIZE.WIDTH, 
                    image_id=img_id, 
                    sem_seg_file_name= self.cfg.DB.PATH + seg_name # paht for segmentation map
                )
                
                if i == 0: 
                    # add the file to the list
                    train_meta.append(meta)
                elif i == 1: 
                    test_meta.append(meta)
            

        return train_meta, test_meta, class_labels

##################################################################################################################
# Functions Available to import into other programs
##################################################################################################################

# Configured this function to be independent of the class to allow outside calls
def map_labels (label, inverse = False):
    """
        Converts the mapping labels on annotation files / prediction masks from 0->19 to 0->34 
        -----------\n
        Params: \n
        label (int tensor): labels to be converted from one scale to another \n
        inverse (bool): sets the direction that the conversion is being made, inverse = True converts 0->34 to 0->19\n
    """

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


    # Code below obtained from Rellis implementation in HRNet
    # Class 1 (Dirt) is omitted due to how sparse it is in the dataset (see Rellis-3D paper)
    
    temp = label.clone().detach() # store the old version of the label
    if inverse: # if (0->18), convert to (0->34)
        for v, k in label_mapping.items():
            label[temp == k] = v
    else: # if (0->34), convert to (0->18)
        for k, v in label_mapping.items():
            label[temp == k] = v
    return label



        
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