# This file provides the ability of registering and configuring the loading the data from rellis
import os, json, cv2
import time
import numpy as np
import random
import patchify
import albumentations as album 
import torch
import yaml

"""@package docstring
Documentation for the dataloader module

More Details.
"""

def get_dataloader(cfg, setType = "train"): 
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
        return Rellis(cfg.DB.PATH, setType=setType, cfg = cfg) # Using the old version of the Rellis-3D config (no need to fix what aint broke)
    elif db_name == "rugd":  # RUGD dataset
        cfg.DB = cfg.DB.RUGD  # overwrite
        return RUGD(cfg, setType)
    else: 
        exit("DB_NAME not properly configured, please review options and update the configuration file. ")

class DataLoader(object): 
    """!
    <p>
        Main parent class providing an inheritable structure for the specialized dataloader implementations.
        All datasets featured within this development environment should employ the Dataloader child class to implement specific loading functions.
        This class handles all functions concerning the dataset such as data fetching, pre-preperation, sorting, and run-time loading operations.
    </p>
    """

    def __init__(self, cfg, setType = "train"): 
        ##Configuration file for the dataloader. 
        self.cfg = cfg 
        ##Represents the application area for this dataloader (ie "train" or "eval"). 
        self.setType = setType 
        ## Toggle preprocessing of the input image, Default: False
        self.preprocessing = False
        ##True Forces a remapping scheme for the index values 
        self.remap = False
        ##True toggles normalization of the input RGB image. (Default: False)
        self.input_norm = False
        ##Number of classes held within the dataset (Default: 0)
        self.num_classes = 0 
        ## Dictionary representing class name and its index ({index: class_name})
        self.class_labels = {} 
    
    ##Configures the metadata concerning image properties (height, width) for the respective dataset
    def setup_data(self): 
        ##Number of elements within the entire dataset: [N(training images), N(testing images)]; N = Number of elements
        self.size = [len(self.train_meta), len(self.test_meta)] # n# of elements in the entire dataset
        ##Height of all images, Note that a *static* dimension is assumed for both height and width of the input images
        self.height = int(self.train_meta[0]["height"])
        ##Width of all images, Note that a *static* dimension is assumed for both height and width of the input images
        self.width = int(self.train_meta[0]["width"])

    def __reg_db(self): 
        """!
        __reg_db() is utilized by all dataloaders to properly configure any dataset-specific information for the dataloader.
        The exact implementation is specific to the dataset implemented as they are inherently different in format from one another. 
        This method is overwritten within each child class (RUGD, Rellis, etc.). 

        Parameters set:
        -----------------
        @param self.num_classes

        Return values:
        ----------------
        @return train_meta: dictionary of all training images 
        @return test_meta: dictionary of all testing images
        @return val_meta: dictionary of all validation images
        @return class_labels: dictionary of all class labels {index: class}

        """
        exit("No implementation for __reg_db() is provided for this dataset. Please update before continuing...")
    
    def load_frame(self, img_path, ann_path): 
        """!
        Reads and returns the selected image alongside its segmentation mask. 

        @param img_path (str) = URL Path to the image 
        @param ann_path (str) = URL Path to the mask annotation file 
        """
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert the color code to use RGB
        mask = cv2.imread(ann_path)
        return img, mask[:,:,0] # keep only the first dimension (since the other 3 are the same thing) 

    def load_batch(self, idx:int = None, batch_size:int = 1, resize = None):
        """!
            Loads an image batch of size "batch_size" returning the images, annotation maps, and the new index value. 
            An index value is provided to allow for an iterable process where the initial index value provided is incremented by the batch_size loaded
            by the *load_batch()* function. An additional optional re-sizing of the image is provided by the function to enable a reduction in input size
            dimension. 

            @param idx (int): the index of the image in the list, this function iterates this index and returns the value later. Default: None \n
            @param batch_size (int): size of batch being retrieved by the program, Default: 1 \n
            @param resize (tuple or None): (height,width) tuple dimension to re-size the input image to. Default: None

            @return orig_images : list of the original (un-modified) images from the dataset. Same as *images* if resize = *None* or resize = (H_in, W_in)
            @return images : list of images in the given batch 
            @return annMap : list of annotation maps in the given batch
            @return idx : new index of the image on the list.
        """

        if self.setType == "train": 
            metadata = self.train_meta
        elif self.setType == "test":
            metadata = self.test_meta 
        elif self.setType == "val": 
            metadata = self.val_meta # TODO - > Update this set type to work
        else: # incorrect version used
            raise Exception("Invalid set type. Please select either train/test/val.")

        #initialize the index
        if idx == None: 
            idx = 1

        # If no new size is requested, employ the original dimensions
        if resize == None: 
            resize = (self.height, self.width)
            
        orig_images = np.empty((batch_size, self.height, self.width, 3)) # stores the original images
        annMap = np.empty((batch_size, self.height, self.width))

        # # Used to select a specifically desired file_name. Comment out when not required
        # metadata[idx]["file_name"] = '../datasets/Rellis-3D/00000/pylon_camera_node/frame000110-1581624663_749.jpg'
        # metadata[idx]["sem_seg_file_name"] = '../datasets/Rellis-3D/00000/pylon_camera_node_label_id/frame000110-1581624663_749.png'

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
        annMap = (torch.from_numpy(annMap)).to(torch.float32).permute(0,1,2)

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
        """! Retrieve the colours/class mapping for the given dataset. Since each class is represented by a given colour, each colour will be unique to the class specified 
        Since the color mapping is specific to a given dataset, an individual implementation of the function is provided for each implemented dataset. For datasets like Rellis-3D, 
        an optional remapping is provided to ensure that the label indexes match the desired value ranges. 
                
        @param remap_labels (bool): True selects for class index remapping to occur within the color map
        @return color_map (dict): dictionary containing the colour (value) in RGB for each class (key) of the dataset. 

        """
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
    def __init__(self, path="../datasets/Rellis-3D/", setType = "train", preprocessing=None, remap = True, input_norm = False, cfg = None):
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
        self.cfg = cfg # save the configuration inputs (useful for later steps)
        self.path = path
        self.label_mapping = False # updated during database registration
        # only configured for the rellis dataset as of right now, would be good to add some configuration for multiple datasets
        self.train_meta, self.test_meta, self.val_meta, self.class_labels = self.__reg_rellis() # register training and test dataset
        self.size = [len(self.train_meta), len(self.test_meta), len(self.val_meta)] # n# of elements in the entire dataset
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

        self.remap_class_labels = {
            0: "void & dirt",
            1: "grass",
            2: "tree",
            3: "pole",
            4: "water",
            5: "sky",
            6: "vehicle",
            7: "object",
            8: "asphalt",
            9: "building",
            10: "log",
            11: "person",
            12: "fence",
            13: "bush",
            14: "concrete",
            15: "barrier",
            16: "puddle",
            17: "mud",
            18: "rubble",
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

        val_meta = []
        # get the list file 
        val_lst = open(path + "val.lst", 'r')
        for line in val_lst.readlines(): 
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
            val_meta.append(meta)


        return train_meta, test_meta, val_meta, class_labels

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
        """get_colors()

        :param remap_labels: Configure whether the label remapping is being used. Defaults to False
        :type remap_labels: bool, optional
        :return: _description_
        :rtype: _type_
        """        
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

        return colors[1:]

class RUGD(DataLoader): 
    """!
    The RUGD implementation of the dataloader provides a specific version of the dataloader suited to loading RUGD image assets.
    Prior to using this class, make sure that the file path and database directory is properly configured in development environment configuration files. 
    """
    def __init__(self, cfg, setType="train"): 
        # Base implementation
        super().__init__(cfg, setType=setType)
        # Complete the registration of the dataset -> obtain the list of images in each set
        self.train_meta, self.test_meta, self.val_meta,self.class_labels = self.__reg_db()
        super().setup_data() # complete the same steps as the DataLoader class

    def __reg_db(self): 
        """!
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
            # 9: "container/generic-object", # reduced size to make fitting to figure a lot easier
            9: "object",
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

        self.num_classes = len(class_labels)

        """ __get_split_config(self):
        Returns the split configuration data (from train.lst, test.lst and val.lst) for the RUGD dataset. Provides sorting information to split the dataset in the 3-way split.\n
        Validates that data config exists and data is valid.
        """
        train_meta, test_meta, val_meta = [], [], []

        # Convert and re-map the annotations to the new 0->Nclasses-1 map from the RGB colouring provided
        # in the dataset
        if not os.path.exists(self.cfg.DB.PATH + self.cfg.DB.MODIF_ANN_DIR):
            print("Re-mapped annotation files not present. Remapping is required.")
            self.__gen_modif_ann_dir() 

        try:
            train_lst = open(self.cfg.DB.PATH + "train.lst", "r")
            test_lst = open(self.cfg.DB.PATH + "test.lst", "r")
            val_lst = open(self.cfg.DB.PATH + "val.lst", 'r')  
            lsts = [train_lst, test_lst, val_lst]
        except FileNotFoundError: # Generate lst files if they don't exist 
            print("No Configuration Files Present\nGenerating a new set of split configuration files...\n")
            lsts = self.__gen_split_config_lsts() 


        for i, lst in enumerate(lsts): 

            for line in lst.readlines():
                # obtain the image file name as well as the associated segmentation mask
                [img_name, seg_name] = line.split(' ')
                seg_name = seg_name[:-1] # remove the eol character
                # img_id = img_name.split("/")[-1] # get the image file name from this
                # Create the new dictionary
                meta = dict(
                    file_name = img_name, # path for image file
                    height=self.cfg.DB.IMG_SIZE.HEIGHT,
                    width=self.cfg.DB.IMG_SIZE.WIDTH, 
                    image_id= img_name, 
                    sem_seg_file_name= seg_name # paht for segmentation map
                )
                
                if i == 0: 
                    # add the file to the list
                    train_meta.append(meta)
                elif i == 1: 
                    test_meta.append(meta)
                elif i == 2: 
                    val_meta.append(meta)
            

        return train_meta, test_meta, val_meta, class_labels
    
    def __gen_modif_ann_dir(self):
        """! Generate a modified annotation directory and re-map the images into the desired format. 
        This function handles the conversion process of all annotation images to use the (0->N_classes -1) numbering scheme rather than the colours present in the original dataset. 
        The function should only be really during first-time initialization of the dataset and currently only triggers if no directory is present for the newly re-mapped output images. 
        Currently only supports a CPU-based implementation of the conversion, so the time required to process the entire dataset is considerable. 
        *Note* that the current implementation employs a GPU-based image conversion to greatly reduce conversion time. 
        """
        import torch # requires torch to speed up the conversion process

        print("The re-mapping process is beginning. This will take a while to complete.")
        modif_ann_path = "{}{}".format(self.cfg.DB.PATH, self.cfg.DB.MODIF_ANN_DIR)
        ann_path = "{}{}".format(self.cfg.DB.PATH, self.cfg.DB.ANN_DIR)
        os.mkdir(modif_ann_path) # TODO -> make sure to turn this back ON!!!!

        seqs = os.listdir(ann_path) # get the sequences in the dataset
        seqs.remove(self.cfg.DB.COLOR_MAP) # only use the sequences, not the color map config file 

        # Get the color/class mapping based on the dataset provided configuration map
        color_map = {}
        with open("{}{}/{}".format(self.cfg.DB.PATH, self.cfg.DB.ANN_DIR, self.cfg.DB.COLOR_MAP), 'r') as file:
            color_config = file.readlines()

            for c in color_config:
                c = c.strip() # remove eol characters such as \n
                c = c.split(' ') 
                color_map[(int(c[-3]), int(c[-2]), int(c[-1]))] = int(c[0])
            
            file.close()

        # Go through all the images in the sequences and re-map them using the retrieved color_map value 
        for seq in seqs: 
            os.mkdir("{}/{}".format(modif_ann_path, seq)) # Create the new sequence directory (preserve old structure) 
            
            # Create the new images with the re-mapped colours
            anns = os.scandir("{}/{}".format(ann_path, seq))
            for ann in anns: 
                if ann.name.split(".")[-1] == "png": # only applies to the png files 
                    # Obtain and convert the image color to use RGB values (same as used in color_map)
                    img = cv2.cvtColor(cv2.imread(ann.path), cv2.COLOR_BGR2RGB)
                    img_shape = img.shape[:-1] # store shape to re-use, only one channel is required for the image  
                    img = torch.tensor(img)

                    # Will be using the GPU if available on the host machine, CPU used if not available
                    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
                    img.to(device)
                    
                    # Easier to handle the data mapping when flattening the image prior to mapping 
                    img = img.flatten(end_dim = 1)
                    img = [color_map[tuple(x.tolist())] for x in img] # re-maps the tensor using the new mapping scheme
                    img = torch.tensor(img).reshape(img_shape).numpy() # convert and re-shape back to the original dimensions

                    cv2.imwrite("{}/{}/{}".format(modif_ann_path, seq, ann.name), img)

        print("Image Re-mapping is Complete and the new files are saved within the 'RUGD_modif_frames-with-annotations' directory")



    def __gen_split_config_lsts(self): 
        """!Generates the train/test/val split configuration lists which outline which sequences are present within each split. 
        *Assumes that the RUGD dataset has been un-modified and obtained from http://rugd.vision/.*
    
        The split configuration divides the dataset based on sequence in the manner outlined in the original RUGD paper publication. 
        The generation of the split lists will occur if they are not present within the dataset directory (such as a newly downloaded dataset.). 
        """

        # Split sets based on original split configuration in source RUGD paper. 
        train_split = ["park-2", "trail", "trail-3", "trail-4", "trail-6", "trail-9", "trail-10", "trail-11", "trail-12", "trail-14", "trail-15", "village"]
        val_split = ["park-8", "trail-5"]
        test_split = ["creek", "park-1", "trail-7", "trail-13"]
        splits = [train_split, test_split, val_split]
        fnames = ["train.lst", "test.lst", "val.lst"] # Output file names

        for i, _ in enumerate(splits): 
            try:
                with open("{}/{}".format(self.cfg.DB.PATH,fnames[i]), "a") as file: # Create new lst files in dataset location
                    for seq in splits[i]: # sequences in splits
                        # Use only the image directory for the image ID. Check that the annotation also exists while doing this
                        img_dir = "{}{}/{}/".format(self.cfg.DB.PATH, self.cfg.DB.IMG_DIR, seq)
                        ann_dir = "{}{}/{}/".format(self.cfg.DB.PATH, self.cfg.DB.MODIF_ANN_DIR, seq)
                        iter = os.scandir(img_dir)
                        for img in iter: 
                            if img.name.endswith(".png"): # solely treat images
                                img_path = img.path 
                                ann_path = ann_dir + img.name # use the same id as the image, ensure duplicates
                                
                                if os.path.exists(ann_path): # only perform this when a corresponding annotation file exists
                                    file.write("{} {}\n".format(img_path, ann_path)) 
            except FileNotFoundError:
                exit("RUGD dataset is missing or invalid path to the dataset is provided. Make sure to update prior to restarting the program")
            file.close() 
    
        # Return all the values on the files themselves
        train_lst = open(self.cfg.DB.PATH + "train.lst", "r")
        test_lst = open(self.cfg.DB.PATH + "test.lst", "r")
        val_lst = open(self.cfg.DB.PATH + "val.lst", "r")


        return [train_lst, test_lst, val_lst]



    def get_colors(self, remap_labels = False):  
        """!Get the colour mapping for each classes within the RUGD dataset.

        \see DataLoader for more information
        """            
        # open the colour map file for RUGD
        with open("{}/RUGD_annotations/RUGD_annotation-colormap.txt".format(self.cfg.DB.PATH)) as f: 
            classes = f.readlines()

            colors = []
            for i, c in enumerate(classes): 
                classes[i] = c.replace("\n", "") # remove the EOL characters
                colors.append(tuple([int(x) for x in classes[i].split()[-3:]])) # convert all the values in the list to int
                
        # Return the color mapping for the dataset
        return colors
    
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