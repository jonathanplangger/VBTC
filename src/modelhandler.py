import datetime
import unet


class Model(object): 
    """
    Parent class used to hold the base code implementation for the models.\n
    Create a child class and overwrite this class for every model being implemented in the study.\n
    -----------------------------------\n
    Init Params: 
    cfg (CfgNode): Config file containing all of the configuration information for the project
    mode (str): selects the mode being employed for the model. |  Options are: "eval", "train". 
    """
    def __init__(self, cfg, mode):
        self.cfg = cfg # configuration file
        self.input_size = self.__input_size(mode)

    def __input_size(self, mode): 
        """
        __input_size(self): 
        -------------------------
        Determines the size of the inputs images for the model based on configuration. 
        """
        # string that represents the desired configuration param
        modestr = "self.cfg." + mode.upper() +"."
        if eval(modestr+'INPUT_SIZE.RESIZE_IMG'): # if image resizing is required
            return (eval(modestr+"INPUT_SIZE.HEIGHT"), eval(modestr+"INPUT_SIZE.WIDTH"))
        else: # if resizing is not required
            return (self.cfg.DB.IMG_SIZE.HEIGHT, self.cfg.DB.IMG_SIZE.WIDTH)

    def load_model(self): 
        """
        load_model()
        ------------------
        For test cases only. Loads the trained model ready for implementation. 
        """
        print("No implementation of load_model() for this class has yet to be configured.\n")
        exit()

    def gen_model(self): 
        """
        Generates model based on the configuration specified within the config file. 
        This method should be overwritten by the child class with the specific relevant implementations    
        """
        print("No implementation of gen_model() for this class has yet to be configured.\n")
        exit()

    def handle_output(self):
        print("No implementation of handle_output() for this class has yet to be configured.\n")
        exit()

    def logTrainParams(self):
        """
        Sets up the logging for tensorboard when training the model. State model params & values here.
        Child class should use return of this function and provide a return which provides parameter values for the specific model. 
        """ 
        return """
        --------------------------------------------------------------- <br />
        Model Training Parameters <br />
        --------------------------------------------------------------- <br />
        Date: {} <br />
        --------------------------------------------------------------- <br />
        Training Params <br />
        ---------------- <br />
        Model Name: {} <br />
        Batch Size: {} <br />
        Epochs: {} <br />
        Loss Function: {} <br />
        --------------------------------------------------------------- <br />
        Image Resizing <br />
        ---------------
        Enabled: {} <br />
        Height: {} |  Width: {} <br />
        --------------------------------------------------------------- <br />
        """.format(datetime.datetime.now(), self.cfg.TRAIN.MODEL_NAME, self.cfg.TRAIN.BATCH_SIZE, self.cfg.TRAIN.TOTAL_EPOCHS, self.cfg.TRAIN.CRITERION,
                   self.cfg.TRAIN.INPUT_SIZE.RESIZE_IMG, self.cfg.TRAIN.INPUT_SIZE.HEIGHT, self.cfg.TRAIN.INPUT_SIZE.WIDTH)
        

class UNet(Model): 

    def gen_model(self):
        """
            Generates the model based on the parameters specifed within the configuration file.
        """
        base = self.cfg.MODELS.UNET.BASE
        kernel_size = self.cfg.MODELS.UNET.KERNEL_SIZE
        num_class = self.cfg.DB.NUM_CLASSES

        return unet.UNet(
            enc_chs=(3,base, base*2, base*4, base*8, base*16),
            dec_chs=(base*16, base*8, base*4, base*2, base), 
            out_sz=self.input_size, retain_dim=True, num_class=num_class, kernel_size=kernel_size
        )   

    def logTrainParams(self):
        # Add onto the tensorboard string the model parameters
        return super().logTrainParams() + """
        U-Net Parameters: <br />
        ------------------ <br />
        Base Value: {} <br />
        Kernel Size: {} <br />
        Learning rate: {} <br />
        -------------------------------------------------------------- <br />
        """.format(self.cfg.MODELS.UNET.BASE, self.cfg.MODELS.UNET.KERNEL_SIZE, self.cfg.MODELS.UNET.LR)

class DeepLabV3Plus(Model): 

    def gen_model(self): 
        import segmentation_models_pytorch as smp # get the library for the model
        return smp.DeepLabV3Plus(
            encoder_name=self.cfg.MODELS.DEEPLABV3PLUS.ENCODER, 
            encoder_weights=self.cfg.MODELS.DEEPLABV3PLUS.ENCODER_WEIGHTS, 
            classes = self.cfg.DB.NUM_CLASSES, 
            activation = "sigmoid"
        )        


class ModelHandler(object): 
    """
    Class ModelHandler:\n
    -----------------------------------\n
    Handles the various operations required for the 'Model' classes.\n
    Serves as a front API for selecting the correct model and performing the required functions.
    ------------------------------------\n
    Params:\n
    cfg (CfgNode): Configuration information from the main programs
    mode (str): Mode of operation for the ModelHandler (Ex: 'eval' & 'train')
    """
    def __init__(self, cfg, mode): 
        self.cfg = cfg
        self.mode = mode
        self.model = self.__register_model()


    def __register_model(self):
        """
            Sets up the correct selection for the model based on configuration and mode selection
        """
        # Retrieve which model is employed through the configuration file. 
        if self.mode == "train": 
            modestr = "self.cfg.TRAIN." # useful later 
            model_name = eval(modestr + "MODEL_NAME")
        elif self.mode == "eval": 
            modestr = "self.cfg.EVAL." # useful later 
            model_name = eval(modestr + "MODEL_NAME")
        else: 
            exit("Invalid mode selected. Please use a valid one (Ex: 'eval' & 'train')")

        # Based on the model name, select obtain the respective model output
        if model_name == "unet": 
            return UNet(self.cfg, self.mode)
        elif model_name == "deeplabv3plus": 
            return DeepLabV3Plus(self.cfg, self.mode)
        elif model_name == "hrnet_ocr": 
            pass
        elif model_name == "gscnn": 
            pass
        else: 
            exit("Invalid model_name specified. Please configure a valid model_name in config file.")
        ######################################################
        # Register more models by adding to the elif above
        ######################################################

    def logTrainParams(self):
        """
            Log the training parameters for the given model employed. This will directly employ the parameters as defined in the configuration file.
        """ 
        return self.model.logTrainParams()

    def load_model(self): 
        """
            Returns the torch.nn model object for the specified model. 
        """
        return self.model.load_model()
    
    def gen_model(self): 
        """
        gen_model(self)
        ----------------
        Generates the model, ready for training, based on the configuration file. \n 
        This should only be used during training as the model retrieved will not contain weights trained upon Rellis-3D dataset
        """
        return self.model.gen_model()

# -------------- Testing the code implementation --------------------- #
if __name__ == "__main__": 
    
    print("Test Main")