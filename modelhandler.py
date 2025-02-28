import datetime
from torch.nn import functional as TF
import torch
from dataloader import map_labels
import sys, os, argparse
import newmodel


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
        self.resize_img = False # default to false, overwritten in next function
        self.input_size = self.__input_size(mode)
        self.mode = mode # preset the mode of operation for the model

    def __input_size(self, mode): 
        """
        __input_size(self): 
        -------------------------
        Determines the size of the inputs images for the model based on configuration. 
        """
        # string that represents the desired configuration param
        modestr = "self.cfg." + mode.upper() +"."
        self.resize_img = eval(modestr+'INPUT_SIZE.RESIZE_IMG')
        if self.resize_img: # if image resizing is required
            return (eval(modestr+"INPUT_SIZE.HEIGHT"), eval(modestr+"INPUT_SIZE.WIDTH"))
        else: # if resizing is not required
            return (self.cfg.DB.IMG_SIZE.HEIGHT, self.cfg.DB.IMG_SIZE.WIDTH)

    def load_model(self): 
        """load_model(self) Loads the network model based on the specific desired file.\n
        This method is not implemented in Parent Model class and will return an error unless overwritten in child class.\n
        :return: Pre-trained torch model 
        :rtype: nn.Module & Child Variants
        """        

        print("No implementation of load_model() for this class has yet to be configured.\n")
        exit()

    def gen_model(self, num_classes): 
        """
        Generates model based on the configuration specified within the config file. 
        This method should be overwritten by the child class with the specific relevant implementations    
        """
        print("No implementation of gen_model() for this class has yet to be configured.\n")
        exit()

    def handle_output_train(self, pred):
        """Default handling for the prediction during training. This class should be overwritten by child classes. 

        :param pred: Prediction (logits) output of the network model 
        :type pred: torch.tensor
        :return: Handled output -> Convert to the desired output size
        :rtype: torch.tensor
        """        
        # If the image was re-sized, regenerate the original size
        if self.resize_img: 
            # retrieve the output size for the db.
            output_size = (self.cfg.DB.IMG_SIZE.HEIGHT, self.cfg.DB.IMG_SIZE.WIDTH)
            # interpolate to regenerate the original output size. 
            pred = TF.interpolate(input=pred, size=output_size, mode="bilinear", align_corners=False)

        # as a default, the direct prediction (logits) is used during training. Overwrite if needed
        return pred

    def handle_output_eval(self, pred): 
        # Default implementation allows for re-sizing if required 
        if self.resize_img: 
            output_size = (self.cfg.DB.IMG_SIZE.HEIGHT, self.cfg.DB.IMG_SIZE.WIDTH)
            pred  = TF.interpolate(input=pred, size = output_size, mode = "bilinear", align_corners=False)

        return pred

    def logTrainParams(self):
        """LogTrainParams() \n 
        Sets up the logging for tensorboard when training the model. State model params & values here.
        Child class should use return of this function and provide a return which provides parameter values for the specific model. 
        
        :return: String Formatted to contain training parameters specific to this model (Later logged). 
        :rtype: string
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
        --------------- <br />
        Enabled: {} <br />
        Height: {} |  Width: {} <br />
        --------------------------------------------------------------- <br />
        """.format(datetime.datetime.now(), self.cfg.TRAIN.MODEL_NAME, self.cfg.TRAIN.BATCH_SIZE, self.cfg.TRAIN.TOTAL_EPOCHS, self.cfg.TRAIN.CRITERION,
                   self.cfg.TRAIN.INPUT_SIZE.RESIZE_IMG, self.cfg.TRAIN.INPUT_SIZE.HEIGHT, self.cfg.TRAIN.INPUT_SIZE.WIDTH)
        
# ----------------------------------------------------------------------------------------------------------- #
#                                           Model Configurations                                              #
# ----------------------------------------------------------------------------------------------------------- #

class UNet(Model):
    def __init__(self, cfg, mode): 
        sys.path.insert(0, cfg.MODELS.UNET.SRC_DIR)
        super().__init__(cfg, mode) 

    def gen_model(self, num_classes):
        """
            Generates the model based on the parameters specifed within the configuration file.
        """
        import unet 

        base = self.cfg.MODELS.UNET.BASE
        kernel_size = self.cfg.MODELS.UNET.KERNEL_SIZE
        num_class = num_classes

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
        """.format(self.cfg.MODELS.UNET.BASE, self.cfg.MODELS.UNET.KERNEL_SIZE, self.cfg.TRAIN.LR)

    def load_model(self): 
        if self.mode == "train": 
            return torch.load(self.cfg.MODELS.UNET.MODEL_FILE)
        elif self.mode == "eval": 
            return torch.load(self.cfg.EVAL.MODEL_FILE)
    
    def handle_output_eval(self, pred):
        pred = super().handle_output_eval(pred)
        return pred.argmax(dim=1)
# --------------------------------------------------------------------------------------------------------------- #

           
# --------------------------------------------------------------------------------------------------------------- #    
class HRNet_OCR(Model): 
    # HRNetv2 Model

    def __init__(self, cfg, mode): 
        # run the initial configuration
        super().__init__(cfg, mode) 
        self.mode = mode

    def load_model(self): 
        if self.mode == "train": # load the desired model 
            return torch.load(self.cfg.MODELS.HRNET_OCR.MODEL_FILE)
        elif self.mode == "eval": 
            return torch.load(self.cfg.EVAL.MODEL_FILE)
        
    def handle_output_eval(self, pred):
        img_size = (self.cfg.DB.IMG_SIZE.HEIGHT, self.cfg.DB.IMG_SIZE.WIDTH)
        pred = pred[0] # hrnet has 2 outputs, whilst only one is used... 
        pred = pred.exp()
        # Use the same interpolation scheme as is used in the source code.
        pred = TF.interpolate(input=pred, size=img_size, mode='bilinear', align_corners=False)
        pred = pred.argmax(dim=1) # obtain the predictions for each layer
        # pred = map_labels(label=pred, inverse = True) # convert to 0->34
        return pred
    
    def handle_output_train(self, pred):
        # Linear upsampling re-sizing should always occur at the output (even when the full size is being used)
        # This is due to how the output of the HRNEt is 1/4 the original size of the image
        # retrieve the output size for the db.
        output_size = (self.cfg.DB.IMG_SIZE.HEIGHT, self.cfg.DB.IMG_SIZE.WIDTH)
        # interpolate to regenerate the original output size. 
        pred = TF.interpolate(input=pred[0], size=output_size, mode="bilinear", align_corners=False)
        return pred
    
    def gen_model(self, num_classes): 
        src_dir = self.cfg.MODELS.HRNET_OCR.SRC_DIR
        sys.path.insert(0, src_dir + "/lib") # add to path to allow for import 
        from hrnet_config import config, update_config # import the default config and update_config f'n

        # import the model library
        import models.hrnet_ocr.lib.models.seg_hrnet_ocr as hrnet_model

        args = argparse.Namespace(cfg = self.cfg.MODELS.HRNET_OCR.CONFIG, opts="")
        update_config(config, args)  #update the configuration file to use the right config 

        # add in our own custom overwrites for the configuration file
        config.defrost()
        config.DATASET.NUM_CLASSES = self.cfg.DB.NUM_CLASSES # update the n# of classes
        config.freeze() # freeze the configuration in place
        model = hrnet_model.get_seg_model(config)

        return model
    
    def logTrainParams(self): # ensures that the right configuration is being logged for review
        return """
        ------------------ <br />
        HRNet & OCR Parameters: <br />
        ------------------ <br />
        Configuration Model Name: {}
        ------------------ <br />
        """.format(self.cfg.MODELS.HRNET_OCR.MODEL_NAME)
# --------------------------------------------------------------------------------------------------------------- #
class GSCNN(Model):

    def load_model(self): 
        src_dir = self.cfg.MODELS.GSCNN.SRC_DIR
        # Add the network files directory to obtain the model 
        sys.path.insert(0,src_dir) # add the source dir to the path.
        sys.path.insert(0,os.path.join(src_dir, "network/"))
        
        # Prep the args to be passed to the model loader (bypass command line handling on their end)
        dataset_cls = argparse.Namespace(num_classes=self.cfg.DB.EFF_NUM_CLASSES) # update based on the configuration file
        args = argparse.Namespace(arch = "network.gscnn.GSCNN", dataset_cls = dataset_cls, trunk='resnet101',
                                    checkpoint_path = self.cfg.EVAL.MODEL_FILE,  
                                    img_wt_loss=False, joint_edgeseg_loss=False, wt_bound=1.0, edge_weight=1.0, 
                                    seg_weight=1.0)
        import network
        from loss import get_loss # their model loading requires the criterion, only use the base one (configured using args)
        model = network.get_net(args, get_loss(args))
        return model

    def handle_output_eval(self, pred):
        img_size = (self.cfg.DB.IMG_SIZE.HEIGHT, self.cfg.DB.IMG_SIZE.WIDTH)
        # GSCNN returns two outputs: the shape & regular stream. Only the regular segmentation is required
        pred, _ = pred  # get the shape stream 
        pred =  pred.data # get the data for the segmentation 
        pred = TF.interpolate(input=pred, size=img_size, mode='bilinear', align_corners=False)
        pred = pred.argmax(dim=1) # convert into label mask 
        pred = map_labels(label=pred, inverse=True) # convert the labels to 0->34 scheme
        return pred
    
    def handle_output_train(self, pred):
        return super().handle_output_train(pred)
    

    def gen_model(self, num_classes):
        src_dir = self.cfg.MODELS.MODELS_DIR # get the source file directory
        # sys.path.append(os.path.abspath(src_dir))
        sys.path.append(os.path.abspath(src_dir + "/gscnn")) # add the path to the gscnn directory 

        import network

        # TODO, update to select LF based on the training setting rather than default value
        criterion = torch.nn.CrossEntropyLoss() 

        # Create the base configuration variables. 
        dataset_cls = argparse.Namespace(num_classes=self.cfg.DB.EFF_NUM_CLASSES)

        # Set up the default arguments for the arguments
        args = argparse.Namespace(
            arch = "network.gscnn.GSCNN", 
            dataset_cls = dataset_cls, 
            trunk = 'resnet50',
            checkpoint_path = False, # no checkpoint is being used when training the model.
            img_wt_loss=False, joint_edgeseg_loss=False, wt_bound=1.0, edge_weight=1.0, seg_weight=1.0,
        ) 

        model = network.get_net(args, criterion)
        
        return model
    

# --------------------------------------------------------------------------------------------------------------- #    
class DeepLabV3Plus(Model): 
    """DeepLabV3Plus: Model implementation for the model handler that implements the configured version of the deeplabv3plus model.\n
    Default configurations can be updated within the project configuration file.
    """
    def __init__(self, cfg, mode): 
        # run the initial configuration
        super().__init__(cfg, mode) 
        self.mode = mode

        # Import the required libraries for the application of the model.
        src_dir = self.cfg.MODELS.DEEPLABV3PLUS.SRC_DIR
        sys.path.insert(0, src_dir) 
        

    def gen_model(self, num_classes):
        """Constructs the model using the methods provided in the source code. \n

        :param num_classes: Number of classes in the dataset -> Defines the parameters of the model constructed
        :type num_classes: int
        :return: Model file for the DeeplabV3+
        :rtype: network._deeplab.DeepLabV3
        """
        import network as net 
        bbn = self.cfg.MODELS.DEEPLABV3PLUS.BACKBONE # retrieve the backbone used from config file 
        model = net.modeling.__dict__["deeplabv3plus_" + bbn](num_classes = self.cfg.DB.EFF_NUM_CLASSES, output_stride = 16, pretrained_backbone = False)
        if self.cfg.MODELS.DEEPLABV3PLUS.SEPARABLE_CONV: 
            net.convert_to_separable_conv(model.classifier)

        return model
        
    def logTrainParams(self):
        return super().logTrainParams() + """
        DeepLabV3+ Parameters: <br />
        ---------------------- <br />
        Backbone Structure: {} <br />
        """.format(self.cfg.MODELS.DEEPLABV3PLUS.BACKBONE) # Add more as necessary. 
    
    def load_model(self):
        if self.mode == "train": # load the desired model 
            return torch.load(self.cfg.MODELS.DEEPLABV3PLUS.MODEL_FILE)
        elif self.mode == "eval": 
            return torch.load(self.cfg.EVAL.MODEL_FILE)


    def handle_output_eval(self, pred):
        # Complete the default resizing
        pred = super().handle_output_eval(pred)
        return torch.argmax(pred,dim=1) # get the argmax representation for the output 


class NewModel(Model): 
    def gen_model(self, num_classes): 
        return newmodel.NewModel()
    
    def handle_output_train(self, pred):
        return pred

    def load_model(self):
        return torch.load(self.cfg.EVAL.MODEL_FILE)

    def handle_output_eval(self, pred):
        pred = super().handle_output_eval(pred)
        return pred.argmax(dim=1)

# -------------------------------------------------------------------------------------------------------------------------- #
#                                                  Model Handler
# -------------------------------------------------------------------------------------------------------------------------- #

class ModelHandler(object): 
    """
    Class ModelHandler:\n
    -----------------------------------\n
    Handles the various operations required for the 'Model' classes.\n
    Serves as a front API for selecting the correct model and performing the required functions.\n
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
            return HRNet_OCR(self.cfg, self.mode)
        elif model_name == "gscnn": 
            return GSCNN(self.cfg, self.mode)
        elif model_name == "newmodel": 
            return NewModel(self.cfg, self.mode) # TODO - Add configuration through config file instead of preset
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
    
    def gen_model(self, num_classes): 
        """
        gen_model(self)
        ----------------
        Generates the model, ready for training, based on the configuration file. \n 
        This should only be used during training as the model retrieved will not contain weights trained upon Rellis-3D dataset
        """
        return self.model.gen_model(num_classes)
    
    def handle_output(self, pred): 
        """
        Handles the output based on the train/eval mode.\n
        --------------------------
        If mode == train, the output will be logits. 
        If mode == eval, the output will be an integer mask which represents each class
        """
        if self.mode == "train": 
            return self.model.handle_output_train(pred)
        elif self.mode == "eval": 
            return self.model.handle_output_eval(pred)

# -------------- Testing the code implementation --------------------- #
if __name__ == "__main__": 
    
    print("Test Main")