from yacs.config import CfgNode as CN


_C = CN()

# ----------------- Database Handling Parameters ------------------------------------#
# These params (other than path) should ideally not be altered since they directly represent the Rellis-3D dataset itself.
_C.DB = CN()
_C.DB.DB_NAME = "rellis" 
_C.DB.PATH = ""
_C.DB.NUM_CLASSES = None
_C.DB.EFF_NUM_CLASSES = None # effective n# of classes, some of the 35 classes are not used at all 
_C.DB.IMG_SIZE = CN()
_C.DB.IMG_SIZE.HEIGHT = None 
_C.DB.IMG_SIZE.WIDTH = None

# --------- Rellis-3D (rellis) --------- #
_C.DB.RELLIS = CN()
_C.DB.RELLIS.DB_NAME = 'rellis'
_C.DB.RELLIS.PATH = "../datasets/Rellis-3D/"
_C.DB.RELLIS.NUM_CLASSES = 35
_C.DB.RELLIS.EFF_NUM_CLASSES = 19 # effective n# of classes, some of the 35 classes are not used at all 
_C.DB.RELLIS.IMG_SIZE = CN()
_C.DB.RELLIS.IMG_SIZE.HEIGHT = 1200 
_C.DB.RELLIS.IMG_SIZE.WIDTH = 1920

# -------- RUGD Dataset (rellis) ---------- #
_C.DB.RUGD = CN()
_C.DB.RUGD.DB_NAME = 'rugd'
_C.DB.RUGD.PATH = "../datasets/RUGD/"
_C.DB.RUGD.NUM_CLASSES = 24 # ? How many are there? 
_C.DB.RUGD.EFF_NUM_CLASSES = 24 # effective n# of classes, some of the 35 classes are not used at all 
_C.DB.RUGD.IMG_SIZE = CN()
_C.DB.RUGD.IMG_SIZE.HEIGHT = 550 
_C.DB.RUGD.IMG_SIZE.WIDTH = 688

# ----------------- Evaluation Parameter Configuration  ---------------------------- #
"""
  hrnet_ocr
  unet
"""
_C.EVAL = CN()
_C.EVAL.MODEL_NAME = "" # Name of the model being used. Options displayed above.
_C.EVAL.MODEL_FILE = "" # File path for the model file. Specify this prior to training
_C.EVAL.DISPLAY_IMAGE = False # Toggle the display of each prediction alongside the annotations 
_C.EVAL.PRED_CERTAINTY = False # display the prediction certainty for each class for the given input image
_C.EVAL.BATCH_SIZE = 1 # Image batch size input to the model
# Re-size the input image to model the set size. Annotation and output image size remains the same as the dataset
_C.EVAL.INPUT_SIZE = CN()
_C.EVAL.INPUT_SIZE.RESIZE_IMG = False # toggle if the input image is re-sized, TRUE will resize the image
_C.EVAL.INPUT_SIZE.HEIGHT = None # new height
_C.EVAL.INPUT_SIZE.WIDTH = None # new width

# ------------------ Model Training Parameter Configuration ------------------------- #

_C.TRAIN = CN() # for all training params
_C.TRAIN.MODEL_NAME = "" # model that is being used during training. Only limited options are available (see list above)
_C.TRAIN.BATCH_SIZE = 2
_C.TRAIN.TOTAL_EPOCHS = 10 # n# of epochs to be used during training.
_C.TRAIN.CRITERION = "crossentropyloss" # loss function employed by the model ("crossentropyloss" or "focalloss")
_C.TRAIN.LR = 1e-5
_C.TRAIN.FINAL_LR = 1e-7
_C.TRAIN.PRETRAINED = False
_C.TRAIN.INPUT_NORM = False # apply batchnormalization to the input image to the model
# Used to set the size for the input image. Output will be re-sized to match the original dimensions
_C.TRAIN.INPUT_SIZE = CN()
_C.TRAIN.INPUT_SIZE.RESIZE_IMG = True # toggle if the input image is re-sized, TRUE will resize the image
_C.TRAIN.INPUT_SIZE.HEIGHT = None # new height
_C.TRAIN.INPUT_SIZE.WIDTH = None # new width


# ----------------- Comparative Study Model Parameters ---------------------------- #
_C.MODELS = CN()
# Base models configuration
_C.MODELS.MODELS_DIR = "models" # location of the Models source code 
# Modified UNet 
_C.MODELS.UNET = CN()
_C.MODELS.UNET.MODEL_FILE = "" # Path to model file 
_C.MODELS.UNET.BASE = 40 # base value for the n# of U-Net channels per layer. (Every succeeding layer increases the n# of channels by 2)\
_C.MODELS.UNET.KERNEL_SIZE = 5 # kernel size employed in the convolution

# DeepLabv3+
_C.MODELS.DEEPLABV3PLUS = CN()
_C.MODELS.DEEPLABV3PLUS.BACKBONE = "resnet101"
_C.MODELS.DEEPLABV3PLUS.SEPARABLE_CONV = True # if separable convolution is being used by the model
_C.MODELS.DEEPLABV3PLUS.SRC_DIR = "models/DeepLabV3Plus-Pytorch" # path to dir containing src code 
_C.MODELS.DEEPLABV3PLUS.MODEL_FILE = "" # Model file being evaluated 
# HRNet + OCR
_C.MODELS.HRNET_OCR = CN()
_C.MODELS.HRNET_OCR.CONFIG = "" # Config file path. Normally located in the HRNet directories
_C.MODELS.HRNET_OCR.MODEL_FILE="" # Model file being tested by the program
_C.MODELS.HRNET_OCR.SRC_DIR="" # Directory that holds all the source code for the model 
_C.MODELS.HRNET_OCR.MODEL_NAME=""
# GSCNN
_C.MODELS.GSCNN = CN()
_C.MODELS.GSCNN.CONFIG = "" # Config file path
_C.MODELS.GSCNN.MODEL_FILE = "" # file path to the model file 
_C.MODELS.GSCNN.SRC_DIR = "" # Source code directory for the GSCNN


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()