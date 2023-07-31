from yacs.config import CfgNode as CN


_C = CN()

# ----------------- Evaluation Paramater Configuration  ---------------------------- #
_C.EVAL = CN()

"""
  hrnet_ocr
  unet
"""
_C.EVAL.MODEL_NAME = "" # Name of the model being used. Options displayed above.
_C.EVAL.DISPLAY_IMAGE = False # Toggle the display of each prediction alongside the annotations 
_C.EVAL.BATCH_SIZE = 1 # Image batch size input to the model

# ----------------- Comparative Study Model Parameters ---------------------------- #
_C.MODELS = CN()
# Modified UNet 
_C.MODELS.UNET_MODIF = CN()
_C.MODELS.UNET_MODIF.MODEL_FILE = "" # Path to model file 
# HRNet + OCR
_C.MODELS.HRNET_OCR = CN()
_C.MODELS.HRNET_OCR.CONFIG = "" # Config file path. Normally located in the HRNet directories
_C.MODELS.HRNET_OCR.MODEL_FILE="" # Model file being tested by the program
_C.MODELS.HRNET_OCR.SRC_DIR="" # Directory that holds all the source code for the model 


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()