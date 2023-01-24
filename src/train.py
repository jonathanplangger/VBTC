import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import matplotlib.pyplot as plt
from dataloader import DataLoader
import cv2
import numpy as np 

print("---------------------------------------------------------------\n")
db = DataLoader("./../../datasets/Rellis-3D/")

# Global configuration variables
BATCH_SIZE = 3
TRAIN_LENGTH = len(db.metadata)
STEPS_PER_EPOCH = TRAIN_LENGTH//BATCH_SIZE # n# of steps within the specific epoch.

# -------------- Data Loading, Transformations, and Augmentations ------------------ #

# Normalize the color values inot the 0,1 range
def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image.numpy(), input_mask

idx = 0 # track location in epoch

images, annMap, idx = db.load_batch(idx, BATCH_SIZE) # load images in a batch 
images, annMap = normalize(images, annMap) # normalize the image files 

# --------------------- Model Configuration ----------------------------------------#







