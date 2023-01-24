import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import matplotlib.pyplot as plt
from dataloader import DataLoader
import cv2

print("---------------------------------------------------------------\n")
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

db = DataLoader("./../../datasets/Rellis-3D/")


# Normalzie the color values into the 0,1 range 
def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask











