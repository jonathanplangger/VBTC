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
# obtain height and width for the images
img_h = db.height
img_w = db.width

# Global configuration variables
BATCH_SIZE = 1
TRAIN_LENGTH = len(db.metadata)
STEPS_PER_EPOCH = TRAIN_LENGTH//BATCH_SIZE # n# of steps within the specific epoch.
EPOCHS = 1

# -------------- Data Loading, Transformations, and Augmentations ------------------ #

# Normalize the color values inot the 0,1 range
def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  input_image = tf.image.resize(input_image, [img_h, img_w])
  input_mask =  tf.image.resize(input_image, [img_h, img_w])
  return input_image.numpy(), input_mask

idx = 0 # track location in epoch

images, annMap, idx = db.load_batch(idx, BATCH_SIZE) # load images in a batch 
images, annMap = normalize(images, annMap) # normalize the image files 

# --------------------- Model Configuration ----------------------------------------#
print("---------------------- Starting Model Training ---------------------------\n")

from tensorflow import keras 
from keras.models import Model 
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input

# https://idiotdeveloper.com/unet-implementation-in-tensorflow-using-keras-api/

def conv_block(input, num_filters):
  """
    input: The input represents the feature maps from the previous block.\n
    num_filters: The num_filters refers to the number of output feature channels for the convolutional layers present in the conv_block function.
  """
  x = Conv2D(num_filters, 3, padding="same")(input)
  x = BatchNormalization()(x)
  x = Activation("relu")(x)

  x = Conv2D(num_filters, 3, padding="same")(x)
  x = BatchNormalization()(x)
  x = Activation("relu")(x)

  return x

def encoder_block(input, num_filters):
  """
    input: The input represents the feature maps from the previous block.\n
    num_filters: The num_filters refers to the number of output feature channels for the convolutional layers present in the conv_block function.
  """
  x = conv_block(input, num_filters)
  p = MaxPool2D((2, 2))(x)
  return x, p

def decoder_block(input, skip_features, num_filters):
  x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
  x = Concatenate()([x, skip_features])
  x = conv_block(x, num_filters)
  return x


def build_unet(input_shape):
  """
    input_shape: It is a tuple of height, width and the number of input channels. For example: (512, 512, 3)
  """
  inputs = Input(input_shape)

  s1, p1 = encoder_block(inputs, 64)
  s2, p2 = encoder_block(p1, 128)
  s3, p3 = encoder_block(p2, 256)
  s4, p4 = encoder_block(p3, 512)

  b1 = conv_block(p4, 1024)

  d1 = decoder_block(b1, s4, 512)
  d2 = decoder_block(d1, s3, 256)
  d3 = decoder_block(d2, s2, 128)
  d4 = decoder_block(d3, s1, 64)

  outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

  model = Model(inputs, outputs, name="U-Net")
  return model


if __name__ == "__main__":
  
  input_shape = (img_h, img_w, 3)
  model = build_unet(input_shape)
  model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

  model.fit(images, annMap, epochs=EPOCHS)

  





