import os
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf

Learning_Rate=1e-5
width=800
height=800 # image width and height
batchSize=3

TrainFolder="LabPics/Simple/Train//"
ListImages=os.listdir(os.path.join(TrainFolder, "Image")) 

transformImg=tf.Compose([tf.ToPILImage(),tf.Resize((height,width)), tf.ToTensor(),tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
transformAnn=tf.Compose([tf.ToPILImage(),tf.Resize((height,width)), tf.ToTensor()])

