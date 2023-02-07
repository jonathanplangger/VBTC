import torch 
import dataloader
import unet 
import cv2
torch.cuda.empty_cache() # liberate the resources 

db = dataloader.DataLoader()
db.randomizeOrder()

# set to cuda if correctly configured on pc
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# load the model to the device 
model = torch.load('model.pt')
model.eval()
model.to(device)


# ------------ Display the Masks ---------------- #
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

images, ann, idx = db.load_batch() # load images
# prep images and load to GPU
images = (torch.from_numpy(images)).to(torch.float32).permute(0,3,1,2)/255.0
images = images.to(device)

# run model
pred = model(images)

# softmax the output to obtain the probability masks
out = torch.nn.functional.softmax(pred, dim=1)

print(pred)








