import os 
import numpy as np 
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as T 
import dataloader
import torchsummary
from dataloader import DataLoader
from torch.nn.functional import normalize


# Empty the cache prior to training the network
torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(1.0)
torch.cuda._lazy_init()


print("---------------------------------------------------------------\n")

rellis_path = "../../datasets/Rellis-3D/" #path ot the dataset directory
db = dataloader.DataLoader(rellis_path)
BATCH_SIZE = 5
TRAIN_LENGTH = len(db.metadata)
STEPS_PER_EPOCH = TRAIN_LENGTH//BATCH_SIZE # n# of steps within the specific epoch.
EPOCHS = 1
TOTAL_BATCHES = STEPS_PER_EPOCH*EPOCHS # total amount of batches that need to be completed for training
LR = 1e-5 # learning rate


# obtain a sample of the database 
sample = db.metadata[0]
# obtain image information 
img_w = int(sample["width"]) # cast to int to ensure valid type
img_h = int(sample["height"])

# ------------------------ Model Configuration ----------------------------- #
from torchsummary import summary
from torch import nn 
import torch.nn.functional as F
    # summary(Net, ())
# set to cuda if correctly configured on pc
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

import unet 

base = 2
Net = unet.UNet(
    enc_chs=(3,base, base*2, base*4, base*8, base*16),
    dec_chs=(base*16, base*8, base*4, base*2, base), 
    out_sz=(img_h,img_w), retain_dim=True
    )


# # place model onto GPU
Net = Net.to(device)

print(torchsummary.summary(Net, (3,img_h,img_w)))
print(torch.cuda.memory_summary())

optimizer = torch.optim.Adam(params=Net.parameters(), lr = LR)

# set the index for handling the images
idx = 0

# transformImg = T.Compose([T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
# transformAnn = T.Compose([T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


# ---- Training loop ---------------#
for i in range(TOTAL_BATCHES): 
    
    # randomize order and load the batch
    db.randomizeOrder()
    images, ann, idx = db.load_batch(idx, batch_size=BATCH_SIZE)

    # prep the images through normalization and re-organization
    images = normalize(torch.from_numpy(images)).to(torch.float32).permute(0,3,1,2)
    ann = normalize(torch.from_numpy(ann)).to(torch.float32).permute(0,3,1,2)[:,0,:,:]
    
    # Create autogradient variables for training
    images = torch.autograd.Variable(images, requires_grad = False).to(device)
    ann = torch.autograd.Variable(ann, requires_grad = False).to(device)
    

    Net.train()

    images = images.to(device)
    ann = ann.to(device)


    Pred = Net(images)

    Pred = Pred[:,0,:,:] # reduce dimensionality of tensor to match label 

    criterion = torch.nn.CrossEntropyLoss() # use cross-entropy loss function 
    loss = criterion(Pred, ann) # calculate the loss 
    loss.backward() # backpropagation for loss 
    optimizer.step() # apply gradient descent to the weights

    



    # print()

    # # save the model at specific intervals
    # if i % 1000 == 0:
    #     print("Saving Model" + str(i) + ".torch")
    #     torch.save(Net.save_dict(), str(i) + ".torch")
