import os 
import numpy as np 
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as T 
import dataloader
import torchsummary
from dataloader import DataLoader
# from torch.nn.functional import normalize
import tqdm
import tools
import torchmetrics

# Empty the cache prior to training the network
torch.cuda.empty_cache()
# Model configuration
from torch import nn 
import torch.nn.functional as F
import unet 
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter() 
# http://localhost:6006/?darkMode=true#timeseries
# tensorboard --logdir=runs

print("---------------------------------------------------------------\n")

rellis_path = "../../datasets/Rellis-3D/" #path ot the dataset directory

db = dataloader.DataLoader(rellis_path)
BATCH_SIZE = 3 #3
TRAIN_LENGTH = len(db.train_meta)
STEPS_PER_EPOCH = TRAIN_LENGTH//BATCH_SIZE # n# of steps within the specific epoch.
EPOCHS = 10 #10
TOTAL_BATCHES = STEPS_PER_EPOCH*EPOCHS # total amount of batches that need to be completed for training
lr = 1e-4 # learning rate
BASE = 2 # base value for the UNet feature sizes
KERNEL_SIZE = 3


# obtain a sample of the database 
sample = db.train_meta[0]
# obtain image information 
img_w = int(sample["width"]) # cast to int to ensure valid type
img_h = int(sample["height"])

# ------------------------ Model Configuration ----------------------------- #

# set to cuda if correctly configured on pc
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

base = BASE
model = unet.UNet(
    enc_chs=(3,base, base*2, base*4, base*8, base*16),
    dec_chs=(base*16, base*8, base*4, base*2, base), 
    out_sz=(img_h,img_w), retain_dim=True, num_class=db.num_classes, kernel_size=KERNEL_SIZE
    )
model.train()

# # place model onto GPU
model = model.to(device)

# Obtain the summary of the model architecture + memory requirements
torchsummary.summary(model, (3,img_h,img_w))

optim = torch.optim.Adam(params=model.parameters(), lr = lr)

criterion = torch.nn.CrossEntropyLoss(reduction='mean') # use cross-entropy loss function 
dice = torchmetrics.Dice().to(device)
# Record the model parameters
writer.add_text("_params/text_summary", tools.logTrainParams(BATCH_SIZE, BASE, STEPS_PER_EPOCH, EPOCHS,lr, criterion, KERNEL_SIZE))

# ------------------------ Training loop ------------------------------------#

for epoch in range(EPOCHS):

    print("---------------------------------------------------------------\n")
    print("Training Epoch {}\n".format(epoch+1))
    print("---------------------------------------------------------------\n")

    # set the index for handling the images
    idx = 0
    db.randomizeOrder()
    # create/wipe the array recording the loss values

    with tqdm.tqdm(total=STEPS_PER_EPOCH, unit="Batch") as pbar:

        for i in range(STEPS_PER_EPOCH): 
            

            images, ann, idx = db.load_batch(idx, batch_size=BATCH_SIZE)

            # prep the images through normalization and re-organization
            images = (torch.from_numpy(images)).to(torch.float32).permute(0,3,1,2)/255.0
            ann = (torch.from_numpy(ann)).to(torch.float32).permute(0,3,1,2)[:,0,:,:]
            
            # Create autogradient variables for training
            images = torch.autograd.Variable(images, requires_grad = False).to(device)
            ann = torch.autograd.Variable(ann, requires_grad = False).to(device)

            # forward pass
            pred = model(images)
            
            loss = criterion(pred, ann.long()) # calculate the loss 
            writer.add_scalar("Loss/train", loss, epoch*STEPS_PER_EPOCH + i) # record current loss 

            
            loss.backward() # backpropagation for loss 

            optim.zero_grad()
            optim.step() # apply gradient descent to the weights

            # update the learning rate based on the amount of error present
            # if optim.param_groups[0]['lr'] == lr and loss.item() < 1.5: 
            #     print("Reducing learning rate")
            #     optim.param_groups[0]['lr'] = 5e-5

            del images # free up some memory, no longer needed here

            # Obtain the performance metrics
            dice_score = dice(pred, ann.long())
            writer.add_scalar("Metrics/Dice", dice_score, epoch*STEPS_PER_EPOCH + i) # record the dice score 

            #update progress bar
            pbar.set_postfix(loss=loss.item())
            pbar.update()

    # finish writing to the buffer 
    writer.flush()
    # save the model at the end of each epoch
    torch.save(model, "saves/epoch{}.pt".format(epoch+1))
    
# flush buffers and close the writer 
writer.close()