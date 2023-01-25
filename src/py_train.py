import os 
import numpy as np 
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf 
import dataloader
# Empty the cache prior to training the network
torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(1.0)
torch.cuda._lazy_init()

rellis_path = "../../datasets/Rellis-3D/" #path ot the dataset directory


Learning_Rate = 1e-5
# get rellis dataloader
db = dataloader.DataLoader(rellis_path)
# obtain a sample of the database 
sample = db.metadata[0]
# obtain image information 
width = int(sample["width"]) # cast to int to ensure valid type
height = int(sample["height"])

batchSize = 3

transformImg=tf.Compose([tf.ToPILImage(), tf.ToTensor()])
transformAnn=tf.Compose([tf.ToPILImage(), tf.ToTensor()])

def ReadRandomImage(db: dataloader.DataLoader):
    """
        Read a random image and converts them into pytorch compatible tensor\n
        ------------------------\n
        db(dataloader.Dataloader): Dataloader for the given database. Provides the database images.\n 
    """
    
    idx = np.random.randint(0, len(db.metadata)) # pick a random image from the index
    img = transformImg(cv2.imread(db.metadata[idx]["file_name"])) # convert to tensor
    annMap = transformAnn(cv2.imread(db.metadata[idx]["sem_seg_file_name"])[:,:,0]) # only take one channel (they are the same throughout)

    return img, annMap

# Load a batch of images
def LoadBatch(db: dataloader.DataLoader):
    """
        LoadBatch(): Load a batch of images. 
    """ 
    images = torch.zeros([batchSize, 3, height, width])
    ann = torch.zeros([batchSize, 3, height, width])
    for i in range(batchSize):
        images[i], ann[i] = ReadRandomImage(db)

    return images, ann



# --------- Load the neural net --------------------- #
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
    out_sz=(height,width), retain_dim=True
    )

# # place model onto GPU
Net = Net.to(device)


optimizer = torch.optim.Adam(params=Net.parameters(), lr = Learning_Rate)

# ---- Training loop ---------------#
for itr in range(20000): 
    images, ann = LoadBatch(db)

    images = torch.autograd.Variable(images, requires_grad = False).to(device)
    ann = torch.autograd.Variable(ann, requires_grad = False).to(device)
    ann = ann[:,0,:,:] # only use one channel (same throughout)
    

    Net.train()

    images = images.to(device)
    ann = ann.to(device)


    Pred = Net(images)

    Pred = Pred[:,0,:,:] # reduce dimensionality of tensor to match label 

    criterion = torch.nn.CrossEntropyLoss() # use cross-entropy loss function 
    loss = criterion(Pred, ann) # calculate the loss 
    loss.backward() # backpropagation for loss 
    optimizer.step() # apply gradient descent to the weights

    

    # save the model at specific intervals
    if itr % 1000 == 0:
        print("Saving Model" + str(itr) + ".torch")
        torch.save(Net.save_dict(), str(itr) + ".torch")
