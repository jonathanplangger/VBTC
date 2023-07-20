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
import torchmetrics
import matplotlib.pyplot as plt 
import focal_loss
# Empty the cache prior to training the network
torch.cuda.empty_cache()
# Model configuration
from torch import nn 
import torch.nn.functional as F
import unet 
import datetime
from patchify import patchify
        
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter() 
# http://localhost:6006/?darkMode=true#timeseries
# tensorboard --logdir=runs


class TrainModel(object):
    """
        Class Provides the basic functionalities required for the training process.\n
        The training process was configured into a class to better suit the needs for modularity and logging purposes\n
        -------------------------------------------------\n
        Parameters: 

    
    
    """
    def __init__(self): 
        
        # Dataloading parameters
        self.dbPath = "../../datasets/Rellis-3D/"
        self.db = db = dataloader.DataLoader("../../datasets/Rellis-3D/")

        # Training Parameters
        self.batch_size = 1 #3
        self.train_length = len(db.train_meta)
        self.steps_per_epoch = int(self.train_length/self.batch_size) # n# of steps within the specific epoch.
        self.epochs = 10 #10
        self.total_batches = self.steps_per_epoch*self.epochs # total amount of batches that need to be completed for training
        self.lr = 1e-5 # learning rate
        self.base = 40 # base value for the UNet feature sizes - ONLY applies for the U-Net
        self.kernel_size = 5
        # self.criterion = focal_loss.FocalLoss()
        self.criterion = torch.nn.CrossEntropyLoss()

        # Image characteristics
        self.img_w = self.db.width
        self.img_h = self.db.height

        # model 
        self.model = unet.UNet(
            enc_chs=(3,self.base, self.base*2, self.base*4, self.base*8, self.base*16),
            dec_chs=(self.base*16, self.base*8, self.base*4, self.base*2, self.base), 
            out_sz=(self.img_h,self.img_w), retain_dim=True, num_class=self.db.num_classes, kernel_size=self.kernel_size
        )

        # Misc Params 
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # use GPU if available 

    def train_model(self): 
        # Use the GPU as the main device if present
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Create a tensorboard writer
        writer = SummaryWriter() 

        


        # Set the model to training mode and place onto GPU
        self.model.train()
        self.model.to(self.device)

        # Obtain the summary of the model architecture + memory requirements
        summary = torchsummary.summary(self.model, (3,self.img_h,self.img_w))
        # record the model parameters on tensorboard
        writer.add_text("Model/", str(summary).replace("\n", " <br \>")) # Print the summary on tensorboard

        # optimizer for the model 
        optim = torch.optim.Adam(params=self.model.parameters(), lr = self.lr)
        #dice performance metric
        dice = torchmetrics.Dice().to(device)
        
        # Log the current training parameters
        writer.add_text("_params/text_summary", self.logTrainParams())
        

        # ------------------------ Training loop ------------------------------------#
        for epoch in range(self.epochs):

            print("---------------------------------------------------------------\n")
            print("Training Epoch {}\n".format(epoch+1))
            print("---------------------------------------------------------------\n")

            # set the index for handling the images
            idx = 0
            self.db.randomizeOrder()
            # create/wipe the array recording the loss values

            with tqdm.tqdm(total=self.steps_per_epoch, unit="Batch") as pbar:

                for i in range(self.steps_per_epoch): 
                    

                    images, ann, idx = self.db.load_batch(idx, batch_size=self.batch_size)
                    

                    # prep the images through normalization and re-organization
                    images = (torch.from_numpy(images)).to(torch.float32).permute(0,3,1,2)/255.0
                    ann = (torch.from_numpy(ann)).to(torch.float32).permute(0,3,1,2)[:,0,:,:]

                    self.getPatches(images, ann)
                    
                    # Create autogradient variables for training
                    images = torch.autograd.Variable(images, requires_grad = False).to(device)
                    ann = torch.autograd.Variable(ann, requires_grad = False).to(device)

                    # forward pass
                    pred = self.model(images)
                    
                    loss = self.criterion(pred, ann.long()) # calculate the loss 
                    writer.add_scalar("Loss/train", loss, epoch*self.steps_per_epoch + i) # record current loss 

                    optim.zero_grad()
                    loss.backward() # backpropagation for loss 
                    optim.step() # apply gradient descent to the weights

                    # Obtain the performance metrics
                    dice_score = dice(pred, ann.long())
                    writer.add_scalar("Metrics/Dice", dice_score, epoch*self.steps_per_epoch + i) # record the dice score 

                    # reduce the learning rate once the 4th epoch is reached 
                    if optim.param_groups[0]['lr'] == self.lr and epoch == 3: 
                        print("Reducing learning rate")
                        optim.param_groups[0]['lr'] = self.lr/4

                    #update progress bar
                    pbar.set_postfix(loss=loss.item())
                    pbar.update()

            # finish writing to the buffer 
            writer.flush()
            # save the model at the end of each epoch
            torch.save(self.model, "saves/epoch{}.pt".format(epoch+1))
            
        # flush buffers and close the writer 
        writer.close()

    # Prepares the markdown to log the parameters of the file
    def logTrainParams(self):
        return """ 
        ------------------------------------------------------------<br />
        Model Training Parameters <br />
        ------------------------------------------------------------  <br />
        Date: {0} <br />
        ------------------------------------------------------------<br />
        Batch Size: {1} <br />
        Learning Rate: {2} <br />
        Base: {3} <br />
        Kernel Size: {4} <br />
        Epochs: {5} <br />
        Steps Per Epochs: {6} <br />
        Loss Function: {7} <br />
        -----------------------------------------------------------<br />  
        """.format(datetime.datetime.now(), self.batch_size, self.lr, self.base, self.kernel_size, self.epochs,
                    self.steps_per_epoch, self.criterion.__class__.__name__)


    # this function is only used during testing to allow for the visual validation of results, no need for it 
    # for the training of the network
    def showTensor(self, image): 
        plt.imshow(image.permute(1,2,0)) # re-format and plot the image         
if __name__ == "__main__": 
    train = TrainModel()
    train.train_model() # train the model