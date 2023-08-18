import os 
import numpy as np 
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as T 
from torch.nn import functional as TF
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
import modelhandler

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter() 
# http://localhost:6006/?darkMode=true#timeseries
# tensorboard --logdir=runs

from config import get_cfg_defaults # obtain model configurations


class TrainModel(object):
    """
        Class Provides the basic functionalities required for the training process.\n
        The training process was configured into a class to better suit the needs for modularity and logging purposes\n
        -------------------------------------------------\n
        Parameters: 

    """
    def __init__(self): 

        # initialize configuration and update using config file
        self.cfg = get_cfg_defaults()
        self.cfg.merge_from_file("configs/config_comparative_study.yaml")
        self.cfg.freeze()

        # obtain the model-specific operations
        self.model_handler = modelhandler.ModelHandler(self.cfg, "train")
    

        # default to not preprocessing the input to the model
        preprocess_input = False 

       
        # Dataloader initialization
        self.db = db = dataloader.DataLoader(self.cfg.DB.PATH, preprocessing=preprocess_input)

        # Training Parameters
        self.batch_size = self.cfg.TRAIN.BATCH_SIZE #3
        self.epochs = self.cfg.TRAIN.TOTAL_EPOCHS #10
        self.train_length = len(db.train_meta)
        self.steps_per_epoch = int(self.train_length/self.batch_size) # n# of steps within the specific epoch.
        self.total_batches = self.steps_per_epoch*self.epochs # total amount of batches that need to be completed for training
        
        # Get the criterion for training
        self.criterion = ""
        self.__get_criterion()

        # Image characteristics
        self.img_w = self.db.width
        self.img_h = self.db.height

        # retrieve the model based on the configuration 
        self.model = self.model_handler.gen_model()

    def train_model(self): 
        # Use the GPU as the main device if present
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Create a tensorboard writer
        writer = SummaryWriter() 

        # Set the model to training mode and place onto GPU
        self.model.train()
        self.model.to(device)

        # optimizer for the model 
        optim = torch.optim.Adam(params=self.model.parameters(), lr = self.cfg.TRAIN.LR)
        #dice performance metric
        dice = torchmetrics.Dice().to(device)

        # Log the training parameters        
        writer.add_text("_params/text_summary", self.model_handler.logTrainParams())

        # If resizing of the input image is required 
        if self.cfg.TRAIN.INPUT_SIZE.RESIZE_IMG:
            # new img size set by config file
            input_size = (self.cfg.TRAIN.INPUT_SIZE.HEIGHT, self.cfg.TRAIN.INPUT_SIZE.WIDTH) 
        else: 
            input_size = (self.db.height, self.db.width) # use the original input size values instead.

        # Obtain the summary of the model architecture + memory requirements
        summary = torchsummary.summary(self.model, (3,input_size[0],input_size[1]))
        # record the model parameters on tensorboard
        writer.add_text("Model/", str(summary).replace("\n", " <br \>")) # Print the summary on tensorboard

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
                    
                    # load the image batch
                    _, images, ann, idx = self.db.load_batch(idx, batch_size=self.batch_size, resize = input_size)
                    
                    # prep the images through normalization and re-organization
                    images = (torch.from_numpy(images)).to(torch.float32).permute(0,3,1,2)/255.0
                    ann = (torch.from_numpy(ann)).to(torch.float32).permute(0,3,1,2)[:,0,:,:]

                    # Create autogradient variables for training
                    images = torch.autograd.Variable(images, requires_grad = False).to(device)
                    ann = torch.autograd.Variable(ann, requires_grad = False).to(device)

                    # forward pass
                    pred = self.model(images)
                    pred = self.model_handler.handle_output(pred) # handle output based on the model
                    
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

    def __get_criterion(self): 
        """
        __get_criterion(self): 
        ---------------------------------------------------
        Retrieves the loss function based on the configuration file as defined in self.cfg.TRAIN.CRITERION
        """
        if self.cfg.TRAIN.CRITERION == 'crossentropyloss': 
            self.criterion = torch.nn.CrossEntropyLoss()
        elif self.cfg.TRAIN.CRITERION == "focalloss": 
            self.criterion = focal_loss.FocalLoss()
        else: 
            exit("Invalid loss function, please select a valid one")


    # this function is only used during testing to allow for the visual validation of results, no need for it 
    # for the training of the network
    def showTensor(self, image): 
        plt.imshow(image.permute(1,2,0)) # re-format and plot the image         

# Main program calling 
if __name__ == "__main__": 
    train = TrainModel()
    train.train_model() # train the model