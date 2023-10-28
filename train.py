import torch.nn as nn
import torch
import torchvision.transforms as T 
from torch.nn import functional as TF
import dataloader
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
import tools 
from io import StringIO as SIO
# Add the loss odyssey loss functions to the implementation
import sys
sys.path.insert(1,"/home/jplangger/Documents/Dev/VBTC/loss_odyssey")

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter() 
# http://localhost:6006/?darkMode=true#timeseries
# tensorboard --logdir=runs

from config import get_cfg_defaults # obtain model configurations

        # Class Provides the basic functionalities required for the training process.\n
        # The training process was configured into a class to better suit the needs for modularity and logging purposes\n
        # -------------------------------------------------\n

class TrainModel(object):
    """ Implements the basic functionalities for the implementation of model training. Configuration of training is defined by the configuration file. \n
    No configuration params are available at the moment.

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
        self.db = db = dataloader.DataLoader(self.cfg.DB.PATH, preprocessing=preprocess_input, remap = True)

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

        if self.cfg.TRAIN.PRETRAINED == True: # Pre-trained model is being used
            self.model = self.model_handler.load_model()
        else: # Generate a new model based on configuration file
            self.model = self.model_handler.gen_model(self.db.num_classes)

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
        # scheduler for the learning rate 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", patience=1, factor=0.1,verbose=True, min_lr=self.cfg.TRAIN.FINAL_LR)
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

        # # Obtain the summary of the model architecture + memory requirements
        ##TODO: Fix Summary not working with DLV3+, execution halts at same location 
        # summary = torchsummary.summary(self.model, (3,input_size[0],input_size[1]))


        from torchinfo import summary 
        model_summary = summary(self.model, input_size=(self.batch_size, 3, input_size[0], input_size[1]))
        # record the model parameters on tensorboard``
        writer.add_text("Model/", str(model_summary).replace("\n", " <br \>")) # Print the summary on tensorboard

        # string used to log when each learning rate values are updated
        lr_str = ""

        # count all the steps during training
        step = 0
        # ------------------------ Training loop ------------------------------------#
        for epoch in range(self.epochs):

            print("---------------------------------------------------------------\n")
            print("Training Epoch {}\n".format(epoch+1))
            print("---------------------------------------------------------------\n")

            # set the index for handling the images
            idx = 0
            self.db.randomizeOrder()
            # create/wipe the array recording the loss values

            with tqdm.tqdm(total=int(self.steps_per_epoch), unit="Batch") as pbar:

                for i in range(self.steps_per_epoch): 
                    
                    # load the image batch
                    _, images, ann, idx = self.db.load_batch(idx, batch_size=self.batch_size, resize = input_size)
                    
                    # Create autogradient variables for training
                    images = torch.autograd.Variable(images, requires_grad = False).to(device)
                    ann = torch.autograd.Variable(ann, requires_grad = False).to(device)

                    # forward pass
                    pred = self.model(images)
                    del images # release images from mem -> no longer needed
                    pred = self.model_handler.handle_output(pred) # handle output based on the model

                    # Zero the gradients for the function
                    optim.zero_grad()

                    dice_score = dice(pred, ann.long())
                    writer.add_scalar("Metric/Dice", dice_score, epoch*self.steps_per_epoch + i)

                    loss = self.criterion(pred, ann.long()) # calculate the loss
                    del ann, pred # release, no longer needed. 

                    # Add a list wrapper if only 1 loss value is obtained as output (due to reduction scheme of LF )
                    if loss.numel() == 1: 
                        loss = [loss] 
                    for l in loss: 
                        writer.add_scalar("Loss/train", l, epoch*self.steps_per_epoch + i) # record current loss 
                        pbar.set_postfix(l = l.item(),lr = optim.param_groups[0]['lr'])
                    
                    sum(loss).backward()

                    # loss.backward() # backpropagation for loss 


                    pbar.update()

                    # del loss # release memory 
                    optim.step() # apply gradient descent to the weights
                    optim.zero_grad()

                    # Measure the total amount of memory that is being reserved training (for optimization purposes)
                    if step == 2: 
                        print("Memory Reserved for Training: {}MB".format(tools.get_memory_reserved()))
                        
                    step += 1 # increment the counter

            ############# Output LR update to Tensorboard ############################
            
            # Update the std out to save to a string 
            temp = sys.stdout
            sys.stdout = o = SIO()
            # update the learning rate based on scheduler scheme (every epoch)
            scheduler.step(loss[0]) # use only one of the obtained loss values 
            sys.stdout = temp # restore original output 

            # get the result of the step update 
            new_lr = o.getvalue()

            # save the lr update to then place it on tensorboard
            if new_lr != "": 
                lr_str += new_lr + "<br />" # store the value on the string
                print(new_lr)

            # write the update to tensorboard
            writer.add_text("_lr/lr_update", lr_str)

            ############# End Output LR update to Tensorboard ############################

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
        import loss 
        
        from loss_odyssey import dice_loss


        criterion = self.cfg.TRAIN.CRITERION
        # Call the loss function based on the configuration file
        if criterion == 'crossentropyloss': 
            self.criterion = torch.nn.CrossEntropyLoss()
        elif criterion == "focalloss": 
            self.criterion = focal_loss.FocalLoss(gamma=2.0)
        elif criterion == "iouloss": 
            self.criterion = loss.IoULoss()
        elif criterion == "dicefocal": # CUrrently not training well -> Requires some testing to make sure that it even works 
            self.criterion = loss.DiceFocal()
        elif criterion == "diceloss": 
            self.criterion = loss.DiceLoss()
        elif criterion == "tverskyloss": 
            self.criterion = loss.TverskyLoss()
        elif criterion == "dicetopk": 
            self.criterion = loss.DiceTopk()
        elif criterion == "powerjaccard":
            self.criterion = loss.PowerJaccard()
        # ------- FCIoU Versions ---------- #
        elif criterion == "fciouv1":
            self.criterion = loss.FCIoUV1()
        elif criterion == "fciouv2": 
            self.criterion = loss.FCIoUV2() 
        elif criterion == "fciouv3": 
            self.criterion = loss.FCIoUV3()
        elif criterion == "fciouv4": 
            self.criterion = loss.FCIoUV4()
        elif criterion == "fciouv5":
            self.criterion = loss.FCIoUV5()
        elif criterion == "fciouv6":
            self.criterion = loss.FCIoUV6()
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