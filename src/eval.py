import os 
# Preset environment variables 
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # ONLY FOR DEBUG -  Allows for the stacktrace to be effectively used
# To run trace: 
# python -m cProfile -o /tmp/tmp.prof eval.py
# snakeviz /tmp/tmp.prof

import sys
import torch 
import dataloader
from torchvision.utils import draw_segmentation_masks, save_image
torch.cuda.empty_cache() # liberate the resources 
import numpy as np
import matplotlib.pyplot as plt
# read yaml file 
import yaml
import torchmetrics
from tqdm import tqdm
import time 
import sklearn

# Display masks
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchsummary
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter() 

# obtain the data
db = dataloader.DataLoader()
db.randomizeOrder()

# Empty the cache prior to training the network
torch.cuda.empty_cache()

TOTAL_NUM_TEST = len(db.test_meta)
NUM_TEST = TOTAL_NUM_TEST
SHOW = False # display the output as an image, turn off to log results 
BATCH_SIZE = 1 # configure the size of the batch
MODEL = "hrnet" # set which model is going to be evaluated



# ------------ GEt Colour Representation for the figure ------------------ #
# open the ontology file for rellis and obtain the colours for them
# ---- TODO -- This needs to be handled by the dataloader and NOT the eval.py
with open("Rellis_3D_ontology/ontology.yaml", "r") as stream: 
    try: 
        ont = yaml.safe_load(stream)
    except yaml.YAMLError as exc: 
        print(exc)
        exit()

# add all the colours to a list object 
colors = []
for i in range(35): 
    try: 
        val = tuple(ont[1][i])
        colors.append(val)
    except: # if the dict element does not exist
        colors.append("#000000") # assign black colour to the unused masks


# ---------------- Prep the model for testing ------------------- # 

# set to cuda if correctly configured on pc
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if MODEL == "unet":
    # load the model to the device 
    model = torch.load('model.pt')
elif MODEL == "hrnet":
    # TODO - Update to relative path -> current iterations of sys.path did not allow this... 
    sys.path.insert(0,"/home/jplangger/Documents/Dev/VBTC/src/models/HRNet-Semantic-Segmentation-HRNet-OCR/tools/")
    from test import load_model
    model = load_model("config/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml")

model.eval()
model.to(device)

# Obtain the summary of the model architecture + memory requirements
torchsummary.summary(model, (3,db.height,db.width))

# Use the Dice score as the performance metric to assess model accuracy
dice = torchmetrics.Dice().to(device)

# Intersection over Union metric calculator 
iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=db.num_classes, average='none')

# Obtain the confusion matrix
# confMat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=db.num_classes)
# confMat = confMat.to(device=device)

# Used to tally the final mean intersection over union score
miou = torch.zeros(db.num_classes).to(device)
# Count the n# of instances with this class in it
count_c = torch.zeros(db.num_classes).to(device)

# Create a blank confusion matrix to be input into 
confusionMatrix = torch.empty([TOTAL_NUM_TEST,db.num_classes, db.num_classes])

from sklearn.metrics import confusion_matrix


with tqdm(total = TOTAL_NUM_TEST, unit = "Batch") as pbar:
# iterate through all tests while using batches 
    for idx in range(0, NUM_TEST, BATCH_SIZE): 

        # ------------ Run the model with the loaded images  ---------------- #

        images, ann, idx = db.load_batch(idx, BATCH_SIZE, isTraining=False) # load images
        orig_images = images # store this for later use
        orig_ann = ann # store for later display
        # prep images and load to GPU
        images = ((torch.from_numpy(images)).to(torch.float32).permute(0,3,1,2)/255.0).to(device)

        ann = (torch.from_numpy(ann)).to(torch.float32).permute(0,3,1,2)[:,0,:,:].to(device)

        
        startTime = time.time() # used to measure prediction time for the model

        # run model
        with torch.no_grad(): # do not calculate gradients for this task
            pred = model(images)

        # Measure the performance of the model
        dice_score = dice(pred, ann.long())
        

        # ----------- Plot the Results ----------------------- #
        # create a blank array
        masks = torch.zeros(35,1200,1920, device=device, dtype=torch.bool)

        # obtain a mask for each class
        for classID in range(masks.shape[0]): 
            masks[classID] = (pred.argmax(dim=1) == classID)

        # move the masks and the image onto the CPU
        images = orig_images # restore original images
        masks = masks.to('cpu')
        del orig_images # no longer needed

        # convert the image to uint8 type 
        # images = images.to(torch.uint8)
        masks = masks.to(torch.bool) # convert to boolean 

        # Get the iou score for each of the classes individually
        iou_score = iou(pred.cpu(), ann.long().cpu()).to(device)

        unique_classes = ann.unique() # get unique classes that are represented in the annotation

        # for each unique class
        for c in unique_classes: 
            c = int(c.item()) # convert to int number 
            miou[c] += iou_score[c] #add to the final sum count for IoU
            count_c[c] += 1 # increment the final count 

        # Obtain and store the confusion matrix
        # confusionMatrix[idx] = confMat(pred.cpu(), ann.long().cpu()) # Running this on CPU actually makes it considerably faster

        # Log the results of the evaluation onto tensorboard
        if not SHOW:
            writer.add_scalar("Metrics/Dice", dice_score.item(), idx) # record the dice score 
            writer.add_scalar("Metrics/Time", time.time() - startTime, idx)


        # -------------- Show the image output for the segmentation  ---------- #
        if SHOW:     

            # Create the output plot
            fig, axs = plt.subplots(ncols=4, squeeze=False, gridspec_kw = {'wspace':0.05, 'hspace':0})

            # Base image
            # img = F.to_pil_image(images[0])
            axs[0, 0].imshow(images[0].astype(int))
            axs[0, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], title="Base Image" )

            # obtain the blended segmentation image 
            seg_img = draw_segmentation_masks(torch.tensor(images[0]).permute(2,0,1).to(torch.uint8), masks, alpha=0.7, colors=colors)
            img = F.to_pil_image(seg_img.detach())
            axs[0, 1].imshow(np.asarray(img))
            axs[0, 1].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], title="Image/Mask Blend" )

            # Ground Truth Annotation masks 
            axs[0, 2].imshow(orig_ann[0].astype(int)[:,:,0], cmap='gray')
            axs[0, 2].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], title="Ground Truth Annotations")

            # Output mask
            axs[0, 3].imshow(torch.argmax(pred,1).cpu().detach().numpy()[0], cmap='gray')
            axs[0, 3].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], title="Output Mask")

            plt.show()

            # exit()

        # Update the progress bar
        pbar.set_postfix(score = dice_score.item())
        pbar.update()

        # Clear memory
        del pred 
        del images 

        # end the testing if the desired # of tests has been obtained
        if idx == NUM_TEST: 
            break


# After the completion of obtaining the metric
miou = miou/count_c # obtain the average amount for each class 

table = {}

# Map the class names to the mIoU values obtained
for c in ont[0]: 
    table[ont[0][c]] = round(miou[c].cpu().item(), 4)



print("End of Evaluation Program")







   