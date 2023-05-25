import torch 
import dataloader
import unet 
import cv2
from torchvision.utils import draw_segmentation_masks, save_image
torch.cuda.empty_cache() # liberate the resources 
import numpy as np
import matplotlib.pyplot as plt
# read yaml file 
import yaml

# Display masks
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchsummary

# obtain the data
db = dataloader.DataLoader()
db.randomizeOrder()

# Empty the cache prior to training the network
torch.cuda.empty_cache()

TOTAL_NUM_TEST = len(db.test_meta)
NUM_TEST = TOTAL_NUM_TEST
SHOW = False # display the output as an image
BATCH_SIZE = 1 # configure the size of the batch

# ------------ GEt Colour Representation for the figure ------------------ #
# open the ontology file for rellis and obtain the colours for them
# ---- TODO -- This needs to be handled by the dataloader and NOT the eval.py
with open("Rellis_3D_ontology/ontology.yaml", "r") as stream: 
    try: 
        ont = yaml.safe_load(stream)[1]
    except yaml.YAMLError as exc: 
        print(exc)
        exit()

# add all the colours to a list object 
colors = []
for i in range(35): 
    try: 
        val = tuple(ont[i])
        colors.append(val)
    except: # if the dict element does not exist
        colors.append("#000000") # assign black colour to the unused masks


# ---------------- Prep the model for testing ------------------- # 



# set to cuda if correctly configured on pc
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# load the model to the device 
model = torch.load('model.pt')
model.eval()
model.to(device)

# Obtain the summary of the model architecture + memory requirements
torchsummary.summary(model, (3,db.height,db.width))

# iterate through all tests while using batches 
for idx in range(0, NUM_TEST, BATCH_SIZE): 

    # ------------ Run the model with the loaded images  ---------------- #

    images, ann, idx = db.load_batch(idx, BATCH_SIZE, isTraining=False) # load images
    orig_images = images # store this for later use
    # prep images and load to GPU
    images = (torch.from_numpy(images)).to(torch.float32).to(device).permute(0,3,1,2)/255.0

    # run model
    pred = model(images)

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




    # -------------- Show the image output for the segmentation  ---------- #
    if SHOW:     

        # Create the output plot
        fig, axs = plt.subplots(ncols=4, squeeze=False, gridspec_kw = {'wspace':0.05, 'hspace':0})

        # Base image
        img = F.to_pil_image(images[0])
        axs[0, 0].imshow(np.asarray(img))
        axs[0, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], title="Base Image" )

        # obtain the blended segmentation image 
        seg_img = draw_segmentation_masks(images[0], masks, alpha=0.7, colors=colors)
        img = F.to_pil_image(seg_img.detach())
        axs[0, 1].imshow(np.asarray(img))
        axs[0, 1].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], title="Image/Mask Blend" )

        # Ground Truth Annotation masks 
        axs[0, 2].imshow(ann.cpu().detach().numpy()[0], cmap='gray')
        axs[0, 2].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], title="Ground Truth Annotations")

        # Output mask
        axs[0, 3].imshow(torch.argmax(pred,1).cpu().detach().numpy()[0], cmap='gray')
        axs[0, 3].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], title="Output Mask")

        plt.show()

    # end the testing if the desired # of tests has been obtained
    if idx == NUM_TEST: 
        break
