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


db = dataloader.DataLoader()
db.randomizeOrder()

# set to cuda if correctly configured on pc
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# load the model to the device 
model = torch.load('model.pt')
model.eval()
model.to(device)


# ------------ Display the Masks ---------------- #

images, ann, idx = db.load_batch(isTraining=False) # load images
# prep images and load to GPU
images = (torch.from_numpy(images)).to(torch.float32).permute(0,3,1,2)/255.0
images = images.to(device)

# run model
pred = model(images)

# softmax the output to obtain the probability masks
pred = torch.nn.functional.softmax(pred, dim=1)

# ------------ Obtain IoU Metrics ------------------- #
# Import the Jaccard IoU class
from torchmetrics import JaccardIndex
# obtain the iou for this specific class
jac = JaccardIndex(task = "multiclass", num_classes = 35, average='micro').to(device)
# convert the annotations to a tensor of the required type (int)
ann = torch.from_numpy(ann[:,:,:,0]).type(torch.int).to(device)
# obtain the iou
iou = jac(pred,ann)

print("IoU Obtained: {:.2f}%".format(iou*100))


# ----------- Plot the Results ----------------------- #
# create a blank array
masks = torch.zeros(35,1200,1920, device=device, dtype=torch.bool)

# obtain a mask for each class
for classID in range(masks.shape[0]): 
    masks[classID] = (pred.argmax(dim=1) == classID)

# move the masks and the image onto the CPU
images = images.to('cpu')*255.0
masks = masks.to('cpu')

# convert the image to uint8 type 
images = images.to(torch.uint8)
masks = masks.to(torch.bool) # convert to boolean 


# obtain the segmented image
seg_img = draw_segmentation_masks(images[0], masks, alpha=0.7, colors=colors)

# code for visualization obtained from https://pytorch.org/vision/main/auto_examples/plot_visualization_utils.html
def show(imgs):
    # Create the figures 
    fig, axs = plt.subplots(ncols=4, squeeze=False, gridspec_kw = {'wspace':0.05, 'hspace':0})

    # Base image
    img = F.to_pil_image(images[0])
    axs[0, 0].imshow(np.asarray(img))
    axs[0, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], title="Base Image" )
   
    # Image/Mask Blended image 
    if not isinstance(imgs, list):
        imgs = [imgs]
    img = F.to_pil_image(imgs[0].detach())
    axs[0, 1].imshow(np.asarray(img))
    axs[0, 1].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], title="Image/Mask Blend" )
   
    # Ground Truth Annotation masks 
    axs[0, 2].imshow(ann.cpu().detach().numpy()[0], cmap='gray')
    axs[0, 2].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], title="Ground Truth Annotations")
   
    # Output mask
    axs[0, 3].imshow(torch.argmax(pred,1).cpu().detach().numpy()[0], cmap='gray')
    axs[0, 3].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], title="Output Mask")



    
# show the plot for the segmentation mask 
show(seg_img)
plt.show()





