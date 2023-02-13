import torch 
import dataloader
import unet 
import cv2
from torchvision.utils import draw_segmentation_masks, save_image
torch.cuda.empty_cache() # liberate the resources 

import numpy as np
import matplotlib.pyplot as plt


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
pred = torch.nn.functional.softmax(pred, dim=1)

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
seg_img = draw_segmentation_masks(images[0], masks, alpha=0.7)

# code obtained from https://pytorch.org/vision/main/auto_examples/plot_visualization_utils.html
def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

show(seg_img)
plt.show()





