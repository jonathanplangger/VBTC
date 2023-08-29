# This file contains the code for any custom-made loss functions

import torch
import torch.nn as nn
import numpy as np # for debugging 


class CustomIoULoss(nn.Module):
    def __init__(self): 

        super(CustomIoULoss, self).__init__()
        self.num_classes = 35


    def forward(self, pred, ann):
        """
            Params: \n
            ann (tensor) = annotation (b,h,w)\n
            pred (tensor) = logit output prediction (b,c,h,w) \n
        """

        # Get the softmax output ([0,1]) for the prediction
        pred = torch.softmax(pred, dim=1)

        # Create the onehot tensor for each class (gic)
        ann = torch.unsqueeze(ann,0)
        ann = ann.long() # convert to required variable type
        ann_onehot = torch.zeros(pred.shape) # obtain blank array w/ same shape as the prediction
        ann_onehot = ann_onehot.cuda() # TODO update to use the device instead. 
        ann_onehot.scatter_(1, ann, 1) # create onehot vector

        num = pred * ann_onehot # numerator
        denom = pred + ann_onehot - num # denominator

        # sum up all the values on a per-class basis
        num = torch.sum(num, dim=(2,3))
        denom = torch.sum(denom, dim=(2,3)) 
        
        # get the IoU score for each class 
        iou_c = torch.div(num, denom) 
        loss_iou = -1.4*torch.log(torch.pow(iou_c, 0.5) + 1) + 1 # fit the error function 

        # turn off contribution to loss by any classes not within the annotation file        
        num_active = 0
        for c in range(self.num_classes): 
            if c not in ann: 
                loss_iou[0,c] = 0
            else: 
                num_active += 1 # increase the count

        #sum up the final loss amount 
        loss_iou = torch.sum(loss_iou, dim=1)
        # Average the iou loss based on the currently active classes within the annotations
        loss_iou = loss_iou/num_active
        

        return loss_iou


# For testing out the loss function first 
if __name__ == "__main__": 
    criterion = CustomIoULoss()
    pred = torch.rand(size=(1,35,1200,1920)).cuda()
    ann = torch.randint(0,34,size = (1,1200,1920)).cuda()
    loss = criterion(pred, ann)
    