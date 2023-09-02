# This file contains the code for any custom-made loss functions

import torch
import torch.nn as nn
import numpy as np # for debugging 


class CustomIoULoss(nn.Module):
    def __init__(self): 

        super(CustomIoULoss, self).__init__()
        self.num_classes = 35
        self.eps = 1e-10 # epsilon value -> used to avoid dividing by zero

    def forward(self, pred, ann):
        """
            Params: \n
            ann (tensor) = annotation (b,h,w)\n
            pred (tensor) = logit output prediction (b,c,h,w) \n
        """

        # Get the softmax output ([0,1]) for the prediction
        pred = torch.softmax(pred, dim=1) + self.eps

        # Create the onehot tensor for each class (gic)
        ann = torch.unsqueeze(ann,0)
        ann = ann.long() # convert to required variable type
        ann_onehot = torch.zeros(pred.shape) # obtain blank array w/ same shape as the prediction
        ann_onehot = ann_onehot.cuda() # TODO update to use the device instead. 
        ann_onehot.scatter_(1, ann, 1) # create onehot vector

        # del ann # free up memory  # TODO, figure out another way to determine which classes are within the annotation file

        num = pred * ann_onehot # numerator
        denom = pred + ann_onehot - num # denominator

        del pred # free up memory 

        # sum up all the values on a per-class basis
        num = torch.sum(num, dim=(2,3))
        denom = torch.sum(denom, dim=(2,3)) 
        
        # get the IoU score for each class 
        loss_iou = torch.div(num, denom) 

        del num, denom 

        # fit the iou values to a more suitable weighted loss value
        loss_iou = -0.3*torch.atan(5*loss_iou - 2.5) + 0.5 
        # loss_iou = 1 - loss_iou # base iou implementation -> Employ the class-based IoU directly  

        # turn off contribution to loss by any classes not within the annotation file        
        num_active = 0
        for c in range(self.num_classes): 
            if c not in ann: 
                loss_iou[0,c] = self.eps
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
    