# This file contains the code for any custom-made loss functions

import torch
import torch.nn as nn
import numpy as np # for debugging 
import loss_odyssey
import focal_loss

class FCIoUV1(nn.Module):
    """
    Base implementation of the FCIoU. No Power Term 
    """

    def __init__(self): 

        super(FCIoUV1, self).__init__()
        self.eps = 1e-10 # epsilon value -> used to avoid dividing by zero

    def forward(self, pred, ann):
        """
            Params: \n
            ann (tensor) = annotation (b,h,w)\n
            pred (tensor) = logit output prediction (b,c,h,w) \n
        """

        # Retrieve the n# of classes based on the n# of classes within the prediction channel
        num_classes = pred.shape[1]

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
        # denom = torch.pow(pred,2) + ann_onehot - num # denominator
        denom = pred + ann_onehot - num # denominator

        del pred # free up memory 

        # sum up all the values on a per-class basis
        num = torch.sum(num, dim=(2,3))
        denom = torch.sum(denom, dim=(2,3)) 
        
        # get the IoU score for each class 
        loss_iou = torch.div(num, denom) 

        del num, denom 

        # fit the iou values to a more suitable weighted loss value
        loss_iou = -torch.log10(torch.pow(loss_iou + self.eps,0.5) + self.eps)

        # turn off contribution to loss by any classes not within the annotation file        
        num_active = 0
        for c in range(num_classes): 
            if c not in ann: 
                loss_iou[0,c] = self.eps
            else: 
                num_active += 1 # increase the count

        #sum up the final loss amount 
        loss_iou = torch.sum(loss_iou, dim=1)
        # Average the iou loss based on the currently active classes within the annotations
        loss_iou = loss_iou/num_active
        

        return loss_iou

class FCIoUV2(nn.Module):
    def __init__(self): 

        super(FCIoUV2, self).__init__()
        self.eps = 1e-10 # epsilon value -> used to avoid dividing by zero

    def forward(self, pred, ann):
        """
            Params: \n
            ann (tensor) = annotation (b,h,w)\n
            pred (tensor) = logit output prediction (b,c,h,w) \n
        """

        # Retrieve the n# of classes based on the n# of classes within the prediction channel
        num_classes = pred.shape[1]

        # Get the softmax output ([0,1]) for the prediction
        pred = torch.softmax(pred, dim=1) + self.eps

        # Create the onehot tensor for each class (gic)
        ann = torch.unsqueeze(ann,0) # add a leading 1 dimension for the tensor 
        ann = ann.long() # convert to required variable type
        ann_onehot = torch.zeros(pred.shape) # obtain blank array w/ same shape as the prediction
        ann_onehot = ann_onehot.cuda() # TODO update to use the device instead. 

        for i, _ in enumerate(ann_onehot): 
            ann_onehot[i].scatter_(0,ann[:,i],1)
        
        num = pred * ann_onehot # numerator
        denom = torch.pow(pred,2) + ann_onehot - num # denominator

        # sum up all the values on a per-class basis
        num = torch.sum(num, dim=(2,3))
        denom = torch.sum(denom, dim=(2,3)) 
        
        # get the IoU score for each class 
        loss_iou = 2*torch.div(num, denom) 

        # fit the iou values to a more suitable weighted loss value
        loss_iou = -torch.log10(torch.pow(loss_iou + self.eps,0.5) + self.eps)

        # turn off contribution to loss by any classes not within the annotation file        
        num_active = 0
        for c in range(num_classes): 
            if c not in ann: 
                loss_iou[0,c] = 0
            else: 
                num_active += 1 # increase the count

        #sum up the final loss amount 
        loss_iou = torch.sum(loss_iou, dim=1)
        # Average the iou loss based on the currently active classes within the annotations
        # loss_iou = loss_iou/num_active
        

        return loss_iou

    # BACKUP Implementation, keep to restore later
    # def forward(self, pred, ann):
    #     """
    #         Params: \n
    #         ann (tensor) = annotation (b,h,w)\n
    #         pred (tensor) = logit output prediction (b,c,h,w) \n
    #     """

    #     # Retrieve the n# of classes based on the n# of classes within the prediction channel
    #     num_classes = pred.shape[1]

    #     # Get the softmax output ([0,1]) for the prediction
    #     pred = torch.softmax(pred, dim=1) + self.eps

    #     # Create the onehot tensor for each class (gic)
    #     ann = torch.unsqueeze(ann,0)
    #     ann = ann.long() # convert to required variable type
    #     ann_onehot = torch.zeros(pred.shape) # obtain blank array w/ same shape as the prediction
    #     ann_onehot = ann_onehot.cuda() # TODO update to use the device instead. 
    #     ann_onehot.scatter_(1, ann, 1) # create onehot vector

    #     num = pred * ann_onehot # numerator
    #     denom = torch.pow(pred,2) + ann_onehot - num # denominator

    #     # sum up all the values on a per-class basis
    #     num = torch.sum(num, dim=(2,3))
    #     denom = torch.sum(denom, dim=(2,3)) 
        
    #     # get the IoU score for each class 
    #     loss_iou = 2*torch.div(num, denom) 

    #     # fit the iou values to a more suitable weighted loss value
    #     loss_iou = -torch.log10(torch.pow(loss_iou + self.eps,0.5) + self.eps)

    #     # turn off contribution to loss by any classes not within the annotation file        
    #     num_active = 0
    #     for c in range(num_classes): 
    #         if c not in ann: 
    #             loss_iou[0,c] = 0
    #         else: 
    #             num_active += 1 # increase the count

    #     #sum up the final loss amount 
    #     loss_iou = torch.sum(loss_iou, dim=1)
    #     # Average the iou loss based on the currently active classes within the annotations
    #     # loss_iou = loss_iou/num_active
        

    #     return loss_iou
    
class FCIoUV3(nn.Module):
    def __init__(self): 

        super(FCIoUV3, self).__init__()
        self.eps = 1e-10 # epsilon value -> used to avoid dividing by zero

    def forward(self, pred, ann):
        """
            Params: \n
            ann (tensor) = annotation (b,h,w)\n
            pred (tensor) = logit output prediction (b,c,h,w) \n
        """

        # Retrieve the n# of classes based on the n# of classes within the prediction channel
        num_classes = pred.shape[1]

        # Get the softmax output ([0,1]) for the prediction
        pred = torch.softmax(pred, dim=1) + self.eps

        # Create the onehot tensor for each class (gic)
        ann = torch.unsqueeze(ann,0)
        ann = ann.long() # convert to required variable type
        ann_onehot = torch.zeros(pred.shape) # obtain blank array w/ same shape as the prediction
        ann_onehot = ann_onehot.cuda() # TODO update to use the device instead. 
        ann_onehot.scatter_(1, ann, 1) # create onehot vector

        num = pred * ann_onehot # numerator
        denom = torch.pow(pred,2) + ann_onehot - num # denominator

        # sum up all the values on a per-class basis
        num = torch.sum(num, dim=(2,3))
        denom = torch.sum(denom, dim=(2,3)) 
        
        # get the IoU score for each class 
        loss_iou = torch.div(num, denom) 

        # fit the iou values to a more suitable weighted loss value
        loss_iou = -torch.log10(torch.pow(loss_iou + self.eps,0.5) + self.eps)

        # turn off contribution to loss by any classes not within the annotation file        
        num_active = 0
        for c in range(num_classes): 
            if c not in ann: 
                loss_iou[0,c] = 0
            else: 
                num_active += 1 # increase the count

        #sum up the final loss amount 
        loss_iou = torch.sum(loss_iou, dim=1)
        # Average the iou loss based on the currently active classes within the annotations
        # loss_iou = loss_iou/num_active
        

        return loss_iou    
    
class FCIoUV4(nn.Module):
    def __init__(self): 

        super(FCIoUV4, self).__init__()
        self.eps = 1e-10 # epsilon value -> used to avoid dividing by zero

    def forward(self, pred, ann):
        """
            Params: \n
            ann (tensor) = annotation (b,h,w)\n
            pred (tensor) = logit output prediction (b,c,h,w) \n
        """

        # Retrieve the n# of classes based on the n# of classes within the prediction channel
        num_classes = pred.shape[1]

        # Get the softmax output ([0,1]) for the prediction
        pred = torch.softmax(pred, dim=1) + self.eps

        # Create the onehot tensor for each class (gic)
        ann = torch.unsqueeze(ann,0)
        ann = ann.long() # convert to required variable type
        ann_onehot = torch.zeros(pred.shape) # obtain blank array w/ same shape as the prediction
        ann_onehot = ann_onehot.cuda() # TODO update to use the device instead. 
        ann_onehot.scatter_(1, ann, 1) # create onehot vector

        num = pred * ann_onehot # numerator
        denom = torch.pow(pred,2) + ann_onehot - num # denominator

        # sum up all the values on a per-class basis
        num = torch.sum(num, dim=(2,3))
        denom = torch.sum(denom, dim=(2,3)) 
        
        # get the IoU score for each class 
        loss_iou = torch.div(num, denom) 

        # fit the iou values to a more suitable weighted loss value
        loss_iou = -torch.log10(torch.pow(10*loss_iou + self.eps,0.5) + self.eps) + 0.51

        # turn off contribution to loss by any classes not within the annotation file        
        num_active = 0
        for c in range(num_classes): 
            if c not in ann: 
                loss_iou[0,c] = 0
            else: 
                num_active += 1 # increase the count

        #sum up the final loss amount 
        loss_iou = torch.sum(loss_iou, dim=1)
        # Average the iou loss based on the currently active classes within the annotations
        # loss_iou = loss_iou/num_active
        

        return loss_iou    

class FCIoUV5(nn.Module):
    def __init__(self): 

        super(FCIoUV5, self).__init__()
        self.eps = 1e-10 # epsilon value -> used to avoid dividing by zero

    def forward(self, pred, ann):
        """
            Params: \n
            ann (tensor) = annotation (b,h,w)\n
            pred (tensor) = logit output prediction (b,c,h,w) \n
        """

        # Retrieve the n# of classes based on the n# of classes within the prediction channel
        num_classes = pred.shape[1]

        # Get the softmax output ([0,1]) for the prediction
        pred = torch.softmax(pred, dim=1) + self.eps

        # Create the onehot tensor for each class (gic)
        ann = torch.unsqueeze(ann,0)
        ann = ann.long() # convert to required variable type
        ann_onehot = torch.zeros(pred.shape) # obtain blank array w/ same shape as the prediction
        ann_onehot = ann_onehot.cuda() # TODO update to use the device instead. 
        ann_onehot.scatter_(1, ann, 1) # create onehot vector

        num = pred * ann_onehot # numerator
        denom = torch.pow(pred,2) + ann_onehot - num # denominator

        # sum up all the values on a per-class basis
        num = torch.sum(num, dim=(2,3))
        denom = torch.sum(denom, dim=(2,3)) 
        
        # get the IoU score for each class 
        loss_iou = torch.div(num, denom) 

        # fit the iou values to a more suitable weighted loss value
        loss_iou = -torch.log10(torch.pow(loss_iou + self.eps, 2) + self.eps)

        # turn off contribution to loss by any classes not present in the annotation image    
        num_active = 0
        for c in range(num_classes): 
            if c not in ann: 
                loss_iou[0,c] = 0
            else: 
                num_active += 1 # increase the count

        #sum up the final loss amount 
        loss_iou = torch.sum(loss_iou, dim=1)


        return loss_iou    

class FCIoUV6(nn.Module):
    def __init__(self): 

        super(FCIoUV6, self).__init__()
        self.eps = 1e-10 # epsilon value -> used to avoid dividing by zero

    def forward(self, pred, ann):
        """
            Params: \n
            ann (tensor) = annotation (b,h,w)\n
            pred (tensor) = logit output prediction (b,c,h,w) \n
        """

        # Retrieve the n# of classes based on the n# of classes within the prediction channel
        num_classes = pred.shape[1]

        # Get the softmax output ([0,1]) for the prediction
        pred = torch.softmax(pred, dim=1) + self.eps

        # Create the onehot tensor for each class (gic)
        ann = torch.unsqueeze(ann,0)
        ann = ann.long() # convert to required variable type
        ann_onehot = torch.zeros(pred.shape) # obtain blank array w/ same shape as the prediction
        ann_onehot = ann_onehot.cuda() # TODO update to use the device instead. 
        ann_onehot.scatter_(1, ann, 1) # create onehot vector

        num = pred * ann_onehot # numerator
        denom = torch.pow(pred,2) + ann_onehot - num # denominator

        # sum up all the values on a per-class basis
        num = torch.sum(num, dim=(2,3))
        denom = torch.sum(denom, dim=(2,3)) 
        
        # get the IoU score for each class 
        loss_iou = torch.div(num, denom) 

        # fit the iou values to a more suitable weighted loss value
        loss_iou = -torch.log10(loss_iou + self.eps) 

        # turn off contribution to loss by any classes not present in the annotation image    
        num_active = 0
        for c in range(num_classes): 
            if c not in ann: 
                loss_iou[0,c] = 0
            else: 
                num_active += 1 # increase the count

        #sum up the final loss amount 
        loss_iou = torch.sum(loss_iou, dim=1)


        return loss_iou   

class PowerJaccard(nn.Module):
    def __init__(self): 

        super(PowerJaccard, self).__init__()
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

        num = pred * ann_onehot  # numerator
        denom = torch.pow(pred,2) + torch.pow(ann_onehot,2) - num + self.eps # denominator

        # sum up all the values on a per-class basis
        num = torch.sum(num, dim=(1,2,3))
        denom = torch.sum(denom, dim=(1,2,3)) 
        
        # get the IoU score for each class 
        loss = 1 - torch.div(num, denom) 

        return loss

#Jaccard index (Intersection over Union) Loss Function Implementation
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()
        self.eps = 1e-10 # epsilon value -> used to avoid dividing by zero

    def forward(self, pred, ann):

        # Get the predictive output for the prediction
        pred = torch.softmax(pred, dim = 1) + self.eps

        # Create the onehot tensor for each class (gic)
        ann = torch.unsqueeze(ann,0)
        ann = ann.long() # convert to required variable type
        ann_onehot = torch.zeros(pred.shape) # obtain blank array w/ same shape as the prediction
        ann_onehot = ann_onehot.cuda() # TODO update to use the device instead. 
        ann_onehot.scatter_(1, ann, 1) # create onehot vector

        # Obtain the numerator and denominator for the IoU
        num = pred * ann_onehot
        denom = pred + ann_onehot - num

        # Calculate the IoU score by summing the num and denom separately (see equation)
        num = torch.sum(num, dim=(1,2,3))
        denom = torch.sum(denom, dim=(1,2,3))
        IoU = num/denom


        return 1 - IoU

class TopKLoss(nn.Module):
    def __init__(self, k = 10):
        super(TopKLoss, self).__init__()
        self.eps = 1e-10 # epsilon value -> used to avoid dividing by zero
        self.k = k # % of max value samples that are chosen 


    def forward(self, pred, ann):

        # Get the predictive output for the prediction
        pred = torch.softmax(pred, dim = 1) + self.eps

        # Create the onehot tensor for each class (gic)
        ann = torch.unsqueeze(ann,0)
        ann = ann.long() # convert to required variable type
        ann_onehot = torch.zeros(pred.shape) # obtain blank array w/ same shape as the prediction
        ann_onehot = ann_onehot.cuda() # TODO update to use the device instead. 
        ann_onehot.scatter_(1, ann, 1) # create onehot vector

        # Get the CE loss value 
        loss = ann_onehot * torch.log(pred)
        # flatten the loss function to format to one dimension 
        loss = loss.flatten()
        
        # Obtain the number of pixels that are to be kept with the Topk function
        n = int(loss.shape[0]*self.k/100)
        # Obtain the values with the largest amount of loss 
        loss = torch.topk(loss,k=n, largest=False).values

        # apply normalization and calculate loss 
        loss = -1/n * torch.sum(loss)



        return loss

class DiceFocal(nn.Module): 
    def __init__(self): 
        super(DiceFocal, self).__init__()
        self.eps = 1e-10

        self.dice = DiceLoss()
        self.focal = focal_loss.FocalLoss(gamma = 2.0)

    def forward(self, pred, ann): 
        # Re-format the input into probability value 
        pred = torch.softmax(pred, dim = 1) + self.eps
        # Return the sum of both loss functions 
        return self.dice(pred,ann) + self.focal(pred,ann)
    
class DiceLoss(nn.Module): 
    def __init__(self): 
        super(DiceLoss, self).__init__()
        self.eps = 1e-10

    def forward(self, pred, ann):     
        # Obtain the prediction values for the outputs 
        pred = torch.softmax(pred, dim=1)

        # Create the onehot tensor for each class (gic)
        ann = torch.unsqueeze(ann,0)
        ann = ann.long() # convert to required variable type
        ann_onehot = torch.zeros(pred.shape) # obtain blank array w/ same shape as the prediction
        ann_onehot = ann_onehot.cuda() # TODO update to use the device instead. 
        ann_onehot.scatter_(1, ann, 1) # create onehot vector
        
        # Calculate the dice loss 
        num = 2 * torch.sum(pred*ann_onehot, dim=(1,2,3))
        denom = torch.sum(ann_onehot, dim=(1,2,3)) + torch.sum(pred, dim=(1,2,3)) + self.eps

        return 1 - num/denom

class TverskyLoss(nn.Module): 
    def __init__(self, alpha = 0.3, beta=0.7): 
        super(TverskyLoss, self).__init__()
        self.eps = 1e-10
        self.alpha = alpha
        self.beta = beta


    def forward(self, pred, ann):     
        # Obtain the prediction values for the outputs 
        pred = torch.softmax(pred, dim=1)

        # Create the onehot tensor for each class (gic)
        ann = torch.unsqueeze(ann,0)
        ann = ann.long() # convert to required variable type
        ann_onehot = torch.zeros(pred.shape) # obtain blank array w/ same shape as the prediction
        ann_onehot = ann_onehot.cuda() # TODO update to use the device instead. 
        ann_onehot.scatter_(1, ann, 1) # create onehot vector
        
        # Calculate the numerator and denominator for the ratio
        num = torch.sum(pred*ann_onehot, dim=(1,2,3))
        denom = self.alpha*torch.sum((1-ann_onehot)*pred,dim =(1,2,3)) + self.beta*torch.sum(ann_onehot*(1-pred))

        return 1 - num/denom  

class DiceTopk(nn.Module): 
    def __init__(self, k=10): 
        """
            k = (k) percent value of samples kept for the training of the model in Topk Loss Function
        """
        super(DiceTopk, self).__init__()
        self.dice = DiceLoss()
        self.topk = TopKLoss(torch.tensor(k))
    
    def forward(self, pred, ann): 
        # obtain the prediction values for the output 
        pred = torch.softmax(pred, dim=1)
        
        dice_loss = self.dice(pred, ann)

        topk_loss = self.topk(pred, ann)

        return dice_loss + topk_loss



# For testing out the loss function first 
if __name__ == "__main__": 
    # criterion = FCIoUV1()
    pred = torch.rand(size=(1,35,1200,1920)).cuda()
    ann = torch.randint(0,34,size = (1,1200,1920)).cuda()
    # handle_lossContribution(pred, ann)
    # loss = criterion(pred, ann)
    