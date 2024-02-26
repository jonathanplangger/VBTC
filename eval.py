import os 
# Preset environment variables 
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # ONLY FOR DEBUG -  Allows for the stacktrace to be effectively used
# To run trace: 
# python -m cProfile -o /tmp/tmp.prof eval.py
# snakeviz /tmp/tmp.prof

import sys
import torch 
import dataloader
from torchvision.utils import draw_segmentation_masks
torch.cuda.empty_cache() # liberate the resources 
import numpy as np
import matplotlib.pyplot as plt
# read yaml file 
import yaml
import torchmetrics
from tqdm import tqdm
import time 
import argparse
from sklearn import metrics # for the confusion matrix
from figures.figures import FigPredictionCertainty

# Display masks
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torch.nn import functional as TF
import torchinfo
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

from config import get_cfg_defaults
import modelhandler

EPS = 1e-10 # used to avoid dividing by zero. 

class ComparativeEvaluation():

    # TODO - Update the code to work in OO fashion 
    def __init__(self):

        # Load and update the configuration file. Serves as the main point of configuration for the testing. 
        self.cfg = get_cfg_defaults()
        self.cfg.merge_from_file("configs/config_comparative_study.yaml")

        # Set up the device where the program is going to be run -> gpu if available
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    

    def eval(self):

        # Dataloader initialization
        self.db = db = dataloader.get_dataloader(self.cfg, setType="train")
        db.randomizeOrder()

        # Obtain the model handler -> used to generate model-specific features 
        model_handler = modelhandler.ModelHandler(self.cfg, "eval")

        writer = SummaryWriter() 

        # Empty the cache prior to training the network
        torch.cuda.empty_cache()

        TOTAL_NUM_TEST = len(db.test_meta)
        NUM_TEST = TOTAL_NUM_TEST
        # NUM_TEST = 100 # for quickly testing the eval program
        
        colors = db.get_colors()

        # Get model and place on GPU. 
        model = model_handler.load_model()
        model.eval()
        model.to(self.device)

        # if the image size needs to be updated
        if self.cfg.EVAL.INPUT_SIZE.RESIZE_IMG:
        # new img size set by config file
            input_size = (self.cfg.EVAL.INPUT_SIZE.HEIGHT, self.cfg.EVAL.INPUT_SIZE.WIDTH) 
        else: 
            input_size = (db.height, db.width) # use the original input size values instead.

        # Get the model summary information
        torchinfo.summary(model, input_size=(1, 3, input_size[0], input_size[1]))

        # Use the Dice score as the performance metric to assess model accuracy
        dice = torchmetrics.Dice().to(self.device)

        # Intersection over Union metric calculator 
        iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=db.num_classes, average='none')

        # Used to tally the final mean intersection over union score
        iou_c = torch.zeros(db.num_classes).to(self.device)
        # Count the n# of instances with this class in it
        count_c = torch.zeros(db.num_classes).to(self.device)


        from sklearn.metrics import confusion_matrix

        mean_dice = torch.tensor(0.).cuda() # used to calculate the mean Dice value for the model
        # init loggers for model time measurement
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings = torch.zeros((NUM_TEST+1, 1)) # prep array to store all the values for time

        # Create blank master confusion matrix
        master_conf_matrix = np.zeros((db.num_classes, db.num_classes))

        # Dummy input tensors employed during the warm-up cycle for the model
        dummy_input = torch.randn(1,3,input_size[0], input_size[1], dtype=torch.float).cuda()

        # Warm-up the execution of the GPU by running a few cycles prior to time measurements
        for _ in range(10):
            _ = model(dummy_input)

        del dummy_input # no longer required for training (free memory)

        log_dice = torch.empty(0) # used to log all values for the dice loss obtained
        log_iou = torch.empty([NUM_TEST+1,25]) # same but for IoU

        with tqdm(total = TOTAL_NUM_TEST, unit = "Batch") as pbar:
        # iterate through all tests while using batches 
            for idx in range(0, NUM_TEST, self.cfg.EVAL.BATCH_SIZE): 

                # ------------ Run the model with the loaded images  ---------------- #
                # load the batch from the dataset
                orig_images, images, ann, idx = db.load_batch(idx, self.cfg.EVAL.BATCH_SIZE, resize=input_size) # load images
                
                # Place the images and annotations on the GPU.
                images = images.cuda()
                ann = ann.cuda()
              
                startTime = time.time() # used to measure prediction time for the model

                # run model
                with torch.no_grad(): # do not calculate gradients for this task
                    starter.record() # measure the starting time
                    pred = model(images)
                    ender.record() # measure the finish time

                torch.cuda.synchronize() # re-synchronize the model prior to continuing (for time meas.)
                timings[idx] = starter.elapsed_time(ender) # measure the total time taken

                # save the old version of pred (before argmax)
                if self.cfg.EVAL.PRED_CERTAINTY: # if enabled in config 
                    pred_raw = pred 

                # Convert the prediction output to argmax labels representing each class predicted
                pred = model_handler.handle_output(pred)

                # Get the iou score for each of the classes individually
                iou_score = iou(pred.cpu(), ann.long().cpu()).to(self.device)

                # # Measure the performance of the model
                dice_score = dice(pred, ann.long())

                # Log the values for storing later
                log_iou[idx] = iou_score.cpu() # store the values onto the tensor
                log_dice = torch.cat((log_dice, dice_score.cpu().unsqueeze(0)), dim=0)
                mean_dice += dice_score

                # Calculate the confusion matrix for this figure.                
                conf_pred = pred.cpu().numpy().flatten()
                conf_ann = ann.cpu().numpy().flatten()
                # Sum the new confusion matrix results together
                master_conf_matrix = master_conf_matrix + metrics.confusion_matrix(conf_ann, conf_pred, labels = range(0,db.num_classes))


                # ----------- Plot the Results ----------------------- #
                unique_classes = ann.unique() # get unique classes that are represented in the annotation

                # for each unique class
                for c in unique_classes: 
                    c = int(c.item()) # convert to int number 
                    iou_c[c] += iou_score[c] #add to the final sum count for IoU
                    count_c[c] += 1 # increment the final count 

                # Log the results of the evaluation onto tensorboard
                if not self.cfg.EVAL.DISPLAY_IMAGE:
                    writer.add_scalar("Metrics/Dice", dice_score.item(), idx) # record the dice score 
                    writer.add_scalar("Metrics/Time", time.time() - startTime, idx)

                # -------------- Show the image output for the segmentation  ---------- #
                if self.cfg.EVAL.DISPLAY_IMAGE:     

                    # create a blank array
                    masks = torch.zeros(self.cfg.DB.NUM_CLASSES, self.cfg.DB.IMG_SIZE.HEIGHT , self.cfg.DB.IMG_SIZE.WIDTH, device=self.device, dtype=torch.bool)

                    if self.cfg.DB.DB_NAME == "rellis":
                        # Re-map back to the 0-35 scheme before displaying the output 
                        pred = db.map_labels(pred, True)
                        ann = db.map_labels(ann,True)

                    # obtain a mask for each class
                    for classID in range(masks.shape[0]): 
                        masks[classID] = (pred == classID)

                    # move the masks and the image onto the CPU
                    images = orig_images # restore original images
                    masks = masks.to('cpu')
                    del orig_images # no longer needed

                    # convert to boolean 
                    masks = masks.to(torch.bool) 

                    # place the annotations back onto the cpu
                    ann = ann.to('cpu')
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
                    axs[0, 2].imshow(ann[0], cmap='gray')
                    axs[0, 2].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], title="Ground Truth Annotations")

                    # Output mask
                    axs[0, 3].imshow(pred.cpu().detach().numpy()[0], cmap='gray')
                    axs[0, 3].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], title="Output Mask")

                    plt.show()

                # display the figure 
                if self.cfg.EVAL.PRED_CERTAINTY: 
                    FigPredictionCertainty(torch.softmax(pred_raw, dim=1), pred, ann, class_labels = db.class_labels, color_map=colors)
                    # FigPredictionCertainty(torch.nn.functional.normalize(pred_raw, dim=1), pred, ann, class_labels = db.class_labels, color_map=colors)



                # Update the progress bar
                pbar.set_postfix(score = dice_score.item())
                pbar.update()

                # Clear memory
                del pred 
                del images 

                # end the testing if the desired # of tests has been obtained
                if idx == NUM_TEST: 
                    break

        

        ### Handle the Confusion Matrix for the Dataset ###
        # Save the output confusion matrix for the dataset
        model_num = db.cfg.EVAL.MODEL_FILE.split("/")[-2] # get the number for the model being evaluated
        fp = "figures/ConfusionMatrix/{}_ConfusionMatrix.csv".format(model_num) # update the file name based on the value
        np.savetxt(fp, master_conf_matrix, delimiter = ",")        

        ### Handle the logging for dice and iou scores
        # Store values on dataframes
        df_log = pd.DataFrame(log_iou[1:].numpy())
        df_log["dice"] = log_dice.numpy()
        df_log.to_csv("figures/ComparativeStudyResults/LoggedResults/{}_LoggedResults.csv".format(model_num))
        df_log.rename(columns = db.class_labels)

        # Get the mean value 
        mean_dice = mean_dice / NUM_TEST           

        # After the completion of obtaining the metric
        iou_c = (iou_c/(count_c+EPS)).cpu() # obtain the average amount for each class 

        table = {}

            # Obtain the class label strings & update to merge "void" and "dirt" classes as done in the model
        class_labels = list(db.class_labels.values())

        # ONLY perform these steps if the rellis dataset is being used here
        if self.cfg.DB.DB_NAME == 'rellis':         
            class_labels.remove("dirt")
            class_labels.remove("void")
            class_labels.insert(0, "void & dirt")

        # Re-format and convert the IoU Object to a table entry
        for c, _ in enumerate(iou_c): 
            table[class_labels[c]] =  round(float(iou_c[c]),4)
        

        # Calculate and output the mean and std dev time.
        mean_time = torch.sum(timings)/NUM_TEST # get the mean time 
        std_time = torch.std(timings)/NUM_TEST
        print("Mean Execution Time: {:.4f} ms".format(mean_time))
        print("Std. Dev. Execution Time: {:.4f} ms".format(std_time))

        print("Mean Dice Value Obtained: {:.4f}".format(mean_dice))
        print(table) # print the raw prediction values for the table
        print(list(table.values()))
        print("End of Evaluation Program")  



if __name__ == "__main__": 
    
    # Run the evaluation program
    eval = ComparativeEvaluation()
    eval.eval( )




    