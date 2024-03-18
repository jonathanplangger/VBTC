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
# from figures.figures import FigPredictionCertainty

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
    def __init__(self, cfg_path = "configs/config_comparative_study.yaml", model_num = None):

        # Load and update the configuration file. Serves as the main point of configuration for the testing. 
        self.cfg = get_cfg_defaults()
        self.cfg.merge_from_file(cfg_path)
        # Load the specified model instead.
        if model_num is not None: 
            self.cfg.EVAL.MODEL_FILE = "../models/1_Winter2024_TrainingResults/{}/model.pt".format(str(model_num).zfill(3))
        # Set up the device where the program is going to be run -> gpu if available
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.db = dataloader.get_dataloader(self.cfg, setType="train")
    

    def eval(self):

        # Dataloader initialization
        db = self.db # hold local version of the dataloader
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
        log_iou = torch.empty([NUM_TEST+1, self.db.num_classes]) # same but for IoU

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

                    if self.cfg.DB.DB_NAME == "rellis":
                        # Re-map back to the 0-35 scheme before displaying the output 
                        pred = db.map_labels(pred, True)
                        ann = db.map_labels(ann,True)

                    # move the masks and the image onto the CPU
                    images = orig_images # restore original images
                    del orig_images # no longer needed

                    # place the annotations back onto the cpu
                    ann = ann.to('cpu')
                    # Create the output plot
                    fig, axs = plt.subplots(ncols=4, squeeze=False, gridspec_kw = {'wspace':0.05, 'hspace':0})

                    # Base image
                    # img = F.to_pil_image(images[0])
                    axs[0, 0].imshow(images[0].astype(int))
                    axs[0, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[], title="Base Image" )

                    # obtain the blended segmentation image 
                    img = self.img_mask_blend(pred=pred, raw_img = torch.tensor(images))
                    axs[0, 1].imshow(img)
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
        df_log = df_log.rename(columns = db.class_labels)
        df_log.to_csv("figures/ComparativeStudyResults/LoggedResults/{}_LoggedResults.csv".format(model_num), index = False)
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

    # Complete the eval process for only a single image. Used for generating figure output
    def single_img_pred(self, idx = 0, model_num = None):
        """(function) single_img_pred\n
        Complete the prediction and output for a single image. Employs the preset configuration for eval.py to determine the correct model settings.\n
        The specific model number examined can be updated programmatically through updating the model_num parameter. This is provided to allow for the obtainment of outputs for a set of the same models. 
        Make sure that the model_num matches the model and dataset configured before usage. Or else the resulting model implemented will be wrong or not work at all.\n

        :param idx: Index value for the specific image input to the model. Ensures the consistency in image employed, defaults to 0
        :type idx: int, optional
        :param model_num: Number of the model evaluated. Please ensure that the number is valid and corresponds to the model/db configured in the configuration file., defaults to 41
        :type model_num: int, optional
        :return: (pred, ann, orig_images) : returns 3 outputs concerning the labelled prediction and annotation images as well as the original input image to the network. 
        :rtype: torch.tensor() : for all 3 return values
        """        


        if model_num is not None: # if the input has been modified, use the specified model
            model_path =  "../models/1_Winter2024_TrainingResults/{}/model.pt".format(str(model_num).zfill(3)) # use the model number in determining the path
            self.cfg.EVAL.MODEL_FILE = model_path # update the configuration to use this other path
    
        # Load the model based on the configuration file settings
        model_handler = modelhandler.ModelHandler(self.cfg, "eval")# load the model handler 
        model = model_handler.load_model()
        model.eval()
        model.to(self.device)#  place on GPU

        # if the image size needs to be updated
        if self.cfg.EVAL.INPUT_SIZE.RESIZE_IMG:
        # new img size set by config file
            input_size = (self.cfg.EVAL.INPUT_SIZE.HEIGHT, self.cfg.EVAL.INPUT_SIZE.WIDTH) 
        else: 
            input_size = (self.db.height, self.db.width) # use the original input size values instead.

        # get the images used for prediction
        orig_images, images, ann, _ = self.db.load_batch(idx, batch_size = 1, resize = input_size)

        # Place the images onto the GPU
        images = images.cuda()
        ann = ann.cuda()

        with torch.no_grad(): 
            pred = model(images)

        # Handle the output based on the specific model configured
        pred = model_handler.handle_output(pred) 
        
        # Update format for ease of handling in other functions
        pred = pred.cpu()[0]
        ann = ann.cpu()[0]
        orig_images = torch.tensor(orig_images)[0].long() # convert everything to torch tensors before sending off

        return pred, ann, orig_images
    
    def img_mask_blend(self, pred, raw_img): 
        """(function) img_mask_blend
        Output a merged representation of the annotated image and the raw input image. This allows for the segmentation to be easily
        viewed on top of the original input image background. 

        :param pred: Segmented (labelled) image based on each class shown. Dimension of (1xHxW) with values ranging from [0,C] classes expected. 
        :type pred: torch.tensor
        :param raw_img: Raw, original input image of dimensions (3xHxW) w/ RGB encoding. Used as the background for the merged image
        :type raw_img: torch.tensor
        """
        masks = torch.zeros(self.cfg.DB.NUM_CLASSES, self.cfg.DB.IMG_SIZE.HEIGHT , self.cfg.DB.IMG_SIZE.WIDTH, device=self.device, dtype=torch.bool)

        if self.cfg.DB.DB_NAME == "rellis":
            # Re-map back to the 0-35 scheme before displaying the output 
            pred = self.db.map_labels(pred, True)

        # obtain a mask for each class
        for classID in range(masks.shape[0]): 
            masks[classID] = (pred == classID)

        # move the masks and the image onto the CPU
        masks = masks.to('cpu')

        # convert to boolean 
        masks = masks.to(torch.bool) 

        seg_img = draw_segmentation_masks(raw_img[0].permute(2,0,1).to(torch.uint8), masks, alpha=0.7, colors=self.db.get_colors())
        img = F.to_pil_image(seg_img.detach())
        blend = np.asarray(img) # get the blended image

        # Return the blended image
        return blend


    def cvt_color(self, sem_img):
        """(function) cvt_color\n
        Convert the output labelled semantic segmentationn mask into a coloured version of it. This allows for the effective output to be displayed in colour rather than values of range [0,C].\n

        :param sem_img: Semantically labelled image containing values [0,C] (C = number of classes). Expected single input of dimensions [1 x H x W]
        :type sem_img: torch.tensor
        """
        # Get the colors mapping for the model
        # colors = self.db.get_colors(remap_labels=True)[1:]
        colors = self.db.get_colors(remap_labels=True)
        input_shape = sem_img.shape # get the shape of the input 

        # storage element for the mapped color elements
        color_img = torch.zeros(3,input_shape[0], input_shape[1])
        zeros = torch.zeros(color_img.shape)

        for c, color in enumerate(colors): 

            # Obtain the pixel applicability mask
            mask = (sem_img == c)
            temp = [(zeros[i] + color[i]) for i,_ in enumerate(color)]
            color_img = color_img + torch.stack(temp) * mask
            

        # Merge the R,G,B colour channels into one tensor
        # color_img = torch.cat(color_img)
        
        #Return the tensor w/ the matplotlib image format to the user
        return color_img
    




if __name__ == "__main__": 
    
    # # Run the evaluation program
    # eval = ComparativeEvaluation()
    # eval.eval( )

    ##### Iteratively run eval.py through all of the desired combinations -> this allows for AFK running of eval.py
    for i in range(1, 6): 

        if i < 6: # Unet & Rellis 
            cfg_path = "configs/001_005_config.yaml"
        elif i  < 11: # HRNet
            cfg_path = "configs/006_010_config.yaml"
        elif i < 16: # Deeplabv3plus
            cfg_path = "configs/011_015_config.yaml"
        
        eval = ComparativeEvaluation(cfg_path, model_num = i)
        eval.eval()
    