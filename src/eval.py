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
from torch.nn import functional as TF

from torch.utils.tensorboard import SummaryWriter

from config_eval import get_cfg_defaults




class ComparativeEvaluation():

    # TODO - Update the code to work in OO fashion 
    def __init__(self):

        # Load and update the configuration file. Serves as the main point of configuration for the testing. 
        self.cfg = get_cfg_defaults()
        self.cfg.merge_from_file("configs/config_comparative_study.yaml")
        self.cfg.freeze()

        # Set up the device where the program is going to be run -> gpu if available
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


        pass

    def __load_model(self):
        """
            __load_model(self):
            ------------------------------------------------------------
            Returns the model based on the configuration selected.
            "self.model" is employed as the deciding factor for the model being used. 

        """
        if self.cfg.EVAL.MODEL_NAME == "unet":
            # load the model to the device 
            model = torch.load('model.pt')
        elif self.cfg.EVAL.MODEL_NAME == "hrnet_ocr":
            # model source code directory
            src_dir = self.cfg.MODELS.HRNET_OCR.SRC_DIR
            # add the source code tools directory to re-use their code
            sys.path.insert(0,os.path.join(src_dir,"tools/"))
            sys.path.insert(0,os.path.join(src_dir,"lib/"))
            from test import load_model
            model = load_model("/home/jplangger/Documents/Dev/VBTC/src/models/HRNet-Semantic-Segmentation-HRNet-OCR/experiments/rellis/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml", 
                               model_file = self.cfg.MODELS.HRNET_OCR.MODEL_FILE)
        else: 
            print("\n\nInvalid model name, please update the configuration file.")
            print("Exiting Program.... ")
            exit()

        return model

    def __handle_output(self, pred, db): 
        """
            __handle_output(self,pred,db): 
            -------------------------------
            Convert the output obtained from the prediction to a set of labeled annotations to be later compared to the annotation labels. 
            This function takes account of the specific structure of the model employed and will convert the output accordingly. 
            --------------------------------
            Inputs: 
            pred (tensor): Prediction tensors of logit format directly obtained from the model output 
            db (dataloader.DataLoader): dataloader for the given dataset. 
        """
        if self.cfg.EVAL.MODEL_NAME == "unet": 
            pred = pred.argmax(dim=1)
        elif self.cfg.EVAL.MODEL_NAME == "hrnet_ocr": 
            pred = pred[0] # hrnet has 2 outputs, whilst only one is used... 
            # Use the same interpolation scheme as is used in the source code.
            pred = TF.interpolate(input=pred, size=(db.height, db.width), mode='bilinear', align_corners=False)
            pred = pred.argmax(dim=1) # obtain the predictions for each layer

            pred = db.map_labels(label=pred, inverse = True) # convert to 0->34

        return pred

    def eval(self):

        writer = SummaryWriter() 

        # obtain the data
        db = dataloader.DataLoader(setType="test") 
        db.randomizeOrder()

        # Empty the cache prior to training the network
        torch.cuda.empty_cache()

        TOTAL_NUM_TEST = len(db.test_meta)
        NUM_TEST = TOTAL_NUM_TEST

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

        model = self.__load_model() # obtain the model based on config

        model.eval()
        model.to(self.device)

        # if the image size needs to be updated
        if self.cfg.EVAL.INPUT_SIZE.RESIZE_IMG:
        # new img size set by config file
            input_size = (self.cfg.EVAL.INPUT_SIZE.HEIGHT, self.cfg.EVAL.INPUT_SIZE.WIDTH) 
        else: 
            input_size = (db.height, db.width) # use the original input size values instead.

        # Obtain the summary of the model architecture + memory requirements
        torchsummary.summary(model, (3,input_size[0],input_size[1]))

        # Use the Dice score as the performance metric to assess model accuracy
        dice = torchmetrics.Dice().to(self.device)

        # Intersection over Union metric calculator 
        iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=db.num_classes, average='none')

        # Used to tally the final mean intersection over union score
        miou = torch.zeros(db.num_classes).to(self.device)
        # Count the n# of instances with this class in it
        count_c = torch.zeros(db.num_classes).to(self.device)

        # Create a blank confusion matrix to be input into 
        confusionMatrix = torch.empty([TOTAL_NUM_TEST,db.num_classes, db.num_classes])

        from sklearn.metrics import confusion_matrix


        with tqdm(total = TOTAL_NUM_TEST, unit = "Batch") as pbar:
        # iterate through all tests while using batches 
            for idx in range(0, NUM_TEST, self.cfg.EVAL.BATCH_SIZE): 

                # ------------ Run the model with the loaded images  ---------------- #
                # load the batch from the dataset
                orig_images, images, ann, idx = db.load_batch(idx, self.cfg.EVAL.BATCH_SIZE, resize=input_size) # load images
                
                # orig_images = images # store this for later use
                orig_ann = ann # store for later display

                # prep images and load to GPU
                images = ((torch.from_numpy(images)).to(torch.float32).permute(0,3,1,2)/255.0).to(self.device)

                ann = (torch.from_numpy(ann)).to(torch.float32).permute(0,3,1,2)[:,0,:,:].to(self.device)

                
                startTime = time.time() # used to measure prediction time for the model

                # run model
                with torch.no_grad(): # do not calculate gradients for this task
                    pred = model(images)

                # Convert the prediction output to argmax labels representing each class predicted
                pred = self.__handle_output(pred, db)

                # Measure the performance of the model
                dice_score = dice(pred, ann.long())
                

                # ----------- Plot the Results ----------------------- #
                # create a blank array
                masks = torch.zeros(35,1200,1920, device=self.device, dtype=torch.bool)

                # obtain a mask for each class
                for classID in range(masks.shape[0]): 
                    masks[classID] = (pred == classID)

                # move the masks and the image onto the CPU
                images = orig_images # restore original images
                masks = masks.to('cpu')
                del orig_images # no longer needed

                # convert the image to uint8 type 
                # images = images.to(torch.uint8)
                masks = masks.to(torch.bool) # convert to boolean 

                # Get the iou score for each of the classes individually
                iou_score = iou(pred.cpu(), ann.long().cpu()).to(self.device)

                unique_classes = ann.unique() # get unique classes that are represented in the annotation

                # for each unique class
                for c in unique_classes: 
                    c = int(c.item()) # convert to int number 
                    miou[c] += iou_score[c] #add to the final sum count for IoU
                    count_c[c] += 1 # increment the final count 

                # Obtain and store the confusion matrix
                # confusionMatrix[idx] = confMat(pred.cpu(), ann.long().cpu()) # Running this on CPU actually makes it considerably faster

                # Log the results of the evaluation onto tensorboard
                if not self.cfg.EVAL.DISPLAY_IMAGE:
                    writer.add_scalar("Metrics/Dice", dice_score.item(), idx) # record the dice score 
                    writer.add_scalar("Metrics/Time", time.time() - startTime, idx)


                # -------------- Show the image output for the segmentation  ---------- #
                if self.cfg.EVAL.DISPLAY_IMAGE:     

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
                    axs[0, 3].imshow(pred.cpu().detach().numpy()[0], cmap='gray')
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

        print(table)


        print("End of Evaluation Program")



if __name__ == "__main__": 
    
    # Run the evaluation program
    eval = ComparativeEvaluation()
    eval.eval( )




    