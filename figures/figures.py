"""
This file contains the code implementation for the figures presented in the research paper. 
"""
from matplotlib import pyplot as plt 
import matplotlib.ticker as mticker
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import os 
import csv   
import pandas as pd 
from PIL import Image

class FigResults(): 
    def __init__(self): 
        """
        Assumes the csv file format is as follows: 
        Slot ID,LF ,0: void,1: dirt,3: grass,4: tree,5: pole,6: water,7: sky,8: vehicle,9: object,10: asphalt,12: building,15: log,17: person,18: fence,19: bush,23: concrete,27: barrier,31: puddle,33: mud,34: rubble,mIoU,Mean Dice
        14,FCIoUV2,0.4372,0,0.8386,0.6658,0.0134,0.1818,0.9595,0.2404,0.4096,0.3384,0.0017,0.0052,0.2578,0.229,0.6192,0.7305,0.2451,0.4939,0.2808,0.4372,0.369255,0.9013
        """
        
        ###############################################################################################
        # Retrieve Results from file
        ###############################################################################################

        results = {} # store all the results data
        # Retrieve the results from the csv file 
        with open ("figures/results.csv", 'r') as file: 
            csvreader = csv.reader(file)
            for i, row in enumerate(csvreader): 
                if i == 0: # Get the header row from the data
                    header = row
                else: # get the data from the rest of the csv file. ID is ignored
                    results[row[1]] = row[2:]

        ################################################################################################
        # Plotting the results
        ################################################################################################
        #----- Prep data ------------- #
        # Create a list containing the results
        data = list(results.values())
        # place as numpy array and update the dtype to float instead of string
        data = np.array(data, dtype=float)

        # Create the figure 
        fig, axs = plt.subplots(1,len(results), dpi=400)
        fig.set_figheight(8)
        fig.set_figwidth(16)
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.suptitle("Loss Function Performance Comparison", va = 'bottom', ha="center",
                    fontsize="x-large", fontweight='bold', y=  0.9)
        axs[0].set_ylabel(ylabel="IoU Performance Score")

        # All x-values will be centered on 0. 
        x_val = np.ones(len(data[0]))
        lf = list(results) # obtain a list containing the keys for the results
        maj_id = [2,3,6,14] # id for majority classes within the dataset

        for i, d in enumerate(data): 
            # Violin plot for the figure
            axs[i].violinplot(d[1:], showmeans=True, )
            # plot the individual points on the violin plot
            axs[i].plot(x_val[1:], d[1:], 'o', color='blue', markersize=4, label="Minority Classes")
            # Replot the majority classes in a different colour 
            axs[i].plot(x_val[maj_id], d[maj_id], 'o', color='red', markersize=4, label="Majority Classes")
            axs[i].set(xlim = (0.5,1.5), ylim=(-0.1, 1.1), xticks=([]))
            axs[i].set_xlabel(lf[i])
            axs[i].margins(x=0)
            axs[i].set_yticks(np.arange(0,1.1,0.1))
            axs[i].grid(True)
            
            if i > 0: 
                axs[i].set(yticklabels=([])) # remove the yticks for all plots except 1st one

        plt.legend()

        plt.savefig("figures/ResultsPerformanceComparison.svg")
        #plt.show()   

class FigLossShaping():
    # Plots the function on construction call
    def __init__(self): 
        fig = plt.figure(dpi=400)
       
        # Plot the two fuunctions 
        x = np.linspace(-0.1, 1.5, 1000) # X-values for the plot
        plt.plot(x, self.f_lossShaping(x), color="red", label="Loss Shaping Function")
        plt.plot(x, self.f_LIoU(x), color="blue", label= "1 − IoU")
        plt.grid()
        plt.xticks(np.arange(0,1.1,0.1))
        plt.xlim(-0.05,1.05)
        plt.ylim(-0.05,1.3)
        plt.legend()
        # plt.tight_layout()
        ax = fig.get_axes()[0]
        ax.set_xlabel("Class IoU Score")
        ax.set_ylabel("Class Generated Error")
        ax.margins(x=0,y=0)


        plt.savefig("figures/LossShapingFunction.svg")
        #plt.show()

    def f_lossShaping(self, x): # Loss shaping function base implementation 
        return -np.log10(pow(x,0.5))
    def f_LIoU(self, x):# IoU-loss base implementation
        return 1-x 

class FigPowerTerm(): 
    def __init__(self): 
        fig = plt.figure(dpi=400)
        x = np.linspace(-0.1,1.5,1000)
        plt.plot(x, self.f_powerJaccard(x), color="red", label="Power Term")
        plt.plot(x, self.f_IoU(x), color="blue", label="IoU")
        plt.grid()
        plt.xticks(np.arange(0,1.1,0.1))
        plt.xlim(-0.05,1.05)
        plt.ylim(-0.05,1.05)
        plt.legend()
        # plt.tight_layout()
        ax = fig.get_axes()[0]
        ax.set_xlabel("Pixel Prediction Probability")
        ax.set_ylabel("Generated Score")



        plt.savefig("figures/PowerTermFunction.svg")
        #plt.show()
        
    def f_powerJaccard(self, x): 
        return x/(pow(x,2) + 1 - x)
    
    def f_IoU(self, x): 
        return x/(x + 1 - x) # IoU score (same as 1/x)

class FigDBDistribution(): 

    def __init__(self, class_labels={}, ignore=[], colors=None, db = ""):
        self.figcfg = self.get_figcfg(db) # get the configuration 
        self.gen_fig(self.figcfg["fpath"], class_labels, ignore, colors) # generate the figure 

    def get_figcfg(self, db): # get the pre-set configuration for the figure 
        # Configuration based on the dataset being used
        if db == "rellis": # Rellis-3D dataset
            return {
                "figheight": 6, 
                "figwidth": 10, 
                "fpath": "figures/Distributions/rellis_distributions.csv"
            } 
        elif db == "rugd":  # RUGD dataset
            return {
                "figheight": 8, 
                "figwidth": 12, 
                "fpath" : "figures/Distributions/rugd_distributions.csv"
            }
        else: 
            exit("Please specify a valid dataset for proper figure configuration to be provided.")

    def gen_fig(self, fpath ="", class_labels={}, ignore=[], colors=None):    
        with open (fpath, 'r') as file: 
            csvreader = csv.reader(file)
            for i, row in enumerate(csvreader):

                if i == 1: 
                    class_labels = np.array(row)
                elif i == 2:  
                    distribution = np.array(row)
                    # Format the csv values into an array of floats
                    distribution = np.char.strip(distribution)
                    distribution.astype("float")

        # Convert the csv file and prep the data
        df = pd.DataFrame({"class_labels": class_labels, "distribution": distribution, "colors": colors})
        df = df.astype({"class_labels": "string", "distribution": "float"})
        df = df.sort_values(by="distribution", ascending=False) # sort the values based on size
        df_lower = df[6:] # Get the lowest values to display them on a separate plot

        plt.rcParams.update({"font.size":11.5})
        # Plot the figure 
        fig, main = plt.subplots()
        plt.xticks(rotation=75)
        
        fig.set_figheight(self.figcfg["figheight"]) 
        fig.set_figwidth(self.figcfg['figwidth']) 
        # Add the subplot axes 
        sub = fig.add_axes([0.35,0.3, 0.6, 0.6])
        plt.xticks(rotation=75)
    
        if colors == None: 
            main.bar(df["class_labels"], df["distribution"])
            sub.bar(df_lower["class_labels"], df_lower["distribution"])
        else: 
            # Normalize the range to fit requirements for plotting
            df.colors = np.asarray(df.colors)
            for i,color in enumerate(df.colors): 
                df["colors"][i] = np.divide(df["colors"][i], 255.0)
                df["colors"][i] = np.around(df.colors[i],2)

            # Plot the main plot and subplot 
            main.bar(df["class_labels"], df["distribution"], color=df.colors)
            sub.bar(df_lower["class_labels"], df_lower["distribution"], color=df.colors[6:])

        # Figure formatting
        main.margins(x=0)
        sub.margins(x=0)
        main.set_ylabel("Number of Pixels")
        plt.tight_layout()

        # Update formatting to use literature scientific notation
        f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
        sub.yaxis.set_major_formatter(f)
        main.yaxis.set_major_formatter(f)

        plt.savefig("figures/rellis3dDistribution.png", dpi = 400)
        plt.show() # display the results


class QualitativeResults(): 
    """
        QualitativeResults()\n
        --------------------------------------\n
        A single image is being displayed alongside all the other segmentation predictions obtained for all loss functions. \n
        Used to qualitatively represent the performance of each loss functions \n
    """
    def __init__(self): 
        from matplotlib.figure import SubplotParams

        fig, axs = plt.subplots(2,6, tight_layout=True, dpi=400)
        fig.set_figheight(4)
        fig.set_figwidth(16)
        #Save description to make overall formatting a lot easier 
        desc = {
                "base_img": "Base Image",
                "ann": "Ground-Truth Annotations",
                "014": "FCIoUV2",
                "018": "Cross Entropy Loss",
                "019": "Jaccard (IoU) Loss",
                "020": "Dice Loss",
                "021": "Focal Loss",
                "023": "DiceFocal",
                "025": "FCIoUV1",
                "026": "Tversky Loss",
                "032": "Power Jaccard",
                "033": "DiceTopk",               
               }
        
        desc_names = list(desc.keys())
        lfnames = list(desc.values())
        i = 0
        for r, _ in enumerate(axs): 
            for c, _ in enumerate(axs[r]):
                axs[r,c].imshow(Image.open("figures/QualitativeResults/{}.png".format(desc_names[i])))
                axs[r,c].set_xlabel(lfnames[i])
                # Remove xticks and yticks alltogether 
                axs[r,c].set_xticks([])
                axs[r,c].set_yticks([])
                axs[r,c].margins(x = 0,y = 0)

                i += 1 # increment the counter
        
        # Update the subplot format
        fig.subplots_adjust(wspace=0, hspace=0)
        plt.savefig("figures/QualitativeResults.svg", dpi=400)
        #plt.show()
        pass

class FigPredictionCertainty(): 
    def __init__(self, pred, pred_map, ann, class_labels=[], color_map = None): 
        # Remove the batching wrapper -> only use one of the image (if >1 present)
        pred = pred[0].cpu()
        ann = ann[0]

        # Convert the labels into a list
        class_labels = list(class_labels.values())
        class_labels = class_labels[1:] # remove the void class since it is the same as the dirt class

        # Configure the n# of rows and cols in the figure
        rows, cols = 4, 5
        fig = plt.figure(tight_layout = True, dpi = 300) # TODO -> Update to create a slot for each class.
        gs = fig.add_gridspec(nrows=rows, ncols=cols, wspace= 0.00, hspace = 0.4)

        # # place the annotation map on the figure. 
        # if color_map is None: 
        # annotation map image
        ax = fig.add_subplot(gs[0,0])
        ax.imshow(ann.cpu(), cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])        
        ax.set_xlabel("Ann Map", fontsize = "xx-small", va="top")   
        # Prediction output map
        ax = fig.add_subplot(gs[0,1])
        ax.imshow(pred_map.cpu()[0], cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])        
        ax.set_xlabel("Pred Map", fontsize = "xx-small", va="top")               


        i = 0
        for row in range(rows): 
            for col in range(cols):

                # Start on the 2nd plot for the first cycle only
                if col < 2 and row == 0: 
                    col = col + 1 
                else:  # ONly complete in the next slots
                    ax = fig.add_subplot(gs[row,col]) # add the new subplot to the figure

                    # r,g,b = self.__conv_color_map(color_map[i]) # rgb colours
                    # cdict ={
                    #     'red':((0,1,r), (1,r,r)),
                    #     'green':((0,1,g), (1,g,g)),
                    #     'blue':((0,b,b), (1,b,b)),
                    # }
                    # cmap = LinearSegmentedColormap(class_labels[i], cdict, N = 256)

                    ax.imshow(1 - pred[i], cmap = "gray")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xlabel("{}: {}".format(i, class_labels[i]), fontsize = "xx-small", va="top") # Add labels for each class
                    ax.margins(x = 0)


                    i = i + 1 # increase the index for pred
                    if i > 18: 
                        break


        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # # Create a figure for each class 
        # for c in pred:
         
        plt.show()

    def __dispfunc(self, pred): 
        """This function is applied to the logit output prediction from the model to display the desired output 
        The function will depend on which relationship we want to investigate in the figure. 

        :param pred: Prediction (logit) output from the model
        :type pred: torch.tensor
        :return: Updated tensor for desired format
        :rtype: torch.tensor
        """        
        return 1 - pred

    def __conv_color_map(self,color): 
        return (color[0]/255.0, color[1]/255.0, color[2]/255.0)

if __name__ == "__main__": 

    # FigLossShaping()
    # FigPowerTerm()

    class_labels = {
        0: "void",
        1: "dirt",
        2: "grass",
        3: "tree",
        4: "pole",
        5: "water",
        6: "sky",
        7: "vehicle",
        8: "object",
        9: "asphalt",
        10: "building",
        11: "log",
        12: "person",
        13: "fence",
        14: "bush",
        15: "concrete",
        16: "barrier",
        17: "puddle",
        18: "mud",
        19: "rubble",
    }

    db_name = "rugd"

    # # Add to path to allow access to the dataloader
    import sys
    import os
    sys.path.append(os.getcwd()) # allow to directly access the other source files
    # Import and configure the data loader
    import dataloader 
    from config import get_cfg_defaults
    cfg = get_cfg_defaults() # get the default configuration
    cfg.merge_from_file("configs/config_comparative_study.yaml") # update cfg based on file
    cfg.DB.DB_NAME = db_name # update this to select the right DB 
    db = dataloader.get_dataloader(cfg, setType = "eval")

    
    colors = db.get_colors(remap_labels=True)
    ignore = []
    if db_name == "rellis": 
        colors = colors[1:] # special case (remove one class)
        ignore = [0,1] # do not represent the void and dirt classes in the figure

    # FigResults()
    # FigLossShaping()
    # FigPowerTerm()
    FigDBDistribution(class_labels=class_labels, ignore=ignore, colors=colors, db = "rugd")
    # QualitativeResults()
