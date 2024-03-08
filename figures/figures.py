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
from sklearn import metrics # for the confusion matrix
import torch

################################################################
# Preset the file paths for the results file here
RELLIS_RESULTS = 'figures/ComparativeStudyResults/rellis_results.csv'
RUGD_RESULTS = "figures/ComparativeStudyResults/rugd_results.csv"
MEMREQ = "figures/ComparativeStudyResults/memory_requirements.csv"

#################################################################
#### Useful Functions

def cvt_torch_2_plt_imgfmt(img, torch_in: bool = True):
    """(function) cvt_torch_2_matplotlib_imgfmt
    Quickly change the format from (C,H,W) used by Pytorch to the format (H,W,C) used in Matplotlib. Code assumes tensor input images as input. However ndarray should work as well\n
    :param img: Image being converted
    :type img: torch.tensor
    :param torch_in: Designates that the input is in the pytorch format. If True, then converts to matplotlib format. False converts the other way around. 
    :type torch_in: bool
    :return: Resulting converted image
    :rtype: torch.tensor
    """         

    if torch_in: # from torch to matplotlib fmt
        return img.permute(1,2,0)
    else: # from matplotlib to torch format
        return img.permute(2,0,1) 
    

def get_model_info(model_num): 
    # Allows fro the re-use of plot_config
    plot_config = { # max and min value of the range
        "rugd": {
            "U-Net" : [31,35],
            "HRNetv2": [36,40],
            "DeepLabv3+":[41,45]
        }, 
        "rellis": {
            "U-Net": [1,5], 
            "HRNetv2":[6,10], 
            "DeepLabv3+":[11,15]
        },
        "db": ["rellis", "rugd"],
        "lf": ["CE", "FCIoUv2", "FocalLoss", "DiceLoss", "PowerJaccard"]
    }   

    # Use the remainder to determine what the loss function is 
    rem = model_num%10
    if rem > 5: 
        rem = rem-5
    
    # get the loss function based on the remainder value
    lf = plot_config["lf"][rem-1] 

    mn = model_num
    # Assign the names of each value
    if model_num > 30: 
        db = "rugd"
        mn = model_num - 30
    else: 
        db = "rellis"

    # Retrieve the model type based on the db and model number selection
    if mn < 6: 
        model_type = list(plot_config[db].keys())[0]
    elif mn < 11: 
        model_type = list(plot_config[db].keys())[1]
    elif mn  < 16: 
        model_type = list(plot_config[db].keys())[2]

    return model_type, lf
    
#################################################################

class FigLFResults(): 
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
        self.db = db
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

        plt.savefig("figures/{}3dDistribution.png".format(self.db), dpi = 400)
        # plt.show() # display the results


class FigQualitativeResults_Paper(): 
    """
        FigQualitativeResults()\n
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
    
##### Comparative Study Results Figures
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np

class FigResults(): 
    """FigResults: Base class for creating figures from the evaluation results obtained using eval.py.  
    """    
    
    def __init__(self,file, db_name, show = False): 
        self.maj = self.get_maj_mapping(db_name) # get the majority class mapping
        self.df = self.load_results(file) # load the results file
        self.show = show # configure whether to display the figure after creating it.
        self.o_file_name = "fig.png" # default output figure name 

    def load_results(self,file):
        """load_results: Loads the results file into a dataframe. Sorts and prep the data.

        Args:
            file (str): File name/path

        Returns:
            pandas.DataFrame: Dataframe containing the organized results 
        """    

        import csv

        with open(file, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            

            data = []
            ids = [] # slot id values (used for indexing)
            for i,row in enumerate(reader):
                if i == 0: 
                    header = row[1:] # Remove the SlotID column (used as index)
                else: 
                    ids.append(int(row.pop(0))) # append the first elem (slotId)
                    row[2:] = [float(e) for e in row[2:]] # convert all values into float (rather than import string)
                    data.append(row) # add the data onto the large list

        df = pd.DataFrame(data,index = ids,columns = header)

        return df 

    def get_maj_mapping(self, db_name):
        """get_maj_mapping(db_name): Returns the specific majority class designations for the dataset selected.

        Args:
            db_name (str): name of the database being selected (i.e. "rellis" or "rugd")

        Returns:
            List: List of majority class names (str)
        """    
        
        if db_name == "rellis":
            maj = ["7: sky", "3: grass", "4: tree", "19: bush"] # majority classes within this dataset
        elif db_name == "rugd": 
            maj = ["tree", "grass", "mulch", "sky", "gravel", "bush"] # which majority class is being used can be disputed

        return maj

    def gen_fig(self, figsize = ()):
        exit("No implementation of gen_fig currently present for this class. Only placeholder is present")

class FigMajMinPerformanceComparison(FigResults): 
    def __init__(self, file, db_name, show = False): 
        super().__init__(file, db_name, show)
        self.gen_fig()
        self.o_file_name = "MajMinPerformanceComparison.jpg" # output file name 

    def gen_fig(self, figsize = (7.2,4.8)): 
        # generate the figure
        # Quickly get the correctly named variables
        maj = self.maj
        df = self.df
        """fig_maj_min_performance_comparison: Creates the figure comparing the majority and minority class performance 
        by the models / Loss function. 

        Args:
            df (pandas.DataFrame): Dataframe containing the data for the figure
            maj (List): Mapping (str) of all the majority classes for that specific dataset
        """    

        ################ Overwrite these values ########################
        figwidth, figheight = figsize # get the params for the figure dimensions
        ##########################################################

        # Get the overall values for dice and mIoU scores
        models = np.unique(df['Model']) # Get the number of unique models handeled by the program
        net_miou = df[["Model", "Loss Function", "mIoU"]]
        net_dice = df["Dice"]
        df = df.drop(columns = ["mIoU", "Dice"])
        df_maj = df[["Loss Function", "Model"]].join(df[maj])
        df_min = df.drop(columns = maj) # get all minority class data

        ### Calculate the mean values for min and maj classes
        df_maj["mean"] = df_maj.iloc[:,2:].mean(axis = 1)
        df_min["mean"] = df_min.iloc[:,2:].mean(axis = 1)

        # Create the figure (subplots based on the amount of models being evaluated)
        fig, axs = plt.subplots(1, len(models))
        fig.set_size_inches(figwidth, figheight)
        fig.subplots_adjust(wspace = 0)
        fig.suptitle("Comparison of Majority, Minority, and Mean Class Performance")

        
        for i, model in enumerate(models): 
            
            #### Data Plotting Functions
            # get the subset tied to the specific model
            m_maj = df_maj[df_maj["Model"] == model] 
            m_min = df_min[df_min["Model"] == model]
            m_miou = net_miou[net_miou["Model"] == model]

            axs[i].scatter(range(0,len(m_maj),1), m_maj["mean"])
            axs[i].scatter(range(0,len(m_min),1), m_min["mean"])
            axs[i].scatter(range(0,len(m_miou),1), m_miou["mIoU"])


            #### Figure Formatting
            axs[i].set_yticks(np.arange(0,1.1,0.1)) # Set the common range for yticks
            axs[i].set_ylim(0,1) # Bind the limits of the plot to the [0,1] range
            axs[i].set_xlim(-0.6, len(m_maj) - 0.4) # give some spacing between the different plots
            axs[i].set_xticks(range(0,len(m_maj),1))
            axs[i].xaxis.set_ticklabels(m_maj["Loss Function"], rotation = 20)
            axs[i].set_xlabel(model, fontweight = "bold") # Set the label to the namne of the loss function
            axs[i].xaxis.set_label_position('top')

            ### Formatting based on position of subplot ####
            if i == 0: # for first plot
                axs[i].set_ylabel("mIoU", fontweight = 'bold') # set the ylabel 
            else: # For all the subsequent plots (not the first one)
                axs[i].yaxis.set_ticklabels([]) # hide the yaxis (use the common axis instead)
            if i == len(models) - 1: # if its the last element 
                axs[i].legend(["Majority", "Minority", "Mean"], loc = "upper right")


            axs[i].grid(True, 'both')

        # Configure whether to show the figure 
        if self.show: 
            plt.show()

        return fig

class FigMinImprovement(FigResults): 
    def __init__(self, file, db_name, show = False): 
        super().__init__(file, db_name, show)
        self.gen_fig(db_name)

    def gen_fig(self, db_name):
        # Apply conversion for easy switch between OO code & function-based code
        df = self.df 
        # Pick 6 of the worst performing classes to be plotted based on performance. (6 chosen for legibility.)
        if db_name == "rellis": 
            # classes = ['5: pole', '18: fence', '12: building', '15: log', '8: vehicle', '27: barrier'] # 6 classes
            classes = ['5: pole', '18: fence', '12: building', '15: log', '8: vehicle', '27: barrier', "17: person","33: mud"] # 8 classes
        elif db_name == "rugd": 
            # classes = ["bicycle","sign","person","pole","object","log"] # 6 classes
            # classes = ["bicycle","sign","person","pole","object","log","concrete","picnic-table","rock-bed","bush"] # 10 classes
            classes = ["bicycle","sign","person","pole","object","log","concrete","picnic-table",] # 8 classes
        else: 
            exit("Please select a valid DB name for the figure (rellis/rugd)")

        # Preset order to use when displaying the results on the figure
        # lf_order = ["DiceLoss", "CE", "PowerJaccard", "FocalLoss", "FCIoUV2"] # just did this manually

        models = np.unique(df['Model'])# get the unique model names

        fig, axs = plt.subplots(1, len(models))
        fig.subplots_adjust(wspace = 0) # remove the space between subplots

        # Create a plot for each model (subplots in the figure)
        for i, model in enumerate(models): 
            ## Prep and plot the data
            df_model = df[df['Model'] == model] # results for the specific model
            lf = df_model["Loss Function"] # get the loss function names
            df_classes = df_model[classes] # retrieve only the scores for each class

            axs[i].set_xlabel(model, fontweight="bold")
            axs[i].xaxis.set_label_position('top')

            for c in df_classes:
                axs[i].scatter(range(0,len(df_model["Loss Function"]),1), df_classes[c].values, label = df_classes.columns)
                axs[i].plot(range(0,len(df_model["Loss Function"]),1), df_classes[c].values, label="_nolegend_")

            ## - Format the figure - ##
            if db_name == "rellis":
                axs[i].set_yticks(np.arange(0,1.1,0.1)) # set the axis to [0,1]
                axs[i].set_ylim(0,1.0) 
            elif db_name == "rugd":
                axs[i].set_yticks(np.arange(0,0.6,0.1)) # set the axis to [0,1]
                axs[i].set_ylim(0,0.6) 

            axs[i].set_xlim(-0.3,len(lf) - 0.5)
            axs[i].set_xticks(range(0,len(lf),1))
            axs[i].xaxis.set_ticklabels(lf, rotation = 20) # show the loss functions on the x-axis 

            # for the first axis
            if i == 0: 
                axs[i].set_ylabel("Class-based mIoU")


            if i > 0: # for all subsequent plots 
                axs[i].yaxis.set_ticklabels([]) # remove the ticks for the nex plots

            if i == len(models) - 1: # for the last subplot
                # axs[i].legend(df_classes.columns,loc = "upper right")
                axs[i].legend(df_classes.columns,loc = "upper right")


            axs[i].grid(True, 'both')

        if self.show: 
            plt.show()

class FigMemReqPerformance(FigResults): 
    def __init__(self, results_file, db_name, show = False, memreq_file = None): 
        super().__init__(results_file, db_name, show) # run super function 
        self.rugd_df = super().load_results(RUGD_RESULTS)# load the results
        self.rellis_df = super().load_results(RELLIS_RESULTS) # rugd loading
        self.memreq = self.load_memreq(memreq_file) # save the memory required
        self.gen_fig() # generate the figure

    def load_memreq(self, memreq_file):  # retrieve the memory required elements 
        import csv
        # Open and read the file 
        with open(memreq_file, newline = "") as csvfile: 
            reader = csv.reader(csvfile, delimiter=",")

            data = []
            for i, row in enumerate(reader): 
                if i == 0: 
                    header = row[:] # get all the header rows
                else: 
                    data.append(row) # retrieve all the file contents

        #  return the dataframe representation of the data
        return pd.DataFrame(data, columns = header)

    def gen_fig(self): 
        fig, axs = plt.subplots(1,2)# subplot for dice & mIoU
        fig.subplots_adjust(wspace = 0)
        fig.suptitle("Comparison of Model Performance and Neural Network Memory Required", fontweight = "bold")

        # Set the performance metrics evaluated in this figure 
        pms = ["mIoU", "Dice"]

        #### Set up the data for plotting
        # Retrieve only the FCIoUV2 results -> simpler to show results 
        rugd_df = self.rugd_df[self.rugd_df["Loss Function"] == "FCIoUV2"]
        rellis_df = self.rellis_df[self.rellis_df["Loss Function"] == "FCIoUV2"]
        # select the specific elements from the dataset
        rugd_df = rugd_df[["Model", "mIoU", "Dice"]] 
        rellis_df = rellis_df[["Model", "mIoU", "Dice"]]
        memreq = self.memreq[["Model", "Dataset","MemoryReq"]]

        for i,pm in enumerate(pms): 
            ### Set formatting for each subplot 
            axs[i].set_ylim(0,1.0)
            axs[i].set_yticks(np.arange(0,1.1,0.1)) # set the axis to [0,1]
            axs[i].set_xlim(0,8)
            axs[i].set_xticks(np.arange(0,8,1.0))
            axs[i].grid(True, 'both')
            axs[i].set_title(pm)
            axs[i].xaxis.set_label_text("GPU Memory Required (GB)")

            if i > 0: # no need to have these repeat between plots 
                axs[i].yaxis.set_ticklabels([]) 

            if i == 0: 
                axs[i].yaxis.set_label_text("Performance Metric Score", fontweight ="bold")



            # For each dataset being covered in the data
            for dataset in np.unique(memreq["Dataset"]):  # for each dataset
                for model in np.unique(memreq["Model"]): # for each model
                    label = "" # Preset the plot label to nothing

                    # Set up the x and y values for the plot
                    x = memreq[(memreq["Model"] == model) & (memreq["Dataset"] == dataset)]["MemoryReq"]
                    
                    if dataset == "Rellis": 
                        y = rellis_df[rellis_df["Model"] == model][pm] # i+1 to determine if mIoU or Dice is used 
                        marker = "o" # diamond marker
                    elif dataset == "RUGD": 
                        y = rugd_df[rugd_df["Model"] == model][pm]
                        marker = "D"
                    else: # need a specific implementation for the df used in the plotting (for future use of the code)
                        exit("No implementation provided for this dataset")

                    if i == 0: 
                        label = "{} & {}".format(model, dataset)

                    if not y.empty and not x.empty:  # make sure there are values available to plot
                        x = np.around(float(x)/1024, 2) # convert the MB values into GB

                        axs[i].scatter(x,y, marker = marker, label = label)

        fig.legend(loc = "center right") # set up the legend for the figure 

        if self.show: 
            plt.show()

class FigConfusionMatrix():
    def __init__(self, model_num, show = False):


        file_name = "0{}_ConfusionMatrix".format(model_num) # get the file name based on model number
        fp = "figures/ConfusionMatrix/{}_ConfusionMatrix.csv".format(str(model_num).zfill(3)) #  get the file path name 

        results = []
        with open(fp, "r") as file: 
            csvreader = csv.reader(file)
            for i, row in enumerate(csvreader): 
                results.append(row)

        # Convert the array to use the proper datatype
        conf = np.array(results)
        conf = conf.astype(float)

        conf = conf + 0.00000001 # add a extremely small value to avoid divide by 0

        # Normalize all values based on the number of ground-truth labels (i.e. the amount of predictions to be expected)
        for i, val in enumerate(conf.sum(axis=1)): # for each of the values in mat
            conf[i] = conf[i] / val # divide by that value

        if db.cfg.DB.DB_NAME == "rellis": 
            class_labels = db.remap_class_labels # use the remapped version of it
        else: 
            class_labels = db.class_labels

        disp = metrics.ConfusionMatrixDisplay(conf, display_labels=class_labels.values())
        disp.plot(include_values = False, cmap = "Blues") # create the plot for the confusion matrix
        disp.ax_.get_images()[0].set_clim(0,1) # update the colour map to be between the range [0,1]
        fig = disp.figure_ # retrieve the figure for later formatting
        axs = fig.get_axes() # get the figure axes
        axs[0].xaxis.set_ticklabels(axs[0].xaxis.get_ticklabels(), rotation = -70)
        axs[0].grid(visible = True) # display the grid on the figure
        axs[1].yaxis.set_ticks(np.arange(0,1.1,0.1))
        fig.set_size_inches(8,7)
        fig.subplots_adjust(bottom = 0.165)
        
        model_type, lf = get_model_info(model_num) # get the information for the specific model
        plt.title("{} & {}".format(model_type, lf)) # update this to feature the actual LF/Model/Dataset combination -> Placeholder ONly

        plt.savefig("figures/ConfusionMatrix/{}.png".format(file_name))

        if show:
            plt.show()
    

class FigPerfBoxPlot(): 
    def __init__(self): 
        """FigPerfBoxPlot()\n
        Generates the box plot for all of the prepared LoggedResults files located in "figures/ComparativeStudyResults/LoggedResults/".\n
        This encompasses all of the provided files. ONLY has the RUGD dataset currently implemented
        """        
        # Load all the model data
        df = {}
        for i in range(0,46): # open all the files and load the dataframe withn the collected information
            fp = "figures/ComparativeStudyResults/LoggedResults/0{}_LoggedResults.csv".format(str(i).zfill(2)) # file path
            # Only add the data to the array if the file actually exists (avoid any errors)
            if os.path.exists(fp): 
                df[i] = pd.read_csv(fp)


        # Configure which db/model is corresponding to model_num 
        plot_config = { # max and min value of the range
            "rugd": {
                "U-Net" : [31,35],
                "HRNetv2": [36,40],
                "DeepLabv3+":[41,45]
            }, 
            "rellis": {
                "U-Net": [1,5], 
                "HRNetv2":[6,10], 
                "DeepLabv3+":[11,15]
            },
            "lf": ["CE", "FCIoUv2", "FocalLoss", "DiceLoss", "PowerJaccard"]
        }

        fig,axs = plt.subplots(1,3) # one subplot per model
        fig.subplots_adjust(wspace = 0, hspace=0)
        fig.set_size_inches(8,6)

        # select the RUGD dataset as preset
        db = "rugd" # start with the RUGD dataset first, update this implementation later
        
        ### Plot the figure 
        models = list(plot_config[db].keys()) # Get the models being evaluated
        for i, _ in enumerate(axs):
            ### Figure formatting
            if i > 0: # for all subsequent plots
                axs[i].set(yticklabels=([]))# remove the ticklabels

            #### - Data plotting & handling
            model_range = plot_config[db][models[i]]
            # Boxplot for each LF used to train this specific model
            model_dice = []
            for j in range(model_range[0], model_range[1] + 1):
                model_dice.append(df[j]["dice"]) # grab the Dice score data from the dataframe
            axs[i].boxplot(model_dice, showfliers=False)# plot the model data
            axs[i].set_xticklabels(plot_config["lf"], rotation = -60)

            axs[i].set(ylim = (0,1))
            axs[i].set_xlabel(models[i], va="top") 
            axs[i].xaxis.set_label_position("top")
            axs[i].set_yticks(np.arange(0,1.1, 0.1))
            axs[i].minorticks_on()
            axs[i].grid(True, which = "both", alpha = 0.25)


        fig.subplots_adjust(bottom = 0.19)
        fig.savefig("figures/{}_PerfBoxPlot.jpg".format(db), dpi = 400)
        # plt.show()

class FigQualitativeResults(): 
    """(class) FigQualitativeResults\n
    Updated version of the Qualitative Results Figure to be employed in the thesis manuscript. Provides a visualization of the prediction results of the semantic 
    segmentation models.

    :param idx: Id specific to a single image in the corresponding dataset. Specify to re-use the same image 
    :type idx: int
    """         
    
    def __init__(self, idx = 0, model_range: list = [41,45]):
        import eval 
        self.eval_handler = eval.ComparativeEvaluation()



        plot_config = { # max and min value of the range
            "rugd": {
                "U-Net" : [31,35],
                "HRNetv2": [36,40],
                "DeepLabv3+":[41,45]
            }, 
            "rellis": {
                "U-Net": [1,5], 
                "HRNetv2":[6,10], 
                "DeepLabv3+":[11,15]
            },
            "lf": ["CE", "FCIoUv2", "FocalLoss", "DiceLoss", "PowerJaccard"]
        }

        # Grab the images for the figure        
        pred, ann, raw_img = self.eval_handler.single_img_pred(idx, model_num = model_range[0])

        fig = plt.figure()
        spec = fig.add_gridspec(2,4)

        axs = []
        for i, _ in enumerate(spec): 
            axs.append(fig.add_subplot(spec[i]))
        
        axs[-1].remove() # no plot required on the last element
        axs[0].imshow(raw_img)
        axs[1].imshow(self.prep_seg(ann))
        axs[2].imshow(self.prep_seg(pred))

        i = 3
        for model_num in range(model_range[0]+1, model_range[1] + 1):
            img, _, _ = self.eval_handler.single_img_pred(idx, model_num = model_num)
            axs[i].imshow(self.prep_seg(img))

            i = i + 1 # increment the axis number
            pass 

        fig.subplots_adjust(wspace = 0, hspace = 0)

        # Apply formatting to all subplots
        for i in range(0,7): 
            axs[i].set_yticks([])
            axs[i].set_xticks([])
            axs[i].xaxis.set_label_position("top")
            if i == 0: 
                axs[i].set_xlabel("Raw Input Image")
            elif i == 1: 
                axs[i].set_xlabel("Ground-Truth")
            elif i > 1: 
                axs[i].set_xlabel(plot_config["lf"][i - 2])
            


        plt.show()
        pass
    
    def prep_seg(self, seg_in): 
        # Convert the segmented labelled image into a colour mapped one -> repeated several times
        return cvt_torch_2_plt_imgfmt(self.eval_handler.cvt_color(seg_in).long())
    
class FigFCIoUComparison():
    def __init__(self, with_v2: bool = False): 
        import eval 
        self.eval_handler = eval.ComparativeEvaluation()

        ids = [0, 2]# FCIoUv1 and v2 with the U-Net

        idx = 52
        v1_pred, ann, raw_img = self.eval_handler.single_img_pred(idx, model_num = ids[0])
        v2_pred, _, _ = self.eval_handler.single_img_pred(idx, ids[1])

        if with_v2: # set the number of columns present in the plot
            cols = 4
            fp = "FigFCIoUComparison"
        else: 
            fp = "FigFCIoUComparison_v1Only"
            cols = 3


        fig, axs = plt.subplots(1,cols)
        fig.set_size_inches(8,6)
        fig.subplots_adjust(wspace = 0.01, hspace = 0, right = 0.96, left = 0.04)


        # Manually display the images in the figure.
        axs[0].imshow(raw_img)
        axs[1].imshow(self.prep_seg(ann))
        axs[2].imshow(self.prep_seg(v1_pred))

        if with_v2: 
            axs[3].imshow(self.prep_seg(v2_pred))

        labels = ["Input Image", "Annotated Image", "FCIoUv1", "FCIoUv2"]

        for i, ax in enumerate(axs): 
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_xlabel(labels[i], va = "top")
            ax.xaxis.set_label_position("top")

        plt.savefig("figures/{}.png".format(fp), dpi = 400)

        # plt.show()

    def prep_seg(self, seg_in): 
        # Convert the segmented labelled image into a colour mapped one -> repeated several times
        return cvt_torch_2_plt_imgfmt(self.eval_handler.cvt_color(seg_in).long())

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



    ##### Configuration Values #####
    db_name = "rellis" # name of db used during eval
    model_num = 34 # for model-specific figures
    ################################

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
    # FigDBDistribution(class_labels=class_labels, ignore=ignore, colors=colors, db = db_name)  
    # FigQualitativeResults_Paper()
    
    #### - New Comparative Study Results Figures - ####
    # FigMajMinPerformanceComparison(RUGD_RESULTS, "rugd", True)
    # FigMinImprovement(RUGD_RESULTS, "rugd", True)
    # FigMemReqPerformance(RELLIS_RESULTS, "rellis", True, "figures/ComparativeStudyResults/memory_requirements.csv")

    # for i in range(1,6): 
    #     FigConfusionMatrix(model_num = i) # create the confusion matrix figure for a specific model
    FigPerfBoxPlot()
    # FigQualitativeResults(idx=203)  
    # FigFCIoUComparison(with_v2=False)