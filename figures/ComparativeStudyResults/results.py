import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np

def load_results(file):
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


def get_maj_mapping(db_name):
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

def fig_maj_min_performance_comparison(df, maj, figsize = (7.2,4.8)): 
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

    
    plt.show()
    return fig

def get_fig_min_improvement(df, db_name = ""):
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
        axs[i].set_yticks(np.arange(0,1.1,0.1)) # set the axis to [0,1]
        axs[i].set_ylim(0,1.0)       
        axs[i].set_xlim(-0.3,len(lf) - 0.5)
        axs[i].set_xticks(range(0,len(lf),1))
        axs[i].xaxis.set_ticklabels(lf, rotation = 20) # show the loss functions on the x-axis 

        if i > 0: # for all subsequent plots 
            axs[i].yaxis.set_ticklabels([]) # remove the ticks for the nex plots

        if i == len(models) - 1: # for the last subplot
            # axs[i].legend(df_classes.columns,loc = "upper right")
            axs[i].legend(df_classes.columns,loc = "upper right")


        axs[i].grid(True, 'both')

    plt.show()

if __name__ == "__main__":

    # ## Rellis-3D
    # maj = get_maj_mapping("rellis") 
    # df = load_results("figures/ComparativeStudyResults/results.csv")
    # fig = fig_maj_min_performance_comparison(df, maj) # run the code 

    # ## RUGD    
    # # maj = get_maj_mapping('rugd')
    # # df = load_results('figures/ComparativeStudyResults/rugd_results.csv')
    # # fig = fig_maj_min_performance_comparison(df, maj, figsize = (11.2 ,4.8))

    # fig.savefig("figures/MajMinMeanComparison.png", dpi = 300) # save the figure

    df = load_results("figures/ComparativeStudyResults/rellis_results.csv")
    fig = get_fig_min_improvement(df, db_name = "rellis")





pass