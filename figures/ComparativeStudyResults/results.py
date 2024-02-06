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

def fig_maj_min_performance_comparison(df, maj): 
    """fig_maj_min_performance_comparison: Creates the figure comparing the majority and minority class performance 
    by the models / Loss function. 

    Args:
        df (pandas.DataFrame): Dataframe containing the data for the figure
        maj (List): Mapping (str) of all the majority classes for that specific dataset
    """    
    # Get the overall values for dice and mIoU scores
    net_miou = df["mIoU"]
    net_dice = df["Dice"]
    df = df.drop(columns = ["mIoU", "Dice"])
    df_maj = df[["Loss Function", "Model"]].join(df[maj])
    df_min = df.drop(columns = maj) # get all minority class data
    
    # store the miou values for the majority classes
    maj_miou = []
    # Calculate the mean scores for majority
    for row in df_maj.iterrows(): 
        vals = row[1][-len(maj):]
        maj_miou.append(np.mean(vals))
    
    # Minority class mIoU values
    min_miou = []
    for row in df_min.iterrows():
        vals = row[1][2:]
        min_miou.append(np.mean(vals))

    # Round all the values
    min_miou = [np.round(e,4) for e in min_miou]

    ##########################
    # Create the figure 
    fig = plt.figure(1, figsize = (3.2,1.8))
    # 
    plt.scatter(range(0,len(maj_miou)), maj_miou)
    plt.scatter(range(0,len(min_miou)), min_miou)
    plt.scatter(range(0,len(net_miou)), net_miou)
    plt.xticks(range(0,len(maj_miou),1), df["Model"])



    # display the image
    plt.show()




if __name__ == "__main__":

    # Set up the figure for the rellis-3d results
    maj = get_maj_mapping("rellis") 
    df = load_results("figures/ComparativeStudyResults/results.csv")

    fig_maj_min_performance_comparison(df, maj) # run the code 
    
    




    





pass