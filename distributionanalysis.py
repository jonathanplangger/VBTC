import torch
import dataloader
from tqdm import tqdm
import numpy as np
import pandas as pd 


class DistributionAnalysis(): 

    def __init__(self, show=False, cfg = None, batch_size=1, split_type = None): 
        self.split_type = split_type # which split set to use
        self.show = show # define if the output is being plotted when analysis completed 
        self.db = dataloader.get_dataloader(cfg, setType = split_type)
        self.batch_size = batch_size
        self.class_labels = np.array(list(self.db.class_labels.values())) # Retrieve a list containing each class labels
        # Run the 
        self.get_distribution()


    def get_distribution(self): 

        # Number of images present within the given set 
        if self.split_type == "train":
            n_img = len(self.db.train_meta)
        elif self.split_type == "test": 
            n_img = len(self.db.test_meta)
        elif self.split_type == "val": 
            n_img = len(self.db.val_meta) # not implemented yet, need to update
        else: 
            exit("Invalid split_type. Please select either train/test/val.")

        # holds the n# of pixels for each class
        pxs = torch.zeros(self.db.num_classes).cuda()

        with tqdm(total=int(n_img/self.batch_size), unit="Batch") as pbar: 
            # Go through every image with batching 
            for idx in range(0, n_img, self.batch_size): 
                # Only retrieve the annotation files 
                _, _, ann, idx = self.db.load_batch(idx, self.batch_size)
                ann.cuda() # place on GPU
                # For each class -> count the amount of entries in the image
                for c in range(0, self.db.num_classes): 
                    pxs[c] += torch.count_nonzero(ann == c) # Count the number of pixels in the batch
                
                pbar.update()
        
        # Save the output to a csv file
        pxs = pxs.cpu().numpy() # convert the tensor to a numpy array
        if self.db.cfg.DB.DB_NAME == "rellis": # special case in the rellis dataset
            df = pd.DataFrame({"classNames":self.class_labels[1:], "distribution": pxs }).transpose()
        else: # for all other cases 
            df = pd.DataFrame({"classNames":self.class_labels, "distribution": pxs }).transpose()

        df.to_csv("figures/distributions.csv", index=False)

        print(df)

if __name__ == "__main__": 
    from config import get_cfg_defaults
    cfg = get_cfg_defaults()
    # Uses eval configuration for the inputs to this function. 
    cfg.merge_from_file("configs/config_comparative_study.yaml") 

    # Which train/test/val split to evaluate the data from.
    split_type = "train"
    DistributionAnalysis(show=True, cfg = cfg, batch_size=1, split_type=split_type)
