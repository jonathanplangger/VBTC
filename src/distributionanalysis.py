import torch
import dataloader
from tqdm import tqdm
import numpy as np
import pandas as pd 


class DistributionAnalysis(): 

    def __init__(self, show=False, db_path = "", batch_size=1): 
        self.show = show # define if the output is being plotted when analysis completed 
        self.db = dataloader.DataLoader(db_path, "train", remap=True)
        self.db.randomizeOrder()
        self.batch_size = batch_size
        self.class_labels = np.array(list(self.db.class_labels.values())) # Retrieve a list containing each class labels
        
        self.get_distribution()


    def get_distribution(self): 
        # Number of images present in the training set 
        n_img = len(self.db.train_meta)


        # holds the n# of pixels for each class
        pxs = torch.zeros(self.db.num_classes).cuda()

        with tqdm(total=int(n_img/self.batch_size), unit="Batch") as pbar: 
            # Go through every image with batching 
            for idx in range(0, n_img, self.batch_size): 
                # Only retrieve the annotation files 
                _, _, ann, idx = self.db.load_batch(idx, self.batch_size)
                ann.cuda() # place on GPU
                # For all classes
                for c in range(0, self.db.num_classes): 
                    pxs[c] += torch.count_nonzero(ann == c) # Count the number of pixels in the batch
                
                pbar.update()
        
        # Save the output to a csv file
        pxs = pxs.cpu().numpy() # convert the tensor to a numpy array
        df = pd.DataFrame({"classNames":self.class_labels[1:], "distribution": pxs }).transpose()
        df.to_csv("figures/distributions.csv", index=False)

        print(df)

if __name__ == "__main__": 
    DistributionAnalysis(show=True, db_path="../../datasets/Rellis-3D/", batch_size=1)
