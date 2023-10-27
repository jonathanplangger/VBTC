<div align="justify">

# Vision based Terrain Classification (VBTC) EMSLab Research Repository
This repository contains the code for the implementation of Vision-based Terrain Classification Research as part of the Embedded Multi-Sensor Resarch Laboratory at Carleton University (Ottawa, ON). As a research repository, the code relevant to several aspects are included within this repository.
Installation instructions for required software solutions and enviroment information is provided in the following sections of the README.MD. We additionally include the implementation instructions for our FCIoU paper below which can be used to recreate the comparative study and experiments detailed in the research manuscript.  

##Environment Setup & Installation
**Python** ver. **3.8.15**, **Pytorch** ver. **1.13.1+cu117**, & **TorchVision** ver. **0.14.1+cu117** were employed during development. Other, more recent versions of these may operate correctly, however only the specified versions were tested. 

Other libraries employed are defined within the *requirements.txt* file included in this repository. Project dependencies can be installed directly through:
```
pip install -r requirements.txt
```



#FCIoU: A Focal Class-based Intersection over Union (IoU) Approach to Improving Minority Class Detection Performance for Off-road Segmentation Systems
## Abstract
In this paper, we present a comparative study of modern semantic segmentation loss functions and their resultant impact when applied with state-of-the-art off-road datasets. Class imbalance, inherent in these datasets, presents a significant challenge to off-road terrain semantic segmentation systems. With numerous environment classes being extremely sparse and underrepresented, model training becomes inefficient and struggles to comprehend the infrequent minority
classes. As a solution to this problem, loss functions have been configured to take class imbalance
into account and counteract this issue. To this end, we present a novel loss function, Focal Classbased Intersection over Union (FCIoU), which directly targets performance imbalance through the
optimization of class-based Intersection over Union (IoU). The new loss function results in a general
increase in class-based performance when compared to state-of-the-art targeted loss functions. 

![Image Comparing the segmentation ability for each loss function for a single image](./figures/QualitativeResults.png)
*Figure: Qualitative Representation of Model Performance for Given Loss Function* 






##Loss Functions
Loss function implementations employed within the study are defined in *loss.py*. Each loss function were recreated within this study to suit the task of multi-class segmentation. Further information regarding the specific loss functions are provided within the paper manuscript. 

Selection of loss function for training can be configured through the *CRITERION* field in the *config_comparative_study.yaml* file in the *configs* directory. The possible selections are as follows: 
<i>
- fciouv1 -> fciouv6 (V1 and V2 are Featured within the manuscript)
- dicefocal
- diceloss
- dicetopk
- iouloss
- powerjaccard
- tverskyloss
- focalloss
</i>

Selecting a specific loss function in the *CRITERION* field results in the use of the corresponding loss function class. 






</div>

