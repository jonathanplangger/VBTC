# Vision-based Terrain Classification (VBTC) EMSLab Research Repository
---------------------------------------------------------------------------------------------------
This repo contains the code for the implementation of Vision-based Terrain Classification Research as part of the Embedded Multi-Sensor Resarch Laboratory at Carleton University (Ottawa, ON). As a research repository, the code relevant to several aspects are included within this repository.
Installation instructions for required software solutions and enviroment information is provided in the following sections of the README.MD. We additionally include the implementation instructions for our FCIoU paper which can be used to recreate the comparative study and experiments detailed in the research manuscript.  



#FCIoU: A Focal Class-based Intersection over Union (IoU) Approach to Improving Minority Class Detection Performance for Off-road Segmentation Systems
--------------------------------------------------------------------------------------------------
## Abstract
In this paper, we present a comparative study of modern semantic segmentation loss functions and their resultant impact when applied with state-of-the-art off-road datasets. Class imbalance, inherent in these datasets, presents a significant challenge to off-road terrain semantic segmentation systems. With numerous environment classes being extremely sparse and underrepresented, model training becomes inefficient and struggles to comprehend the infrequent minority
classes. As a solution to this problem, loss functions have been configured to take class imbalance
into account and counteract this issue. To this end, we present a novel loss function, Focal Classbased Intersection over Union (FCIoU), which directly targets performance imbalance through the
optimization of class-based Intersection over Union (IoU). The new loss function results in a general
increase in class-based performance when compared to state-of-the-art targeted loss functions. 





