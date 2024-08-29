<div align="justify">

# Vision based Terrain Classification (VBTC) EMSLab Research Repository
This repository contains the code for the implementation of Vision-based Terrain Classification Research as part of the Embedded Multi-Sensor Resarch Laboratory at Carleton University (Ottawa, ON). As a research repository, the code relevant to several aspects are included within this repository.
Installation instructions for required software solutions and enviroment information is provided in the following sections of the README.MD. We additionally include the implementation instructions for our FCIoU paper below which can be used to recreate the comparative study and experiments detailed in the research manuscript.  

#### Publications
Plangger, Jonathan, Mohamed Atia, and Hicham Chaoui. “FCIoU: A Targeted Approach for Improving Minority Class Detection in Semantic Segmentation Systems.” Machine Learning and Knowledge Extraction 5, no. 4 (November 23, 2023): 1746–59. https://doi.org/10.3390/make5040085.


### Repository Structure
The structure of the repository code is represented as: 

![Structural block diagram of the code contained in the repository](github/RepoCodeStructure.png)

Auto-generated documentation for the project is developed using the Doxygen documentation generator and can be accessed by opening the index.html file in the ./docs/html directory. From there the relevant hyperlinks to the classes and functions within the code repository are present. Note that the documentation is a work in progress and is focused solely on the novel code developed within this repository. As such, classes presented in the models/ directory will not have hand-configured documentation, but are kept for future reference. Only local file hosting of the html documents is currently provided and no plans on web hosting the documentation are currently planned. 

### FCIoU: A Focal Class-based Intersection over Union (IoU) Approach to Improving Minority Class Detection Performance for Off-road Segmentation Systems
## Abstract
In this paper, we present a comparative study of modern semantic segmentation loss functions and their resultant impact when applied with state-of-the-art off-road datasets. Class imbalance, inherent in these datasets, presents a significant challenge to off-road terrain semantic segmentation systems. With numerous environment classes being extremely sparse and underrepresented, model training becomes inefficient and struggles to comprehend the infrequent minority classes. As a solution to this problem, loss functions have been configured to take class imbalance into account and counteract this issue. To this end, we present a novel loss function, Focal Classbased Intersection over Union (FCIoU), which directly targets performance imbalance through the optimization of class-based Intersection over Union (IoU). The new loss function results in a general increase in class-based performance when compared to state-of-the-art targeted loss functions. 

### Getting Started 
#### Environment Setup & Installation
**Python** ver. **3.8.15**, **Pytorch** ver. **1.13.1+cu117**, & **TorchVision** ver. **0.14.1+cu117** were employed during development. Other, more recent versions of these may operate correctly, however only the specified versions were tested. 

Other libraries employed are defined within the *requirements.txt* file included in this repository. Project dependencies can be installed directly through:
```
pip install -r requirements.txt
```

##### Running the program
Once all the dependencies are installed and resolved, the code can be started by directly running the eval.py and train.py code. The specific configurations by these programs can be configured using the config/config_comparative_study.yaml.

#### First Time Setup

## Datasets
Before running the program, install one or more of the datasets supported in this development kit. Note that at this moment only the RUGD dataset and Rellis-3D dataset are supported. 

### RUGD Dataset
To use the RUGD dataset, the following options must be configured in the *config_comparative_study.yaml* file: 
- DB: 
  - DB_NAME: "rugd"
  - PATH: "/path/to/rugd/dataset/"

For each of switching, the _C.DB.RUGD.PATH parameter may be configured within the *config.py* file instead. In doing so, only the DB_NAME parameter in the yaml configuration file has to be altered to switch between datasets. 

When first using the dataset, pre-conversion of the original data is required to properly utilize the dataset. This is automatically completed by the dataloader and will trigger if a newly obtained dataset missing the conversions is detected. As the conversion is applied to all annotation images within the dataset, the conversion process is expected to take a considerable (~10-25 mins), but will only have to be completed once. The conversion process will result in a new directory "RUGD_modif_frames-with-annotations" which contains re-mapped annotation images. As such, the original directory containing the annotation images "RUGD_annotations" is no longer used by the package and can be safely removed as desired. This pre-conversion was selected to reduce the overhead resulting from the conversion and reduce the overall latency of the dataloader module. 

### Rellis-3D dataset
To use the Rellis-3D dataset, the following options must be configured in the *config_comparative_study.yaml* file:
- DB:
  - DB_NAME: "rellis"
  -  PATH: "/path/to/rellis/dataset/"

For each of switching, the _C.DB.RUGD.PATH parameter may be configured within the *config.py* file instead. In doing so, only the DB_NAME parameter in the yaml configuration file has to be altered to switch between datasets. 


### Loss Functions
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

### Models for Comparative Study
Contained within the models/ directory is a selection of state-of-the-art semantic segmentation modeles employed in the comparative study. The original implementations of the models were obtained from: 

HRNetv2 + OCR: https://github.com/HRNet/HRNet-Semantic-Segmentation
DeepLabv3+: https://github.com/VainF/DeepLabV3Plus-Pytorch
U-Net: https://amaarora.github.io/posts/2020-09-13-unet.html


#### Model Configuration
The configuration for the model utilized in training or evaluation can be selected and modified through the *MODEL_NAME* field in the configuration file. 

The strings to select each of the specified model are as follows: 
- *fciouvX*, where *X* refers to the desired version (fciouv1,etc.)
- *deeplabv3plus*,  Deeplabv3+
- *hrnet_ocr*, HRNetv2 & OCR





