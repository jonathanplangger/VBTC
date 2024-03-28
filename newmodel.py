import torch, torchvision
import torch.nn as nn
import sys, os
from torchinfo import summary
import unet
import torch.autograd.profiler as profiler 
from lib.partialconv.models.partialconv2d import PartialConv2d # import the partial convolutions
# Add the resnet to the available backbone to run the program
sys.path.append("models/DeepLabV3Plus-Pytorch/network/backbone/")
from resnet import resnet34


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class NewModel(nn.Module):
    def __init__(self): 
        super().__init__()
        self.peri = PeripheralUnit()  # get the peripheral viewing unit

    def forward(self, x):  
        return self.peri(x) 



class CentralUnit(nn.Module):
    # TODO: Update the names of each layer to be more fitting a variable (set) amount of channels 
    def __init__(self, depth = 6, num_classes=19): 
        super().__init__()

    def forward(self, x, mask=torch.zeros(19,1200,1920)): 
        pass

class PeripheralUnit(nn.Module): 
    def __init__(self): 
        super().__init__()
        backbone = resnet34() # use the resnet 34 as the backbone for the network
        # remove avg pooling layers on output (constraining it to 1000 classes output)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2]) 

    def forward(self,x):
        return self.backbone(x)






if __name__ == "__main__": 
    # Used for testing the models themselves
    import torch.autograd.profiler as profiler
 
    # model = NewModel().to(device)
    model = NewModel().to(device)

    # Run the model summary to estimate the size for the model.
    model_summary = summary(model, input_size=(1, 3, 1200, 1920))

    # Fake input to be used in benchmarking
    input = torch.rand(1,3,1200,1920).to(device)

    with profiler.profile(with_stack=True, profile_memory=True, use_cuda = True, record_shapes=True) as prof:     
        pred = model(input)

    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=20))
