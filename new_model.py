import torch, torchvision
import torch.nn as nn
import sys, os
from torchinfo import summary



class Encoder(nn.Module): 
    def __init__(self):
        super().__init__()
        self.input_handling = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=100, stride=1, padding=1)
    
    def forward(self,x): 
        return self.input_handling(x) 

class NewModel(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.encoder = Encoder()

    def forward(self, x):
        return self.encoder(x)










if __name__ == "__main__": 

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Use for testing the memory requirements for the new model
    model = NewModel().to(device)

    # Run the model summary to estimate the size for the model.
    model_summary = summary(model, input_size=(1, 3, 1200, 1920))