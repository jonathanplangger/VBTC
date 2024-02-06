import torch, torchvision
import torch.nn as nn
import sys, os
from torchinfo import summary
import unet
import torch.autograd.profiler as profiler 
from lib.partialconv.models.partialconv2d import PartialConv2d # import the partial convolutions

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Encoder(nn.Module): 
    def __init__(self):
        super().__init__()
        self.input_handling = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)
    
    def forward(self,x): 
        return self.input_handling(x) 

class TestLayer(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.encoder = Encoder()

    def forward(self, x):
        return self.encoder(x)
    

class NewModel(nn.Module): 
    def __init__(self, num_classes = 19): 
        super().__init__()
        self.peripheral = PeripheralUnit()
        self.central = CentralUnit()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)

    def forward(self, x):
        peripheral, mask = self.peripheral(x)
        central = self.central(x, mask)
        cat = torch.cat((peripheral, central), 0)
        cat = torch.permute(cat, (1,0,2,3)) # shift around before completing convolution
        conv1 = self.conv1(cat)
        conv1 = torch.permute(conv1, (1,0,2,3)) # shift the positions back to original location. 
        return conv1 # remove the extra dimension

class Lense(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation): 
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding = padding, dilation = dilation)

    def forward(self, x): 
        return self.conv(x) 

class CentralBlock(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super().__init__()
        self.conv1 = Lense(in_channels=in_channels, out_channels=out_channels, kernel_size=7, padding=3, dilation = 1)
        self.conv2 = Lense(in_channels=in_channels, out_channels=out_channels, kernel_size=7, padding=6, dilation = 2)
        self.conv3 = Lense(in_channels=in_channels, out_channels=out_channels, kernel_size=7, padding=9, dilation = 3)
        self.conv4 = Lense(in_channels=in_channels, out_channels=out_channels, kernel_size=7, padding=12, dilation = 4)
        self.conv5 = Lense(in_channels=in_channels, out_channels=out_channels, kernel_size=7, padding=15, dilation = 5)
    
    def forward(self, x): 
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = self.conv5(x)
        # Stack the resulting convolutions together into one stack layer
        return torch.cat((conv1, conv2, conv3, conv4, conv5), 1)

class CentralUnit(nn.Module):
    # TODO: Update the names of each layer to be more fitting a variable (set) amount of channels 
    def __init__(self, depth = 6, num_classes=19): 
        super().__init__()
        self.num_classes = num_classes # set the n# of classes which will be featured on output
        self.cb1 = CentralBlock(3, depth)
        self.conv1d1 = nn.Conv2d(in_channels=depth*5, out_channels=20, kernel_size=1) #intermediary layer convolutions
        self.cb2 = CentralBlock(20, depth)
        self.conv1d2 = nn.Conv2d(in_channels=depth*5, out_channels=num_classes, kernel_size=1)


    def forward(self, x, mask=torch.zeros(19,1200,1920)): 
        # expand the tensor to include mask for all classes. 
        mask = mask.repeat(self.num_classes,1,1) 
        
        #First stack calculation
        s1 = self.cb1(x)
        # apply k=1 convolution to obtain a single resulting map
        c1 = self.conv1d1(s1)
        s2 = self.cb2(c1)
        # c2 = self.conv1d2(s2, mask_in = mask)
        c2 = self.conv1d2(s2)

        # Apply the mask to the convolution and return output
        return torch.mul(c2, mask)



class PeripheralUnit(nn.Module): 
    def __init__(self): 
        super().__init__()
        base = 10
        kernel_size = 5
        num_class = 19
        self.topk = 10

        self.peripheral = unet.UNet(
            enc_chs=(3,base, base*2, base*4, base*8, base*16),
            dec_chs=(base*16, base*8, base*4, base*2, base), 
            out_sz=(1200,1920), retain_dim=True, num_class=num_class, kernel_size=kernel_size
        )

    def forward(self,x):
        
        # obtain the segmentatoin results for the peripheral backbone network
        pred = self.peripheral(x)
        # Get the probabilistic prediction for each class. 
        prob = torch.softmax(pred, dim=1)
        
        max_vals,_ = prob.max(dim=1) # get the max values for each pixels
        max_vals = max_vals.flatten() # flatten the matrix to make the topk operation work
        num_k = self.topk/100*max_vals.shape[0] # obtain the n# of values to be represented 
        _, topk = torch.topk(max_vals, dim = 0, k = int(num_k), largest=False)

        # Create the mask based on the topk indexes. 
        mask = torch.zeros(max_vals.shape, device=device)
        mask.scatter_(0, topk, 1.)
        mask = torch.reshape(mask, (1,prob.shape[2],prob.shape[3])) # reshape into the original dimensions
    
        return pred, mask
    





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