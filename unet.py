import torch.nn as nn 
import torchvision 
import torch 
import torchsummary
import torch.nn.functional as F

# Code obtained from: 
# https://amaarora.github.io/2020/09/13/unet.html

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, padding=2)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding=2)
    
    # configure forward pass through the Block
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(nn.Module):
    def __init__(self, chs=(3,64,128,256,512,1024), kernel_size = 3):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1], kernel_size) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64), kernel_size = 3):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1], kernel_size) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    """
        UNet Pytorch model developed using code from https://amaarora.github.io/2020/09/13/unet.html.\n 
        --------------------------------------------------\n 
        Parameters:\n 
            - en_chs (Tuple[int,..]): 6-elem tuple representing the n# of features (channels) for each encoding layer\n 
            - dec_chs (Tuple[int,..]): 5-elem tuple representing the n# of features (channels) for each decoding layer\n
            - retain_dim (bool): Image output retains same size as original input if == True\n
            - out_sz (Tuple[int,int]): (height, width) - dimensions of the desired output image, only applied if retain_dim==True\n
    """
    def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1, retain_dim=False, out_sz=(572,572), kernel_size = 2):
        super().__init__()
        self.encoder     = Encoder(enc_chs, kernel_size=kernel_size)
        self.decoder     = Decoder(dec_chs, kernel_size=kernel_size)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim # bool - does the output retain the same dimension as the input?
        self.out_sz = out_sz # output size value   

    def forward(self, x):
        out = self.encoder(x)
        out      = self.decoder(out[::-1][0], out[::-1][1:])
        out      = self.head(out)
        # print(self.head.weight)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        return out

#used to test the model
if __name__ == "__main__":
    img_h = 1920
    img_w = 1200

    base = 2
    model = UNet(
        enc_chs=(3,base, base*2, base*4, base*8, base*16),
        dec_chs=(base*16, base*8, base*4, base*2, base), 
        out_sz=(img_h,img_w), retain_dim=True, num_class=35, kernel_size = 7
        )
    device = torch.device("cuda")
    model = model.to(device)
    torchsummary.summary(model, (3,1920,1200))