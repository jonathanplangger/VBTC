#################################################################
# Tools.py                                                      #
# --------                                                      #
# File contains useful tools for development                    #
#################################################################
import torch
import numpy as np

def get_memory_allocated():
    """
        Returns the current amount of memory allocated to the CUDA device in MB.
    """ 
    return torch.cuda.memory_allocated() / pow(1024,2)

def get_memory_reserved(): 
    """
        Returns the current amount of memory reserved by Pytorch & CUDA
    """
    return torch.cuda.memory_reserved() / pow(1024,2)

def convTorch2Onnx(model_path): 
    """Creates a new ONNX file for the selected pytorch file. \nThis can then be imported for visualization in Netron.\n

    :param model_path: Path to the desired model file 
    :type model_path: string
    """

    #load the torch model with the given path
    torch_model = torch.load(model_path)

    x = torch.rand(1, 3, 1200, 1920).cuda() # random sample input of the CORRECT dimension
    torch.onnx.export(torch_model, x, 'test.onnx', input_names=["input"], output_names=['output'])


def getPossibleResolutions(): 
    ar = 1.6 # aspect-ratio for the source images 

    heights = np.arange(0,1200,1) # go through all values 
    widths = ar * heights # obtain all corresponding heights 

    for i,w in enumerate(widths): 
        if w.is_integer(): 
            print("{} , {}".format(heights[i], w))





if __name__ == "__main__": 
    # getPossibleResolutions()
    import torchvision 
    from torchinfo import summary
    
    model = torchvision.models.resnet34()
    
    model_summary = summary(model, input_size=(1, 3, 1200, 1920))
        

    
    
    
    
    pass 



