#################################################################
# Tools.py                                                      #
# --------                                                      #
# File contains useful tools for development                    #
#################################################################
import torch

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



if __name__ == "__main__": 
    convTorch2Onnx("saves/epoch1.pt")