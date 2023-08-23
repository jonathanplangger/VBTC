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
