import torch
import functools

if torch.__version__.startswith('0'):
    from .sync_bn.inplace_abn.bn import InPlaceABNSync
    BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
    BatchNorm2d_class = InPlaceABNSync
    relu_inplace = False
else:
    # BatchNorm2d_class = BatchNorm2d = torch.nn.SyncBatchNorm
    BatchNorm2d_class = BatchNorm2d = torch.nn.InstanceNorm2d # use instance norm to increase the size for the input
    relu_inplace = True