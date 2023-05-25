 # This file contains useful tools employed in the development process
import datetime
def logTrainParams(
        batch_size = None, 
        base=None, 
        steps_per_epoch = None, 
        epochs = None,
        lr = None, 
        criterion = None, 
        kernel_size = None
):
    log = """ Model Training Parameters <br />
    ------------------------------------------------------------  <br />
    Date: {0} <br />
    ------------------------------------------------------------<br />
    Batch Size: {1} <br />
    Learning Rate: {2} <br />
    Base: {3} <br />
    Kernel Size: {4} <br />
    Epochs: {5} <br />
    Steps Per Epochs: {6} <br />
    Loss Function: {7} <br />
    -----------------------------------------------------------<br />  
    """.format(datetime.datetime.now(), batch_size, lr, base, kernel_size, epochs,
                steps_per_epoch, criterion.__class__.__name__)
    return log 
