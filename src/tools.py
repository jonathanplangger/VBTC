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
    ------------------------------------------------------------<br />
    Date: {0}
    ------------------------------------------------------------<br />
    Batch Size: {1}
    Learning Rate: {2}
    Base: {3}
    Kernel Size: {4}
    Epochs: {5}
    Steps Per Epochs: {6}
    Loss Function: {7}
    -----------------------------------------------------------<br />  
    """.format(datetime.now(), batch_size, lr, base, kernel_size, epochs,
                steps_per_epoch, criterion.__class__.__name__)
    return log 
