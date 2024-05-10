import torch

## This class is a wrapper for the PyTorch learning rate scheduler. It is used to allow for the use of the ReduceLROnPlateau scheduler 
## or any scheduler that requires an argument in the same way as the other schedulers.
class SchedulerWrapper:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    ## This method is used to update the learning rate of the optimizer. If the scheduler is an instance of ReduceLROnPlateau,
    ## the metric is used to update the learning rate. Otherwise, the learning rate is updated without using the metric.
    def step(self, metric=None):
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(metric)
        else:
            self.scheduler.step()