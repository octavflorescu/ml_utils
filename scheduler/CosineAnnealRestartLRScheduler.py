from torch.optim import lr_scheduler
import warnings
import math
import numpy as np


class CosineAnnealRestartLRScheduler(lr_scheduler._LRScheduler):
    """
    Cosine annealing learning rate scheduling with restart.
    Also, the top of the LR after each resrt will respect the same cosine annealing config, but relative to the whole number of training steps.
    Can be looked at as 2 threads:
    - 1 to cosine aneal the maximum LR
    - 1 to cosine aneal the lr per each restart
    Overall, minimum is kept at initial_lr*cosine_final_lr_ratio, while maximum cosinely decreases.
    Between restarts, the LR moves from maximum of current batch, to the minimum LR
    Check experiments/results/ts_detection_readme.md  ' s CosineAnnealRestartLRScheduler chapter for more
    """

    def __init__(self, optimizer, max_epochs=10000, last_epoch=-1, initial_lr=1, warmup_epochs=1,
                 cosine_restart_epochs=10, cosine_top_final_lr_ratio=0.5, cosine_final_lr_ratio=0.333, cosine_warmup_perc=0.6, lr_depth_decay=True,
                 **kwargs):
        """
        :param lr_depth_decay: to decay the lr based on the model layers (backbone/neck/heads). heads will have highest lr.
        """
        self.max_epochs = max_epochs
        self.cosine_top_final_lr_ratio = cosine_top_final_lr_ratio
        self.cosine_final_lr_ratio = cosine_final_lr_ratio
        self.cosine_restart_epochs = cosine_restart_epochs
        self.cosine_warmup_perc = cosine_warmup_perc
        self.initial_lr = initial_lr
        self.warmup_lr = initial_lr * 0.1  # change this from 1e-1 to 10, for functionality as in ref images
        self.warmup_epochs = warmup_epochs
        self.last_epoch = last_epoch
        self.lr_depth_decay = lr_depth_decay
        
        super(CosineAnnealRestartLRScheduler, self).__init__(optimizer, last_epoch, verbose="deprecated")


    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        return [self.perform_scheduling(initial_lr=self.initial_lr, epoch=self.last_epoch)/(pow(3, ((len(self.optimizer.param_groups)-1) - indx)) if self.lr_depth_decay else 1)
                for indx, group in enumerate(self.optimizer.param_groups)]

    def _get_closed_form_lr(self):
        return [self.perform_scheduling(initial_lr=base_lr, epoch=self.last_epoch)
                for base_lr in self.base_lrs]

    def perform_scheduling(self, initial_lr, epoch):
        if epoch < self.warmup_epochs:
            return self.warmup_lr
        annealing_epoch = epoch - self.warmup_epochs
        annealing_max_epochs = self.max_epochs - self.warmup_epochs
        # top LR cosine annealing thread
        restart_batch_len = self.cosine_restart_epochs
        restart_start_iter = max(0, (annealing_epoch // restart_batch_len) * restart_batch_len)
        restart_max_iter = min(restart_start_iter + restart_batch_len, annealing_max_epochs)
        # batch_top_lr is the LR of the first item in restart batch
        batch_top_lr = self.compute_learning_rate(restart_start_iter, annealing_max_epochs, initial_lr,
                                                  bottom_lr_ratio=self.cosine_top_final_lr_ratio, warmup_perc=0)
        batch_bottom_lr = self.compute_learning_rate(restart_start_iter+restart_batch_len, annealing_max_epochs, initial_lr,
                                                     bottom_lr_ratio=self.cosine_final_lr_ratio, warmup_perc=0)
        current_iter = max(0, annealing_epoch) % restart_batch_len
        current_resart_max_iter = restart_batch_len \
            if current_iter < (restart_max_iter-(restart_max_iter%restart_batch_len)) \
            else (restart_max_iter%restart_batch_len)
        
        lr = self.compute_learning_rate(current_iter, current_resart_max_iter, batch_top_lr,
                                        bottom_lr_ratio=batch_bottom_lr/batch_top_lr,
                                        warmup_perc=self.cosine_warmup_perc)
        # deploy
        return lr

    def compute_learning_rate(cls, step, total_steps: float, initial_lr: float, bottom_lr_ratio: float, warmup_perc=0.4):
        # the cosine starts from initial_lr and reaches initial_lr * cosine_final_lr_ratio in last epoch
        warmup = (warmup_perc*total_steps)
        step = 0 if step < warmup else ((step-warmup)/(total_steps-warmup))*total_steps
        lr = 0.5 * initial_lr * (1.0 + np.cos(step / (total_steps + 1) * math.pi))
        return lr * (1 - bottom_lr_ratio) + (initial_lr * bottom_lr_ratio)

# add it to pytorch for easy access
lr_scheduler.CosineAnnealRestartLRScheduler = CosineAnnealRestartLRScheduler
