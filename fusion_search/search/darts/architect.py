import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

from fusion_search.search.darts.utils import count_supernet_hardware_metrics

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

class Architect(object):
    def __init__(self, model, args, criterion, optimizer):
        self.network_weight_decay = args.weight_decay
        self.criterion = criterion
        self.model = model
        self.optimizer = optimizer
    
    def log_learning_rate(self, logger):
        for param_group in self.optimizer.param_groups:
            logger.info("Architecture Learning Rate: {}".format(param_group['lr']))
            break
    
    def step(self, input_valid, target_valid, logger, args):
        self.optimizer.zero_grad()
        self._backward_step(input_valid, target_valid, args)
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid, args):
        
        
        # hardware_evaluation = count_supernet_hardware_metrics(self.model, args)
        # a = 0.5
        # b = 0.5
        # c = 0.5
        # loss = (self.criterion(logits, target_valid)**a) + (hardware_evaluation['lat']**b) + (hardware_evaluation['enrg']**c)
        
        
        logits = self.model(input_valid)
        loss = self.criterion(logits, target_valid)
        # why loss has no grad?
        loss.backward()
    