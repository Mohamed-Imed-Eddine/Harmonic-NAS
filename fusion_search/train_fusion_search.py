import torch
import torch.nn as nn
import torch.optim as op
import os


import fusion_search.auxiliary.scheduler as sc
import fusion_search.auxiliary.aux_models as aux

import fusion_search.search.train_searchable.search as tr


import numpy as np
import random
from fusion_search.search.darts.model_search import FusionNetwork
from fusion_search.search.plot_genotype import Plotter
from fusion_search.search.darts.architect import Architect
import torch.backends.cudnn as cudnn
import time
import logging
import sys
import os
from fusion_search.search.darts import utils as utils



def init_seed(args):
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)


def train_darts_model(dataloaders, args, gpu, 
        network1, network2,
        chosen_channels_idx1, chosen_channels_idx2,
        subnet1_channels, subnet2_channels,
        save_logger=False, save_model=False, plot_arch=False, phases=['train', 'dev', 'test'], steps=0, node_steps=0):

    
    device = torch.device("cuda:" + str(gpu) if torch.cuda.is_available() else "cpu")
    init_seed(args)
    
    
    if(save_model or save_logger or plot_arch):
        args.save = 'fusion_search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
        args.save = os.path.join('./results__fusion', args.save)
        utils.create_exp_dir(args.save, save_logger=save_logger, save_model=save_model, plot_arch=plot_arch)

    
    if(save_logger):    
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logger = logging.getLogger()
        logger.addHandler(fh)
    else:
        logger = None
    
    
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in phases}
    num_batches_per_epoch = dataset_sizes['train'] / args.batch_size
    
    if(args.task=='multilabel'):
        criterion = torch.nn.BCEWithLogitsLoss()
    elif(args.task=='classification'):
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError
        
    

    C_ins = []
    for i in chosen_channels_idx1:
        C_ins.append(subnet1_channels[i])
    
    for i in chosen_channels_idx2:
        C_ins.append(subnet2_channels[i])

    
    model = Searchable_Image_Text_Net(args, criterion, network1, network2, C_ins, chosen_channels_idx1, chosen_channels_idx2, steps=steps, node_steps=node_steps)
    params = model.central_params()
    # optimizer and scheduler
    optimizer = op.Adam(params, lr=args.eta_max, weight_decay=args.weight_decay)
    scheduler = sc.LRCosineAnnealingScheduler(args.eta_max, args.eta_min, args.Ti, args.Tm,
                                              num_batches_per_epoch)

    arch_optimizer = op.Adam(model.arch_parameters(),
            lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)



    model.to(device)
    architect = Architect(model, args, criterion, arch_optimizer)

    plotter = Plotter(args)
    start_time = time.time()
    best_f1, best_genotype = tr.train_mmimdb_track_f1(model, architect,
                                            criterion, optimizer, scheduler, dataloaders,
                                            dataset_sizes,
                                            device=device, 
                                            num_epochs=args.fusion_epochs, 
                                            parallel=args.parallel,
                                            logger=logger,
                                            plotter=plotter,
                                            args=args,
                                            init_f1=0.0, th_fscore=0.3, save_logger=save_logger, save_model=save_model, plot_arch=plot_arch, phases=phases, steps=steps, node_steps=node_steps)

    time_elapsed = time.time() - start_time
    if(logger is not None):
        logger.info("*" * 50)
        logger.info('Searching complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        logger.info('Now listing best fusion_net genotype:')
        logger.info(best_genotype)
        
        logger.info('Now listing best f1 :')
        logger.info(best_f1)
    
    return best_f1*100, best_genotype


class Searchable_Image_Text_Net(nn.Module):
    def __init__(self, args, criterion, network1, network2, C_ins, chosen_channels_idx1, chosen_channels_idx2, steps, node_steps):
        super().__init__()
        
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        self.args = args
        self.criterion = criterion        
        self.network1 = network1
        self.network2 = network2
        self.reshape_layers = self.create_reshape_layers(args, C_ins)
        self.multiplier = args.multiplier
        self.steps = steps
        self.parallel = args.parallel
        self.chosen_subnet_channels_idx1 = chosen_channels_idx1
        self.chosen_subnet_channels_idx2 = chosen_channels_idx2
        self.num_input_nodes = args.num_input_nodes
        self.num_keep_edges = args.num_keep_edges
        self.C_ins = C_ins
        
        self._criterion = criterion

        self.fusion_net = FusionNetwork( steps=self.steps, node_steps=node_steps, multiplier=self.multiplier, 
                                         num_input_nodes=self.num_input_nodes, num_keep_edges=self.num_keep_edges,
                                         args=self.args,
                                         criterion=self.criterion)
        
        self.central_classifier = nn.Linear(self.args.C * self.args.L * self.multiplier,
                                            args.num_outputs)

    def create_reshape_layers(self, args, C_ins):
        
        # resnet
        # C_ins = [64, 128, 256, 512, 128, 256]
        
        #mbv3 small
        # C_ins = [24, 40, 96, 576, 128, 256]
        
        #mbv3 large
        # C_ins = [40, 80, 112, 960, 128, 256]
        
        reshape_layers = nn.ModuleList()
        for i in range(len(C_ins)):
            reshape_layers.append(aux.ReshapeInputLayer_MMIMDB(C_ins[i], args.C, args.L, args))
        return reshape_layers

    def reshape_input_features(self, input_features):
        ret = []
        for i, input_feature in enumerate(input_features):
            reshaped_feature = self.reshape_layers[i](input_feature)
            ret.append(reshaped_feature)
        return ret

    def forward(self, tensor_tuple):
        modality1, modality2 = tensor_tuple

        
        # apply net1 on input 1
        temp = self.network1(modality1)
        modality1_features = []
        for i in self.chosen_subnet_channels_idx1:
            modality1_features.append(temp[i])

        
        
        # apply net2 on input 2
        temp = self.network2(modality2)
        modality2_features = []
        for i in self.chosen_subnet_channels_idx2:
            modality2_features.append(temp[i])
            


        
        input_features = list(modality1_features) + list(modality2_features)
        input_features = self.reshape_input_features(input_features)

        out = self.fusion_net(input_features)
        out = self.central_classifier(out)
        return out

    def genotype(self):
        return self.fusion_net.genotype()
    
    def central_params(self):
        central_parameters = [
            {'params': self.reshape_layers.parameters()},
            {'params': self.fusion_net.parameters()},
            {'params': self.central_classifier.parameters()}
        ]
        return central_parameters
    
    def _loss(self, input_features, labels):
        logits = self(input_features)
        return self._criterion(logits, labels) 

    def arch_parameters(self):
        return self.fusion_net.arch_parameters() 
