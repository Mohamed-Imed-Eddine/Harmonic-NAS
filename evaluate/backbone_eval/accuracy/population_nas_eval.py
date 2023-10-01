import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import time
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import random
import utils.comm as comm

from .single_subnet_eval import new_validate_one_subnet
from evaluate.backbone_eval.efficiency import look_up_ofa_proxy





def validate_population(train_loader, val_loader, supernet, population, args,lut_data, modal_num=0, bn_calibration=True, in_channels=3, resolution=28):
    new_population = []
    supernet = supernet.module \
        if isinstance(supernet, torch.nn.parallel.DistributedDataParallel) else supernet
    for indiv in population:
        
        supernet.set_active_subnet(
            indiv['ks'],
            indiv['e'],
            indiv['d'],
        )
        
        subnet = supernet.get_active_subnet()
        subnet.cuda(args.gpu)
        if bn_calibration:
            subnet.eval()
            subnet.reset_running_stats_for_calibration()
            
            for batch_idx, data in enumerate(train_loader):
                if batch_idx >= args.post_bn_calibration_batch_num:
                    break
                modality1, modality2, target = data
                if modal_num == 0:
                    input_data = modality1
                elif modal_num == 1:
                    input_data = modality2
                input_data = input_data.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)   
                subnet(input_data)
                
                
        performance = new_validate_one_subnet(val_loader, subnet, args, modal_num=modal_num)
        print("GPU {}, evaluated backbone :{} ".format(args.gpu, performance))
        # print("GPU {}, the Acc: ".format(args.gpu), Acc)
        Lat, Enrg = look_up_ofa_proxy(net=subnet, lut=lut_data, resolution=resolution, supernet=args.supernet_arch, num_channels=in_channels)
        indiv['latency'] = Lat
        indiv['energy'] = Enrg
        if(args.task=='classification'):
            indiv['Acc@1'] = performance
        elif(args.task=='multilabel'):
            indiv['F1-W@1'] = performance

        if(args.task=='classification'):
            summary = str({'ks':indiv['ks'], 
                        'e':indiv['e'], 
                        'd':indiv['d'], 
                        'Acc@1':indiv['Acc@1'], 
                        'latency':indiv['latency'], 
                        'energy':indiv['energy'],
                        'net_id':indiv['net_id']})
        
        elif(args.task=='multilabel'):
            
            summary = str({'ks':indiv['ks'], 
                        'e':indiv['e'], 
                        'd':indiv['d'], 
                        'F1-W@1':indiv['F1-W@1'], 
                        'latency':indiv['latency'], 
                        'energy':indiv['energy'],
                        'net_id':indiv['net_id']})        

        
        if args.distributed and getattr(args, 'distributed_val', True):
                results += [summary]
        else:
            group = comm.reduce_eval_results(summary, args.gpu)
            new_population += group
        
        
    return new_population