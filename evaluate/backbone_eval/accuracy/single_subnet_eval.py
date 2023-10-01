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

from .utils.progress import AverageMeter
# from .utils.flops_counter import count_net_flops_and_params

# def log_helper(summary, logger=None):
#     if logger:
#         logger.info(summary)
#     else:
#         print(summary)


def validate_one_subnet(
    val_loader,
    subnet,
    args, 
):
    top1 = AverageMeter('F1-W@1', ':6.2f')
    top5 = AverageMeter('F1-W@1', ':6.2f')

    subnet.cuda(args.gpu)
    subnet.eval() # freeze again all running stats
   
    for batch_idx, (modal1, modal2, target) in enumerate(val_loader):
        input_data = modal1
        input_data = input_data.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = subnet(input_data)
        output = output[-1]
        

        
        preds_th = torch.sigmoid(output) > 0.25
        batch_output = preds_th.data.cpu().numpy()
        batch_target = target.data.cpu().numpy()
        acc1 = acc5 = f1_score(batch_output, batch_target, average='weighted', zero_division=1) * 100
        
        batch_size = input_data.size(0)
        


        top1.update(acc1, batch_size)
        top5.update(acc5, batch_size)

    return float(top1.avg), float(top5.avg)




def new_validate_one_subnet(val_loader,subnet,args,modal_num=0):
    
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    mm_model = subnet.cuda(args.gpu)
    mm_model.eval()
    th_fscore = 0.3
    list_preds = [] 
    list_label = []  
    
    for data in val_loader:
        
        modality1, modality2, label = data
        if modal_num==0:
            modality = modality1
        elif modal_num==1:
            modality = modality2
            
        
        modality = modality.cuda(args.gpu, non_blocking=True)           
        label = label.cuda(args.gpu, non_blocking=True)
        # with torch.no_grad():
        output = mm_model(modality)[-1]
        if(args.task=='multilabel'):
            preds_th = torch.sigmoid(output) > th_fscore
        elif(args.task=='classification'):
            preds_th = output
        list_preds.append(preds_th.cpu())
        list_label.append(label.cpu())     
    
    y_pred = torch.cat(list_preds, dim=0)
    y_true = torch.cat(list_label, dim=0)

    if(args.task=='multilabel'):
        performance = f1_score(y_true.numpy(), y_pred.numpy(), average='weighted', zero_division=1)*100
    elif(args.task=='classification'):
        y_pred = torch.argmax(y_pred, 1)
        performance = accuracy_score(y_true.detach().numpy(), y_pred.detach().numpy())*100

    return performance




# def nodist_validate_one_subnet(
#     val_loader,
#     subnet,
#     args, 
# ):
#     top1 = AverageMeter('F1-W@1', ':6.2f')
#     top5 = AverageMeter('F1-W@1', ':6.2f')

#     subnet.cuda(args.gpu)
#     subnet.eval() # freeze again all running stats
   
#     for batch_idx, (modal1, modal2, target) in enumerate(val_loader):
#         input_data = modal1
#         input_data = input_data.cuda(args.gpu, non_blocking=True)
#         target = target.cuda(args.gpu, non_blocking=True)

#         # compute output
#         output = subnet(input_data)
#         output = output[-1]
        

        
#         preds_th = torch.sigmoid(output) > 0.25
#         batch_output = preds_th.data.cpu().numpy()
#         batch_target = target.data.cpu().numpy()
#         acc1 = acc5 = f1_score(batch_output, batch_target, average='weighted', zero_division=1) * 100
        
#         batch_size = input_data.size(0)
        


#         top1.update(acc1, batch_size)
#         top5.update(acc5, batch_size)

#     return float(top1.avg), float(top5.avg)