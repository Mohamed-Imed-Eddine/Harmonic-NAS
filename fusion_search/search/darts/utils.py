import os
import numpy as np
import torch
import torch.nn.functional as F
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import pickle
import json

class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res

class Cutout(object):
        def __init__(self, length):
                self.length = length

        def __call__(self, img):
                h, w = img.size(1), img.size(2)
                mask = np.ones((h, w), np.float32)
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)

                mask[y1: y2, x1: x2] = 0.
                mask = torch.from_numpy(mask)
                mask = mask.expand_as(img)
                img *= mask
                return img

def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    return train_transform, valid_transform

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def count_parameters(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)

def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)

def save(model, model_path):
    torch.save(model.state_dict(), model_path)

def load(model, model_path):
    model.load_state_dict(torch.load(model_path))

def save_pickle(obj, obj_path):
    obj_file = open(obj_path, "wb")
    pickle.dump(obj, obj_file)
    obj_file.close()

def load_pickle(obj_path):
    obj_file = open(obj_path, "rb")
    obj = pickle.load(obj_file)
    obj_file.close()
    return obj

def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1.-drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x

def create_exp_dir(path, save_logger, save_model, plot_arch):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))    
    
    if(plot_arch):
        os.mkdir(os.path.join(path, 'architectures'))
    if(save_model):
        os.mkdir(os.path.join(path, 'best'))


def count_supernet_hardware_metrics(model, args, steps, node_steps):

    # the order of the fusion operators is as follows: 'Sum','ScaleDotAttn','LinearGLU','ConcatFC','SE1','CatConvMish' 

    
    with open(args.fusion_lut, 'r') as f:
        data = json.load(f)

    latencies = torch.tensor([data[i]['lat'] for i in data])
    enrg_consumptions = torch.tensor([data[i]['enrg'] for i in data])
    latencies = torch.cat((latencies[:5], latencies[6:]))
    enrg_consumptions = torch.cat((enrg_consumptions[:5], enrg_consumptions[6:]))

    lat = 0.0
    enrg = 0.0
    
    for i in range(len(model.fusion_net.cell._step_nodes)):
        gamma = model.fusion_net.cell._step_nodes[i].gammas
        gamma = F.softmax(gamma, dim=-1)
        lat += torch.sum(torch.matmul(gamma,latencies))
        enrg += torch.sum(torch.matmul(gamma,enrg_consumptions))


    
    normalized_lat = min_max_normalize(lat, steps * node_steps * 0.05961958312988281, steps * node_steps * 0.6174259033203126)
    normalized_enrg = min_max_normalize(enrg, steps * node_steps * 0.2972280216217041, steps * node_steps * 1.1336467642784118)



    return {"lat": normalized_lat , "enrg" : normalized_enrg} 




def min_max_normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def count_genotype_hardware_metrics(genotype, args):

    all_fusions = []
    total_lat = 0.0
    total_enrg = 0.0
    
    with open(args.fusion_lut, 'r') as f:
        data = json.load(f)

    steps = genotype[1]
    
    for step in steps:
        inner_steps = step.inner_steps
        
        for fusion_step in inner_steps:
            # print("fusion step :", fusion_step)
            total_lat += data['{}-input:1_{}_{}-N:1-C:{}-L:{}'.format(fusion_step, args.C, args.L, args.C, args.L)]['lat']
            total_enrg += data['{}-input:1_{}_{}-N:1-C:{}-L:{}'.format(fusion_step, args.C, args.L, args.C, args.L)]['enrg']

    return {"lat": total_lat , "enrg" : total_enrg} 