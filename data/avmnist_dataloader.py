import torch
import numpy as np
from torch.utils.data import DataLoader
import sys
from .avmnist.sound import Sound
from .avmnist.mnist import MNIST
from .avmnist.soundmnist import SoundMNIST

def build_avmnist_image_loader(args):
    
    dataset_training = MNIST(root=args.dataset_dir+'mnist/', per_class_num=105, train=True)
    dataset_test = MNIST(root=args.dataset_dir+'mnist/', train=False)
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_training)
    else:
        train_sampler = None
    
    if args.distributed and getattr(args, 'distributed_val', True):
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        test_sampler = None
    
    eval_batch_size = min(args.batch_size, 16) \
        if not getattr(args, 'eval_only', False) else args.batch_size
    
    
    train_loader = DataLoader(
            dataset_training, 
            batch_size=args.batch_size, 
            shuffle=(train_sampler is None), 
            sampler = train_sampler,
            drop_last = getattr(args, 'drop_last', True),
            num_workers=args.data_loader_workers_per_gpu,
            pin_memory=True
        ) 
        
    test_loader = DataLoader(
            dataset_test, 
            batch_size=eval_batch_size, 
            shuffle=False, 
            num_workers=args.data_loader_workers_per_gpu,
            drop_last=False,
            pin_memory=True,
            sampler=test_sampler
        ) 
    # print("in avmnist dataloader ",len(train_loader))
    return train_loader, test_loader, train_sampler


def build_avmnist_sound_loader(args):
    
    dataset_training = Sound(sound_root=args.dataset_dir+'sound_450/', per_class_num=100, train=True)
    dataset_test = Sound(sound_root=args.dataset_dir+'sound_450/', per_class_num=100, train=False)
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_training)
    else:
        train_sampler = None
    
    if args.distributed and getattr(args, 'distributed_val', True):
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        test_sampler = None
    
    eval_batch_size = min(args.batch_size, 16) \
        if not getattr(args, 'eval_only', False) else args.batch_size
    
    
    train_loader = DataLoader(
            dataset_training, 
            batch_size=args.batch_size, 
            shuffle=(train_sampler is None), 
            sampler = train_sampler,
            drop_last = getattr(args, 'drop_last', True),
            num_workers=args.data_loader_workers_per_gpu,
            pin_memory=True
        ) 
        
    test_loader = DataLoader(
            dataset_test, 
            batch_size=eval_batch_size, 
            shuffle=False, 
            num_workers=args.data_loader_workers_per_gpu,
            drop_last=False,
            pin_memory=True,
            sampler=test_sampler
        ) 
    # print("in avmnist dataloader ",len(train_loader))
    return train_loader, test_loader, train_sampler



def build_avmnist_loader(args, train_shuffle=True, flatten_audio=False, flatten_image=False, \
                         unsqueeze_channel=True, generate_sample=False, normalize_image=True, normalize_audio=True):   
    
    
    if(args.small_dataset):
            dataset_training = SoundMNIST(img_root=args.dataset_dir+'mnist/',sound_root=args.dataset_dir+'sound_450/', per_class_num=60, train=True)
            dataset_test = SoundMNIST(img_root=args.dataset_dir+'mnist/',sound_root=args.dataset_dir+'sound_450/', per_class_num=60, train=False)
    else:     
        dataset_training = SoundMNIST(img_root=args.dataset_dir+'mnist/',sound_root=args.dataset_dir+'sound_450/', per_class_num=105, train=True)
        dataset_test = SoundMNIST(img_root=args.dataset_dir+'mnist/',sound_root=args.dataset_dir+'sound_450/', per_class_num=150, train=False)
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_training)
    else: 
        train_sampler = None
    
    if args.distributed and getattr(args, 'distributed_val', True):
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        test_sampler = None
        
    eval_batch_size = min(args.batch_size, 16) \
        if not getattr(args, 'eval_only', False) else args.batch_size     
    
    train_loader = DataLoader(
            dataset_training, 
            batch_size=args.batch_size, 
            shuffle=(train_sampler is None), 
            sampler = train_sampler,
            drop_last = getattr(args, 'drop_last', True),
            num_workers=args.data_loader_workers_per_gpu,
            pin_memory=True
        )
    
    test_loader = DataLoader(
            dataset_test, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.data_loader_workers_per_gpu,
            drop_last=False,
            pin_memory=True,
            sampler=test_sampler
        )
    
    return train_loader, test_loader, train_sampler



