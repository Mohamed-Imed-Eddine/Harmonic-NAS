from __future__ import print_function

import torch
import torchvision.transforms.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import math
import sys
import random
from PIL import Image

from torch.utils.data.distributed import DistributedSampler
import os
import numpy as np

from .data_transform import get_data_transform

from data.imdb_dataloader import build_imdb_loader
from data.avmnist_dataloader import build_avmnist_image_loader, build_avmnist_loader, build_avmnist_sound_loader
from data.memes_dataloader import build_cov_memes_loader, build_pol_memes_loader


def build_data_loader(args):
    if args.dataset == 'avmnist':
        return build_avmnist_loader(args)
    elif args.dataset =='imdb':
        return build_imdb_loader(args)
    elif args.dataset == 'avmnist_image':
        return build_avmnist_image_loader(args)
    elif args.dataset == 'avmnist_sound':
        return build_avmnist_sound_loader(args)
    elif args.dataset == "memes_covid":
        return build_cov_memes_loader(args)
    elif args.dataset == "memes_politics":
        return build_pol_memes_loader(args)
    else:
        raise NotImplementedError
    
