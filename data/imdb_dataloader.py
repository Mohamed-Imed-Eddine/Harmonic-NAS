import torch
from torch.utils.data import Dataset
import os
import numpy as np
from torchvision import datasets, transforms
from torchvision.transforms import Resize

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        image, text, label = sample['image'], sample['text'], sample['label']

        return [torch.from_numpy(image.astype(np.float32)),
                torch.from_numpy(text.astype(np.float32)),
                torch.from_numpy(label.astype(np.float32))]



class MM_IMDB(torch.utils.data.Dataset):

    def __init__(self, root_dir='', 
                 transform=None,
                 stage='train',
                 feat_dim=100,
                 args=None):
        """
        Args:
            root_dir (string): Directory where data is.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        if stage == 'train':
            self.len_data = 15552
        elif stage == 'test':
            self.len_data = 7799
        elif stage == 'dev':
            self.len_data = 2608    
        if args.small_dataset:
            self.len_data = 64



        self.transform = transform
        self.root_dir = root_dir
        self.stage = stage


        global fdim
        fdim = feat_dim

    def __len__(self):
        return self.len_data

    def __getitem__(self, idx):

        imagepath = os.path.join(self.root_dir, self.stage, 'image_{:06}.npy'.format(idx))
        labelpath = os.path.join(self.root_dir, self.stage, 'label_{:06}.npy'.format(idx))
        textpath = os.path.join(self.root_dir, self.stage, 'text_{:06}.npy'.format(idx))

        image = np.load(imagepath)
        label = np.load(labelpath)
        text = np.load(textpath)


        # Resize the images to 224x224
        torch_image = torch.from_numpy(image).long()
        torch_image = Resize(size=(224,224), antialias=None)(torch_image)
        image = torch_image.numpy()

        textlen = text.shape[0]
        # modification
        sample = {'image': image, 
                  
                  'text': text, 
                  'label': label
                #   , 
                #   'textlen': textlen
                }

        if self.transform:
            sample = self.transform(sample)

        return sample




def build_imdb_loader(args, train_shuffle=True, flatten_audio=False, flatten_image=False, \
                         unsqueeze_channel=True, generate_sample=False, normalize_image=True, normalize_audio=True):

    transformer_val = transforms.Compose([ToTensor()])
    transformer_tra = transforms.Compose([ToTensor()])
    
    dataset_training = MM_IMDB(args.dataset_dir, transform=transformer_tra, stage='train', feat_dim=300, args=args)
    dataset_test = MM_IMDB(args.dataset_dir, transform=transformer_val, stage='test', feat_dim=300, args=args)
    

    
    
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
    
    
    train_loader = torch.utils.data.DataLoader(
            dataset_training, 
            batch_size=args.batch_size, 
            shuffle=(train_sampler is None), 
            sampler = train_sampler,
            drop_last = getattr(args, 'drop_last', True),
            num_workers=args.data_loader_workers_per_gpu,
            pin_memory=True
        ) 
    

    
    test_loader = torch.utils.data.DataLoader(
            dataset_test, 
            batch_size=eval_batch_size, 
            shuffle=False, 
            num_workers=args.data_loader_workers_per_gpu,
            drop_last=False,
            pin_memory=True,
            sampler=test_sampler
        ) 

    return train_loader, test_loader, train_sampler