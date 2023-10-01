import torch
from data.harmfull_memes.covid_memes import HarmemeMemesDatasetCov
from data.harmfull_memes.policitical_memes import HarmemeMemesDatasetPol

def build_cov_memes_loader(args):

    # # Load the ROI features (Covid)
    train_ROI = torch.load(args.dir_roi+"/harmeme_cov_train_ROI.pt")
    val_ROI = torch.load(args.dir_roi+"/harmeme_cov_val_ROI.pt")
    test_ROI = torch.load(args.dir_roi+"/harmeme_cov_test_ROI.pt")

    # # Load the ENT features
    train_ENT = torch.load(args.dir_ent+"/harmeme_cov_train_ent.pt")
    val_ENT = torch.load(args.dir_ent+"/harmeme_cov_val_ent.pt")
    test_ENT = torch.load(args.dir_ent+"/harmeme_cov_test_ent.pt")

    # Harmful Meme dataset (Covid-Ternary data)
    data_dir_cov = args.dir_data+"/images"
    train_path_cov = args.dir_data+"/annotations/train.jsonl"
    dev_path_cov   = args.dir_data+"/annotations/val.jsonl"
    test_path_cov  = args.dir_data+"/annotations/test.jsonl"

    # Import the dataloader for the the Covid memes dataset
    dataset_train = HarmemeMemesDatasetCov(train_path_cov, data_dir_cov, train_ROI, train_ENT, args.context_length, args.bpe_path, args.extend, 'train')
    #dataset_val = HarmemeMemesDatasetCov(dev_path_cov, data_dir_cov, val_ROI, val_ENT, args.context_length, args.bpe_path, args.extend, 'val')
    dataset_test = HarmemeMemesDatasetCov(test_path_cov, data_dir_cov, test_ROI, test_ENT, args.context_length, args.bpe_path, args.extend, 'test')
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    else:
        train_sampler = None
    
    if args.distributed and getattr(args, 'distributed_val', True):
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        test_sampler = None
    
    eval_batch_size = args.batch_size_per_gpu  
    
    train_loader = torch.utils.data.DataLoader(
            dataset_train, 
            batch_size=args.batch_size_per_gpu, 
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


def build_pol_memes_loader(args):

    # # # Load the ROI features (Political)
    train_ROI = torch.load(args.dir_roi+"/harmeme_pol_train_ROI.pt")
    val_ROI = torch.load(args.dir_roi+"/harmeme_pol_val_ROI.pt")
    test_ROI = torch.load(args.dir_roi+"/harmeme_pol_test_ROI.pt")

    # # # Load the ENT features
    train_ENT = torch.load(args.dir_ent+"/harmeme_pol_train_ent.pt")
    val_ENT = torch.load(args.dir_ent+"/harmeme_pol_val_ent.pt")
    test_ENT = torch.load(args.dir_ent+"/harmeme_pol_test_ent.pt")


    # Harmful Meme dataset (Political-Binary)
    data_dir_pol = args.dir_data+'/images'
    train_path_pol = args.dir_data+'/annotations/train.jsonl'
    dev_path_pol = args.dir_data+'/annotations/val.jsonl'
    test_path_pol = args.dir_data+'/annotations/test.jsonl'

    # Import the dataloader for the the Covid memes dataset
    dataset_train = HarmemeMemesDatasetPol(train_path_pol, data_dir_pol, train_ROI, train_ENT, args.context_length, args.bpe_path, args.extend, 'train')
    #dataset_val = HarmemeMemesDatasetPol(dev_path_pol, data_dir_pol, val_ROI, val_ENT, args.context_length, args.bpe_path, args.extend, 'val')
    dataset_test = HarmemeMemesDatasetPol(test_path_pol, data_dir_pol, test_ROI, test_ENT, args.context_length, args.bpe_path, args.extend, 'test')
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    else:
        train_sampler = None
    
    if args.distributed and getattr(args, 'distributed_val', True):
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        test_sampler = None
    
    eval_batch_size = args.batch_size_per_gpu   
    

    train_loader = torch.utils.data.DataLoader(
            dataset_train, 
            batch_size=args.batch_size_per_gpu, 
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



if __name__ == "__main__":

    import sys

    # build args
    import argparse
    parser = argparse.ArgumentParser(description='NAS for GNN')
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--dataset', default="memes_politics")
    parser.add_argument('--distributed', default=False)
    parser.add_argument('--dir_roi',default="/home/mghebriout/datasets/hatefull_memes/harmeme_ROI_MOMENTA/pol/harmfulness")
    parser.add_argument('--dir_ent',default="/home/mghebriout/datasets/hatefull_memes/harmeme_ENT_MOMENTA/pol/harmeme_pol_harmfulness")
    parser.add_argument('--dir_data',default="/home/mghebriout/datasets/hatefull_memes/Harmeme_HarmP_Data/data/datasets/memes/defaults")
    parser.add_argument('--bpe_path',default="/home/mghebriout/datasets/hatefull_memes/bpe_simple_vocab_16e6.txt.gz")
    parser.add_argument('--data_loader_workers_per_gpu', default=1)
    parser.add_argument('--context_length', default=77)
    parser.add_argument('--extend', default=0)
    parser.add_argument('--batch_size_per_gpu', default=1)
    parser.add_argument('--drop_last', default=True)
    parser.add_argument('--small_dataset', default=False)
    


    
    
    args = parser.parse_args()
    
    train, test, _ = build_pol_memes_loader(args)
    print("len train :",len(train))
    print("len test :",len(test))