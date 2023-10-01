import argparse
import random
import torch
from utils.config import setup


from backbones.ofa.model_zoo import ofa_net
from backbones.ofa.utils.layers import LinearLayer, ConvLayer
from fusion_search.train_fusion_search import train_darts_model
from fusion_search.search.darts.utils import count_genotype_hardware_metrics
from evaluate.backbone_eval.accuracy.population_nas_eval import new_validate_one_subnet
from evaluate.backbone_eval.efficiency import EfficiencyEstimator, look_up_ofa_proxy
from data.data_loader import build_data_loader
import os
directory = os.path.dirname(os.path.abspath(__name__))
directory = "../Harmonic-NAS"
exp_name = "Best_AVMNIST"




parser = argparse.ArgumentParser(description='NAS for the Fusion Network Micro-Architecture')
parser.add_argument('--config-file', default=directory+'/configs/search_config_avmnist.yml')
parser.add_argument('--seed', default=42, type=int, help='default random seed')
parser.add_argument("--net", metavar="OFANET", default= "ofa_mbv3_d234_e346_k357_w1.0", help="OFA networks")


run_args = parser.parse_args()

def eval_worker(args):
        


    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)

    # Build the supernets for both modalities
    image_supernet = ofa_net(args.net, resolution=args.image_resolution, pretrained=False, in_ch=args.in_channels, _type='avmnist')
    image_supernet.classifier = LinearLayer(image_supernet.classifier.in_features, args.n_classes, dropout_rate=args.dropout) 
    image_supernet.first_conv = ConvLayer(args.in_channels, image_supernet.first_conv.out_channels, kernel_size=image_supernet.first_conv.kernel_size,
                                 stride=image_supernet.first_conv.stride, act_func="h_swish")
    image_supernet.cuda(args.gpu)


    sound_supernet = ofa_net(args.net, resolution=args.sound_resolution, pretrained=False, in_ch=args.in_channels, _type='avmnist')
    sound_supernet.classifier = LinearLayer(sound_supernet.classifier.in_features, args.n_classes, dropout_rate=args.dropout) 
    sound_supernet.first_conv = ConvLayer(args.in_channels, sound_supernet.first_conv.out_channels, kernel_size=sound_supernet.first_conv.kernel_size,
                                 stride=sound_supernet.first_conv.stride, act_func="h_swish")
    sound_supernet.cuda(args.gpu)


    # Load LUTs for latency/energy characterization on the targeted Edge device
    lut_data_image = EfficiencyEstimator(fname=args.hw_lut_path, supernet=args.supernet_arch)
    lut_data_sound = EfficiencyEstimator(fname=args.hw_lut_path, supernet=args.supernet_arch)
    
    # Load dataset
    train_loader, test_loader, train_sampler = build_data_loader(args)
    # Reloading supernet pretrained weights
    assert args.pretrained_path1 and args.pretrained_path2
    image_supernet.load_weights_from_pretrained_supernet(args.pretrained_path1)
    sound_supernet.load_weights_from_pretrained_supernet(args.pretrained_path2)

        
        
        
    # Select the unimodal backbones that gave the best accuracy in the multimodal setting     
    image_supernet.set_active_subnet(
    [3,5,7,3],
    [3,3,3,6],
    [2],
    )

    sound_supernet.set_active_subnet(
    [3,5,5,5],
    [3,3,3,3],
    [2],
    )
    
    

    

    image_subnet = image_supernet.get_active_subnet()
    sound_subnet = sound_supernet.get_active_subnet()
    image_subnet.cuda(args.gpu)
    sound_subnet.cuda(args.gpu)
    image_subnet.eval()
    sound_subnet.eval()
    image_subnet.reset_running_stats_for_calibration()
    sound_subnet.reset_running_stats_for_calibration()

    for batch_idx, data in enumerate(train_loader):
        if batch_idx >= args.post_bn_calibration_batch_num:
            break
        modality1, modality2, target = data
        modality1 = modality1.cuda(args.gpu, non_blocking=True)
        modality2 = modality2.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)   
        image_subnet(modality1)
        sound_subnet(modality2)
                
            
            
    # Compute the accuracy and the latency/energy of the selected unimodal backbones        
    acc1 = new_validate_one_subnet(test_loader, image_subnet, args, modal_num=0)
    acc2 = new_validate_one_subnet(test_loader, sound_subnet, args, modal_num=1)
    Lat1, Enrg1 = look_up_ofa_proxy(net=image_subnet, lut=lut_data_image, resolution=args.image_resolution, supernet=args.supernet_arch, num_channels=args.in_channels)
    Lat2, Enrg2 = look_up_ofa_proxy(net=sound_subnet, lut=lut_data_sound, resolution=args.sound_resolution, supernet=args.supernet_arch, num_channels=args.in_channels)
    print("Unimodal Performance:")
    print("Image Backbone: Acc: {:.3f}%, Latency: {:.3f}ms, Energy: {:.3f}mJ".format(acc1, Lat1, Enrg1))
    print("Audio Backbone: Acc: {:.3f}%, Latency: {:.3f}ms, Energy: {:.3f}mJ".format(acc2, Lat2, Enrg2))
        
                    
    dataloaders = {
            'train': train_loader,
            'test': test_loader
            }

    # Select the fusion macro-architecture that gave the best accuracy in the multimodal setting
    args.steps = 3
    args.node_steps = 4







    num_chosen_blocks1 = 4
    num_chosen_blocks2 = 4
    subnet1_channels = []
    out = image_subnet(torch.randn(2, 1, 28, 28).cuda(args.gpu))
    for i in range(len(out)):
        subnet1_channels.append(out[i].shape[1])
    chosen_channels_idx1 = []
    offset = (len(subnet1_channels)-1) // num_chosen_blocks1
    idx = len(subnet1_channels)-2
    chosen_channels_idx1.append(idx)
    for i in range(num_chosen_blocks1-1):
        idx -= offset
        chosen_channels_idx1.append(idx)
    chosen_channels_idx1.reverse()
    subnet2_channels = []
    out = sound_subnet(torch.randn(2, 1, 20, 20).cuda(args.gpu))
    for i in range(len(out)):
        subnet2_channels.append(out[i].shape[1])
    chosen_channels_idx2 = []
    offset = (len(subnet2_channels)-1) // num_chosen_blocks2
    idx = len(subnet2_channels)-2
    chosen_channels_idx2.append(idx)
    for i in range(num_chosen_blocks2-1):
        idx -= offset
        chosen_channels_idx2.append(idx)
    chosen_channels_idx2.reverse()
    args.num_input_nodes = num_chosen_blocks1 + num_chosen_blocks2


    # Run the search for the fusion micro-architecure
    MM_Acc, fusion_genotype = train_darts_model(dataloaders=dataloaders,args=args, gpu=args.gpu, 
                            network1=image_subnet, 
                            network2=sound_subnet, 
                            chosen_channels_idx1=chosen_channels_idx1, 
                            chosen_channels_idx2=chosen_channels_idx2,
                            subnet1_channels=subnet1_channels, 
                            subnet2_channels=subnet2_channels, 
                            phases=['train', 'test'], steps=args.steps, node_steps=args.node_steps)


    print("Multimodal Performance:")
    print("Multimodal Accuracy: {:.3f}% ".format(MM_Acc))    
    fusion_metrics = count_genotype_hardware_metrics(fusion_genotype, args)
    print("Fusion Network: Latency: {:.3f}ms, Energy: {:.3f}mJ".format(fusion_metrics['lat'], fusion_metrics['enrg']))


            

        
        
        



        
if __name__ == '__main__':
        
    args = setup(run_args.config_file)
    args.net = run_args.net
    args.gpu = 0
    eval_worker(args=args)
