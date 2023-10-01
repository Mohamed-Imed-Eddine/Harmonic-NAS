import argparse
import random
import torch
from utils.config import setup


from backbones.ofa.model_zoo import ofa_net
from backbones.maxout.maxout import Our_Maxout
from fusion_search.train_fusion_search import train_darts_model
from backbones.ofa.utils.layers import LinearLayer, ConvLayer
from fusion_search.search.darts.utils import count_genotype_hardware_metrics
from evaluate.backbone_eval.accuracy.population_nas_eval import new_validate_one_subnet
from evaluate.backbone_eval.efficiency import EfficiencyEstimator, look_up_ofa_proxy
from data.data_loader import build_data_loader
import os
directory = os.path.dirname(os.path.abspath(__name__))
directory = "../Harmonic-NAS"
exp_name = "Best_AVMNIST"




parser = argparse.ArgumentParser(description='NAS for the Fusion Network Micro-Architecture')
parser.add_argument('--config-file', default=directory+'/configs/search_config_memes_politics.yml')
parser.add_argument('--seed', default=42, type=int, help='default random seed')
parser.add_argument("--net", metavar="OFANET", default= "ofa_mbv3_d234_e346_k357_w1.0", help="OFA networks")


run_args = parser.parse_args()

def eval_worker(args):
        


    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)

    # Build the image supernet
    supernet = ofa_net(args.net, resolution=args.resolution, pretrained=False, in_ch=args.in_channels, _type=args._type)
    supernet.classifier = LinearLayer(supernet.classifier.in_features, args.n_classes, dropout_rate=args.dropout) 
    supernet.first_conv = ConvLayer(args.in_channels, supernet.first_conv.out_channels, kernel_size=supernet.first_conv.kernel_size,
                                 stride=supernet.first_conv.stride, act_func="h_swish")
    supernet.cuda(args.gpu)


    textnet = Our_Maxout(number_input_feats=77, num_outputs=3)
    with open(args.maxout_weights, 'rb') as f:
        checkpoint = torch.load(f, map_location='cpu')
        textnet.load_state_dict(checkpoint['state_dict'])



    # Load LUT for latency/energy characterization on the targeted Edge device
    lut_data = EfficiencyEstimator(fname=args.hw_lut_path, supernet=args.supernet_arch)

    
    # Load dataset
    train_loader, test_loader, train_sampler = build_data_loader(args)
    
    # Reloading supernet pretrained weights
    assert args.pretrained_path
    supernet.load_weights_from_pretrained_supernet(args.pretrained_path)


	
        
    # Select the unimodal backbone that gave the best accuracy in the multimodal setting     
    supernet.set_active_subnet(
    [3,3,3,3,5,3,3,3,5,5,3,5],
    [4,3,4,6,4,4,3,6,6,6,3,4],
    [2,3,2],
    )
    

    image_subnet = supernet.get_active_subnet()

    image_subnet.cuda(args.gpu)
    textnet.cuda(args.gpu)
    image_subnet.eval()
    textnet.eval()
    image_subnet.reset_running_stats_for_calibration()

    for batch_idx, data in enumerate(train_loader):
        if batch_idx >= args.post_bn_calibration_batch_num:
            break
        modality1, _, _ = data
        modality1 = modality1.cuda(args.gpu, non_blocking=True) 
        image_subnet(modality1)

                
                
    # Compute the accuracy and the latency/energy of the selected unimodal backbones        
    acc1 = new_validate_one_subnet(test_loader, image_subnet, args, modal_num=0)
    acc2 = new_validate_one_subnet(test_loader, textnet, args, modal_num=1)
    Lat1, Enrg1 = look_up_ofa_proxy(net=image_subnet, lut=lut_data, resolution=args.resolution, supernet=args.supernet_arch, num_channels=args.in_channels)

    if("tx2" in args.hw_lut_path):
        print("Unimodal Performance: (NVIDIA Jetson TX2)")
    elif("agx" in args.hw_lut_path):
        print("Unimodal Performance: (NVIDIA Jetson AGX)")
    
    print("Image Backbone: Acc: {:.3f}%, Latency: {:.3f}ms, Energy: {:.3f}mJ".format(acc1, Lat1, Enrg1))
    print("Text Backbone: Acc: {:.3f}%, Latency: 1.09ms, Energy: 1.40mJ".format(acc2))
        
         
    dataloaders = {
            'train': train_loader,
            'test': test_loader
            }

    # Select the fusion macro-architecture that gave the best accuracy in the multimodal setting
    args.steps = 2
    args.node_steps = 3







    num_chosen_blocks1 = 7
    subnet1_channels = []
    out = image_subnet(torch.randn(1, 3, 112, 112).cuda(args.gpu))
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
    

    args.num_input_nodes = num_chosen_blocks1 + 2


    # Run the search for the fusion micro-architecure
    MM_Acc, fusion_genotype = train_darts_model(dataloaders=dataloaders,args=args, gpu=args.gpu, 
                            network1=image_subnet, 
                            network2=textnet, 
                            chosen_channels_idx1=chosen_channels_idx1, 
                            chosen_channels_idx2=[0, 1],
                            subnet1_channels=subnet1_channels, 
                            subnet2_channels=[128, 256, 3], 
                            phases=['train', 'test'], steps=args.steps, node_steps=args.node_steps)


    print("Multimodal Performance:")
    print("Multimodal F1-W: {:.3f}% ".format(MM_Acc))    
    fusion_metrics = count_genotype_hardware_metrics(fusion_genotype, args)
    print("Fusion Network: Latency: {:.3f}ms, Energy: {:.3f}mJ".format(fusion_metrics['lat'], fusion_metrics['enrg']))


            

        
        

        
if __name__ == '__main__':
        
    args = setup(run_args.config_file)
    args.net = run_args.net
    args.gpu = 0
    eval_worker(args=args)
