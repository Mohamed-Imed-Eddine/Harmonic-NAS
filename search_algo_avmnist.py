import argparse
import random
import math
import pickle

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from utils.config import setup
import utils.comm as comm

from backbones.ofa.model_zoo import ofa_net
from backbones.ofa.utils.layers import LinearLayer, ConvLayer

from utils.optim import RankAndCrowdingSurvival_Outer_Acc, RankAndCrowdingSurvival_Inner_Acc
from utils.save import save_ooe_population, save_results, save_resume_population

from evaluate.backbone_eval.accuracy import subnets_nas_eval
from evaluate.backbone_eval.accuracy.population_nas_eval import validate_population
from evaluate.backbone_eval.efficiency import EfficiencyEstimator
from data.data_loader import build_data_loader
from datetime import datetime
import os
directory = os.path.dirname(os.path.abspath(__name__))
directory = "../Harmonic-NAS"
exp_name = "tx2_avmnist"


# f = directory+'/results/'+exp_name
# if not os.path.exists(f):
#     os.makedirs(f)

parser = argparse.ArgumentParser(description='Harmonic-NAS Search for the Complete MM-NN Architecure')
parser.add_argument('--config-file', default=directory+'/configs/search_config_avmnist.yml')
parser.add_argument('--machine-rank', default=0, type=int, help='machine rank, distributed setting')
parser.add_argument('--num-machines', default=1, type=int, help='number of nodes, distributed setting')
parser.add_argument('--dist-url', default="tcp://localhost:8080", type=str, help='init method, distributed setting')
parser.add_argument('--dist-backend', default="nccl", type=str, help='init method, distributed setting')
parser.add_argument('--seed', default=42, type=int, help='default random seed')
parser.add_argument('--resume-evo', default=0, type=int, help='Resume previous search')
parser.add_argument('--start-evo', default=0, type=int, help='evolution to resume')
parser.add_argument("--net", metavar="OFANET", default= "ofa_mbv3_d234_e346_k357_w1.0", help="OFA networks")


run_args = parser.parse_args()

def eval_worker(gpu, ngpus_per_node, args):
        
    args.gpu = gpu  # local rank, local machine cuda id
    args.local_rank = args.gpu
    args.batch_size = args.batch_size

    # set random seed, make sure all random subgraph generated would be the same 
    global_rank = args.gpu + args.machine_rank * ngpus_per_node
    dist.init_process_group(
        backend=args.dist_backend, 
        init_method=args.dist_url,
        world_size=args.world_size, 
        rank=global_rank
    )

    comm.synchronize()

    args.rank = comm.get_rank() # global rank

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)

    # Building  the supernets
    image_supernet = ofa_net(args.net, resolution=args.image_resolution, pretrained=False, in_ch=args.in_channels, _type='avmnist')
    image_supernet.classifier = LinearLayer(image_supernet.classifier.in_features, args.n_classes, dropout_rate=args.dropout) 
    image_supernet.first_conv = ConvLayer(args.in_channels, image_supernet.first_conv.out_channels, kernel_size=image_supernet.first_conv.kernel_size,
                                 stride=image_supernet.first_conv.stride, act_func="h_swish")
    image_supernet.cuda(args.gpu)
    image_supernet = comm.get_parallel_model(image_supernet, args.gpu) #local rank

    sound_supernet = ofa_net(args.net, resolution=args.sound_resolution, pretrained=False, in_ch=args.in_channels, _type='avmnist')
    sound_supernet.classifier = LinearLayer(sound_supernet.classifier.in_features, args.n_classes, dropout_rate=args.dropout) 
    sound_supernet.first_conv = ConvLayer(args.in_channels, sound_supernet.first_conv.out_channels, kernel_size=sound_supernet.first_conv.kernel_size,
                                 stride=sound_supernet.first_conv.stride, act_func="h_swish")
    sound_supernet.cuda(args.gpu)
    sound_supernet = comm.get_parallel_model(sound_supernet, args.gpu) #local rank

    # Load LUTs for latency/energy characterization on the targeted Edge device
    lut_data_image = EfficiencyEstimator(fname=args.hw_lut_path, supernet=args.supernet_arch)
    lut_data_sound = EfficiencyEstimator(fname=args.hw_lut_path, supernet=args.supernet_arch)
    
    ## load dataset, train_sampler: distributed
    train_loader, test_loader, train_sampler = build_data_loader(args)
    val_loader = None
    
    # Reloading supernet pretrained weights
    assert args.pretrained_path1 and args.pretrained_path2
    image_supernet.module.load_weights_from_pretrained_supernet(args.pretrained_path1)
    sound_supernet.module.load_weights_from_pretrained_supernet(args.pretrained_path2)
    comm.synchronize()
        
    # Initialize different first populations form the different backbones
    if(args.resume_evo == 0): # First population (evolution)
        parent_ooe_popu = []
        parent_ooe_popu1 = image_supernet.module.init_population(n_samples=args.evo_search_outer.parent_popu_size)
        parent_ooe_popu2 = sound_supernet.module.init_population(n_samples=args.evo_search_outer.parent_popu_size)
        for idx in range(len(parent_ooe_popu1)):
            parent_ooe_popu1[idx]['net_id'] = f'net_{idx % args.world_size}_evo_0_{idx}'
            parent_ooe_popu2[idx]['net_id'] = f'net_{idx % args.world_size}_evo_0_{idx}'
            couple = {'backbone1': parent_ooe_popu1[idx], 'backbone2': parent_ooe_popu2[idx], 'net_id': f'net_{idx % args.world_size}_evo_0_{idx}'}
            parent_ooe_popu.append(couple)
        args.start_evo = 0
        save_ooe_population(directory, 0, parent_ooe_popu,exp_name)

        
    
    # Start the search from a previous population    
    else:
        
        if(args.rank == 0):
            print('Resuming from population ',args.start_evo) # To resume the search from an already saved population
        f = open(directory+'/results/' + exp_name + '/popu/resume_'+str(args.start_evo)+'.popu', 'rb')                 
        parent_ooe_popu = pickle.load(f)
        for idx in range(len(parent_ooe_popu)): # to adapt to the number of GPUs available
            parent_ooe_popu[idx]['net_id'] = f'net_{idx % args.world_size}_evo_{args.start_evo}_{idx}'
        print(len(parent_ooe_popu), args.evo_search_outer.parent_popu_size, args.world_size)
        assert len(parent_ooe_popu) == args.evo_search_outer.parent_popu_size

    # Run the first optimization step here --> explore backbones
    for evo_outer in range(args.start_evo, args.evo_search_outer.evo_iter):
        
        if(args.rank == 0):
            print("The evolution is at iteration: {} has population {} ".format(evo_outer, len(parent_ooe_popu)))

        
        backbones1 = []
        backbones2 = []
        n_evaluated = len(parent_ooe_popu) // args.world_size * args.world_size
        print("n_evaluated: ", n_evaluated)
        for cfg in parent_ooe_popu[:n_evaluated]:
            if cfg['net_id'].startswith(f'net_{args.rank}_'):
                backbones1.append(cfg['backbone1'])
                backbones2.append(cfg['backbone2'])
    
        if(args.rank == 0):
            print("Before validation Evolution {} Len Initial Backbones1: {} Len Initial Backbones2: {}".format(evo_outer, len(backbones1), len(backbones2)))   

        
        # Evauate the backbones population on our performance metrics for the Image modality
        backbones1 = validate_population(train_loader=train_loader, val_loader=test_loader, 
                        supernet=image_supernet, population=backbones1, 
                        args=args, lut_data=lut_data_image,
                        modal_num=0, bn_calibration=True, in_channels=1, resolution=28)
        comm.synchronize()
    
        # Evauate the backbones population on our performance metrics for the Audio modality
        backbones2 = validate_population(train_loader=train_loader, val_loader=test_loader, 
                        supernet=sound_supernet, population=backbones2, 
                        args=args, lut_data=lut_data_sound,
                        modal_num=1, bn_calibration=True, in_channels=1, resolution=20)

        comm.synchronize()

        
    
        if(args.rank == 0):
            print("Evolution {} Len Initial Backbones1: {} Len Initial Backbones2: {}".format(evo_outer, len(backbones1), len(backbones2)))
            print("###################################################")
            save_results(directory, evo_outer, 'Init_B1', backbones1,exp_name)
            save_results(directory, evo_outer, 'Init_B2', backbones2,exp_name)

        
        # Selection the promising backbones for the second stage of fusion search
            
        backbones1 = RankAndCrowdingSurvival_Inner_Acc(backbones1, normalize=None, 
                                                      n_survive=math.ceil(args.evo_search_outer.parent_popu_size*args.evo_search_inner.survival_ratio))

        
        backbones2 = RankAndCrowdingSurvival_Inner_Acc(backbones2, normalize=None, 
                                                      n_survive=math.ceil(args.evo_search_outer.parent_popu_size*args.evo_search_inner.survival_ratio))
        
        comm.synchronize()
        
        if(args.rank == 0):
            print("Evolution {} Len Survived Backbones1: {} \n Len Selected Backbones2: {}".format(evo_outer, len(backbones1), len(backbones2)))
        
            print("###################################################")
            save_results(directory, evo_outer, 'Elites_B1', backbones1,exp_name)
            save_results(directory, evo_outer, 'Elites_B2', backbones2,exp_name)
        


        parent_ooe_popu = []
        for i in range(len(backbones1)):
            id = f'net_{i % args.world_size}_evo_{evo_outer}_{i}'
            b1 = backbones1[i]
            b2 = backbones2[i]
            b1['net_id'] = id
            b2['net_id'] = id
            
            # Exploring the fusion network macro-architecure
            
            steps = 2          # the number of fusions cells 
            node_steps = 1     # the number of fusion operators inside the cell 
            steps_candidates = [1, 3, 4]
            node_steps_candidates = [2, 3, 4]
            
            if(random.random() < 0.4):
              steps = random.choice(steps_candidates)  
            if(random.random() < 0.4):
                node_steps = random.choice(node_steps_candidates)
            
            
            
            couple = {'backbone1': b1, 'backbone2': b2, 'net_id': id, 'steps': steps, 'node_steps': node_steps}
            parent_ooe_popu.append(couple)
            
        comm.synchronize()
        
        if(args.rank == 0):
            print("Evolution {} Len MM-Population before the fusion {} ".format(evo_outer, len(parent_ooe_popu)))
            print("###################################################")
            save_results(directory, evo_outer, 'Fusion_Popu', parent_ooe_popu,exp_name)
        

            

        

        

        save_ooe_population(directory, evo_outer, parent_ooe_popu,exp_name)
            
        
  
             
        my_subnets_to_be_evaluated = {}

        n_evaluated = len(parent_ooe_popu)
        for cfg in parent_ooe_popu[:n_evaluated]:
            if cfg['net_id'].startswith(f'net_{args.rank}_'):
                my_subnets_to_be_evaluated[cfg['net_id']] = cfg
                
        

        
        if args.rank == 0:
            print('evolution: ', evo_outer) 

        # Fusion search: Deriving the MM-NNs
        eval_results = subnets_nas_eval.avmnist_fusion_validate(
            eval_subnets1=my_subnets_to_be_evaluated,
            train_loader=train_loader,
            test_loader=test_loader,
            model1=image_supernet,
            model2=sound_supernet,
            lut_data1=lut_data_image,
            lut_data2=lut_data_sound,
            args=args,
            bn_calibration=True
        )
        comm.synchronize()
    

     
        
        f = open(directory+'/results/'+exp_name+'/popu/evo_'+str(evo_outer)+'.popu', 'rb')   
        actual_popu = pickle.load(f)
        assert len(actual_popu) == math.ceil(args.evo_search_outer.parent_popu_size*args.evo_search_inner.survival_ratio)

        comm.synchronize()



        for i, row in enumerate(actual_popu, start=0):
            
            mm_id = actual_popu[i]['net_id']
            
            b1_id = str(eval_results[i]['net_id1'])
            b2_id = str(eval_results[i]['net_id2'])
            
            steps = int(eval_results[i]['steps'])
            node_steps = int(eval_results[i]['node_steps'])
            genotype = eval_results[i]['genotype']
            
            for mm in actual_popu:
                if(mm['backbone1']['net_id'] == b1_id):
                    b1 = mm['backbone1'].copy()
                    break
                
            for mm in actual_popu:
                if(mm['backbone2']['net_id'] == b2_id):
                    b2 = mm['backbone2'].copy()
                    break
            
            
            
            b1['net_id'] = mm_id
            b2['net_id'] = mm_id
            
            actual_popu[i]['Acc@1'] = eval_results[i]['Acc@1']
            actual_popu[i]['latency'] = eval_results[i]['latency']
            actual_popu[i]['energy'] = eval_results[i]['energy']
            actual_popu[i]['backbone1'] = b1
            actual_popu[i]['backbone2'] = b2
            actual_popu[i]['steps'] = steps
            actual_popu[i]['node_steps'] = node_steps
            actual_popu[i]['genotype'] = genotype
            



        comm.synchronize()
        if(args.rank == 0):
            print("Evolution {} Len Fusion Results {}".format(evo_outer, len(eval_results)))
            print("###################################################")
            save_results(directory, evo_outer, 'Fusion_Popu_Results', actual_popu,exp_name)



        n_survive =  math.ceil( math.ceil(args.evo_search_outer.parent_popu_size * args.evo_search_inner.survival_ratio) * args.evo_search_outer.survival_ratio)
        survivals_ooe = RankAndCrowdingSurvival_Outer_Acc(pop=actual_popu, normalize=None, 
                                                      n_survive=n_survive)



        
        comm.synchronize()
        
        if args.rank == 0:
            print("Evolution {} Len Fusion Survivals {}".format(evo_outer, len(survivals_ooe)))
            print("###################################################")
            save_results(directory, evo_outer, 'Elites_MM', survivals_ooe,exp_name)
        
        


        comm.synchronize()
            
        # Generate the next population of the backbones for the next evolution
        parent_ooe_popu = []

        # crossover  (this removes the net_id key)
        for idx in range(args.evo_search_outer.crossover_size):
            cfg1 = random.choice(survivals_ooe)
            cfg2 = random.choice(survivals_ooe)
            # print("config 1",cfg1)
            cfg_backbone1 = image_supernet.module.crossover_and_reset1(cfg1['backbone1'], cfg2['backbone1'], crx_prob=args.evo_search_outer.crossover_prob)
            cfg_backbone2 = image_supernet.module.crossover_and_reset1(cfg1['backbone2'], cfg2['backbone2'], crx_prob=args.evo_search_outer.crossover_prob)
            cfg = {'backbone1': cfg_backbone1, 'backbone2': cfg_backbone2}
            
            parent_ooe_popu.append(cfg)

        # Mutation  
        for idx in range(args.evo_search_outer.mutate_size):          
            old_cfg = random.choice(survivals_ooe)
            
            cfg_backbone1 = image_supernet.module.mutate_and_reset(old_cfg['backbone1'], prob=args.evo_search_outer.mutate_prob)
            cfg_backbone2 = image_supernet.module.mutate_and_reset(old_cfg['backbone2'], prob=args.evo_search_outer.mutate_prob)
            cfg = {'backbone1': cfg_backbone1, 'backbone2': cfg_backbone2}
            parent_ooe_popu.append(cfg)

        if args.rank == 0:
            print("len parent_ooe_popu: {} /  the correct: {}".format(len(parent_ooe_popu), args.evo_search_outer.parent_popu_size))
        assert len(parent_ooe_popu) == args.evo_search_outer.parent_popu_size

        for idx in range(len(parent_ooe_popu)):
            parent_ooe_popu[idx]['net_id'] = f'net_{idx % args.world_size}_evo_{evo_outer}_{idx}'
            parent_ooe_popu[idx]['backbone1']['net_id'] = f'net_{idx % args.world_size}_evo_{evo_outer}_{idx}'
            parent_ooe_popu[idx]['backbone2']['net_id'] = f'net_{idx % args.world_size}_evo_{evo_outer}_{idx}'
            
        
        if(args.rank==0):
            print("Generation {} the parent popu is {}: ".format(evo_outer, len(parent_ooe_popu)))
    
        if args.rank not in [-1, 0]:
            comm.synchronize()
        

        save_resume_population(directory, evo_outer+1, parent_ooe_popu,exp_name)
        if args.rank == 0:
            comm.synchronize()
            print("Evolution {} has finished {}:".format(evo_outer, datetime.now().strftime('%Y-%m-%d-%H:%M:%S')))
    
    print("The search has ended at :",datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))


        
if __name__ == '__main__':
        
    args = setup(run_args.config_file)
    args.resume_evo = run_args.resume_evo
    args.start_evo = run_args.start_evo
    args.dist_url = run_args.dist_url
    args.dist_backend = run_args.dist_backend
    args.machine_rank = run_args.machine_rank
    args.num_nodes = run_args.num_machines
    args.net = run_args.net

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.num_nodes
        print("world size: ", args.world_size, "GPUs")
        print("The search has started at :",datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
        assert args.world_size > 1, "only support DDP settings"
        mp.spawn(eval_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        
    else:
        raise NotImplementedError
        