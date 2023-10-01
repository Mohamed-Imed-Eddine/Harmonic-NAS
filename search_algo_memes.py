import argparse
import random
import math
import pickle

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from utils.config import setup
import utils.comm as comm
import pandas as pd

from backbones.ofa.model_zoo import ofa_net
from backbones.ofa.utils.layers import LinearLayer, ConvLayer
from backbones.maxout.maxout import Our_Maxout

from utils.optim import RankAndCrowdingSurvival_Outer_Acc, RankAndCrowdingSurvival_Inner_Acc
# , RankAndCrowdingSurvival_Inner, calc_hypervolume, find_pareto_front
from utils.save import save_ooe_population, save_results, save_resume_population

from evaluate.backbone_eval.accuracy import subnets_nas_eval
from evaluate.backbone_eval.accuracy.population_nas_eval import validate_population
from evaluate.backbone_eval.efficiency import EfficiencyEstimator, look_up_ofa_proxy
from data.data_loader import build_data_loader
from datetime import datetime
import os
directory = os.path.dirname(os.path.abspath(__name__))
directory = "../Harmonic-NAS"

parser = argparse.ArgumentParser(description='Harmonic-NAS Search for the Complete MM-NN Architecure') 
parser.add_argument('--config-file', default=directory+'/configs/eval_supernet_models.yml')
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
    args.batch_size = args.batch_size_per_gpu
    
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


    # build the supernet
    supernet = ofa_net(args.net, resolution=args.resolution, pretrained=False, in_ch=args.in_channels, _type=args._type)
    supernet.classifier = LinearLayer(supernet.classifier.in_features, args.n_classes, dropout_rate=args.dropout) 
    supernet.first_conv = ConvLayer(args.in_channels, supernet.first_conv.out_channels, kernel_size=supernet.first_conv.kernel_size,
                                 stride=supernet.first_conv.stride, act_func="h_swish")
    supernet.cuda(args.gpu)
    supernet = comm.get_parallel_model(supernet, args.gpu) #local rank

    # Load LUTs for latency/energy characterization on the edge device
    lut_data = EfficiencyEstimator(fname=args.hw_lut_path, supernet=args.supernet_arch)
    
    ## load dataset, train_sampler: distributed
    train_loader, test_loader, train_sampler = build_data_loader(args)
    # Reloading supernet pretrained weights

    textnet = Our_Maxout()
    with open(args.maxout_weights, 'rb') as f:
        checkpoint = torch.load(f, map_location='cpu')
        textnet.load_state_dict(checkpoint['state_dict'])

    assert args.pretrained_path
    supernet.module.load_weights_from_pretrained_supernet(args.pretrained_path)
        
    if(args.resume_evo == 0): # First population (evolution)
        parent_ooe_popu = []
        parent_ooe_popu1 = supernet.module.init_population(n_samples=args.evo_search_outer.parent_popu_size)
        for idx in range(len(parent_ooe_popu1)):
            parent_ooe_popu1[idx]['net_id'] = f'net_{idx % args.world_size}_evo_0_{idx}'
            config = {'backbone1':parent_ooe_popu1[idx], 'net_id':f'net_{idx % args.world_size}_evo_0_{idx}'}
            parent_ooe_popu.append(config)
        args.start_evo = 0
        # save_ooe_population(directory, 0, parent_ooe_popu, 'memes')
    else:
        print('resuming...') # To resume the search from an already saved population
        f = open(directory+'/results/memes/popu/resume_'+str(args.start_evo)+'.popu', 'rb')   
        parent_ooe_popu = pickle.load(f)
        for idx in range(len(parent_ooe_popu)): # to adapt to the number of GPUs available
            parent_ooe_popu[idx]['net_id'] = f'net_{idx % args.world_size}_evo_{args.start_evo}_{idx}'
        print(len(parent_ooe_popu), args.evo_search_outer.parent_popu_size, args.world_size)
        assert len(parent_ooe_popu) ==   args.evo_search_outer.parent_popu_size


    # Run the first optimization step here --> explore backbones
    for evo_outer in range(args.start_evo, args.evo_search_outer.evo_iter):
        
        if(args.rank == 0):
            print("The evolution is at iteration: {} has population {} ".format(evo_outer, len(parent_ooe_popu)))
        
        backbones = []
        n_evaluated = len(parent_ooe_popu) // args.world_size * args.world_size
        for cfg in parent_ooe_popu[:n_evaluated]:
            if cfg['net_id'].startswith(f'net_{args.rank}_'):
                backbones.append(cfg['backbone1'])
                # print("the net_id is: ", cfg)

        backbones = validate_population(train_loader=train_loader, val_loader=test_loader, 
                        supernet=supernet, population=backbones, 
                        args=args, lut_data=lut_data,
                        modal_num=0, bn_calibration=True, in_channels=args.in_channels, resolution=args.resolution)
        
        comm.synchronize()
        

    
        if(args.rank == 0):
            print("Evolution {} Len Initial Backbones: {}".format(evo_outer, len(backbones)))
            print("###################################################")
            save_results(directory, evo_outer, 'Init_B1', backbones, 'memes')
        
        # Selection based on the hypervolume of each backbone-PF from the IOE --> select the backbone w/ the best fusion for next generation        
        backbones = RankAndCrowdingSurvival_Inner_Acc(backbones, normalize=None, 
                                                      n_survive=math.ceil(args.evo_search_outer.parent_popu_size*args.evo_search_inner.survival_ratio), perfo_metric='Acc@1')

        comm.synchronize()
        
        
        if(args.rank==0):
            print("Evolution {} Len Survived Backbones: {}".format(evo_outer, len(backbones)))
            print("###################################################")
            save_results(directory, evo_outer, 'Elites_B1', backbones, 'memes')
            
            
        parent_ooe_popu = []
        for i in range(len(backbones)):
            id = f'net_{i % args.world_size}_evo_{evo_outer}_{i}'
            b = backbones[i]
            b['net_id'] = id

            # Fusion steps & inner_steps
            
            # steps = 2          # the number of fusions cells 
            # node_steps = 1     # the number of fusion operators in each fusion cell 
            # steps_candidates = [1, 3, 4]
            # node_steps_candidates = [2, 3, 4]
            
            steps = 2          # the number of fusions cells 
            node_steps = 2     # the number of fusion operators in each fusion cell 
            steps_candidates = [1, 3, 4, 5]
            node_steps_candidates = [1, 3, 4, 5]
            
            #steps --> (1, 2, 3, 4) | #node_steps --> (1, 2, 3, 4) : Fusion search space
            if(random.random() < 0.4):
              steps = random.choice(steps_candidates)  
            if(random.random() < 0.4):
                node_steps = random.choice(node_steps_candidates)
                
                
            parent_ooe_popu.append({'backbone1': b, 'net_id': id, 'steps': steps, 'node_steps': node_steps})

        comm.synchronize()
        
        if(args.rank==0):
            print("Evolution {} Len MM-Population before the fusion {} ".format(evo_outer, len(parent_ooe_popu)))
            # print("Evolution {} \nMM-Population before the fusion {} ".format(evo_outer, parent_ooe_popu))
            print("###################################################")
            save_results(directory, evo_outer, 'Fusion_Popu', parent_ooe_popu, 'memes')



        save_ooe_population(directory, evo_outer, parent_ooe_popu, 'memes')

        
        my_subnets_to_be_evaluated = {}
        n_evaluated = len(parent_ooe_popu)
        
        for cfg in parent_ooe_popu[:n_evaluated]:
            if cfg['net_id'].startswith(f'net_{args.rank}_'):
                my_subnets_to_be_evaluated[cfg['net_id']] = cfg
        
        
        
        
    
        if args.rank == 0:
            print('evolution: ', evo_outer)

        
        
        print("gpu {} subnets {}".format(args.gpu, len(my_subnets_to_be_evaluated)))
        
        eval_results = subnets_nas_eval.memes_fusion_validate(
            my_subnets_to_be_evaluated,
            train_loader,
            test_loader,
            supernet,
            lut_data,
            args, 
            textnet,
            bn_calibration=True,
        )
        comm.synchronize()
        
    
     
        
        f = open(directory+'/results/memes/popu/evo_'+str(evo_outer)+'.popu', 'rb')   
        actual_popu = pickle.load(f)
        print("len of actual pop is: ", len(actual_popu))
        assert len(actual_popu) == math.ceil(args.evo_search_outer.parent_popu_size*args.evo_search_inner.survival_ratio)
        comm.synchronize()
        print("len(eval_results) {}  len(actual_popu) {} / gpu {}  ".format(len(eval_results),  len(actual_popu) , args.gpu ))



        # Reading evaluation results from all GPUs
        for i, row in enumerate(actual_popu, start=0):
            
            mm_id = actual_popu[i]['net_id']
            b_id = str(eval_results[i]['net_id1'])
            steps = int(eval_results[i]['steps'])
            node_steps = int(eval_results[i]['node_steps'])
            genotype = eval_results[i]['genotype'] 
            
            for mm in actual_popu:
                if(mm['backbone1']['net_id']==b_id):
                    b = mm['backbone1'].copy()
                    break
            
            b['net_id'] = mm_id
                       
            
            actual_popu[i]['Acc@1'] = eval_results[i]['Acc@1']
            actual_popu[i]['latency'] = eval_results[i]['latency']
            actual_popu[i]['energy'] = eval_results[i]['energy']
            actual_popu[i]['genotype'] = genotype
            actual_popu[i]['steps'] = steps
            actual_popu[i]['node_steps'] = node_steps
            actual_popu[i]['backbone1'] = b

        comm.synchronize()
        if(args.rank == 0):
            print("Evolution {} Len Fusion Results {}".format(evo_outer, len(eval_results)))
            print("###################################################")
            save_results(directory, evo_outer, 'Fusion_Popu_Results', actual_popu, 'memes')        


        n_survive =  math.ceil( math.ceil(args.evo_search_outer.parent_popu_size * args.evo_search_inner.survival_ratio) * args.evo_search_outer.survival_ratio)
        survivals_ooe = RankAndCrowdingSurvival_Outer_Acc(pop=actual_popu, normalize=None, 
                                                      n_survive=n_survive, perfo_metric='Acc@1')

        
        comm.synchronize()

        # in : models, gpu, dataloader, 
        # out : best_metric, best_mm_archi (genotype) 
        
        if args.rank == 0:
            print("Evolution {} Len Fusion Survivals {}".format(evo_outer, len(survivals_ooe)))
            print("###################################################")
            save_results(directory, evo_outer, 'Elites_MM', survivals_ooe, 'memes')
            
            
        comm.synchronize()
        
        # Generate the next generation of GNN to be evaluated
        parent_ooe_popu = []

        # crossover  (this removes the net_id key)
        for idx in range(args.evo_search_outer.crossover_size):
            cfg1 = random.choice(survivals_ooe)
            cfg2 = random.choice(survivals_ooe)
            # print("config 1",cfg1)
            cfg_backbone1 = supernet.module.crossover_and_reset1(cfg1['backbone1'], cfg2['backbone1'], crx_prob=args.evo_search_outer.crossover_prob)
            cfg = {'backbone1': cfg_backbone1}
            parent_ooe_popu.append(cfg)

        # mutate  
        for idx in range(args.evo_search_outer.mutate_size):          
            old_cfg = random.choice(survivals_ooe)
            
            cfg_backbone1 = supernet.module.mutate_and_reset(old_cfg['backbone1'], prob=args.evo_search_outer.mutate_prob)
            cfg = {'backbone1': cfg_backbone1}
            parent_ooe_popu.append(cfg)

        if args.rank == 0:
            print("len parent_ooe_popu: {} /  the correct: {}".format(len(parent_ooe_popu), args.evo_search_outer.parent_popu_size))
        assert len(parent_ooe_popu) == args.evo_search_outer.parent_popu_size

        for idx in range(len(parent_ooe_popu)):
            parent_ooe_popu[idx]['net_id'] = f'net_{idx % args.world_size}_evo_{evo_outer}_{idx}'
            parent_ooe_popu[idx]['backbone1']['net_id'] = f'net_{idx % args.world_size}_evo_{evo_outer}_{idx}'
        
        if(args.rank==0):
            print("Generation {} the parent popu is {}: ".format(evo_outer, len(parent_ooe_popu)))
    
        if args.rank not in [-1, 0]:
            comm.synchronize()

        save_resume_population(directory, evo_outer+1, parent_ooe_popu, 'memes')
        
        
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
        