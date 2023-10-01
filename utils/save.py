import json
import pickle
import os

def save_ooe_population(dir, evo, popu, dataset):
    folder = dir+'/results/' + dataset + '/popu'
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(dir+'/results/' + dataset + '/popu/evo_'+str(evo)+'.popu', 'wb') as f: 
        pickle.dump(popu, f)


def save_results(dir, evo, name, popu, dataset):
    
    folder = dir+'/results/'+dataset+'/ooe/evo_'+str(evo)
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    with open(folder+'/evo_'+str(evo)+'_'+name+'.json', 'w') as f:
        json.dump(popu, f)


def save_resume_population(dir, evo, popu, dataset):
    folder = dir+'/results/' + dataset + '/popu'
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(dir+'/results/' + dataset + '/popu/resume_'+str(evo)+'.popu', 'wb') as f: 
        pickle.dump(popu, f)





# def save_results_ioe(dir, evo, popu):
#     with open(dir+'/results/ioe/evo_'+str(evo)+'_eval.json', 'w') as f:
#         json.dump(popu, f)

# def save_results_ooe(dir, evo, popu):
#     with open(dir+'/results/ooe/evo_'+str(evo)+'_eval.json', 'w') as f:
#         json.dump(popu, f)
        
        
