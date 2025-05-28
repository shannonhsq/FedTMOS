


import itertools
from itertools import combinations

import numpy as np
import copy 
from time import time 
import os

from pyTsetlinMachineParallel_vote.tm import MultiClassTsetlinMachine, MultiClassConvolutionalTsetlinMachine2D
import os , pickle 
import argparse 
import random

from sklearn.cluster import KMeans

from oneshot_client import * 
from oneshot_data import * 



parser = argparse.ArgumentParser(description='One-Shot FedTM')
parser.add_argument("--epochs", type=int, default = 1)
parser.add_argument("--dirichlet", type=float, default = 0.1)
parser.add_argument("--n_clauses", type=int, default = 1000)
parser.add_argument("--local_epochs", type=int, default = 100)
parser.add_argument("--dataset", type=str, default = 'MNIST')
parser.add_argument("--num_models", type=int, default = 3)
parser.add_argument("--num_clients", type=int, default = 10)
parser.add_argument("--T", type=int, default = 10000)
parser.add_argument("--patch_dim", type=int, default = 5)
parser.add_argument("--s", type=float, default = 5)
parser.add_argument('--data_dir', type =str, default = '/data/dir/')
parser.add_argument('--k', type=int, default = 10)
parser.add_argument('--c', type=str, default = '4')
parser.add_argument('--seed', type=int, default = 123)
parser.add_argument('--load_pretrain', type = int, default = 0)
parser.add_argument('--dir_type', type = str, default = 'dir') #choose from dir or cls
parser.add_argument('--ensemble', type = int, default = 0) 
parser.add_argument('--alg_seed', type = int, default = 1) 
parser.add_argument('--participation', type = float, default = 1.0) 



args = parser.parse_args()
print(args)
num_models = args.num_models
numclients = args.num_clients
ALG_SEED = args.alg_seed
random.seed(ALG_SEED)
data_dir = args.data_dir

patch_dim = (args.patch_dim, args.patch_dim)


args.c = [int(x) for x in args.c.split(',')]
print(args.c)

device = "CPU"
max_included_literals = 32
resolution = 8
factor = 1
epochs = args.epochs
k_cluster = args.k










model_name = 'pretrain' + '_seed_' + str(args.seed) +'_' +  str(args.dataset) + str(num_models) +str(args.local_epochs) + '_local_epochs_' +str(args.dirichlet) + '__' + str(args.n_clauses) +'__clauses' + str(args.num_clients) +'clients__'+ str(args.T) + '__T__pd__'  +str(args.patch_dim) + '.pkl'


model_name =  os.path.join('split_models/', model_name)


print(model_name)

accuracy_dict = {}
model_list = []
  

epochs = args.epochs
past_epochs = len(accuracy_dict)


def gini_index(p):
    p = np.array(p)
    return 1 - np.sum(p ** 2)

def normalized_gini(p):
    gini = gini_index(p)
    return 1 - gini


def size(model,numclasses=10):
  storages = {}
  uploads = {}
  downloads = {}
  #return storage size of TM in MB
  for m in range(len(model)):
    total_m_num = len(model[m])

    if total_m_num >=1 :
      num_clauses = model[m][0].__dict__['number_of_clauses']
      num_states = len(model[m][0].get_state()[0][1])
      ta_chunk = model[m][0].__dict__['number_of_ta_chunks']
      clause_weights  = np.ones(num_clauses, dtype = 'uint32')
      ta_state = np.zeros(num_states, dtype = 'uint32')
      num_state_bits = model[m][0].__dict__['number_of_state_bits']
      total_num_non_state = 0

      pos = 0
      sig_bit_num =0 
      for j in range(num_clauses):
        for k in range(ta_chunk):
          for b in range(num_state_bits-1):
            ta_state[pos] = ~0
            pos +=1
            total_num_non_state+=1
          ta_state[pos] = 0 #this is the most significant bit 
          pos+=1
          sig_bit_num +=1


        storage = ((num_clauses*numclasses) + (sig_bit_num*numclasses)) *4  *1e-6 * total_m_num 
        upload = ((num_clauses*numclasses) + (sig_bit_num*numclasses)) *4*1e-6
        download = storage 
        storages[m] = storage
        uploads[m] = upload
        downloads[m] = download

  print('Total Upload: ', sum(uploads.values()))
  print('Total download ',  sum(downloads.values()))
  print('Total storage ',  sum(storages.values()))

  return storages, uploads, downloads







  
          
       
def get_client_models(dataset_name , num_models, T, s, patch_dim): 
  models = [] 

  if 'SVHN' in dataset_name:
    num_models = 1
    tm_thermometer_3 = MultiClassConvolutionalTsetlinMachine2D(
    number_of_clauses = int(args.n_clauses),
    T=2000,
    s = s,
    patch_dim = (3,3),
    weighted_clauses = True
    )

    tm_model = MultiClassConvolutionalTsetlinMachine2D(
        number_of_clauses = args.n_clauses,
        T=T,
        s = s,
        patch_dim = patch_dim,
        weighted_clauses = True
    )
    
    models.append(tm_model)
    models = [tm_model][0:num_models]
  #  models.append(tm_thermometer_3)

  
  elif 'CIFAR' in dataset_name:

    max_included_literals = 32 
    tm_hog = MultiClassTsetlinMachine(
    number_of_clauses = 20,
    T=int(50*T),
    s = 5,
    weighted_clauses = True
    )


    tm_threshold = MultiClassConvolutionalTsetlinMachine2D(
      number_of_clauses = args.n_clauses,
      T=int(30*T), #(30*T)
      s =s,
      patch_dim = (10,10),
      weighted_clauses = True
    )

    tm_threshold_5 = MultiClassConvolutionalTsetlinMachine2D(
        number_of_clauses = args.n_clauses,
        T=int(20*T),
        s = s,
        patch_dim = (5,5),
        weighted_clauses = True
    )


    tm_thermometer_3 = MultiClassConvolutionalTsetlinMachine2D(
    number_of_clauses = int(args.n_clauses),
    T=int(15*T),
    s = s,
    patch_dim = (3,3),
    weighted_clauses = True
    )

    tm_thermometer_4 = MultiClassConvolutionalTsetlinMachine2D(
        number_of_clauses = int(args.n_clauses),
        T=int(15*T),
        s = s,
        patch_dim = (4,4),
        weighted_clauses = True
    )

    models = [tm_threshold, tm_thermometer_3, tm_thermometer_4 , tm_hog][0:num_models]
  
  elif 'ImageNet' in dataset_name:

    max_included_literals = 32 
    tm_hog = MultiClassTsetlinMachine(
    number_of_clauses = 20,
    T=int(50*T),
    s = 5,
    weighted_clauses = True
    )


    tm_threshold = MultiClassConvolutionalTsetlinMachine2D(
      number_of_clauses = args.n_clauses,
      T=int(30*T), #(30*T)
      s =s,
      patch_dim = (10,10),
      weighted_clauses = True
    )

    tm_threshold_5 = MultiClassConvolutionalTsetlinMachine2D(
        number_of_clauses = args.n_clauses,
        T=int(20*T),
        s = s,
        patch_dim = (10,10),
        weighted_clauses = True
    )


    tm_thermometer_3 = MultiClassConvolutionalTsetlinMachine2D(
    number_of_clauses = int(args.n_clauses),
    T=int(15*T),
    s = s,
    patch_dim = (3,3),
    weighted_clauses = True
    )

    tm_thermometer_4 = MultiClassConvolutionalTsetlinMachine2D(
        number_of_clauses = int(args.n_clauses),
        T=int(15*T),
        s = s,
        patch_dim = (4,4),
        weighted_clauses = True
    )

    models = [tm_threshold, tm_thermometer_3, tm_thermometer_4 , tm_hog][0:num_models]
    
  else:
    tm_model = MultiClassConvolutionalTsetlinMachine2D(
        number_of_clauses = args.n_clauses,
        T=T,
        s = s,
        patch_dim = patch_dim,
        weighted_clauses = True
    )
    for i in range(num_models):
      models.append(tm_model)
  
  print('model len: ', len(models))
  for i in range(num_models):
    print(models[i].__dict__)

  return models 






def get_sig_bits(model,ta_states):
  sig_bits = []
  pos = 0
  for j in range(model.number_of_clauses):
    for k in range(model.number_of_ta_chunks):
      for b in range(model.number_of_state_bits-1):
        pos +=1
      sig_bits.append(ta_states[pos])
      pos+=1   

  return sig_bits



def flatten_model_params_participation(global_model, gm, ratio_count, c, participation=1):
    class_counter = np.zeros(len(ratio_count[0]))
    
    for m in range(len(global_model)):
        single_global_model = global_model[m][gm]
        
        for j in range(len(ratio_count[0])):
            if ratio_count[m][j]:
                if np.floor(np.mean(single_global_model.get_state()[j][0])) > 0:
                    class_counter[j] += 1     

    print('class counter: ', class_counter)

    # Flatten all weights into a single vector
    all_params = [] 
    model_indices = {}
   
    v = 0
    duplicate_indices = [] 

    # Store parameters and their magnitudes
    param_vector = [[] for _ in range(len(ratio_count[0]))]
    model_indices_class =  [[] for _ in range(len(ratio_count[0]))]
    for m in range(len(global_model)):
        single_global_model = global_model[m][gm]
        print(m)
 
        for i in range(len(ratio_count[0])):
            if ratio_count[m][i]:        
                if np.floor(np.mean(single_global_model.get_state()[i][0])) > 0:
                    print('exists ', i, ': ', np.mean(np.mean(single_global_model.get_state()[i][0])))    
                    param_vector[i].append(single_global_model.get_state()[i][0])
                    model_indices_class[i].append(m)

        print('_________')

   
    for i in range(len(param_vector)):
      if len(param_vector[i]) ==0:
        continue
       
      num_to_select = max(1, int(participation * class_counter[i]))  
      
 
      top_indices = np.random.choice(list(range(len(param_vector[i]))), size=num_to_select, replace=False)
      
      for index in top_indices:
          all_params.append(param_vector[i][index])
          model_indices[v] = (model_indices_class[i][index], i)  
          v += 1

    all_params = np.array(all_params)

    return all_params, model_indices, duplicate_indices



def kmeans_clustering(weights,global_model, ratio_count, k, mean_gini, return_mean = False, defined_c = [3], numclasses =10,participation = 1, num_models = 1):
    cluster_info = []
    kmeans_means=[]
    new_c = [] 
    duplicate_indices= []
    num_clients = len(ratio_count[0])

    for gm in range(num_models):
      cluster_model = {} 
      print(defined_c)
      print(gm)

      c = int(defined_c[gm])

  
      vectors, model_indices, di =flatten_model_params_participation(global_model,gm,ratio_count,c,participation)#
     
      normalized_vectors = vectors 
      print('len clusters: ', len(normalized_vectors))
      
      

      compute_c = np.ceil(len(normalized_vectors)/numclasses)-1
      chose_c = np.min([int(compute_c), int(defined_c[gm])])
      chose_c =  int(defined_c[gm])
      print('compute c: ', int(compute_c), 'defined_c: ', int(defined_c[gm]))
      
      duplicate_indices.append(di)
      new_c.append(int(chose_c))

      
   

      max_k = np.min([len(model_indices)-1, k])
      print(max_k)
      
      cluster_k = max_k


      if c != 0:

        print('cluster k: ', cluster_k)
        kmeans = KMeans(n_clusters=cluster_k, random_state= ALG_SEED)
        labels = kmeans.fit_predict(normalized_vectors)
        kmeans_mean = kmeans.cluster_centers_

        for i in range(cluster_k):
          cluster_model[i] = {}

        for idx, label in enumerate(labels):
            
            (client, c) = model_indices[idx]
            #print('label: ', ratio_count[client])
            #print(centroids[label])
            if client not in cluster_model[label].keys():
              cluster_model[label][client] = []
              cluster_model[label][client].append(c)
            else:
              cluster_model[label][client].append(c)
        
        cluster_info.append(cluster_model)
        kmeans_means.append(kmeans_mean)
      
      else:
        kmeans_means.append([])
        cluster_info.append({})
    
    if return_mean:
      return cluster_info, kmeans_means , new_c, duplicate_indices
    else:
      return cluster_info, new_c, duplicate_indices


def calculate_distance(mean1, mean2):
    """
    Calculate Euclidean distance between two cluster means.
    """
    return np.linalg.norm(np.array(mean1) - np.array(mean2))

def find_best_model_to_add_class(current_cluster_idx, model_means, potential_idx, previous_clusters):
    """
    Find the best model to add a new class based on the distance from the current cluster.
    """
    best_idx = -1
    max_dist = 0
 
    for i in potential_idx:
      avg_dist = 0
      if current_cluster_idx not in previous_clusters[i]:
        potential_means = list(previous_clusters[i])#list(set(previous_clusters[i])       
        for k in potential_means:
          avg_dist+=abs(calculate_distance(model_means[k],model_means[current_cluster_idx]))   
        avg_dist/=len(potential_means)
        if avg_dist > max_dist:
          max_dist = avg_dist
          best_idx = i 

    return best_idx 







def cluster_global_model_best(weights , global_model,ratio_count,  unique_counts, k, c ,mean_gini, num_models=1):
  print('cluster global model with the best model each')

  
  new_mappings = [] 
  cluster_info, kmeans_mean , new_c ,duplicate_indices = kmeans_clustering(weights,global_model, ratio_count, k ,mean_gini=mean_gini, defined_c= c, return_mean = True, participation=args.participation,numclasses=len(ratio_count[0]), num_models = num_models)
  print(cluster_info)
  print('computed c: ',new_c )


  for gm in range(num_models):
    min_class_cover = 2
    print('min class cover: ',  min_class_cover)
    print('combined_models_with_k_clusters: ', int(new_c[gm]),int(len(ratio_count[0])*(1-mean_gini)))
    
 
    combined_cluster_info =combine_classes_into_models_by_distance_fill(cluster_info[gm], kmeans_mean[gm], int(new_c[gm]), min_class_cover)
    

    new_mappings.append( combined_cluster_info)



  print(new_mappings)
  
  cluster_info  = new_mappings


  
  combined_models = [] 
  new_combined_ratio = [] 
  for gm in range(len(global_model[0])): 
    weights_counts = []
    for c in range(len(unique_counts)):
      wc = []
      for num_cls in range(len(unique_counts[0])):
        wc.append(np.mean( global_model[c][gm].get_state()[num_cls][0]))
      
      weights_counts.append(wc)
  for gm in range(num_models): 
    print('global model: ', gm)

    combined_ratio = []
    all_m = []  
    cid_dup = {}

    frac_dup = [] 
    for (cid,cls,frac) in duplicate_indices[gm]:
      cid_dup[(cid,cls)] = frac

    for i in range(len(new_mappings[gm])):
      print('-----------------')
      print('cluster key: ', i)
      ratio_cluster = np.zeros(len(ratio_count[0]))
      cluster_k = cluster_info[gm][i] 
      temp_model = copy.deepcopy(global_model[0][gm])
      temp_params = temp_model.get_state()
      clause_weights  = np.zeros(len(temp_params[0][0]), dtype = 'float32')
      ta_state = np.zeros(len(temp_params[0][1]), dtype = 'uint32')
      new_params = ([(clause_weights,ta_state)] * len(ratio_count[0]))
      avg_num = np.zeros(len(temp_params[0][1]))
      init_parameters = [[np.copy(clause_weights), np.copy(ta_state)] for _ in range(len(ratio_count[0]))]
      max_counts = np.zeros(len(ratio_count[0]))
      last_max_counts = [ta_state for _ in range(len(ratio_count[0]))]
      maj_weights = [[] for _ in range(len(ratio_count[0]))]

      for idx in cluster_k.keys():
        for classes in cluster_k[idx]:
          if (idx,classes) in cid_dup.keys(): 
            if idx >= len(ratio_count[0]):
              idx = idx % len(ratio_count[0])
              init_parameters[classes][0] +=(global_model[idx][gm].get_state()[classes][0]) *  cid_dup[(idx,classes)]
            else:
              init_parameters[classes][0] += global_model[idx][gm].get_state()[classes][0]
          else:   
            init_parameters[classes][0] += global_model[idx][gm].get_state()[classes][0]
          
          avg_num[classes] += 1
          ratio_cluster[classes] = 1 
          maj_weights[classes].append(global_model[idx][gm].get_state()[classes][1])
          if weights_counts[idx][classes]  >= max_counts[classes]:
          
            if idx == list(cluster_k.keys())[len(cluster_k.keys())-1]:
             init_parameters[classes][1] = global_model[idx][gm].get_state()[classes][1] | last_max_counts[classes] 
            else:
   
             init_parameters[classes][1] = global_model[idx][gm].get_state()[classes][1]
             max_counts[classes] =  weights_counts[idx][classes]#unique_counts[idx][classes]  #
             last_max_counts[classes] = global_model[idx][gm].get_state()[classes][1]
      
      for c in range(len(ratio_count[0])):
        
        if avg_num[c] >=1:
          init_parameters[c][0] /= (avg_num[c])
   
          init_parameters[c][0] = np.uint32(init_parameters[c][0])
          init_parameters[c][1] = np.uint32(init_parameters[c][1])
    
        else:
          init_parameters[c][0] = np.zeros(len(temp_params[0][0]), dtype = 'uint32')
          init_parameters[c][1] = np.zeros(len(temp_params[0][1]), dtype = 'uint32')
        
        new_params[c] = (init_parameters[c][0],init_parameters[c][1])

      temp_model.set_state(new_params)
      all_m.append(temp_model)
      combined_ratio.append(ratio_cluster)

    new_combined_ratio.append(combined_ratio)
    combined_models.append(all_m)
  
  print(new_combined_ratio)
  print(len(combined_models[0]))
  return combined_models,new_combined_ratio


def sorted_clusters(cluster_info):

  """
  Sort clusters by the number of unique classes they contain.
  """

  cluster_data = cluster_info
  reverse = False

  
  cluster_unique_classes = {}


  for cluster_key, clients in cluster_data.items():
     
      all_classes = []
      for client_classes in clients.values():
          all_classes.extend(client_classes)  
      

      unique_classes = set(all_classes)
      

      cluster_unique_classes[cluster_key] = len(unique_classes)

  
  clusters_to_process = sorted(cluster_unique_classes.keys(), 
                                key=lambda x: cluster_unique_classes[x], 
                                reverse=reverse)
 
  return clusters_to_process

 

def combine_classes_into_models_by_distance_fill(cluster_info, kmeans_means, c, N ):
    num_clusters = len(cluster_info)
    new_model_mean = [0 for _ in range(c)]
    combined_models = [{} for _ in range(c)]  
    used_clusters = set()
    track_cluster_num = [[] for _ in range(c)]
    classes_in_clusters = [] 
    for key in cluster_info:
      
      for sub_key in cluster_info[key]:
          classes_in_clusters.append(cluster_info[key][sub_key])


    max_iterations = 1000  

    def update_class_mean_in_model(model_means, new_class_mean, class_idx):
      if class_idx in model_means:
          current_mean = model_means[class_idx]
          updated_mean = np.mean([current_mean, new_class_mean], axis=0)
          model_means[class_idx] = updated_mean
      else:
          
          model_means[class_idx] = new_class_mean

    def calculate_average_linkage_distance(model_means):
      """
      Calculate the average sum of squared distances (average linkage) between all pairs of class means in the model.
      """
      class_means = list(model_means.values())
      sum_squared_distances = 0
      num_pairs = 0
      
     
      for i in range(len(class_means)):
          for j in range(i + 1, len(class_means)):
              distance = np.linalg.norm(class_means[i] - class_means[j]) ** 2
              sum_squared_distances += distance
              num_pairs += 1 
              

      return sum_squared_distances / num_pairs if num_pairs > 0 else 0
     
    def model_has_class(model, class_to_check):
        for client_classes in model.values():
            if class_to_check in client_classes:
                return True
        return False

    def find_best_model_for_class(cluster_idx, class_idx, existing_models, model_means_dict):
        """
        Find the best model to add the new class (based on maximum sum of squared distances).
        """
        best_model_idx = -1
        max_distance = -1  

        new_class_mean = kmeans_means[cluster_idx]

        for model_idx, model in enumerate(existing_models):
          if model_has_class(model,class_idx):
            continue
           
          temp_model_means = model_means_dict[model_idx].copy()


          update_class_mean_in_model(temp_model_means, new_class_mean, class_idx)

  
          distance = calculate_average_linkage_distance(temp_model_means)

        
          if distance > max_distance:
              max_distance = distance
              best_model_idx = model_idx

        return best_model_idx

    def count_distinct_classes(model):
        return len(set(cls for client_classes in model.values() for cls in client_classes))

    def find_model_with_least_classes(existing_models, class_cluster_idx):
        min_class_count = float('inf')
        best_model_idx = -1

        for model_idx, model in enumerate(existing_models):
            current_class_count = count_distinct_classes(model)
            if current_class_count < min_class_count:
                min_class_count = current_class_count
                best_model_idx = model_idx
            elif current_class_count == min_class_count:
              
                cluster_class_count = sum(1 for cluster in model.values() if class_cluster_idx in cluster)
                if cluster_class_count < min_class_count:
                    best_model_idx = model_idx

        return best_model_idx
    
    def calculate_combination_linkage(model_means, kmeans_means, class_combination, track_cluster_num, current_model_mean):
      temp_mean = np.mean(current_model_mean, axis=0)
      
    
      for cls, cluster_idx in class_combination:
          temp_mean += kmeans_means[cluster_idx]  
      updated_mean = temp_mean / (len(track_cluster_num) + len(class_combination)) 

    
      updated_model_means = model_means.copy()  # Create a copy of the model means
      for cls, cluster_idx in class_combination:
          update_class_mean_in_model(updated_model_means, kmeans_means[cluster_idx], cls)

      total_wcss = calculate_average_linkage_distance(updated_model_means)  # Use updated model means for WCSS
      return total_wcss


   
    clusters_to_process = sorted_clusters(cluster_info=cluster_info) 

    model_means_dict = {i: {} for i in range(len(combined_models))}  
    iteration_count = 0 
    while len(used_clusters) < num_clusters and iteration_count < max_iterations:
        for cluster_idx in clusters_to_process:
            if cluster_idx in used_clusters:
                continue 

          
            for client, classes in cluster_info[cluster_idx].items():
                for class_idx in classes:
                   
                    best_model_idx = find_best_model_for_class(cluster_idx, class_idx, combined_models, model_means_dict)

                    if best_model_idx == -1:
                        
                        best_model_idx = find_model_with_least_classes(combined_models, cluster_idx)

                  
                    if client not in combined_models[best_model_idx]:
                        combined_models[best_model_idx][client] = []

                    combined_models[best_model_idx][client].append(class_idx)
                    
                   
                    update_class_mean_in_model(model_means_dict[best_model_idx], kmeans_means[cluster_idx], class_idx)

                    track_cluster_num[best_model_idx].append(cluster_idx)

 
            used_clusters.add(cluster_idx)

        iteration_count += 1
   
    for m in range(len(combined_models)):
        model = combined_models[m]
        current_classes = set(cls for client_classes in model.values() for cls in client_classes)
        current_count = len(current_classes)

        if current_count < N:
            print(f"Model {m} is missing {N - current_count} classes.")
            missing_classes_count = N - current_count
            potential_classes = []

            print(f"Track cluster num for model {m}: ", track_cluster_num[m])

           
            unused_clusters_for_model = list(set(range(num_clusters)) - set(track_cluster_num[m]))
            for cluster_idx in unused_clusters_for_model:
                all_classes = [cls for client_classes in cluster_info[cluster_idx].values() for cls in client_classes]
                new_classes = set(all_classes) - current_classes 
                potential_classes.extend(new_classes)

            # Remove duplicates
            potential_classes = list(set(potential_classes))
            print(f"Potential classes for model {m}: ", potential_classes)

 
            potential_clusters_per_class = {cls: [] for cls in potential_classes}
            for cls in potential_classes:
                for cluster_idx in unused_clusters_for_model:
                    if cls in [c for client_classes in cluster_info[cluster_idx].values() for c in client_classes]:
                        potential_clusters_per_class[cls].append(cluster_idx)

            print(f"Potential clusters per class for model {m}: ", potential_clusters_per_class)

      
            best_class_combination = []
            max_combination_linkage = -1

  
    
            potential_class_combinations = list(combinations(potential_classes, missing_classes_count))

            for class_combination in potential_class_combinations:

                class_cluster_pairs = [(cls, potential_clusters_per_class[cls][0]) for cls in class_combination]

            
                current_model_mean = [kmeans_means[cid] for cid in track_cluster_num[m]]
                combination_linkage = calculate_combination_linkage(model_means_dict[m], kmeans_means, class_cluster_pairs, track_cluster_num[m], current_model_mean)

          
                if combination_linkage > max_combination_linkage:
                    max_combination_linkage = combination_linkage
                    best_class_combination = class_cluster_pairs

          
            for best_class_to_add, best_cluster_for_class_to_add in best_class_combination:
                print(f"Adding class {best_class_to_add} from cluster {best_cluster_for_class_to_add} to model {m}")

                client = list(cluster_info[best_cluster_for_class_to_add].keys())[0] 
                if client not in combined_models[m]:
                    combined_models[m][client] = []

                combined_models[m][client].append(best_class_to_add)

                track_cluster_num[m].append(best_cluster_for_class_to_add)
                update_class_mean_in_model(model_means_dict[m], kmeans_means[best_cluster_for_class_to_add], best_class_to_add)

            print(f"Updated model {m} after adding combination: ", combined_models[m])
    
  
    return combined_models








def sum_models_zero_scaled(global_model, unique_counts):
  print('sum models zero scaled ')
  
  new_global_model=  global_model 
  
  for m in range(len(global_model)):
    for gm in range(len(global_model[m])):
      params = global_model[m][gm].get_state()
      new_params = params 
      for i in range(len(params)):
        weights = np.array(params[i][0], dtype = 'uint32')
        new_params[i] = (weights,params[i][1])
      global_model[m][gm].set_state(new_params)
      new_global_model[m][gm] = global_model[m][gm]
  return new_global_model 


def sum_models_scaled(global_model, unique_counts, mean_gini, dataset='F-MNIST'):
  all_weights = [] 
  print('sum models scaled ')
  threshold_values = {'MNIST': 0.5, 'F-MNIST': 0.5, 'CIFAR-10':0.6, 'SVHN': 0.3 }
  thres = threshold_values[dataset ]



  print('less than :', thres)

  #inequality is directly proportional to normalized gini 
  
  if mean_gini <= thres:
    print('less than thres')
    mean_gini = 1 

  
  print('scaling x mean gini: ', mean_gini)

  new_global_model=  global_model 
  
  for m in range(len(global_model)):
    model_weights = [] 
    for gm in range(len(global_model[m])):
      params = global_model[m][gm].get_state()
      new_params = params 
      for i in range(len(params)):
        weights = np.array(mean_gini*params[i][0] * unique_counts[m][i]/np.sum(unique_counts[m]), dtype = 'uint32')
        new_params[i] = (weights,params[i][1])
      global_model[m][gm].set_state(new_params)
      new_global_model[m][gm] = global_model[m][gm]
      model_weights.append(weights)
    all_weights.append(model_weights)
  return new_global_model , weights 


def sum_models(global_model):
  print('simple sum')
  sum_model = [] 
    
  for m in range(len(global_model[0])):
    model_num = [] 
    for gm in range(len(global_model)):
      model_num.append(global_model[gm][m])
    sum_model.append(model_num)

  return sum_model




def testing_global_model(global_model, X_test_data, Y_test_data, ratio_count, unique_count, numclasses):  
 
  votes = np.zeros((len(Y_test_data), numclasses), dtype= np.float32)
  

  for m in range(len(global_model)):
    print('gm: ', m)
    denominator_list = [] 
    for gm in range(len(global_model[m])):
      print('model num: ', gm )
      votes_local = np.zeros((len(Y_test_data), numclasses), dtype= np.float32)


      Y_test_threshold , Y_test_scores_threshold = global_model[m][gm].get_score(X_test_data[m])
    
      
      for i in range(Y_test_scores_threshold.shape[0]):
   
          
          denominator = (np.max(Y_test_scores_threshold) - np.min(Y_test_scores_threshold))
          
          denominator_list.append(denominator)
          if denominator == 0:
            denominator = 1 
          vote =1.0*Y_test_scores_threshold[i]/denominator
          votes[i] += vote 
          votes_local[i] += vote 
             
      Y_test_team = votes.argmax(axis=1)
      print('accuracy: ',  100*(Y_test_team == Y_test_data).mean())
    
  Y_test_team = votes.argmax(axis=1)
  print(np.unique(Y_test_team,return_counts=True))
 
  accuracy_global = 100*(Y_test_team == Y_test_data).mean()
  return accuracy_global 
  






all_traindata, all_testdata , numclasses = get_datasets(name=args.dataset, transform = None, numclients =numclients , TM = 'TM',
                                             distribution = None ,dirichlet = args.dirichlet,seed=args.seed, dir_type = args.dir_type, data_dir= data_dir)



print(len(all_traindata))
print(len(all_testdata))

print('NUM MODELS: ',num_models)
X_train_data, X_test_data,Y_train, Y_test = get_dataset_all(dataset_name=args.dataset,num_models=num_models, data_dir= data_dir )


print('len(global test): ', len(Y_test))
all_client_train_X , all_client_test_X ,all_client_train_Y ,all_client_test_Y  = make_client_data(args.dataset, all_traindata,all_testdata, numclients , num_models)
print(len(all_client_test_X))

for i in range(num_models):
  print('len global model ', i )
  print(X_train_data[i].shape)
  
models = get_client_models(args.dataset, num_models, args.T, args.s, patch_dim)
print('no. of clients models: ' , len(models))
print(model_list)

#make clients:

all_clients =[]
print(len(all_client_train_X),len(models))
for i in range(len(all_client_test_Y)):
  all_clients.append(Client(all_client_train_X[i], all_client_train_Y[i],all_client_test_X[i], all_client_test_Y[i], [], numclasses=numclasses))
  



global_model = model_list
global_parameters = [] 

avg_client_acc = {}
all_train_client = []
all_test_client = [] 
train_data_client =[]
test_data_client = [] 
votes = [] 
counter_votes_global = 0
unique_client_counts = [] 
ratio_count = [] 
gini_indexes = [] 
copy_data = 0
all_counts = np.zeros(numclasses)

for epoch in range(past_epochs, epochs):

    if epoch == past_epochs: 
      print('Initializing models: ')
      if len(global_model) == 0:
        local_epochs = 1
        client_m = [] 
        for m in range(len(all_client_test_X)):
          if np.max(all_client_train_Y[m]) == numclasses-1:
            copy_data  = m
            break

      for m in range(len(all_client_test_X)): 
        unique_classes = np.zeros(numclasses)
        unique, counts = np.unique(all_client_train_Y[m], return_counts=True)
         
        all_counts[unique] += counts 
     
        unique_classes[unique] = counts

        unique_counts = unique_classes / np.sum(counts)
        unique_client_counts.append(unique_classes)

        

        ratio_classes = np.where(unique_counts > 0, 1, 0)
        print(ratio_classes)

        ratio_count.append(ratio_classes)
        gi = normalized_gini(unique_counts)
        new_T=  args.T
        print('NORMALIZED GINI: ', gi, 'NEW_T: ', new_T)

        client_mod = [] 
        client_params = [] 
        for gm in range(len(models)):
          gm_mod = get_client_models(args.dataset, num_models, new_T, args.s, patch_dim)[gm]
          client_mod.append(gm_mod)
          client_params.append([])

          all_clients[m].model.append(gm_mod) 
        global_model.append(client_mod)
        gini_indexes.append(gi)
        global_parameters.append(client_params)
    
    avg_gini =  np.mean(gini_indexes)
    print('mean gini: ', avg_gini)
    if accuracy_dict != {}:
      print('Resume training from ', past_epochs)
    
        
    
    print('<><><><><><><><><><><><><><><><><><><><><><>')
    print('____________________________________________')
    print('++++++++Start of Training Round ', epoch, '++++++++++')


    if args.load_pretrain:
      if os.path.exists(model_name):
        with open(model_name, 'rb') as fp:
          global_model = pickle.load(fp)
          if args.dir_type == 'cls':
            for m in range(len(global_model)):
              unique_classes = np.zeros(numclasses)
              for i in range(len(unique_classes)):
                if np.mean(global_model[m][0].get_state()[i][0], axis = 0) > 0:
                  unique_classes[i] = 1
                  
              unique_client_counts[m] = unique_classes
              ratio_classes = np.where(unique_classes > 0, 1, 0)
              ratio_count[m] = ratio_classes
              print(ratio_classes)            
      else: 
        args.load_pretrain = 0 
    
    if not args.load_pretrain: 
      total_avg_time = [] 
      for m in range(len(all_client_test_X)): 
      
          
        client_votes = [] 
        counter_votes_client = 0
        
        print('................')
        print('copydata: ', copy_data)
        avg_time_train = 0
        for gm in range(len(global_model[m])):
          if epoch == past_epochs: 
            dc = all_clients[m].model[gm]
            dc.fit(all_client_train_X[copy_data][gm], all_client_train_Y[copy_data], epochs = 1,  incremental = True)
        

          start_time = time()
         # for i in range(args.local_epochs):
         # all_clients[m].model[gm].fit(all_client_train_X[m][gm], all_client_train_Y[m], epochs =args.local_epochs,  incremental = True)
          all_clients[m].train(epochs  =args.local_epochs, model_num = gm)
          prediction = all_clients[m].model[gm].predict(all_client_train_X[m][gm])
    
    
         #global_model[m][gm].fit(all_client_train_X[m][gm], all_client_train_Y[m], epochs =args.local_epochs,  incremental = True)
          
          end_time = time() 
          global_model[m][gm] = all_clients[m].model[gm]
          copy_params = global_model[m][gm].get_state()
          avg_time_train += end_time-start_time
          

 
       
        
          Y_train_threshold ,Y_train_scores_threshold= global_model[m][gm].get_score(all_client_train_X[m][gm])
          print('Client ', m , " - Train Accuracy: %.1f%%" % (100*(Y_train_threshold== all_client_train_Y[m]).mean()), end=' ')
          
          Y_test_threshold ,Y_test_scores_threshold= global_model[m][gm].get_score(all_client_test_X[m][gm])
          print('Client ', m , " - Test Accuracy: %.1f%%" % (100*(Y_test_threshold==  all_client_test_Y[m]).mean()), end=' ')
          
          if len(client_votes) != len(Y_test_scores_threshold):
            client_votes = np.zeros(np.array(Y_test_scores_threshold).shape, dtype=np.float32)
            client_count = np.zeros(np.array(Y_test_scores_threshold).shape, dtype=np.float32)
          
          
          global_parameters[m][gm] = all_clients[m].handle_uploading(model_num = gm)
          #handle_uploading(global_model[m][gm], all_client_train_Y[m], copy_params, numclasses=numclasses)

    
          for i in range(Y_test_scores_threshold.shape[0]):
              counter_votes_client +=1 
            
              vote  = 1.0*Y_test_scores_threshold[i]/(np.max(Y_test_scores_threshold) - np.min(Y_test_scores_threshold)) 
    
              client_votes[i] += vote #* ratio_classes

          Y_team_local = client_votes.argmax(axis = 1)
          global_parameters[m][gm] = all_clients[m].handle_non_iid(model_num = gm)
          #global_parameters[m][gm] = handle_non_iid(global_model[m][gm], all_client_train_Y[m], copy_params, numclasses=numclasses)
          print('Client ARGMAX Accuracy: ', "%.1f%%" % (100*(Y_team_local == all_client_test_Y[m]).mean()))
        
          print('.........')

          if len(votes) != len(Y_test_scores_threshold):
              votes = np.zeros(np.array(Y_test_scores_threshold).shape, dtype=np.float32)
          
        
          for i in range(Y_test_scores_threshold.shape[0]):
              
              vote =1.0*Y_test_scores_threshold[i]/(np.max(Y_test_scores_threshold) - np.min(Y_test_scores_threshold)) 
              votes[i] += vote #* ratio_classes
              counter_votes_global+=1 
              
        
        Y_team_local = client_votes.argmax(axis = 1)
        total_avg_time.append(avg_time_train)
        print('Client total training time: ', avg_time_train)
        print("Team LOCAL ACCURACY:  %.1f%%" % (100*(Y_team_local == all_client_test_Y[m]).mean()))
      # print('counter votes client: ', counter_votes_client)
     
      Y_test_team = votes.argmax(axis=1)
      #print('counter_votes_global: ', counter_votes_global)
      

      print('______________')
      
      #print("Team GLOBAL ACCURACY: %.1f%%" % (100*(Y_test_team == Y_test).mean()))
    # accuracy_dict[epoch] = 100*(Y_test_team == Y_test).mean()
  
      print('Average client training time: ', np.mean(total_avg_time))
      print('++++++++End of Training Round ', epoch, '++++++++++')
      

      with open(model_name, 'wb') as fp:
          pickle.dump(global_model, fp)
          print('model list saved')

     # with open(accuracy_dict_name, 'wb') as fp:
         # pickle.dump(accuracy_dict, fp)
         # print('accuracy dict saved')
      
    start_time = time()
    print('++++++++Evaluating Round ', epoch, '++++++++++')
    
    combined_ratio = [ratio_count] 
    combined_counts = [unique_client_counts]
  
    
    if args.ensemble:
      global_model_round = sum_models(global_model)
    
    else: 
      global_model_round  , weights = sum_models_scaled(global_model, unique_client_counts,avg_gini, dataset =args.dataset)
      global_model_round, combined_ratio = cluster_global_model_best(weights, global_model_round,ratio_count,unique_client_counts,k=k_cluster, c= args.c, mean_gini=avg_gini,num_models = num_models)

    
    end_time = time()
    print('Round ', epoch ,' Time for aggregation: ', end_time-start_time)
    



    global_acc = testing_global_model(global_model_round, X_test_data, Y_test , combined_ratio , combined_counts, numclasses=numclasses)
    accuracy_dict[epoch] = global_acc
    print('Global Accuracy for Round ', epoch, ': ', global_acc)
    print('------------------')
    print('Size: ')
    
    storage, upload, download = size(global_model_round,numclasses)
    print('Num models: ', len(global_model_round))


    print('Size (MB): ', storage)
    print('Upload (MB): ', upload)

    print('Download (MB): ', download)
    print('------------------')
  

    mean_client_acc = []
   
    print('Global Model Accuracy: ', accuracy_dict[epoch])
    print(accuracy_dict)



    print('++++++++End of Round ', epoch, '++++++++++')
    print('<><><><><><><><><><><><><><><><><><><><><><>')
    print('____________________________________________')
    

    









   
   