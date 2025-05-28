



from torch.utils.data import Dataset
import os 
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.datasets import ImageFolder, DatasetFolder
from torch.utils.data import DataLoader

from PIL import Image
import numpy as np 
from collections import Counter

import random
import cv2 
import pickle 





def get_datasets(name, split='train', transform=None,
                target_transform=None, download=False, numclients=4, dataset_path=None, TM = None, load_cids = None,balanced = True,dirichlet = 1e5, distribution = None, seed = 12345, dir_type = 'dir', data_dir = None):
    train = (split == 'train')
    test_dataset = None
    _DATASETS_MAIN_PATH = data_dir
    _dataset_path = {

    'MNIST':  _DATASETS_MAIN_PATH,
   'F-MNIST': _DATASETS_MAIN_PATH,
   'SVHN':  os.path.join(_DATASETS_MAIN_PATH,'SVHN/'),
   'CIFAR-10':  os.path.join(_DATASETS_MAIN_PATH,'CIFAR10/'),
    'CIFAR-100':  os.path.join(_DATASETS_MAIN_PATH,'CIFAR100/'),
   'ImageNet': os.path.join(_DATASETS_MAIN_PATH,'tiny-imagenet-200/'),
}

    
    
  
    if name == 'MNIST':
        dataset = datasets.MNIST(root=_dataset_path['MNIST'],
                                 train=True,
                                 transform = transform
                       )
        test_dataset = datasets.MNIST(root=_dataset_path['MNIST'],
                                 train=False,
                                 transform = transform)
        numclasses = 10
        data_x_train = dataset.data
        data_x_test = test_dataset.data 
        data_y_train = np.array(dataset.targets, dtype = np.uint8)
        data_y_test = np.array(test_dataset.targets, dtype = np.uint8)
        
    
 
    elif name == 'F-MNIST':
        dataset = datasets.FashionMNIST(root=_dataset_path['F-MNIST'],
                                 train=True
                                 )
        test_dataset = datasets.FashionMNIST(root=_dataset_path['F-MNIST'],
                                 train=False)
        numclasses = 10
        data_x_train = dataset.data
        data_x_test = test_dataset.data 
        print(data_x_train.shape)
        data_y_train = np.array(dataset.targets, dtype = np.uint8)
        data_y_test = np.array(test_dataset.targets, dtype = np.uint8)
       
         

    elif name == "SVHN":
      
       dataset = datasets.SVHN(root=_dataset_path['SVHN'],
                                 split='train')
                      
                             
       test_dataset = datasets.SVHN(root=_dataset_path['SVHN'],
                                 split = 'test')
       numclasses = 10
      

       data_y_test = np.array(test_dataset.labels,dtype=np.uint8)
        
       data_y_train = np.array(dataset.labels,dtype=np.uint8)

       
       data_x_train = dataset.data
       data_x_test= test_dataset.data
    
    elif 'ImageNet' in name:
      dataset,test_dataset = load_tinyimagenet_data(data_dir=_DATASETS_MAIN_PATH)
      data_x_train = dataset.data
      data_x_test= test_dataset.data
      data_y_test = np.array(test_dataset.targets,dtype=np.uint8)
        
      data_y_train = np.array(dataset.targets,dtype=np.uint8)
      
      numclasses=200
      
     

    elif "CIFAR" in name:
      
      if name == "CIFAR-10":
        dataset = datasets.CIFAR10(root=_dataset_path['CIFAR-10'],
                                  train = True)
        
        test_dataset = datasets.CIFAR10(root=_dataset_path['CIFAR-10'],
                                  train = False)
        numclasses = 10
      
      if name == "CIFAR-100":
        dataset = datasets.CIFAR100(root=_dataset_path['CIFAR-100'],
                                  train = True)
        
        test_dataset = datasets.CIFAR100(root=_dataset_path['CIFAR-100'],
                                  train = False)
        numclasses = 100 

      
      data_y_test = np.array(test_dataset.targets,dtype=np.uint8)
        
      data_y_train = np.array(dataset.targets,dtype=np.uint8)

      
      data_x_train = dataset.data
      data_x_test= test_dataset.data
      print(data_x_train.shape)
      print(data_x_test.shape)
    
    
    if  "synthetic" != name:
      dataset_train = []   
      dataset_test = []
    
      if distribution =='iid':
        dirichlet = 1e4

      elif distribution == 'noniid':
        dirichlet = 1e-2

      if load_cids != None: 
        indexes_ =  load_distributed_ids(load_cids)
      
      if load_cids == None:
  
        indexes_ =  distribute_dir_classes(numclients=numclients,numclasses=numclasses, dataset_train = data_y_train, dataset_test=data_y_test ,dirichlet = dirichlet,balanced = balanced, seed = seed,dir_type=dir_type)
        index_list_name = f'{name}{numclients}numclients_{dirichlet}dirichlet_{balanced}balanced' +'.pkl'
        with open(index_list_name, 'wb') as fp:
          pickle.dump(indexes_, fp)
          print('Index List saved successfully to file')
      
      
      for indexes in indexes_['train']:
        i_data_data = []
        i_data_labels = []
        for i in indexes:
          #i_data.append([data_x_train[i], data_y_train[i]])
          i_data_data.append(data_x_train[i])
          i_data_labels.append(data_y_train[i])
        dataset_train.append([np.array(i_data_data),np.array(i_data_labels)])
        #dataset_train.append(i_data)
      for indexes in indexes_['test']:
        i_data_data = []
        i_data_labels = []
        for i in indexes:
          i_data_data.append(data_x_test[i])
          i_data_labels.append(data_y_test[i])
        #dataset_test.append(i_data)
        dataset_test.append([np.array(i_data_data),np.array(i_data_labels)])

    

    return dataset_train, dataset_test, numclasses





def distribute_dir_classes(numclients,numclasses, dataset_train, dataset_test, dirichlet = 1e5,balanced = True, seed = 12345, dir_type = 'dir'):
  
  if dir_type == 'dir': 
    """Distribute data across clients by sampling class priors from a Dirichlet Distribution"""
    client_ids = {'train':[],'test':[]}

    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed) 

    if np.all(dataset_test == None):
      all_ids = np.array(dataset_train)
    # Determine locations of different classes
      class_ids = {class_num: np.where(all_ids== class_num)[0] for class_num in range(numclasses)}
    
      dist_of_client = np.random.dirichlet(np.repeat(dirichlet, numclients), size=numclasses).transpose()
      dist_of_client /= dist_of_client.sum()
    
      n_samples = len(dataset_train)

      if(balanced):
        for i in range(numclients):
          s0 = dist_of_client.sum(axis=0, keepdims=True)
          s1 = dist_of_client.sum(axis=1, keepdims=True)
          dist_of_client /= s0
          dist_of_client /= s1

    #Allocate number of samples per class to each client based on distribution
      samples_per_class= (np.floor(dist_of_client * n_samples))
    
      
      start_ids = np.zeros((numclients+1, numclasses), dtype=np.int32)
      for i in range(0, numclients):
        start_ids[i+1] = start_ids[i] + samples_per_class[i]
        
  
      for client_num in range(numclients):
        l = np.array([], dtype=np.int32)
        client_temp=l
    
        for class_num in range(numclasses):
          start, end = start_ids[client_num, class_num], start_ids[client_num+1, class_num]
          l = np.concatenate((l, class_ids[class_num][start:end].tolist())).astype(np.int32)
        
        client_temp = l
        client_ids['train'].append(client_temp)
        
        print('client_num', client_num)
        print(len(l))

    else:
    
  
        all_ids_train = np.array(dataset_train)
        all_ids_test = np.array(dataset_test)
      
        class_ids_train = {class_num: np.where(all_ids_train == class_num)[0] for class_num in range(numclasses)}
        class_ids_test = {class_num: np.where(all_ids_test == class_num)[0] for class_num in range(numclasses)}
      
        dist_of_client = np.random.dirichlet(np.repeat(dirichlet, numclients), size=numclasses).transpose()
        dist_of_client /= dist_of_client.sum()
    
    
        n_samples_train = len(dataset_train.data)
        n_samples_test= len(dataset_test.data)
  
        if(balanced):
          for i in range(numclients):
              s0 = dist_of_client.sum(axis=0, keepdims=True)
              s1 = dist_of_client.sum(axis=1, keepdims=True)
              dist_of_client /= s0
              dist_of_client /= s1

    
        samples_per_class_train = (np.floor(dist_of_client * n_samples_train))
        samples_per_class_test = (np.floor(dist_of_client * n_samples_test))
    
        start_ids_train = np.zeros((numclients+1, numclasses), dtype=np.int32)
        start_ids_test = np.zeros((numclients+1, numclasses), dtype=np.int32)
      
        for i in range(0, numclients):
          start_ids_train[i+1] = start_ids_train[i] + samples_per_class_train[i]
          start_ids_test[i+1] = start_ids_test[i] + samples_per_class_test[i]
        
      
        for client_num in range(numclients):
          l = np.array([], dtype=np.int32)
          k = np.array([], dtype=np.int32)
        
          client_temp = l
          client_temp_test = k
        
    
          for class_num in range(numclasses):
      
              start, end = start_ids_train[client_num, class_num], start_ids_train[client_num+1, class_num]
              l = np.concatenate((l, class_ids_train[class_num][start:end].tolist())).astype(np.int32)
              start, end = start_ids_test[client_num, class_num], start_ids_test[client_num+1, class_num]
              k = np.concatenate((k, class_ids_test[class_num][start:end].tolist())).astype(np.int32)
          
        
          client_temp = l
          client_ids['train'].append(client_temp)
          client_temp_test = k
          client_ids['test'].append(client_temp_test) 
        
          print('client_num', client_num)
          print(len(l))
          print(len(k))


  elif 'cls' in dir_type:
     
      np.random.seed(seed)
      client_ids = {'train': [], 'test': []}
      K = numclasses  # Total number of classes
      n_parties = numclients  # Number of clients

      num = int(dirichlet)
      # Ensure num does not exceed K
      if num > K:
          raise ValueError("Number of classes per client (num) cannot exceed the total number of classes (K).")

      times = [0 for _ in range(K)]  
      contain = []  

      # Initialize data structures
     
      
      if n_parties == K: #distribute each class to each client first
        for i in range(n_parties):
          client_ids['train'].append([])
          client_ids['test'].append([])
          times[i] += 1 
          current = random.sample([c for c in range(K) if c != i], num - 1)    
          for cls in current:
              times[cls] += 1
          current.append(i)
          contain.append(current)
      # Distribute remaining classes
      else:
        for i in range(n_parties):
            client_ids['train'][i] = []
            client_ids['test'][i] = []
            current = random.sample(range(K), num)
            for cls in current:
                times[cls] += 1
            contain.append(current)
    
      # Split dataset indices by class and distribute to clients
      for i in range(K):
        idx_k_train = np.where(dataset_train == i)[0]
        idx_k_test = np.where(dataset_test == i)[0]
        np.random.shuffle(idx_k_train)
        np.random.shuffle(idx_k_test)

        # Only split if times[i] is greater than 0
        if times[i] > 0:
            # Split data among clients for the current class
            split_train = np.array_split(idx_k_train, times[i])
            split_test = np.array_split(idx_k_test, times[i])

            ids = 0
            for j in range(n_parties):
                if i in contain[j]:
                    if ids < len(split_train) and len(split_train[ids]) > 0:
                        client_ids['train'][j].extend(split_train[ids])
                    if ids < len(split_test) and len(split_test[ids]) > 0:
                        client_ids['test'][j].extend(split_test[ids])
                    ids += 1

  return client_ids
       



def get_dataset_all(dataset_name, num_models, data_dir ): 
 

  _DATASETS_MAIN_PATH = data_dir

  _dataset_path = {

        'MNIST':  _DATASETS_MAIN_PATH,
    'F-MNIST': _DATASETS_MAIN_PATH,
    'SVHN':  os.path.join(_DATASETS_MAIN_PATH,'SVHN/'),
    'CIFAR-10':  os.path.join(_DATASETS_MAIN_PATH,'CIFAR10/'),
        'CIFAR-100':  os.path.join(_DATASETS_MAIN_PATH,'CIFAR100/'),
    'ImageNet': os.path.join(_DATASETS_MAIN_PATH,'tiny-imagenet-200/'),
    }


  if "CIFAR" in dataset_name or "ImageNet" in dataset_name : 
    if dataset_name == 'ImageNet':
      dataset,test_dataset = load_tinyimagenet_data(data_dir= _DATASETS_MAIN_PATH)
      data_x_train = dataset.data
      data_x_test= test_dataset.data
      data_y_test = np.array(test_dataset.targets,dtype=np.uint8)
        
      data_y_train = np.array(dataset.targets,dtype=np.uint8)
      X_train_org, Y_train, X_test_org, Y_test = dataset.data,dataset.targets, test_dataset.data, test_dataset.targets

      numclasses=200
      imageSize = 32
      
    if dataset_name == "CIFAR-10":

      dataset = datasets.CIFAR10(root=_dataset_path['CIFAR-10'],
                                  train = True)
        
      test_dataset = datasets.CIFAR10(root=_dataset_path['CIFAR-10'],
                                      train = False)
      numclasses = 10
      imageSize = 32
      X_train_org, Y_train, X_test_org, Y_test = dataset.data,dataset.targets, test_dataset.data, test_dataset.targets
 
    
    if dataset_name == "CIFAR-100":

      dataset = datasets.CIFAR100(root=_dataset_path['CIFAR-100'],
                                  train = True)
        
      test_dataset = datasets.CIFAR100(root=_dataset_path['CIFAR-100'],
                                      train = False)
      numclasses = 100
      imageSize = 32
    
      X_train_org, Y_train, X_test_org, Y_test = dataset.data,dataset.targets, test_dataset.data, test_dataset.targets
    
    Y_train=np.array(Y_train)#.reshape(Y_train.shape[0])
    Y_test=np.array(Y_test)#.reshape(Y_test.shape[0])

      #The size of the original image - in pixels - assuming this is a square image
    channels = 3    #The number of channels of the image. A RBG color image, has 3 channels
    classes = 10    #The number of classes available for this dataset

    winSize = imageSize
    blockSize = 12
    blockStride = 4
    cellSize = 4
    nbins = 18
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = True
    nlevels = 64
    signedGradient = True
    device = "CPU"
    max_included_literals = 32
    resolution = 8
    factor = 1
    hog = cv2.HOGDescriptor((winSize,winSize),(blockSize, blockSize),(blockStride,blockStride),(cellSize,cellSize),nbins,derivAperture, winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels,signedGradient)


    fd = hog.compute(X_test_org[0])
    X_test_hog = np.empty((X_test_org.shape[0], fd.shape[0]), dtype=np.uint32)
    for i in range(X_test_org.shape[0]):
        fd = hog.compute(X_test_org[i])
        X_test_hog[i] = fd >= 0.1

    X_test_threshold = np.copy(X_test_org)
    for i in range(X_test_threshold.shape[0]):
        for j in range(X_test_threshold.shape[3]):
            X_test_threshold[i,:,:,j] = cv2.adaptiveThreshold(X_test_org[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)#cv2.adaptiveThreshold(X_test[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)

    X_test_thermometer3 = np.empty((X_test_org.shape[0], X_test_org.shape[1], X_test_org.shape[2], X_test_org.shape[3], resolution), dtype=np.uint8)
    for z in range(resolution):
        X_test_thermometer3[:,:,:,:,z] = X_test_org[:,:,:,:] >= (z+1)*255/(resolution+1)
    X_test_thermometer3 = X_test_thermometer3.reshape((X_test_org.shape[0], X_test_org.shape[1], X_test_org.shape[2], 3*resolution))

    X_test_thermometer4 = np.empty((X_test_org.shape[0], X_test_org.shape[1], X_test_org.shape[2], X_test_org.shape[3], resolution), dtype=np.uint8)
    for z in range(resolution):
        X_test_thermometer4[:,:,:,:,z] = X_test_org[:,:,:,:] >= (z+1)*255/(resolution+1)
    X_test_thermometer4 = X_test_thermometer4.reshape((X_test_org.shape[0], X_test_org.shape[1], X_test_org.shape[2], 3*resolution))
            
    fd = hog.compute(X_train_org[0])
    X_train_hog = np.empty((X_train_org.shape[0], fd.shape[0]), dtype=np.uint32)

    for i in range(X_train_hog.shape[0]):
        fd = hog.compute(X_train_org[i]) 
        X_train_hog[i] = fd >= 0.1


    X_train_threshold = np.copy(X_train_org)
    for i in range(X_train_threshold.shape[0]):
        for j in range(X_train_threshold.shape[3]):
            #X_train_threshold[i,:,:,j] = cv2.adaptiveThreshold(X_train_org[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) #cv2.adaptiveThreshold(X_train[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
            X_train_threshold[i,:,:,j] = cv2.adaptiveThreshold(X_train_threshold[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) #cv2.adaptiveThreshold(X_train[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)


    X_train_thermometer3 = np.empty((X_train_org.shape[0], X_train_org.shape[1], X_train_org.shape[2], X_train_org.shape[3], resolution), dtype=np.uint8)

    for z in range(resolution):
        X_train_thermometer3[:,:,:,:,z] = X_train_org[:,:,:,:] >= (z+1)*255/(resolution+1)
    X_train_thermometer3 = X_train_thermometer3.reshape((X_train_org.shape[0], X_train_org.shape[1], X_train_org.shape[2], 3*resolution))

    X_train_thermometer4 = np.empty((X_train_org.shape[0], X_train_org.shape[1], X_train_org.shape[2], X_train_org.shape[3], resolution), dtype=np.uint8)

    for z in range(resolution):
        X_train_thermometer4[:,:,:,:,z] = X_train_org[:,:,:,:] >= (z+1)*255/(resolution+1)

    X_train_thermometer4 = X_train_thermometer4.reshape((X_train_org.shape[0], X_train_org.shape[1], X_train_org.shape[2], 3*resolution))

    X_train_thermometer5 = np.copy(X_train_thermometer4)
    X_test_thermometer5 = np.copy(X_test_thermometer4)
    X_train_data  = []
    
    X_train_data.append(X_train_threshold)
    X_train_data.append(X_train_thermometer3)
    X_train_data.append(X_train_thermometer4)
    #X_train_data.append(X_train_thermometer5)
    X_train_data.append(X_train_hog)

    X_test_data  = []
    
    X_test_data.append(X_test_threshold)
    X_test_data.append(X_test_thermometer3)
    X_test_data.append(X_test_thermometer4)
   # X_test_data.append(X_test_thermometer5)
    X_test_data.append(X_test_hog)

  elif dataset_name == 'MNIST':
    dataset = datasets.MNIST(root=_dataset_path['MNIST'],
                              train=True)
    test_dataset = datasets.MNIST(root=_dataset_path['MNIST'],
                              train=False)
    numclasses = 10
    data_x_train = dataset.data
    data_x_test = test_dataset.data 
    data_y_train = np.array(dataset.targets, dtype = np.uint8)
    data_y_test = np.array(test_dataset.targets, dtype = np.uint8)
    train_data_mnist = np.where(data_x_train>= 75, 1, 0) 
    test_data_mnist = np.where(data_x_test>= 75, 1, 0) 

    X_train_data=[]
    X_train_data.append(train_data_mnist)
    X_train_data.append(train_data_mnist)
    X_test_data = [] 
    X_test_data.append(test_data_mnist)
    X_test_data.append(test_data_mnist)
    Y_test = data_y_test
    Y_train = data_y_train
    
        
    

  elif dataset_name == 'F-MNIST':
      dataset = datasets.FashionMNIST(root=_dataset_path['F-MNIST'],
                                train=True)
      test_dataset = datasets.FashionMNIST(root=_dataset_path['F-MNIST'],
                                train=False)
      numclasses = 10
      data_train = dataset.data
      data_test = test_dataset.data 
    
      data_y_train = np.array(dataset.targets, dtype = np.uint8)
      data_y_test = np.array(test_dataset.targets, dtype = np.uint8)

      data_x_train = np.copy(data_train)
      data_x_test = np.copy(data_test)
      for i in range(data_x_train.shape[0]):
        data_x_train[i,:]=(cv2.adaptiveThreshold(data_x_train[i], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))
      for i in range(data_x_test.data.shape[0]):
        data_x_test[i,:] = cv2.adaptiveThreshold(data_x_test[i], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
      
      X_train_data=[]
      X_train_data.append(data_x_train)

      X_test_data = [] 
      X_test_data.append(data_x_test)
      Y_test = data_y_test
      Y_train = data_y_train
    
  
  elif dataset_name == "SVHN":
      resolution = 8
    
      dataset = datasets.SVHN(root=_dataset_path['SVHN'],
                                split='train')
      test_dataset = datasets.SVHN(root=_dataset_path['SVHN'],
                                split = 'test')
      numclasses = 10
    

      data_y_test = np.array(test_dataset.labels,dtype=np.uint8)
      
      data_y_train = np.array(dataset.labels,dtype=np.uint8)

      
      X_train = dataset.data.reshape(dataset.data.shape[0], dataset.data.shape[2], dataset.data.shape[3], 3)
   

      X_test = test_dataset.data.reshape(test_dataset.data.shape[0], test_dataset.data.shape[2], test_dataset.data.shape[3], 3)
  

      
      data_x_train = apply_adaptive_threshold(X_train)
      
      X_train_thermometer4 = np.empty((X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3], resolution), dtype=np.uint8)

      for z in range(resolution):
          X_train_thermometer4[:,:,:,:,z] = X_train[:,:,:,:] >= (z+1)*255/(resolution+1)

      X_train_thermometer4 = X_train_thermometer4.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 3*resolution))

      X_test_thermometer4 = np.empty((X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3], resolution), dtype=np.uint8)

      for z in range(resolution):
          X_test_thermometer4[:,:,:,:,z] = X_test[:,:,:,:] >= (z+1)*255/(resolution+1)

      X_test_thermometer4 = X_test_thermometer4.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 3*resolution))


      data_x_test = apply_adaptive_threshold(X_test)




      X_train_data=[]
      X_train_data.append(data_x_train)
   
      X_test_data = [] 
      X_test_data.append(data_x_test)
  
      Y_test = data_y_test
      Y_train = data_y_train
  
  
  Y_test_half  = []
  X_test_data_half = [] 
  
  for i in range(num_models):
    test_dataloader = DataLoader(list(zip(X_test_data[i],Y_test)), shuffle = False, batch_size = 32)
    half= int(len(test_dataloader)/2 )
    X_test_temp = [] 
    for batch_idx, (X_batch, y_batch) in enumerate(test_dataloader):
      if batch_idx  < half:
        continue
      if i == 0:
        Y_test_half.extend(y_batch)
      X_test_temp.extend(X_batch)
    X_test_data_half.append(np.array(X_test_temp))
    
  print('len test: ', len(Y_test_half))
  for i in range(len(X_train_data)):
    print('shape ', i ,  X_train_data[i].shape)
 
  return X_train_data[0:num_models], X_test_data_half[0:num_models],Y_train, np.array(Y_test_half)
    


      
def apply_adaptive_threshold(images):
    thresholded_images = np.copy(images)
    for i in range(images.shape[0]):
        for channel in range(images.shape[-1]):
            thresholded_images[i, :, :, channel] = cv2.adaptiveThreshold(
                images[i, :, :, channel], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

    return thresholded_images

def make_client_data(dataset, all_traindata, all_testdata, numclients, num_models ):
  all_client_train_X = []
  all_client_test_X  = [] 
  all_client_train_Y = []
  all_client_test_Y  = [] 

  if 'CIFAR' in dataset or 'ImageNet' in dataset:
    
    imageSize = 32  #The size of the original image - in pixels - assuming this is a square image
    if 'ImageNet' in dataset:
      imageSize = 32
    channels = 3    #The number of channels of the image. A RBG color image, has 3 channels
   

    winSize = imageSize
    blockSize = 12
    blockStride = 4
    cellSize = 4
    nbins = 18
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = True
    nlevels = 64
    signedGradient = True
    device = "CPU"
    max_included_literals = 32
    resolution = 8
    factor = 1
    hog = cv2.HOGDescriptor((winSize,winSize),(blockSize, blockSize),(blockStride,blockStride),(cellSize,cellSize),nbins,derivAperture, winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels,signedGradient)


    for m in range(numclients):
      if len(all_traindata[m][1]) > 0 and len(all_testdata[m][1]) > 0 :
        all_client_train_Y.append(all_traindata[m][1])
        all_client_test_Y.append(all_testdata[m][1])
        train_data = all_traindata[m][0]
        test_data = all_testdata[m][0]
        train_data_threshold = np.copy(train_data)
        test_data_threshold = np.copy(test_data)
        
    

        fd = hog.compute(train_data[0])
        train_data_hog  = np.empty((train_data.shape[0], fd.shape[0]), dtype=np.uint32)

        for i in range(train_data.shape[0]):
            fd = hog.compute(train_data[i]) 
            train_data_hog[i] = fd >= 0.1



        train_thermometer3 = np.empty((train_data.shape[0], train_data.shape[1], train_data.shape[2], train_data.shape[3], resolution), dtype=np.uint8)

        for z in range(resolution):
            train_thermometer3[:,:,:,:,z] = train_data[:,:,:,:] >= (z+1)*255/(resolution+1)
        
        train_thermometer3 = train_thermometer3.reshape((train_data.shape[0], train_data.shape[1], train_data.shape[2], 3*resolution))

        test_thermometer3 = np.empty((test_data.shape[0], test_data.shape[1], test_data.shape[2], test_data.shape[3], resolution), dtype=np.uint8)

        for z in range(resolution):
            test_thermometer3[:,:,:,:,z] = test_data[:,:,:,:] >= (z+1)*255/(resolution+1)
        
        test_thermometer3 = test_thermometer3.reshape((test_data.shape[0], test_data.shape[1], test_data.shape[2], 3*resolution))


        train_thermometer4 = np.empty((train_data.shape[0], train_data.shape[1], train_data.shape[2], train_data.shape[3], resolution), dtype=np.uint8)

        for z in range(resolution):
            train_thermometer4[:,:,:,:,z] = train_data[:,:,:,:] >= (z+1)*255/(resolution+1)
        
        train_thermometer4 = train_thermometer4.reshape((train_data.shape[0], train_data.shape[1], train_data.shape[2], 3*resolution))


        fd = hog.compute(test_data[0])
        test_data_hog  = np.empty((test_data.shape[0], fd.shape[0]), dtype=np.uint32)

        for i in range(test_data.shape[0]):
            fd = hog.compute(test_data[i]) 
            test_data_hog[i] = fd >= 0.1


        test_thermometer4 = np.empty((test_data.shape[0], test_data.shape[1], test_data.shape[2], test_data.shape[3], resolution), dtype=np.uint8)

        for z in range(resolution):
            test_thermometer4[:,:,:,:,z] = test_data[:,:,:,:] >= (z+1)*255/(resolution+1)
        
        test_thermometer4 = test_thermometer4.reshape((test_data.shape[0], test_data.shape[1], test_data.shape[2], 3*resolution))

        
        for i in range(train_data_threshold.shape[0]):
            for j in range(train_data_threshold.shape[3]):
                train_data_threshold[i,:,:,j] = cv2.adaptiveThreshold(train_data_threshold[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) #cv2.adaptiveThreshold(X_train[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)

        for i in range(test_data_threshold.shape[0]):
            for j in range(test_data_threshold.shape[3]):
                test_data_threshold[i,:,:,j] = cv2.adaptiveThreshold(test_data_threshold[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)#cv2.adaptiveThreshold(X_test[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
        
        train_thermometer5 = np.copy(train_thermometer4)
        test_thermometer5 = np.copy(test_thermometer4)

        all_client_train_X.append([ train_data_threshold, train_thermometer3,train_thermometer4, train_data_hog][0:num_models])
        all_client_test_X.append([test_data_threshold, test_thermometer3,test_thermometer4, test_data_hog][0:num_models])
        print('client num: ', m)
      
        print('train: ', Counter(all_client_train_Y[-1]))
        print('test: ', Counter(all_client_test_Y[-1]))

  if dataset=="MNIST":
    for m in range(numclients):
      if len(all_traindata[m][1]) > 0 and len(all_testdata[m][1]) > 0 :
        all_client_train_Y.append(all_traindata[m][1])
        all_client_test_Y.append(all_testdata[m][1])
        train_data = all_traindata[m][0]
        test_data = all_testdata[m][0]
      
        train_data_mnist = np.where(train_data>= 75, 1, 0) 
        test_data_mnist = np.where(test_data>= 75, 1, 0) 
        all_client_train_X.append([train_data_mnist,train_data_mnist])
        all_client_test_X.append([test_data_mnist,test_data_mnist])
        print('client num: ', m)
        
        print('train: ', Counter(all_client_train_Y[-1]))
        print('test: ', Counter(all_client_test_Y[-1]))
 
  if dataset=="EMNIST":
    for m in range(numclients):
      if len(all_traindata[m][1]) > 0 and len(all_testdata[m][1]) > 0 :
        all_client_train_Y.append(all_traindata[m][1])
        all_client_test_Y.append(all_testdata[m][1])
        train_data = all_traindata[m][0]
        test_data = all_testdata[m][0]
        
        train_data_emnist = np.where(train_data>= 75, 1, 0) 
        test_data_emnist = np.where(test_data>= 75, 1, 0) 
        all_client_train_X.append([train_data_emnist,train_data_emnist])
        all_client_test_X.append([test_data_emnist,test_data_emnist])
        print('client num: ', m)
        
        print('train: ', Counter(all_client_train_Y[-1]))
        print('test: ', Counter(all_client_test_Y[-1]))
  
  if dataset=='F-MNIST':
    for m in range(numclients):
      if len(all_traindata[m][1]) > 0 and len(all_testdata[m][1]) > 0 :
      
        all_client_train_Y.append(all_traindata[m][1])
        all_client_test_Y.append(all_testdata[m][1])
        data_train = all_traindata[m][0]
        data_test = all_testdata[m][0]

        data_x_train = np.copy(data_train)
        for i in range(data_x_train.shape[0]):
          data_x_train[i,:]=(cv2.adaptiveThreshold(data_x_train[i], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))
        data_x_test =np.copy(data_test)
        for i in range(data_x_test.shape[0]):
          data_x_test[i,:] = cv2.adaptiveThreshold(data_x_test[i], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        all_client_train_X.append([data_x_train])
        all_client_test_X.append([data_x_test])
        print('client num: ', m)
        
        print('train: ', Counter(all_client_train_Y[-1]))
        print('test: ', Counter(all_client_test_Y[-1]))

  if dataset=='SVHN':
    for m in range(numclients):
      if len(all_traindata[m][1]) > 0 and len(all_testdata[m][1]) > 0 :
        all_client_train_Y.append(all_traindata[m][1])
        all_client_test_Y.append(all_testdata[m][1])
        train_data = all_traindata[m][0]
        test_data = all_testdata[m][0]
    


        X_train =train_data.reshape(train_data.shape[0], train_data.data.shape[2], train_data.shape[3], 3)

        X_test= test_data.reshape(test_data.data.shape[0], test_data.shape[2],test_data.shape[3], 3)

    
        data_x_train = apply_adaptive_threshold(X_train)
        data_x_test = apply_adaptive_threshold(X_test)

        all_client_train_X.append([data_x_train,data_x_train])
        all_client_test_X.append([data_x_test,data_x_test])
        print('client num: ', m)
        
        print('train: ', Counter(all_client_train_Y[-1]))
        print('test: ', Counter(all_client_test_Y[-1]))

  
  return all_client_train_X, all_client_test_X, all_client_train_Y, all_client_test_Y
        






def load_tinyimagenet_data(data_dir ='/data/'):
  
    data_root =   os.path.join(data_dir,'tiny-imagenet-200/')
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize images to 32x32
        transforms.ToTensor()         # Convert images to tensors
    ])

    
    xray_train_ds = TinyImageNetDataset(data_root, train=True, transform=transform)
    xray_test_ds = TinyImageNetDataset(data_root, train=False, transform=transform)
    
    # Convert images to 8-bit format after loading
    X_train = np.array([np.array(xray_train_ds[i][0] * 255, dtype=np.uint8).transpose(1, 2, 0) for i in range(len(xray_train_ds))])
    y_train = np.array([int(xray_train_ds[i][1]) for i in range(len(xray_train_ds))])
    X_test = np.array([np.array(xray_test_ds[i][0] * 255, dtype=np.uint8).transpose(1, 2, 0) for i in range(len(xray_test_ds))])
    y_test = np.array([int(xray_test_ds[i][1]) for i in range(len(xray_test_ds))])

    # Wrap the data in the custom class
    train_data = DatasetWithDataAttribute(X_train, y_train)
    test_data = DatasetWithDataAttribute(X_test, y_test)

    return train_data, test_data


class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)
            

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:

            return len(self.dataidxs)

class DatasetWithDataAttribute:
    def __init__(self, X, y):
        self.data = X
        self.targets = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]



class TinyImageNetDataset(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        """
        Args:
            root (str): Root directory path for the Tiny ImageNet dataset.
            train (bool): If True, loads the training set; if False, loads the validation set.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_transform (callable, optional): Optional transform to be applied on the target.
        """
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if self.train:
            self.data_dir = os.path.join(root, 'train')
        else:
            self.data_dir = os.path.join(root, 'val')

        self.samples, self.labels = self._load_data()

    def _load_data(self):
        """Loads data and labels from the dataset directory."""
        samples = []
        labels = []

        if self.train:
            classes = sorted(os.listdir(self.data_dir))
            class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
            for cls_name in classes:
                cls_dir = os.path.join(self.data_dir, cls_name, 'images')
                for img_name in os.listdir(cls_dir):
                    img_path = os.path.join(cls_dir, img_name)
                    samples.append(img_path)
                    labels.append(class_to_idx[cls_name])
        else:
            # For validation, use a separate directory structure or annotation file if needed
            val_dir = os.path.join(self.data_dir, 'images')
            with open(os.path.join(self.data_dir, 'val_annotations.txt'), 'r') as f:
                annotations = f.readlines()
                img_to_label = {line.split('\t')[0]: line.split('\t')[1] for line in annotations}
                classes = sorted(set(img_to_label.values()))
                class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
                for img_name, cls_name in img_to_label.items():
                    img_path = os.path.join(val_dir, img_name)
                    samples.append(img_path)
                    labels.append(class_to_idx[cls_name])

        return samples, labels

    def __getitem__(self, index):
        """Fetches an image and its label."""
        img_path = self.samples[index]
        label = self.labels[index]
        image = Image.open(img_path).convert('RGB')  # Convert to RGB to ensure consistency

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.samples)
