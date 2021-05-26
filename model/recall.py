# -*- coding: utf-8 -*-

import torch
import numpy as np
from model.utils import split_train_test
from model.dataset.dataset import CustomDataLoader
from model.collaborative.lightfm import LightFM
from model.collaborative.mlp import MLP
from model.collaborative.ease import EASE
from model.collaborative.neu import NeuCF
from model.helper.cuda import gpu, cpu
from model.helper.loss import hinge_loss
from model.evaluate import EvaluateRec_all
from model.helper.negative_sampling import get_negative_batch



class Recall(torch.nn.Module):
    
    """
    Encodes users (or item sequences) and items in a low dimensional space, using dot products as similarity measure 
    and Collaborative Filterings or Sequence Models as backend.
    
    ----------
    dataset: class
        instance of dataset class
    n_factors: int
        dimensionality of the embedding space
    net_type: string
        type of the model/net.
        "LightFM" -> Collaborative Filtering with (optional) item metadata with add operator and dot product
        "MLP" -> Collaborative Filtering with (optional) item metadata with concat operator and a stack of linear layers
        "NeuCF" -> Combine LightFM and MLP with a concat layer and a stack of linear layers 
        "EASE" -> Embarassingly Shallow Auto-Encoder
        "LSTM" -> Sequence Model using LSTM
    use_metadata: boolean
        Use True to add metadata to training procedure
    use_cuda: boolean, optional
        Use CUDA as backend. Default to False
    """
    
    def __init__(self,
                 dataset, 
                 n_factors,
                 net_type = 'light_fm',
                 optimizer = 'Adam',
                 lr = 3e-3,
                 use_metadata = False,
                 use_cuda=False, 
                 verbose = True):

        super().__init__()
             
        self.dataset = dataset
        self.n_users = int(self.dataset.users_id.max()) + 1
        self.n_items = int(self.dataset.items_id.max()) + 1
        self.epoch = 0
        self.use_metadata = use_metadata
        self.mapping_item_metadata = dataset.get_item_metadata_mapping if use_metadata else None
        

        self.verbose = verbose

        self.n_factors = n_factors
        
        self.use_cuda = use_cuda

        self.net_type = net_type
        

        self._init_net(optimizer, lr)

    @property
    def get_n_metadata(self):
        return [i + 1 for i in self.dataset.metadata_id.max(axis=0).values.tolist()]
        
    def _init_net(self, optimizer, lr):

        if self.net_type == 'hybrid_cf':
          print('Training LightFM')
          self.net = LightFM(n_users=self.n_users, 
                              n_items=self.n_items, 
                              n_metadata=self.get_n_metadata if self.use_metadata else None, 
                              n_factors=self.n_factors, 
                              use_metadata=self.use_metadata, 
                              use_cuda=self.use_cuda)

        elif net_type == 'mlp':
          print('MLP under_construction')
          #net = MLP(n_users, n_items, n_metadata, n_metadata_type, n_factors, use_metadata=True, use_cuda=False)
          
        elif net_type == 'ease':
          print('EASE under construction')
          
        elif net_type == 'neucf':
          print('NeuCF under construction')
        
        self.optimizer = self.get_optimizer(optimizer, lr=lr)
        
    def get_optimizer(self, optimizer, lr=3e-3):

        if optimizer == 'Adam':
            return torch.optim.Adam(self.net.parameters(),
                             lr=lr)
        else:
            pass

    def forward(self, net, batch, batch_size):

        score = gpu(net.forward(batch, batch_size), self.use_cuda)

        return score
    
    
    def backward(self, positive, negative):
                
        self.optimizer.zero_grad()
                
        loss_value = hinge_loss(positive, negative)                
        loss_value.backward()
        
        self.optimizer.step()
        
        return loss_value.item()
    
    def fit_partial(self, users, items, metadata=None, verbose=False):
        
        self.epoch+=1

        self.net = self.net.train()

        positive = gpu(self.net(users, 
                                items, 
                                metadata = metadata if self.use_metadata else None), 
                        self.use_cuda)

        neg_items, neg_metadata = get_negative_batch(users, self.n_items,self.mapping_item_metadata, use_metadata = self.use_metadata)
        negative = gpu(self.net(users, 
                                neg_items, 
                                metadata = neg_metadata), 
                        self.use_cuda) 
                                                        
        loss_value = self.backward(positive, negative)

        if verbose:
            print(f'epoch fitting : {self.epoch}')

        return loss_value

    
    def fit(self, batch_size=1024, epochs=10, splitting_train_test=False, eval_bool = False, k=3):

        if splitting_train_test:
            
            print('|== Splitting Train/Test ==|')

            train, test = split_train_test(self.dataset, test_percentage=0.25, random_state=None)
            
        else:
            
            train = self.dataset
        
        
        train_loader = CustomDataLoader(dataset=train, batch_size=batch_size, shuffle = False)
        
        # self.total_train_auc = []
        # self.total_test_auc = []
        self.total_loss = []

        if eval_bool and splitting_train_test: 
            self.evaluation = EvaluateRec_all(mapping_item_metadata=self.mapping_item_metadata, k=k, kind='AUC')
            

        for epoch in range(epochs):
            
            
            for e, (users, items, metadata, weights) in enumerate(train_loader):                
                loss_value = self.fit_partial(users, items, metadata=metadata)

            
            self.total_loss.append(loss_value)

            if self.verbose:
                print(f' Epoch {epoch}: loss {loss_value}')
        
            if eval_bool and splitting_train_test:
                self.evaluation.evaluation(self.net, test.users_id, test.items_id, metadata = test.metadata_id)
                self.evaluation.show()
                
    
    def predict(self,user, items=None):
        self.net.train(False)
        
        user = np.atleast_2d(user)
        user = gpu(torch.from_numpy(user.astype(np.int64).reshape(1, -1)), self.use_cuda)
        
        if items is None:
            items = gpu(torch.arange(0,self.n_items).reshape(-1,1), self.use_cuda)

        if self.use_metadata:
            metadata = gpu(self.mapping_item_metadata[items,:].reshape(-1,len(self.get_n_metadata)), self.use_cuda)
        
        out = self.net(user, items, metadata=metadata)

        return cpu(out).detach().numpy().flatten()
    
    def history(self):
        
        return {'train_loss': self.total_loss,
                'train_auc': self.total_train_auc,
                'test_auc': self.total_test_auc}
    
    def get_item_representation(self):
        
        return self.item.weight.cpu().detach().numpy()