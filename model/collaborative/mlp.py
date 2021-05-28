# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
from model.embeddings.init_embeddings import ScaledEmbedding
from model.helper.cuda import gpu
import pandas as pd
import numpy as np


class MLP(torch.nn.Module):

    ### UNDER COSTRUCTION
    
    def __init__(self, n_users, n_items, n_metadata, n_factors, n_linear_neurons=[1024,126,64], activation_functions=['relu','relu','relu'],  use_metadata=True, use_cuda=False):
        super(MLP, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        
        self.n_factors = n_factors
        self.n_linear_layers = len(n_linear_neurons)
        
        self.use_metadata = use_metadata
        self.use_cuda = use_cuda

        self.activation_func = [self.get_activation_function(i) for i in activation_functions]
        
        self.user_emb = gpu(ScaledEmbedding(self.n_users, self.n_factors), self.use_cuda)
        self.item_emb = gpu(ScaledEmbedding(self.n_items, self.n_factors), self.use_cuda)

        if use_metadata:    
            self.n_metadata = n_metadata
            self.metadata_emb = torch.nn.ModuleList([gpu(ScaledEmbedding(i, self.n_factors), self.use_cuda) for i in self.n_metadata])
            
        
        n_linear_neurons = [n_factors*(2+len(self.n_metadata))] + n_linear_neurons + [1]
        linear_layer = []
       
        for layer in range(len(n_linear_neurons)-1):
            linear_layer.append(gpu(torch.nn.Linear(int(n_linear_neurons[layer]), int(n_linear_neurons[layer+1])), self.use_cuda))

        self.linear_layers = torch.nn.ModuleList(linear_layer)
    
    def get_activation_function(self,activation_func):
        return eval('torch.nn.functional.'+activation_func)
    
    
    def forward(self, users, items, metadata = None):
        
        """
        """

        user = Variable(gpu(users, self.use_cuda))
        item = Variable(gpu(items, self.use_cuda))

        user_emb = self.user_emb(user).squeeze(1)
        item_emb = self.item_emb(item).squeeze(1)
        
        if self.use_metadata:
            metadata_emb = []
            metadata = Variable(gpu(metadata, self.use_cuda))

            for i,e in enumerate(self.metadata_emb):
                metadata_out = e(metadata[:,i].view(-1,1)).squeeze(1)
                metadata_emb.append(metadata_out)
            
            metadata_emb = torch.cat(metadata_emb,1)
            out = torch.cat([user_emb, item_emb, metadata_emb], 1)

        else:
            out = torch.cat([user_emb, item_emb], 1)
        
        for i,l in enumerate(self.linear_layers):
             out = l(out)

             if i<self.n_linear_layers:
                out = self.activation_func[i](out)
        
        return out

    
    def get_item_representation(self):
        
        if self.use_metadata:
            
            data = (self.dataset
                    .dataset[['item_id'] + self.dataset.metadata_id]
                    .drop_duplicates())
            
            mapping = pd.get_dummies(data, columns=[*self.dataset.metadata_id]).values[:, 1:]
            identity = np.identity(self.dataset.dataset['item_id'].max() + 1)
            binary = np.hstack([identity, mapping])
            
            metadata_representation = np.vstack([self.item.weight.detach().numpy(), self.metadata.weight.detach().numpy()])
            
            return np.dot(binary, metadata_representation), binary, metadata_representation
        
        else:
            return self.item.weight.cpu().detach().numpy()