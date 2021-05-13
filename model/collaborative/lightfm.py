# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
from model.embeddings.init_embeddings import ScaledEmbedding, ZeroEmbedding
from model.helper.cuda import gpu
import pandas as pd
import numpy as np

class LightFM(torch.nn.Module):
    
    def __init__(self, n_users, n_items, n_factors, n_metadata = None, use_metadata=True, use_cuda=False):
        super(LightFM, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        
        self.n_factors = n_factors
        
        self.use_metadata = use_metadata
        self.use_cuda = use_cuda
        
        if use_metadata:
            self.n_metadata = n_metadata
            self.metadata_emb = torch.nn.ModuleList([gpu(ScaledEmbedding(i, self.n_factors), self.use_cuda) for i in self.n_metadata])
        
        
        self.user_emb = gpu(ScaledEmbedding(self.n_users, self.n_factors), self.use_cuda)
        self.item_emb = gpu(ScaledEmbedding(self.n_items, self.n_factors), self.use_cuda)
        
        self.user_bias_emb = gpu(ZeroEmbedding(self.n_users, 1), self.use_cuda)
        self.item_bias_emb = gpu(ZeroEmbedding(self.n_items, 1), self.use_cuda)
    
    
    def forward(self, users, items, metadata = None):
        
        """
        Forward method that express the model as the dot product of user and item embeddings, plus the biases. 
        Item Embeddings itself is the sum of the embeddings of the item ID and its metadata
        """
        
        user = Variable(gpu(users, self.use_cuda))
        item = Variable(gpu(items, self.use_cuda))

        user_bias_emb = self.user_bias_emb(user).view(-1,1)
        item_bias_emb = self.item_bias_emb(item).view(-1,1)
        
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        
        
        if self.use_metadata:
            metadata_emb = []
            metadata = Variable(gpu(metadata, self.use_cuda))
            for i,e in enumerate(self.metadata_emb):
                metadata_out = e(metadata[:,i].view(-1,1))
                metadata_emb.append(metadata_out)

            ### concatenate metadata emb    
            metadata_emb = torch.cat(metadata_emb,1)
    
            ### Reshaping in order to match metadata tensor
            item_emb = [metadata_emb, item_emb]
            item_emb = torch.cat(item_emb,1)
            ### sum of latent dimensions
            item_emb = item_emb.sum(1).unsqueeze(1)
        
        out = (user_emb * item_emb).sum(2).view(-1,1) + user_bias_emb + item_bias_emb
        
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
        
        
    # def predict(self, users):
        
    #     """
    #     It takes a user vector representation (based on user_idx arg) and it takes the dot product with
    #     the item representation
    #     """
        
    #     item_repr, _, _ = self.get_item_representation()
    #     user_repr = self.user.weight.detach().numpy()
        
    #     item_bias = self.item_bias.weight.detach().numpy()
    #     user_bias = self.user_bias[torch.tensor([user_idx])].detach().numpy()
        
    #     return np.dot(user_pred[user_idx, :], item_repr) + item_bias + user_bias