# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

class DatasetLoader():

    def __init__(self,kind,
                 users_col,
                 items_col,
                 weights_col = None,
                 metadata_cols = None,
                 path=None,
                 format='csv',
                 header='infer',
                 delimiter=','):

        self.kind = kind
        self.path = path
        self.format = format
        self.header = header
        self.users = users_col
        self.items = items_col
        self.weights = weights_col
        self.metadata = metadata_cols

        self.__initialize()

    def __initialize(self):

        #test
        if self.path:
            self.custom()

        elif self.kind == 'movie_lens':
            self.upload_movie_lens()
        
        else:
            raise ValueError('set an existing path or use an example dataset!')

    def upload_movie_lens(self):
        pass
    
    def custom(self):
        df = pd.read_csv(self.path,header=self.header,delimiter=self.delimiter)

        user = df[self.users].values
        item = df[self.items].values
        weights = df[self.weights].values
        metadata = df[self.metadata].values

        self.dataset = Dataset(user,item,weights=weights,metadata=metadata_ids)




        
class Dataset(object):
    
    def __init__(self, users, items, weights=None, metadata=None, metadata_name=None,encoder = None):

        self.users = users
        self.items = items
        self.n_interactions = users.shape[0]

        if weights is not None:
            self.weights_numpy = weights
        
        if metadata is not None:
            self.metadata = metadata
            self.num_metadata = self.metadata.shape[1]
            self.metadata_name = metadata_name
        
        if encoder is None:
            self.encoder = EncoderDataset()
            self.users_id_numpy, self.items_id_numpy, self.metadata_id_numpy = self.encoder.encode_initialize(users,items,metadata,metadata_name)
        else:
            self.encoder = encoder
            self.users_id_numpy, self.items_id_numpy, self.metadata_id_numpy = self.encoder.encode(users,items,metadata,metadata_name)

        self._converting_torch_attr()


    def _converting_torch_attr(self):
        self.users_id = torch.from_numpy(self.users_id_numpy).view(-1,1)
        self.items_id = torch.from_numpy(self.items_id_numpy).view(-1,1)
        
        if hasattr(self, 'weights_numpy'):
            self.weights = torch.from_numpy(self.weights_numpy).view(-1,1)

        if hasattr(self, 'metadata'):
            self.metadata_id  = torch.from_numpy(self.metadata_id_numpy.astype(np.int64))
    
    @property
    def get_item_metadata_mapping(self):
        ## the index of tensor represent item id ---> value is its metadata

        unique_items_ids, indices_items = np.unique(self.items_id.numpy().reshape(-1,),
                                            return_index=True)

        related_metadata_id = torch.zeros((self.items_id.max()+1,self.metadata_id.shape[1])).type(torch.int64)

        related_metadata_id[unique_items_ids] = self.metadata_id[indices_items,:]

        assert (unique_items_ids == self.items_id[indices_items].numpy().reshape(-1,)).all() , 'error in dictionary metadata'
        
        return related_metadata_id

    def __len__(self):
        return self.users.shape[0]
    

    def __getitem__(self, index):
        
        if hasattr(self, 'metadata_id') and  hasattr(self, 'weights_id'):
            return self.users_id[index], self.items_id[index], self.metadata_id[index], self.weights[index]

        elif hasattr(self, 'metadata_id') and ~hasattr(self, 'weights_id'):
            # assert (self.get_item_metadata_mapping[int(self.items_id[index])] == self.metadata_id[index]).all()
            return self.users_id[index], self.items_id[index], self.metadata_id[index], np.array([])
        
        elif ~hasattr(self, 'metadata_id') and hasattr(self, 'weights_id'):
            return self.users_id[index], self.items_id[index], np.array([]), self.weights[index]
        
        else:
            return self.users_id[index], self.items_id[index], np.array([]), np.array([])
        
        
class CustomDataLoader(DataLoader):
    
    def __init__(self, *args, **kwargs):
        super(CustomDataLoader, self).__init__(*args, **kwargs)
        
        self.data = kwargs['dataset']

class EncoderDataset(object):

    def __init__(self, name=None):
        self.name = name

    def _encondig_label(self,inpt):
        encoder = LabelEncoder()
        opt_encoder = encoder.fit_transform(inpt)
        
        return opt_encoder, encoder
 
    def encode_initialize(self,users,items,metadata=None,metadata_name=None):
        users_id_numpy, self.users_encoder = self._encondig_label(users)
        items_id_numpy, self.items_encoder = self._encondig_label(items)

        if metadata is not None:
            metadata_id_numpy = np.zeros_like(metadata)
            self.metadata_encoder = dict()
            
            for i in range(metadata.shape[1]):
                metadata_id_numpy[:,i], self.metadata_encoder[metadata_name[i]] = self._encondig_label(metadata[:,i])
        
        else:
            metadata_id_numpy = None

        return users_id_numpy, items_id_numpy, metadata_id_numpy
    
    def encode(self, user, items, metadata=None, metadata_name=None):
        users_id_numpy = self.users_encoder.transform(user)
        items_id_numpy = self.items_encoder.transform(items)

        if hasattr(self, 'metadata_encoder'):
            metadata_id_numpy = np.zeros_like(metadata)
            
            for i in range(metadata.shape[1]):
                metadata_id_numpy[:,i] = self.metadata_encoder[metadata_name[i]].transform(metadata[:,i])
        
        else:
            metadata_id_numpy = None
        
        return users_id_numpy, items_id_numpy, metadata_id_numpy

