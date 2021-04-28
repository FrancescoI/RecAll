# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
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



class MyDataset():
    
    def __init__(self, brand):
        
        self.brand = brand
    
    def _get_interactions(self):

        bucket_uri = f'C:/Users/imbrigliaf/Documents/GitRepo/RecAll/example/data/'

        if self.brand == 'missoni':
        
          dataset = pd.read_csv(bucket_uri + f'{self.brand}.csv')


        elif self.brand == 'ton':
          
          clickstream = pd.read_csv(bucket_uri + f'{self.brand}.csv')
          metadata = pd.read_csv(bucket_uri + 'anagrafica_ton.csv')

          clickstream = (clickstream\
                         .groupby(['user_ids', 'product_code'])['brand'].count() 
                         ).reset_index()
          
          clickstream.columns = ['hashedEmail', 'product', 'actions']

          dataset = pd.merge(clickstream, metadata, left_on='product', right_on='pty_pim_variant')
          
          dataset = dataset[['hashedEmail', 'product', 'macro', 'saleline', 'actions']]
          dataset.columns = ['hashedEmail', 'product', 'macro', 'saleLine', 'actions']

          dataset['gender'] = 'W'

        return dataset 
    
    
    def _encondig_label(self, dataset, input_col, output_col):
        
        encoder = LabelEncoder()
        dataset[output_col] = encoder.fit(dataset[input_col]).transform(dataset[input_col])
        
        return dataset, encoder
    
    
    def fit(self, metadata=None, seasons=None):
        
        dataset = self._get_interactions()
        self.metadata = metadata
        
        if seasons:
            dataset = dataset[dataset['season'].isin(seasons)]
        
        ### Label Encoding
        dataset, _ = self._encondig_label(dataset, input_col='hashedEmail', output_col='user_id')
        dataset, _ = self._encondig_label(dataset, input_col='product', output_col='item_id')
        
        if metadata is not None:
            output_list_name = []
            
            for meta in metadata:
                output_name = meta + '_id'
                dataset, _ = self._encondig_label(dataset, input_col=meta, output_col=output_name)
                output_list_name.append(output_name)                
            
            dataset['metadata'] = dataset[output_list_name].values.tolist()
            self.metadata_id = output_list_name
            
        self.dataset = dataset
        
    def get_item_metadata_dict(self):
        
        if self.metadata is not None:
        
            return self.dataset.set_index('item_id')['metadata'].to_dict()
        
        else:
            
            return None
        
        
class Dataset():
    
    def __init__(self,users,items,weights=None,metadata=None,metadata_name =None):

        self.users = users
        self.items = items
        
        if weights is not None:
            self.weights_numpy = weights
        
        if metadata is not None:
            self.metadata = metadata
            self.num_metadata = self.metadata.shape[1]
            self.metadata_name = metadata_name

        self.encoding_label()

        
    def _encondig_label(self,inpt):
        
        encoder = LabelEncoder()
        opt_encoder = encoder.fit_transform(inpt)
        
        return opt_encoder, encoder
    
    def _converting_torch_attr(self):
        self.users_id = torch.from_numpy(self.users_id_numpy)
        self.items_id = torch.from_numpy(self.items_id_numpy)
        
        if hasattr(self, 'weights_numpy'):
            self.weights = torch.from_numpy(self.weights_numpy)

        if hasattr(self, 'metadata'):
            self.metadata_id  = torch.from_numpy(list(self.metadata_id_numpy))


    def encoding_label(self):
        self.users_id_numpy, self.users_encoder = self._encondig_label(self.users)
        self.items_id_numpy, self.items_encoder = self._encondig_label(self.items)

        if hasattr(self, 'metadata'):
            self.metadata_id_numpy = np.zeros_like(self.metadata)
            self.metadata_encoder = dict()
            
            for i in self.metadata.shape[1]:
                self.metadata_id_numpy[:,i], self.metadata_encoder[self.metadata_name[i]] = self._encondig_label(self.metadata[:,i])

        self._converting_torch_attr()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        
        if hasattr(self, 'metadata_id') and  hasattr(self, 'weights_id'):
            return self.user_id[index : (index+1)], self.item_id[index: (index+1)], self.metadata_id[index: (index+1)], self.weights[index: (index+1)]

        elif hasattr(self, 'metadata_id') and ~hasattr(self, 'weights_id'):
            return self.user_id[index : (index+1)], self.item_id[index: (index+1)], self.metadata_id[index: (index+1)]
        
        elif ~hasattr(self, 'metadata_id') and hasattr(self, 'weights_id'):
            return self.user_id[index : (index+1)], self.item_id[index: (index+1)], self.weights[index: (index+1)]
        
        else:
            return self.user_id[index : (index+1)], self.item_id[index: (index+1)]
        
        
    
    
    
    
    
    
    
    