# -*- coding: utf-8 -*-

from model.dataset.dataset import EncoderDataset
from model.dataset.dataset import Dataset
from model.dataset.dataset import CustomDataLoader
import torch
import numpy as np
import pytest

users = np.array(['u1', 'u2', 'u3'])
items = np.array(['i1', 'i2', 'i3'])
metadata = np.array([['g1', 'c1'], ['g1', 'c2'], ['g2', 'c1']])
metadata_name = np.array(['gender', 'category'])


class TestEncoderDataset:
    
    def test_init_encoding_returns_numpy(self):
        
        le = EncoderDataset()
        
        (user_ids, 
         item_ids, 
         metadata_ids) = (le
                          .encode_initialize(users=users,
                                             items=items,
                                             metadata=metadata,
                                             metadata_name=metadata_name)
                          )
                          
        assert isinstance(user_ids, np.ndarray), 'user_ids is not a numpy array'
        assert isinstance(item_ids, np.ndarray), 'item_ids is not a numpy array'
        assert isinstance(metadata_ids, np.ndarray), 'metadata_ids is not a numpy array'                                    
        
        
    def test_init_encoding_returns_correct_lengths(self):
        
        le = EncoderDataset()
        
        (user_ids, 
         item_ids, 
         metadata_ids) = (le
                          .encode_initialize(users=users,
                                             items=items,
                                             metadata=metadata,
                                             metadata_name=metadata_name)
                          )
                          
        assert len(user_ids) == len(users), 'user_ids encoding of wronged length'
        assert len(item_ids) == len(items), 'item_ids encoding of wronged length'
        assert len(metadata_ids) == len(metadata), 'metadata_ids encoding of wronged length'
        assert metadata_ids.shape[1] == len(metadata_name), 'metadata_ids encoding of wronged shape on axis=1'
        
    
    def test_init_encoding_returns_none_if_no_metadata(self):
        
        le = EncoderDataset()
        
        (user_ids, 
         item_ids, 
         metadata_ids) = (le
                          .encode_initialize(users=users,
                                             items=items)
                          )
                          
        assert metadata_ids is None, 'metadata_ids should be None'
        

class TestDataset:
    
    def test_numpy_are_converted_to_torch(self):
        
        dataset = Dataset(users=users, 
                          items=items, 
                          metadata=metadata, 
                          metadata_name=metadata_name)
        
        assert isinstance(dataset.users_id, torch.Tensor), 'users_id is not a torch tensor'
        assert isinstance(dataset.items_id, torch.Tensor), 'items_id is not a torch tensor'
        assert isinstance(dataset.metadata_id, torch.Tensor), 'metadata_id is not a torch tensor'
     
        
    def test_istance_has_no_metadata_attribute_if_not_provided(self):
        
        dataset = Dataset(users=users, 
                          items=items)
        
        assert not hasattr(dataset, 'metadata'), 'istance of class Dataset has metadata attribute when not init'
        
    
    def test_item_metadata_mapping_shape_is_correct(self):
        
        dataset = Dataset(users=users, 
                          items=items, 
                          metadata=metadata, 
                          metadata_name=metadata_name)
                
        (unique_items_ids, 
         indices_items) = np.unique(dataset.items_id.numpy().reshape(-1,), return_index=True)
        
        assert (unique_items_ids == dataset.items_id[indices_items].numpy().reshape(-1,)).all() , 'error in dictionary metadata'
        assert dataset.get_item_metadata_mapping.shape == (3,2), 'shape of dictionary metadata is wrong'

    
    def test_len_dataset_is_correct(self):
        
        dataset = Dataset(users=users, 
                          items=items, 
                          metadata=metadata, 
                          metadata_name=metadata_name)
        
        assert dataset.__len__() == 3, 'Lenghts of input and output dont match'
        
    
    def test_getitem_is_working_properly(self):
        
        dataset = Dataset(users=users, 
                          items=items, 
                          metadata=metadata, 
                          metadata_name=metadata_name)
        
        assert dataset.__getitem__(1)[0] == torch.Tensor([1]), 'user tensor output has wronged values'
        assert dataset.__getitem__(1)[1] == torch.Tensor([1]), 'item tensor output has wronged values'
        assert (dataset.__getitem__(1)[2] == torch.Tensor([0, 1])).all(), 'metadata tensor output has wronged values'
            
        
class TestCustomDataLoader:
    
    def test_custom_dataloader_has_dataset_attribute(self):
        
        dataset = Dataset(users=users, 
                          items=items, 
                          metadata=metadata, 
                          metadata_name=metadata_name)
        
        loader = CustomDataLoader(dataset=dataset, batch_size=1)
        
        assert hasattr(loader, 'data'), 'custom data loader has not inherit dataset attributes'