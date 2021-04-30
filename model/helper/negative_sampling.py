# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch

def get_negative_batch(users, n_items, mapping_item_metadata, use_metadata = False):
    
    neg_batch = None

    neg_item_id = torch.randint(0, n_items-1, (len(users),1))
    
    if use_metadata:
        neg_metadata_id = mapping_item_metadata[neg_item_id]   
    else:
        neg_metadata_id = None
    
            
    return neg_item_id, neg_metadata_id