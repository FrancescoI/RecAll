import torch
import numpy as np
import random


def split_train_test(dataset, test_percentage=0.25, random_state=None):

    random.seed(random_state)

    idx_sample = random.sample(range(dataset.n_interactions), dataset.n_interactions-1)
    
    thresh = int((1.0 - test_percentage) * dataset.n_interactions)

    train_idx = idx_sample[:thresh]
    test_idx = idx_sample[thresh:]

    print(f'Shape Train: {len(train_idx)} \nShape Test: {len(test_idx)}')

    train = dataset(dataset.users[train_idx],
                    dataset.items[train_idx],
                    weights = dataset.weights[train_idx] if hasattr(dataset,'weights') else None,
                    metadata = dataset.metadata[train_idx] if hasattr(dataset,'metadata') else None,
                    metadata_name = dataset.metadata_name if hasattr(dataset,'metadata_name') else None)

    test = dataset(dataset.users[test_idx],
                    dataset.items[test_idx],
                    weights = dataset.weights[test_idx] if hasattr(dataset,'weights') else None,
                    metadata = dataset.metadata[test_idx] if hasattr(dataset,'metadata') else None,
                    metadata_name = dataset.metadata_name if hasattr(dataset,'metadata_name') else None)
    
    return train,test