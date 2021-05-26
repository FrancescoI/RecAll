# -*- coding: utf-8 -*-

import numpy as np
import torch
from model.helper.negative_sampling import get_negative_batch
from model.helper.cuda import gpu, cpu
import pandas as pd
import scipy.sparse as sp



class EvaluateRec_all(object):

    def __init__(self, mapping_item_metadata=None, k=3, kind='AUC'):

        self.mapping_item_metadata = mapping_item_metadata
        self.kind = kind
        self.history = {'AUC':[],'precision_recall':[]}
        self.k = k
        

    def evaluation(self, net, users, items, metadata = None):

        
        net.eval()

        if self.kind=='AUC':
            score = self.auc(net, users, items, metadata = metadata)
            self.history['AUC'].append(score)
            

        elif self.kind=='precision_recall':
            score = self.precision_recall_k(net, users, items, metadata = metadata)
            self.history['precision_recall'].append(score)
            

        else:
            pass

    def show(self):
        if self.kind == 'AUC':
            print(f'AUC: {np.mean(self.total_auc)} \n')
        
        elif self.kind == 'precision_recall':
            print(f'Precision@{self.k}: {"%.4f" % np.mean(self.total_precision)} \nRecall@{self.k}: {"%.4f" % np.mean(self.total_recall)} \n')
    
    def auc(self, net, users, items, metadata=None):

        with torch.no_grad():

            positive_score = gpu(net(users, 
                                    items, 
                                    metadata = metadata), 
                                net.use_cuda)

            neg_items, neg_metadata = get_negative_batch(users, net.n_items,self.mapping_item_metadata, use_metadata = net.use_metadata)

            negative_score = gpu(net(users, 
                                    neg_items, 
                                    metadata = neg_metadata), 
                        net.use_cuda) 
            
            self.total_auc = self.auc_score(positive_score,negative_score)

            return self.total_auc

    def auc_score(self,positive, negative):
        
        total_auc = []
        
        positive = positive.cpu().detach().numpy()
        negative = negative.cpu().detach().numpy()

        batch_auc = (positive > negative).sum() / len(positive)
        total_auc.append(batch_auc)
            
        return np.mean(total_auc)


    def precision_recall_k(self,  net, users, targets, metadata=None):
        
        self.total_precision = []
        self.total_recall = []

        users_unique = torch.unique(users)
        
        for i, user in enumerate(users_unique):

            target = targets[users==user]
            prediction = np.argsort(-net.predict(user))[:self.k]

            n_matching = len(set(target).intersection(set(prediction)))
            precision = float(n_matching) / self.k
            recall = float(n_matching) / len(target)

            self.total_precision.append(precision)
            self.total_recall.append(recall)

        return self.total_precision, self.total_recall