# -*- coding: utf-8 -*-

from model.recall import *
from model.dataset.dataset import *

missoni = MyDataset(brand='missoni')

missoni.fit(metadata=['saleLine', 'macro'])

recall = Recall(dataset=missoni, 
                n_factors=80, 
                net_type='lightfm', 
                use_metadata=True, 
                use_cuda=False)

optimizer = torch.optim.Adam(recall.net.parameters(),
                             lr=3e-3)

recall.fit(optimizer=optimizer,
           batch_size=10_240,
           epochs=20,
           split_train_test=True,
           verbose=True)