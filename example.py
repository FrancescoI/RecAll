# -*- coding: utf-8 -*-

from model.recall import *
from model.dataset.dataset import *

missoni = pd.read_csv('missoni.csv').sample(n = 1000)


dataset = Dataset(missoni['hashedEmail'].values,
                    missoni['product'].values,
                    metadata=missoni[['saleLine', 'gender','macro']].values,
                    metadata_name=['saleline','gender','macro']
                    )

recall = Recall(dataset, 
                80,
                net_type = 'hybrid_cf', 
                use_metadata = True,
                use_cuda=False, 
                verbose = True)

optimizer = torch.optim.Adam(recall.net.parameters(),
                             lr=3e-3)

recall.fit(optimizer=optimizer,
           batch_size=128,
           epochs=20,
           splitting_train_test=True,
           eval_bool=False,
           kind_eval='precision_recall')

evaluation = EvaluateRec_all(mapping_item_metadata=recall.mapping_item_metadata, k=3, kind='precision_recall')
evaluation.evaluation(recall, dataset.users_id, dataset.items_id, metadata = dataset.metadata_id)
evaluation.show()


recall.predict(1)

# from model.dataset.dataset import *
# import pandas as pd
# import numpy as np

# df = pd.DataFrame(data=np.arange(0,50).reshape(10,5),columns=['A','B','C', 'metadata_1', 'metadata_2'])

# mydataset = Dataset(df['A'].values,
#                     df['B'].values
#                     #weights=df['C'].values,
#                     #metadata=df[['metadata_1', 'metadata_2']].values,
#                     #metadata_name =['metadata_1','metadata_2']
#                     )


# from torch.utils.data import DataLoader
# data_loader = CustomDataLoader(dataset=mydataset, batch_size=1)

# for i, k in enumerate(data_loader):
#     print(k)