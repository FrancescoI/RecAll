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


recall.fit(batch_size=128,
           epochs=20,
           splitting_train_test=True,
           eval_bool=True)

print('--- TRIAL ---\n')

for i in range(1,10):
    user = dataset.encoder.users_encoder.inverse_transform([i])
    items = dataset.encoder.items_encoder.inverse_transform(np.argsort(-recall.predict(i))[:5])

    print(f'user_id : {user[0]} - sugg {items}')