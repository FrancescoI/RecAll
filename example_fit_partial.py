# -*- coding: utf-8 -*-

from model.recall import *
from model.dataset.dataset import *

missoni = pd.read_csv('missoni.csv').sample(10000)


dataset = Dataset(missoni['hashedEmail'].values,
                    missoni['product'].values,
                    metadata=missoni[['saleLine', 'gender','macro']].values,
                    metadata_name=['saleline','gender','macro']
                    )

## splitting train, test
train, test = split_train_test(dataset, test_percentage=0.25, random_state=1234)

recall = Recall(dataset, 
                80,
                net_type = 'hybrid_cf', 
                use_metadata = True,
                use_cuda=False, 
                verbose = True)

## custom dataloader
train_loader = CustomDataLoader(dataset=train, batch_size=128, shuffle = True)

#initialize evaluation
evaluation = EvaluateRec_all(mapping_item_metadata=recall.mapping_item_metadata, kind='precision_recall', k=10)

for epoch in range(1,21):
    for i, (users, items, metadata, weights) in enumerate(train_loader):

        loss = recall.fit_partial(users, items, metadata=metadata, verbose=False)

    print(f'epoch number : {epoch} -- loss: {loss}\n')

    #evaluation
    evaluation.evaluation(recall, test.users_id, test.items_id, metadata = test.metadata_id)
    evaluation.show()



print('\n--- TRIAL ---\n')

for i in range(1,10):
    user = dataset.encoder.users_encoder.inverse_transform([i])
    items = dataset.encoder.items_encoder.inverse_transform(np.argsort(-recall.predict(i))[:5])

    print(f'user_id : {user[0]} - sugg {items}')