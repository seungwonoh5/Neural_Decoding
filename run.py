"""
run.py
"""
from neural_decoding.utils import plot_performance
from neural_decoding.data import NeuralData
from neural_decoding.models import Batch_Model, Online_Model

# select datasets used for training and testing
subject = '1004'
train_datasets_info = [('3','1'), ('3','2')]
test_datasets_info = [('4','1'), ('4','2'), ('4','3'), ('5','3')]

task = 2
time_step = 10
path = '/content/drive/MyDrive/github/Online_Learning_Neural_Decoding/data'

h_params = [{'n_layers':2, 'units':10, 'lr':1e-4, 'drop_rate':0.1, 'opt':'adam', 'w_decay':0.05, 'epochs':20, 'batch':32, 'p':5, 'update_epoch':1}]


# list of dicts
train_datasets = []
test_datasets = []

# load dataset and preprocess according to the model you want to use
for day, inj in train_datasets_info:
    train_data = NeuralData(path, subject, day, inj, task)
    print(train_data)
    dataset = train_data.preprocess(time_step, True)
    train_datasets.append(dataset)

for day, inj in test_datasets_info:
    test_data = NeuralData(path, subject, day, inj, task) 
    print(test_data)
    dataset = test_data.preprocess(time_step, True)
    test_datasets.append(dataset)

print(len(train_datasets))
print(len(test_datasets))

for params in h_params:
    batch_model = Batch_Model(model='lstm', time_steps=time_step, num_features = train_datasets[0]['input'].shape[-1], units=params['units'], lr_rate=params['lr'], drop_rate=params['drop_rate'], opt=params['opt'])
    online_model = Online_Model(model='lstm', time_steps=time_step, num_features = train_datasets[0]['input'].shape[-1], units=params['units'], lr_rate=params['lr'], drop_rate=params['drop_rate'], opt=params['opt'])
    
    batch_model.run(params, train_datasets, test_datasets, False)
    online_model.run(params, train_datasets, test_datasets, False)
    
    plot_performance(batch_model.acc, online_model.acc)