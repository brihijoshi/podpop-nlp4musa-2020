import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import functools
import json
import random
import pickle
import time
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import numpy as np
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from tqdm.notebook import tqdm
from livelossplot.outputs import MatplotlibPlot
import gc


class PodcastClassifier(nn.Module):
    def __init__(self, in_dim=768):
        super(PodcastClassifier, self).__init__()
        self.in_dim=in_dim
        self.block1 = nn.Sequential(nn.Linear(768, 512),
                                    nn.ReLU(),
                                    nn.Dropout(0.5))
        self.block2 = nn.Sequential(nn.Linear(512, 256),
                                   nn.ReLU())

    def forward(self, inp):
        inp = self.block1(inp)
        inp = self.block2(inp)

        return inp
    
    
cuda = 1
device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")


checkpoint = torch.load('../models/ckpt.pt')

model = PodcastClassifier()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)


def make_dataframes(split, path = '../data/'):
    fl = open(path+'popularity_'+split+'.txt','r')
    labels = []
    data = []

    count = 0
    for line in fl:
        line = json.loads(line)
        labels.append(line['label'])
        try:
            podcast = 'e_'+line['id']+'.txt'
            with open(path+'processed/'+podcast,'r') as f:
                content = f.read()
                data.append(content)
        except:
            print('e_'+line['id']+'.txt')  
    df = pd.DataFrame({'data':data, 'label':labels})

    return df

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def get_vectors(row):
    with torch.no_grad():
        data = row['data']
        tokened = tokenizer.encode(data.replace("\n", " "),\
                                         max_length=512, truncation=True, padding='max_length')
        vec = torch.LongTensor(tokened).unsqueeze(0).to(device)
        out = bert(vec)[0][:,0]

        res = model(out)

        vec = vec.cpu()
        out = out.cpu()
        res = res.cpu()
        

        
        return res.numpy()[0]
    
train_set['vecs'] = train_set.apply(lambda x: get_vectors(x), axis=1)
test_set['vecs'] = test_set.apply(lambda x: get_vectors(x), axis=1)


from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


train_set = train_set.sample(frac=1)
test_set = test_set.sample(frac=1)


X_train, X_test = train_test_split(
    test_set, test_size=0.33, random_state=42, stratify=test_set['label'])


neigh = KNeighborsClassifier(n_neighbors=3) # you can use any supervised classifier of your choice. This demo code is with kNN.
    
neigh.fit(np.array(list(X_train['vecs'])), X_train['label'])

print(classification_report(X_test['label'], neigh.predict(np.array(list(X_test['vecs'])))))