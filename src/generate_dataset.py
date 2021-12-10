# coding: utf-8





import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import functools
import json
import random
import pickle
from livelossplot import PlotLosses
import time
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import numpy as np
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from tqdm.notebook import tqdm
import gc





def dump_tensors(gpu_only=True):
    torch.cuda.empty_cache()
    total_size = 0
    for obj in gc.get_objects():
        # print(obj)
        try:
            if torch.is_tensor(obj):
                if obj.is_cuda:
                    del obj
                    gc.collect()
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    del obj
                    gc.collect()
        except Exception as e:
            pass


# In[ ]:


class TripletDataset(Dataset):
    
    def get_tokens(self, elem):
        return self.tokenizer.encode(elem.replace("\n", " "), max_length=512, truncation=True, padding='max_length')
    
    def get_embeds(self, row):
        with torch.no_grad():
            self.model.to(device)
            ref = torch.LongTensor(self.get_tokens(row['ref'])).unsqueeze(0).to(device)
            pos = torch.LongTensor(self.get_tokens(row['pos'])).unsqueeze(0).to(device)
            neg = torch.LongTensor(self.get_tokens(row['neg'])).unsqueeze(0).to(device)
            
            out1 = self.model(ref)
            out2 = self.model(pos)
            out3 = self.model(neg)
            
            ref.cpu()
            pos.cpu()
            neg.cpu()
            
            dump_tensors()
            
            self.counter += 1
            
            return (out1[0][:,0],out2[0][:,0],out3[0][:,0])
            
            
    def __init__(self, split, sample_size=20000):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.split = split
        self.counter = 0
        
        self.triplet_df = pd.read_pickle('../data/triplet_'+self.split+'.pkl')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.model.eval()
        self.model.to(device)
        
        self.triplet_df = self.triplet_df.sample(n=sample_size, random_state=0).reset_index(drop=True)


        self.triplet_df['embs'] = self.triplet_df.apply(lambda x: self.get_embeds(x), axis=1).reset_index(drop=True)
        self.model.to('cpu')

    def __len__(self):
        return len(self.triplet_df)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        embs = self.triplet_df.loc[idx]['embs']
        
        ref_ = embs[0]
        pos_ = embs[1]
        neg_ = embs[2]
        
        return (ref_, pos_, neg_)


# In[ ]:


NUM_EPOCHS = 500


LEARNING_RATE = 1e-3

MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
CUDA = True
MARGIN = 1.


cuda = 1
device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
NAME = 'triple_training_1'




train_dataset = TripletDataset(split = "train", sample_size=20000)
with open('../data/triple_dataset_bert_train_20k.pt','wb') as f:
    torch.save(train_dataset, f)




test_dataset = TripletDataset(split = "test", sample_size=10000)
with open('../data/triple_dataset_bert_test_10k.pt','wb') as f:
    torch.save(test_dataset, f)


