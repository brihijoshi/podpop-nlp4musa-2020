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
from livelossplot.outputs import MatplotlibPlot
import gc



class TripletDataset(Dataset):
    
    def get_tokens(self, elem):
        return self.tokenizer.encode(elem.replace("\n", " "), max_length=512, truncation=True, padding='max_length')
    
    def get_embeds(self, row):
        with torch.no_grad():
            self.model.to(device)
            ref = torch.LongTensor(self.get_tokens(row['ref'])).unsqueeze(0).to(device)
            pos = torch.LongTensor(self.get_tokens(row['pos'])).unsqueeze(0).to(device)
            neg = torch.LongTensor(self.get_tokens(row['neg'])).unsqueeze(0).to(device)
            
            out1 = self.model(ref)[0][:,0]
            out2 = self.model(pos)[0][:,0]
            out3 = self.model(neg)[0][:,0]
            
            ref = ref.cpu()
            pos = pos.cpu()
            neg = neg.cpu()
            
            out1 = out1.cpu()
            out2 = out2.cpu()
            out3 = out3.cpu()
            
            self.counter += 1
            
            return (out1,out2,out3)
            
            
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






TIME_STAMP = time.strftime("%d%m%Y-%H%M%S")
BATCH_SIZE = 512




train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
val_loader = torch.utils.data.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)





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





class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()





train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
val_loader = torch.utils.data.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)



model = PodcastClassifier()


criterion = TripletLoss(MARGIN)


optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)



is_better = True
prev_loss = float('inf')

liveloss = PlotLosses(outputs=[MatplotlibPlot(figpath='../figures/plot.png')])





dataloaders = {'train':train_loader, 'validation':val_loader}





for epoch in range(NUM_EPOCHS):
    logs = {}
    t_start = time.time()
    

    
    for phase in ['train', 'validation']:
        if phase == 'train':
            model.train()
            
        else:
            model.eval()
        model.to(device)
        
        print("Started Phase")

        running_loss = 0.0
        
        if phase == 'validation':
            
            with torch.no_grad():
                for (i,batch) in tqdm(enumerate(dataloaders[phase])):
                  
                    ref = batch[0].to(device)
                    pos = batch[1].to(device)
                    neg = batch[2].to(device)

                    ref_emb = model(ref)
                    pos_emb = model(pos)
                    neg_emb = model(neg)

                    loss = criterion(ref_emb, pos_emb, neg_emb)

                    ref = ref.cpu()
                    pos = pos.cpu()
                    neg = neg.cpu()

                    running_loss += loss.detach() * ref.size(0)
                    
        else:
            
            for (i,batch) in tqdm(enumerate(dataloaders[phase])):
                ref = batch[0].to(device)
                pos = batch[1].to(device)
                neg = batch[2].to(device)
                
                ref_emb = model(ref)
                pos_emb = model(pos)
                neg_emb = model(neg)

                loss = criterion(ref_emb, pos_emb, neg_emb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ref = ref.cpu()
                pos = pos.cpu()
                neg = neg.cpu()

                running_loss += loss.detach() * ref.size(0)
    

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        
        model.to('cpu')

        prefix = ''
        if phase == 'validation':
            prefix = 'val_'

        logs[prefix + 'log loss'] = epoch_loss.item()

        
        print('Phase time - ',time.time() - t_start)

    delta = time.time() - t_start
    is_better = logs['val_log loss'] < prev_loss
    if is_better:
        prev_loss = logs['val_log loss']
        torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(), 'loss': logs['log loss']}, "../models/"+NAME+"_"+TIME_STAMP+"_"+str(logs['val_log loss'])+".pth")

    liveloss.update(logs)
    liveloss.send()

