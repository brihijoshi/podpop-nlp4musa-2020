#!/usr/bin/env python
# coding: utf-8




import pandas as pd
import numpy as np
import pickle
import json
import itertools
import random

from sklearn.feature_extraction.text import TfidfVectorizer

import spacy

from EstimatorSelectionHelper import EstimatorSelectionHelper
from classifier_setup import *





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





train_df = make_dataframes("train")
train_df = train_df.sample(frac=1)



test_df = make_dataframes("test")
test_df = test_df.sample(frac=1)





nlp = spacy.load("en_core_web_lg")





def get_wv_embed(doc):
    """get avg wv embedding of a doc"""
    doc = nlp(doc)
    
    avg_embed = list()
    
    for tok in doc:
        avg_embed.append(tok.vector)
    
    avg_embed = np.array(avg_embed)
    return np.mean(avg_embed, axis=0)





train_df["embed"] = train_df["data"].apply(get_wv_embed)




test_df["embed"] = test_df["data"].apply(get_wv_embed)





train_embeds = train_df["embed"].tolist()
train_fm = np.zeros((len(train_embeds), len(train_embeds[0])))
for i in range(len(train_embeds)):
    train_fm[i] = train_embeds[i]





helper = EstimatorSelectionHelper(models, params)





helper.fit(train_fm,
            train_df['label'],
            cv = 5,
            scoring=make_scorer(custom_scorer, greater_is_better=True), n_jobs=16, refit=True)





helper.save_models('../models/','w2vmean_baseline')





test_mat = tfidf_vectorizer.transform(test_df["data"].tolist())





helper.summary(test_mat, test_df["label"])

