#!/usr/bin/env python
# coding: utf-8




import pandas as pd
import numpy as np
import pickle
import json
import itertools
import random

from sklearn.feature_extraction.text import TfidfVectorizer


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





def tokenize(text):
    return text.split()

tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize)
tfidf_mat = tfidf_vectorizer.fit_transform(train_df["data"].tolist())




tfidf_mat = tfidf_mat.toarray()





helper = EstimatorSelectionHelper(models, params)





helper.fit(tfidf_mat,
            train_df['label'],
            cv = 5,
            scoring=make_scorer(custom_scorer, greater_is_better=True), n_jobs=16, refit=True)





helper.save_models('../models/','tfidf_baseline')





test_mat = tfidf_vectorizer.transform(test_df["data"].tolist())





helper.summary(test_mat, test_df["label"])

