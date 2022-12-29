import pandas as pd
import numpy as np
import pickle
import json
import itertools
import random

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

def create_triplets(df, pair_lengths = 5000, triplet_lengths = 20000):
  groups = df.groupby('label')

  pair_dfs = []

  for name, group in groups:
    group_df = group.reset_index(drop=True)
    combinations = list(itertools.combinations(list(group_df.index), 2))
    combinations = random.choices(combinations, k=pair_lengths)
    print(len(combinations))
    complement = groups.get_group(1-name).reset_index(drop=True)
    triplets = list(itertools.product(combinations, list(complement.index)))
    triplets = random.choices(triplets, k=triplet_lengths)
    pair_df = pd.DataFrame({'triplet_ids':triplets})
    pair_df['ref'] = pair_df['triplet_ids'].apply(lambda x: group_df.loc[x[0][0]]['data'])
    pair_df['pos'] = pair_df['triplet_ids'].apply(lambda x: group_df.loc[x[0][1]]['data'])
    pair_df['neg'] = pair_df['triplet_ids'].apply(lambda x: complement.loc[x[1]]['data'])
    pair_df['ref_class'] = pair_df['triplet_ids'].apply(lambda x: name)
    pair_dfs.append(pair_df)

  return pd.concat(pair_dfs).reset_index(drop=True)


train_df = create_triplets(make_dataframes('train'), pair_lengths = 10000, triplet_lengths = 40000)
train_df.to_pickle('../data/triplet_train.pkl')

test_df = create_triplets(make_dataframes('test'), pair_lengths = 5000, triplet_lengths = 20000)
test_df.to_pickle('../data/triplet_test.pkl')
