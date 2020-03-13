import os
import pickle

import spacy
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertTokenizer

import pandas as pd
import pdb
import numpy as np

key_vals = {}   #  key -> set of values

def make_dataloader(data, batch_size, sampler=None):
    data = TensorDataset(*data)
    data_loader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return data_loader

def add_val(key_name, val):
    if key_name not in key_vals:
        key_vals[key_name] = {}
    if val not in key_vals[key_name]:
        key_vals[key_name][val] = len(key_vals[key_name])

def date_to_year(df, split_ind):
#    print(df['date'][split_ind][0:100])
    return [d.split('-')[0] for d in df['date'][split_ind]]

def load_and_cache_data(
        raw_data_file='data/all_gs_comments_by_school.csv',
        prepared_data_file='data/all_mediantest.p',
#        raw_data_file='data/tiny.csv',
#        prepared_data_file='data/tiny_mediantest.p',
        max_len=512,
        train_frac = 0.8
):

    if os.path.isfile(prepared_data_file):
#    if False:
        with open(prepared_data_file, 'rb') as f:
            input_ids, labels_test_score, attention_masks = pickle.load(f)
        print('data loaded from cache!')
    else:

        print('Loading data ...')

        df = pd.read_csv(raw_data_file).dropna(subset=['mn_avg_eb', 'review_text']).reset_index()

        all_ind = list(range(0, len(df)))
        np.random.shuffle(all_ind)

        train_ind = all_ind[0:int(train_frac*len(all_ind))]
        val_ind = all_ind[int(train_frac*len(all_ind)):]

        median_test_score = float(df['mn_avg_eb'][train_ind].median())
        median_toplevel_score = float(df['top_level'][train_ind].median())
        print("Median test score", median_test_score)
        print("Median toplevel score", median_toplevel_score)
    
        data = {'train': list(df['review_text'][train_ind]), 'validation': list(df['review_text'][val_ind])}
        labels_test_score = {'train': [int(x >= median_toplevel_score) for x in list(df['top_level'][train_ind])],
                             'validation': [int(x >= median_toplevel_score) for x in list(df['top_level'][val_ind])]}

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        input_ids = {}   # split -> list of list of ids
        attention_masks = {}  # split -> list of attention masks
        for d in data:
            input_ids[d] = []
            attention_masks[d] = []
            for review in data[d]:
                token_ids = tokenizer.encode(review.decode('utf-8'), max_length=max_len)
                if len(token_ids) < max_len:
                    token_ids += [0] * (max_len - len(token_ids))
                attention_mask = [float(id>0) for id in token_ids]
                input_ids[d].append(token_ids)
                attention_masks[d].append(attention_mask)

        with open(prepared_data_file, 'wb') as f:
            pickle.dump((input_ids, labels_test_score, attention_masks), f)
            print('Data written to disk')

    # tensorize
    for dataset in [labels_test_score, input_ids, attention_masks]:
        for d in dataset:
            dataset[d] = torch.tensor(dataset[d])

    return input_ids, labels_test_score, attention_masks
