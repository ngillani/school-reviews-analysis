import os
import pickle

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
#        raw_data_file='data/tiny_seda_ratings.csv', 
#        prepared_data_file='data/tiny_prepared_data_for_bert.p',
        raw_data_file='data/all_gs_and_seda_with_comments.csv',
        prepared_data_file='data/all_gs_and_seda_with_comments.p',
#        raw_data_file='data/tiny_gs_review_ratings.csv',
#        prepared_data_file='data/tiny_gs_review_ratings.p',
        max_len=512,
        train_frac = 0.8
    ):

    if os.path.isfile(prepared_data_file):
        with open(prepared_data_file, 'rb') as f:
            input_ids, year_ids, labels_test, labels_progress, attention_masks = pickle.load(f)

        print('data loaded!')

    else:

        print('Loading data ...')

        df = pd.read_csv(raw_data_file).dropna(subset=['mn_avg_eb', 'review_text', 'date']).reset_index()

        all_ind = list(range(0, len(df)))
        np.random.shuffle(all_ind)

        train_ind = all_ind[0:int(train_frac*len(all_ind))]
        val_ind = all_ind[int(train_frac*len(all_ind)):]

        data = {'train': list(df['review_text'][train_ind]), 'validation': list(df['review_text'][val_ind])}
        years = {'train': date_to_year(df, train_ind), 'validation': date_to_year(df, val_ind)}
        labels_test = {'train': list(df['mn_avg_eb'][train_ind]), 'validation': list(df['mn_avg_eb'][val_ind])}
        labels_progress = {'train': list(df['mn_grd_eb'][train_ind]), 'validation': list(df['mn_grd_eb'][val_ind])}

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenized_texts = {}  # split -> list of list of tokens

        year_ids = {}
        for d in data:
            year_ids[d] = []
            tokenized_texts[d] = []
            for year in years[d]:
                add_val("year", year)
                year_ids[d].append(key_vals["year"][year])
            for sent in data[d]:
                try:
                    tokenized_texts[d].append(tokenizer.tokenize(sent.decode('utf-8')))
                except:
                    pdb.set_trace()

        input_ids = {}   # split -> list of list of token ids
        for d in tokenized_texts:
            ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts[d]]
            for idx in range(len(ids)):
                if len(ids[idx]) > max_len:
                    ids[idx] = ids[idx][:max_len]
                if len(ids[idx]) < max_len:
                    ids[idx] = ids[idx] + [0]*(max_len-len(ids[idx]))
            input_ids[d] = ids
            #input_ids[d] = pad_sequences(ids, maxlen=max_len, dtype="long", truncating="pre", padding="pre")

        attention_masks = {}
        for d in input_ids:
            masks = []
            for seq in input_ids[d]:
                seq_mask = [float(i>0) for i in seq]
                masks.append(seq_mask)
            attention_masks[d] = masks

        with open(prepared_data_file, 'wb') as f:
            pickle.dump((input_ids, year_ids, labels_test, labels_progress, attention_masks), f)
            print('Data written to disk')

    for dataset in [input_ids, year_ids, labels_test, labels_progress, attention_masks]:
        for d in dataset:
            dataset[d] = torch.tensor(dataset[d])

    return input_ids, year_ids, labels_test, labels_progress, attention_masks
