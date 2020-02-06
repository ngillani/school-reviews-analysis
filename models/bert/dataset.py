import os
import pickle

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertTokenizer

import pandas as pd
import pdb
import numpy as np

def make_dataloader(data, batch_size, sampler=None):
    data = TensorDataset(*data)
    data_loader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return data_loader

def load_and_cache_data(
#        raw_data_file='data/all_gs_reviews_ratings.csv', 
#        prepared_data_file='data/prepared_data_for_bert.p',
        raw_data_file='data/tiny_gs_review_ratings.csv',
        prepared_data_file='data/tiny_gs_review_ratings.p',
        max_len=512,
        train_frac = 0.8
    ):

    if os.path.isfile(prepared_data_file):
    # if False:
        with open(prepared_data_file, 'rb') as f:
            input_ids, labels_t, labels_a, attention_masks = pickle.load(f)

        print('data loaded!')

    else:

        print('Loading data ...')

        df = pd.read_csv(raw_data_file).dropna(subset=['progress_rating',
                                                       'test_score_rating', 'review_text']).reset_index()

        all_ind = list(range(0, len(df)))
        np.random.shuffle(all_ind)

        train_ind = all_ind[0:int(train_frac*len(all_ind))]
        val_ind = all_ind[int(train_frac*len(all_ind)):]

        data = {'train': list(df['review_text'][train_ind]), 'validation': list(df['review_text'][val_ind])}
        labels_t = {'train': list(df['progress_rating'][train_ind]), 'validation': list(df['progress_rating'][val_ind])}
        labels_a = {'train': list(df['test_score_rating'][train_ind]), 'validation': list(df['test_score_rating'][val_ind])}

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenized_texts = {}  # split -> list of list of tokens

        for d in data:
            tokenized_texts[d] = []
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
            pickle.dump((input_ids, labels_t, labels_a, attention_masks), f)
            print('Data written to disk')


    for dataset in [input_ids, labels_t, labels_a, attention_masks]:
        for d in dataset:
            dataset[d] = torch.tensor(dataset[d])

    return input_ids, labels_t, labels_a, attention_masks
