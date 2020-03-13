import os
import pickle

import spacy
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertTokenizer

import json
import pandas as pd
import pdb
import numpy as np

from collections import defaultdict

key_vals = {}   #  key -> set of values

MAX_SENTENCES_PER_SCHOOL = 100


def make_dataloader(data, batch_size, shuffle=True, sampler=None):
	data = TensorDataset(*data)
	data_loader = DataLoader(data, shuffle=shuffle, sampler=sampler, batch_size=batch_size)
	return data_loader


def load_and_cache_data_truncated(
		raw_data_file='data/all_gs_comments_by_school.csv',
		prepared_data_file='data/all_gs_comments_by_school_for_emb.p',
#        raw_data_file='data/tiny_by_school_test_scores.csv',
#        prepared_data_file='data/tiny_by_school_for_emb.p',        
		max_len=30,
		outcome='mn_avg_eb',
	):


	if os.path.isfile(prepared_data_file):
		with open(prepared_data_file, 'rb') as f:
			input_ids, attention_masks, urls = pickle.load(f)
		print('data loaded from cache!')

	else:
		print('Loading data ...')

		df = pd.read_csv(raw_data_file).dropna(subset=['review_text']).reset_index()


		urls = []
		input_ids = []
		attention_masks = []

		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		spacy_nlp = spacy.load('en_core_web_sm')  # For sentence segmentation

		for j in range(0, len(df)):

			print (j)
			review = df['review_text'][j]
			text_sentences = spacy_nlp(review.decode('utf-8'))
			# Choose at most MAX_SENTENCES_PER_SCHOOL to include.
			# Reviews are sorted by recency in the input (newest first).
			token_id_vectors = []
			attention_mask_vectors = []
			for i, sentence in enumerate(text_sentences.sents):
				# Convert to bert IDs
				ids = tokenizer.encode(sentence.text, max_length=max_len)
				# Pad words if needed
				if len(ids) < max_len:
					ids += [0] * (max_len - len(ids))
				attention_mask = [float(id>0) for id in ids]
				token_id_vectors.append(ids)
				attention_mask_vectors.append(attention_mask)
				if i >= MAX_SENTENCES_PER_SCHOOL - 1:
					break
			# Pad sentences if needed
			while len(token_id_vectors) < MAX_SENTENCES_PER_SCHOOL:
				token_id_vectors.append([0] * max_len)
				attention_mask_vectors.append([0.0] * max_len)

			input_ids.append(token_id_vectors)
			attention_masks.append(attention_mask_vectors)
			urls.append(df['url'][j])
	
		print ("Writing data ...")
		with open(prepared_data_file, 'wb') as f:
			pickle.dump((input_ids, attention_masks, urls), f)
			print('Data written to disk')

#    for dataset in [input_ids, attention_masks, urls]:
#        dataset = torch.tensor(dataset)

	return input_ids, attention_masks, urls


def bert_encode_reviews_truncated(
		output_file='data/bert_school_embeddings.json'
	):

	from robert_regressor import BertEncoder
	from transformers import BertConfig

	#torch.cuda.set_device(1)
	device = torch.device("cpu")
	if torch.cuda.is_available():
		device = torch.device("cuda")

	MAX_LEN = 30
	BATCH_SIZE = 32
	NUM_EPOCHS = 10


	input_ids, attention_masks, urls = load_and_cache_data_truncated(max_len=MAX_LEN)
	input_ids = torch.tensor(input_ids)
	attention_masks = torch.tensor(attention_masks)
	url_ids = []
	id_to_url = {}
	for i, u in enumerate(urls):
	url_ids.append(i)
		id_to_url[i] = u
	url_ids = torch.tensor(url_ids)

	data_loader = make_dataloader((input_ids, attention_masks, url_ids), BATCH_SIZE)
	config = BertConfig(output_attentions=True)

	model = BertEncoder(config)
	model.to(device)
	model.eval()

	all_embeddings = {}

	for step, batch in enumerate(data_loader):

		print ('Batch # ', step)
		input_ids, attention_masks, curr_url_ids = batch
		curr_url_ids = curr_url_ids.tolist()
		print (curr_url_ids)
		input_ids = input_ids.to(device)
		attention_masks = attention_masks.to(device)

		n_schools, n_sent, max_len = input_ids.size()
		with torch.no_grad():
			emb = model(input_ids, attention_mask=attention_masks)
			for i in range(0, n_schools):
				all_embeddings[id_to_url[curr_url_ids[i]]] = emb[i, :].tolist()

	f = open(output_file, 'w')
	f.write(json.dumps(all_embeddings, indent=4))
	f.close()


def load_and_cache_data_full(
#		raw_data_file='data/all_gs_and_seda_with_comments.csv',
#		prepared_data_file='data/all_gs_and_seda_with_comments.p',
        raw_data_file='data/tiny_gs_and_seda_with_comments.csv',
        prepared_data_file='data/tiny_gs_and_seda_with_comments.p',        
		max_len=512,
		outcome='mn_avg_eb',
	):


	if os.path.isfile(prepared_data_file):
		with open(prepared_data_file, 'rb') as f:
			input_ids, attention_masks, urls = pickle.load(f)
		print('data loaded from cache!')

	else:
		print('Loading data ...')

		df = pd.read_csv(raw_data_file).dropna(subset=['review_text']).reset_index()


		urls = []
		input_ids = []
		attention_masks = []

		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

		for j in range(0, len(df)):

			print (j)

			# Convert to bert IDs
			ids = tokenizer.encode(df['review_text'][j], max_length=max_len)
			
			# Pad words if needed
			if len(ids) < max_len:
				ids += [0] * (max_len - len(ids))

			attention_mask = [float(id>0) for id in ids]

			input_ids.append(ids)
			attention_masks.append(attention_mask)
			urls.append(df['url'][j])
	
		print ("Writing data ...")
		with open(prepared_data_file, 'wb') as f:
			pickle.dump((input_ids, attention_masks, urls), f)
			print('Data written to disk')

#    for dataset in [input_ids, attention_masks, urls]:
#        dataset = torch.tensor(dataset)

	return input_ids, attention_masks, urls


def bert_encode_reviews_full(
		output_file='data/bert_school_embeddings_full.json'
	):

	from robert_regressor import BertEncoderForComments
	from transformers import BertConfig

	#torch.cuda.set_device(1)
	device = torch.device("cpu")
	if torch.cuda.is_available():
		device = torch.device("cuda")

	MAX_LEN = 512
	BATCH_SIZE = 64
	NUM_EPOCHS = 10


	input_ids, attention_masks, urls = load_and_cache_data_full(max_len=MAX_LEN)
	input_ids = torch.tensor(input_ids)
	attention_masks = torch.tensor(attention_masks)
	url_ids = []
	id_to_url = {}
	for i, u in enumerate(urls):
	url_ids.append(i)
		id_to_url[i] = u
	url_ids = torch.tensor(url_ids)

	data_loader = make_dataloader((input_ids, attention_masks, url_ids), BATCH_SIZE)
	config = BertConfig(output_attentions=True)

	model = BertEncoder(config)
	model.to(device)
	model.eval()

	all_embeddings = defaultdict(list)

	for step, batch in enumerate(data_loader):

		print ('Batch # ', step)
		input_ids, attention_masks, curr_url_ids = batch
		curr_url_ids = curr_url_ids.tolist()
		print (curr_url_ids)
		input_ids = input_ids.to(device)
		attention_masks = attention_masks.to(device)

		n_comments, max_len = input_ids.size()
		with torch.no_grad():
			emb = model(input_ids, attention_mask=attention_masks)
			for i in range(0, n_comments):
				all_embeddings[id_to_url[curr_url_ids[i]]].append(emb[i, :].tolist())

	# Take the mean of all comment embeddings per school
	for u in all_embeddings:
		all_embeddings[u] = np.mean(all_embeddings[u], axis=0)

	f = open(output_file, 'w')
	f.write(json.dumps(all_embeddings, indent=4))
	f.close()

if __name__ == "__main__":
	# bert_encode_reviews_truncated()


