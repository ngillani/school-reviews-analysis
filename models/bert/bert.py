#!/usr/bin/env python
# coding: utf-8
#from keras.preprocessing.sequence import pad_sequences
import logging
#logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
import numpy as np
import os
import pickle
from sklearn.metrics import (
    f1_score, classification_report, roc_auc_score,
    accuracy_score, hamming_loss,
    average_precision_score,
    precision_score, recall_score,
    precision_recall_fscore_support)
import tensorflow as tf
#print(tf.__version__)
import torch
from torch.autograd.variable import Variable
import torch.nn as nn
from transformers import BertModel, BertConfig
from transformers import AdamW
from pytorch_pretrained_bert import BertAdam
from torch.optim import Adam

from tensorboardX import SummaryWriter

from robert_regressor import RobertForSequenceRegression, MeanBertForSequenceRegression
from loss_function import AdversarialLoss, StraightUpLoss
from metric import accuracy, AUC, f1
from dataset import make_dataloader, load_and_cache_data
from transformers import BertTokenizer

import fire

def start(
        loss='adv',
        outcome='mn_avg_eb'
    ):
    
    print('loss function: {}'.format(loss))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    device = "cpu"
    n_gpu = torch.cuda.device_count()
    #torch.cuda.get_device_name(0)
    print(device)
    print(n_gpu)

    MAX_LEN = 30   # max words per sentence
    BATCH_SIZE = 16
    NUM_EPOCHS = 10

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_ids, labels_test_score, attention_masks, num_sentences_per_school = load_and_cache_data(outcome=outcome, max_len=MAX_LEN)

    train_dataloader = make_dataloader(
            (input_ids['train'], attention_masks['train'], labels_test_score['train'], num_sentences_per_school['train']),
            BATCH_SIZE)

    validation_dataloader = make_dataloader(
            (input_ids['validation'],
             attention_masks['validation'], labels_test_score['validation'], num_sentences_per_school['validation']),
            BATCH_SIZE, shuffle=False)

    config = BertConfig(output_attentions=True)

    # TODO(nabeel) what is this stuff doing
    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'gamma', 'beta']
    # optimizer_grouped_parameters = [
    #    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
    #     'weight_decay_rate': 0.01},
    #    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
    #     'weight_decay_rate': 0.0}
    #]
    #optimizer = BertAdam(optimizer_grouped_parameters, lr=2e-5, warmup=.1)

    # Loss function
    loss_fct = StraightUpLoss()

    learning_rates = [1e-3, 5e-4, 1e-4]

    for lr in learning_rates:
        
        # model = RobertForSequenceRegression(config, num_output=1)
	model = MeanBertForSequenceRegression(config, num_output=1)
        model.cuda()
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

	writer = SummaryWriter('models/checkpoints/{}'.format(lr))
	global_num_forward_passes = 0
        for epoch in range(NUM_EPOCHS):

            # Training
            # Set our model to training mode (as opposed to evaluation mode)
            model.train()

            for name, param in model.bert.named_parameters():
                param.requires_grad = False

            # Tracking variables
            tr_loss = 0
            tr_adv_loss = 0
            acc = 0
            nb_tr_steps = 0

            save_path = 'models/checkpoints/{}/max_len_{}_lr_{}/'.format(outcome, MAX_LEN, lr)
            if not os.path.exists(save_path):
                os.mkdir(save_path)


            # Train the data for one epoch
            for step, batch in enumerate(train_dataloader):

		optimizer.zero_grad()

		global_num_forward_passes += 1

                input_ids, input_mask, test_scores, num_sentences_per_school = batch
                num_sentences_per_school, perm = torch.sort(num_sentences_per_school, descending=True)
                input_ids = input_ids[perm, :, :].to(device)
                input_mask = input_mask[perm, :, :].to(device)
                test_scores = test_scores[perm].to(device)

                num_sentences_per_school = num_sentences_per_school.to(device)
                
		# print (input_ids[1,0,:], tokenizer.decode(input_ids[1, 0, :].tolist()), test_scores[1])

                # Forward pass — do not store attentions during training
                # predicted = model(input_ids, num_sentences_per_school, attention_mask=input_mask)
                predicted = model(input_ids, attention_mask=None)

		# composite = torch.concat(predicted, test_scores, predicted - test_scores)
		# print ('prediction: ', predicted, 'ground truth: ', test_scores)
                t_loss = loss_fct.compute_loss(predicted, test_scores)

                # Compute new gradients
                t_loss.backward()

		torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient
                optimizer.step()

                # Update tracking variables
                tr_loss += t_loss.item()

                nb_tr_steps += 1

                if step % 10  == 0:
                    print("Epoch {}, average training loss so far: {}, {}".format(epoch, tr_loss/nb_tr_steps, tr_adv_loss/nb_tr_steps))
		    
		    all_grads = []
		    for name, param in model.named_parameters():
			if param.requires_grad == True:
				if 'output_layer2' in name:
				    print (name, param.grad.cpu().numpy())
		        	writer.add_histogram(name, param.grad.cpu().numpy(), global_num_forward_passes) 
				all_grads.extend(param.grad.cpu().numpy().reshape(-1).tolist())

		writer.add_scalar('batch_loss', t_loss.item(), global_num_forward_passes) 	
		writer.add_scalar('mean_grad', np.mean(all_grads), global_num_forward_passes)
		writer.add_scalar('percent_zeros', (np.array(all_grads) == 0).sum() / float(len(all_grads)), global_num_forward_passes)

            print("Average training loss: {}".format(tr_loss/nb_tr_steps))
            torch.save(model, '{}epoch_{}_training_loss_{}.pt'.format(save_path, epoch, tr_loss/nb_tr_steps))

            # Validation
            # Put model in evaluation mode to evaluate loss on the validation set
            model.eval()

            # Tracking variables
            eval_loss, eval_adv_loss, eval_accuracy = 0, 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            # Evaluate data for one epoch
            for step, batch in enumerate(validation_dataloader):

                # Unpack the inputs from our dataloader
                input_ids, input_mask, test_scores, num_sentences_per_school = batch
                num_sentences_per_school, perm = torch.sort(num_sentences_per_school, descending=True)
                input_ids = input_ids[perm, :, :].to(device)
                input_mask = input_mask[perm, :, :].to(device)
                test_scores = test_scores[perm].to(device)
                num_sentences_per_school = num_sentences_per_school.to(device)
                
                # Telling the model not to compute or store gradients, saving memory and speeding up validation
                with torch.no_grad():

                  # predicted = model(input_ids, num_sentences_per_school, attention_mask=input_mask)
                  predicted = model(input_ids, attention_mask=input_mask)

                  t_loss = loss_fct.compute_loss(predicted, test_scores)

                eval_loss += t_loss.item()

                nb_eval_steps += 1
                if step % 10 == 0:
		    print ("Epoch {}, average validation loss so far: {}, {}".format(epoch, eval_loss/nb_eval_steps, 0))

            print("Average validation loss: {}, {}".format(eval_loss/nb_eval_steps, eval_adv_loss/nb_eval_steps))
            torch.save(model, '{}epoch_{}_training_loss_{}_val_loss_{}.pt'.format(save_path, epoch, tr_loss/nb_tr_steps, eval_loss/nb_eval_steps))
            # torch.cuda.empty_cache()
	    writer.add_scalar('training_loss', tr_loss/nb_tr_steps, epoch)	
	    writer.add_scalar('validation_loss', eval_loss/nb_eval_steps, epoch)

if __name__ == '__main__':
    fire.Fire()
