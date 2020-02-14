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

from robert_regressor import RobertForSequenceRegression
from loss_function import AdversarialLoss, StraightUpLoss
from metric import accuracy, AUC, f1
from dataset import make_dataloader, load_and_cache_data

import fire

def start(
        loss='adv'
    ):
    
    print('loss function: {}'.format(loss))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    device = "cpu"
    n_gpu = torch.cuda.device_count()
    #torch.cuda.get_device_name(0)
    print(device)
    print(n_gpu)

    MAX_LEN = 30   # max words per sentence
    BATCH_SIZE = 4
    NUM_EPOCHS = 5
    RETRAIN = 0

    input_ids, labels_progress, attention_masks, num_sentences_per_school = load_and_cache_data(max_len=MAX_LEN)

    train_dataloader = make_dataloader(
            (input_ids['train'], attention_masks['train'], labels_progress['train'], num_sentences_per_school['train']),
            BATCH_SIZE)

    validation_dataloader = make_dataloader(
            (input_ids['validation'],
             attention_masks['validation'], labels_progress['validation'], num_sentences_per_school['validation']),
            BATCH_SIZE)

    config = BertConfig(output_attentions=True)

    model = RobertForSequenceRegression(config, num_output=1)
    model.cuda()

    # TODO(nabeel) what is this stuff doing
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=2e-5, warmup=.1)

    # Loss function
    loss_fct = StraightUpLoss()

    print('Training')
    for epoch in range(NUM_EPOCHS):

        # Training
        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # Tracking variables
        tr_loss = 0
        tr_adv_loss = 0
        acc = 0
        nb_tr_steps = 0

        save_path = 'models/checkpoints/max_len_{}_threshold_loss_{}/'.format(MAX_LEN, loss)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        if not RETRAIN and os.path.exists('{}e{}.pt'.format(save_path, epoch)):
            print('Model for epoch {} exists. Skipping'.format(epoch))
            model = torch.load('{}e{}.pt'.format(save_path, epoch))
            continue

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):

            input_ids, input_mask, progress_scores, num_sentences_per_school = batch

            num_sentences_per_school, perm = torch.sort(num_sentences_per_school, descending=True)
            input_ids = input_ids[perm, :, :]
            input_mask = input_mask[perm, :, :]
            progress_scores = progress_scores[perm]

            num_sentences_per_school = num_sentences_per_school.to(device)
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            progress_scores = progress_scores.to(device)

            # Forward pass — do not store attentions during training
            predicted = model(input_ids, num_sentences_per_school, attention_mask=input_mask)

            t_loss = loss_fct.compute_loss(predicted, progress_scores)

            # Clear out the old accumulated gradients
            optimizer.zero_grad()
            
            # Compute new gradients
            t_loss.backward()

            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update tracking variables
            tr_loss += t_loss.item()

            nb_tr_steps += 1

            if step % 10  == 0:
                print("Epoch {}, average training loss so far: {}, {}".format(epoch, tr_loss/nb_tr_steps, tr_adv_loss/nb_tr_steps))

        print("Average training loss: {}".format(tr_loss/nb_tr_steps))
        torch.save(model, '{}epoch_{}_training_loss_{}.pt'.format(save_path, epoch, tr_loss/nb_tr_steps))

        # Validation
        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Tracking variables
        eval_loss, eval_adv_loss, eval_accuracy = 0, 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Unpack the inputs from our dataloader
            input_ids, input_mask, progress_scores, num_sentences_per_school = batch
            num_sentences_per_school, perm = torch.sort(num_sentences_per_school, descending=True)
            input_ids = input_ids[perm, :, :]
            input_mask = input_mask[perm, :, :]
            progress_scores = progress_scores[perm]
            for data in [num_sentences_per_school, input_ids, input_mask, progress_scores]:
                data.to(device)            
            
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
              # Forward pass, calculate logit predictions
              predicted = model(input_ids, num_sentences_per_school, attention_mask=input_mask)
              t_loss = loss_fct.compute_loss(predicted, progress_scores)

            eval_loss += t_loss.item()

            # Move logits and labels to CPU
            nb_eval_steps += 1

        print("Average validation loss: {}, {}".format(eval_loss/nb_eval_steps, eval_adv_loss/nb_eval_steps))
        torch.save(model, '{}epoch_{}_training_loss_{}_val_loss_{}.pt'.format(save_path, epoch, tr_loss/nb_tr_steps, eval_loss/nb_eval_steps))
        torch.cuda.empty_cache()


if __name__ == '__main__':
    fire.Fire()
