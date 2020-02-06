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

from bert_regressor import BertForSequenceRegression
from loss_function import AdversarialLoss, StraightUpLoss
from metric import accuracy, AUC, f1
from dataset import make_dataloader, load_and_cache_data

import fire

def start(
        loss='adv'
    ):
    
    print('loss function: {}'.format(loss))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    #torch.cuda.get_device_name(0)
    print(device)
    print(n_gpu)

    MAX_LEN = 512
    BATCH_SIZE = 4
    NUM_EPOCHS = 5
    RETRAIN = 0
    TEST_SCORE_MEAN = 5.72
    TEST_SCORE_STD = 2.52

    input_ids, labels_t, labels_a, attention_masks = load_and_cache_data(max_len=MAX_LEN)

    train_dataloader = make_dataloader(
            (input_ids['train'], attention_masks['train'], labels_t['train'], labels_a['train']),
            BATCH_SIZE)

    validation_dataloader = make_dataloader(
            (input_ids['validation'], attention_masks['validation'], labels_t['validation'], labels_a['validation']),
            BATCH_SIZE)

    config = BertConfig(output_attentions=True)
    model = BertForSequenceRegression(config, num_output=2)
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
    if loss == 'adv':
        print 'Adversarial loss ...'
        loss_fct = AdversarialLoss()
    else:
        print 'Straight up loss ...'
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

                
            input_ids, input_mask, t_scores, a_scores = batch

            # Forward pass — do not store attentions during training
            predicted, _ = model(input_ids, attention_mask=input_mask)
            random_preds = torch.empty(input_ids.size(0)).normal_(mean=TEST_SCORE_MEAN,std=TEST_SCORE_STD).to(device)
            # t_loss, a_loss, loss = loss_fct.compute_loss(predicted[:, 0], t_scores, predicted[:, 1], a_scores)
            t_loss, a_loss, loss = loss_fct.compute_loss(predicted[:, 0], t_scores, predicted[:, 1], random_preds)

            # Clear out the old accumulated gradients
            optimizer.zero_grad()
            
            # Compute new gradients
            loss.backward()

            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update tracking variables
            tr_loss += t_loss.item()
            tr_adv_loss += a_loss.item()

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
            batch = tuple(t.to(device) for t in batch)
            
            # Unpack the inputs from our dataloader
            input_ids, input_mask, t_scores, a_scores = batch
            
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
              # Forward pass, calculate logit predictions
              predicted, attn_mask = model(input_ids, attention_mask=input_mask)
              random_preds = torch.empty(BATCH_SIZE).normal_(mean=TEST_SCORE_MEAN,std=TEST_SCORE_STD).to(device)
              # t_loss, a_loss, loss = loss_fct.compute_loss(predicted[:, 0], t_scores, predicted[:, 1], a_scores)
              t_loss, a_loss, loss = loss_fct.compute_loss(predicted[:, 0], t_scores, predicted[:, 1], random_preds)

            eval_loss += t_loss.item()
            eval_adv_loss += a_loss.item()

            # Move logits and labels to CPU
            nb_eval_steps += 1

        print("Average validation loss: {}, {}".format(eval_loss/nb_eval_steps, eval_adv_loss/nb_eval_steps))
        torch.save(model, '{}epoch_{}_training_loss_{}_val_loss_{}.pt'.format(save_path, epoch, tr_loss/nb_tr_steps, eval_loss/nb_eval_steps))
        torch.cuda.empty_cache()


if __name__ == '__main__':
    fire.Fire()
