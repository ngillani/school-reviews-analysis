import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

def accuracy(logits, y_true, threshold):
    acc = ((logits>threshold)==y_true.byte()).cpu().numpy().mean(axis=1).sum()
    return acc / logits.size(0)

def f1(y_true, logits, threshold):
    '''
    logits: numpy array of prediction logits
    y_true: numpy array of binary label indicators
    return f1 score
    '''
    return f1_score(y_true,
                   (logits>threshold),
                   average='macro')

def AUC(y_true, logits=None, probs=None):
    '''
    logits: numpy array of prediction logits
    y_true: numpy array of binary label indicators
    return ROC AUC
    '''
    if logits is not None:
        probs = 1 / (1+np.exp(-logits))
    scores = []
    for i in range(y_true.shape[1]):
        if not y_true[:,i].sum():
            continue
        scores.append(
                roc_auc_score(
                    y_score=probs[:,i], y_true=y_true[:,i]))
    return np.mean(scores)
