import torch
import torch.nn as nn
from transformers import BertModel

class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def compute_loss(self, predicted_t, actual_t, predicted_a, actual_a):
        t_loss = self.mse_loss(predicted_t, actual_t)
        a_loss = self.mse_loss(predicted_a, actual_a)
        
        loss = t_loss + a_loss

        return t_loss, a_loss, loss


class StraightUpLoss(nn.Module):
    def __init__(self):
        super(StraightUpLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def compute_loss(self, predicted_t, actual_t):
        t_loss = self.mse_loss(predicted_t, actual_t)
        return t_loss
