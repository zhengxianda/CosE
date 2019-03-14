import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model


class Dis(Model):
    def __init__(self, config):
        super(Dis, self).__init__(config)
        self.ent_embeddings = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        # self.rel_embeddings = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.ent_transfer = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.criterion = nn.MarginRankingLoss(self.config.margin, False)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform(self.ent_embeddings.weight.data)
        # nn.init.xavier_uniform(self.rel_embeddings.weight.data)
        nn.init.xavier_uniform(self.ent_transfer.weight.data)

    def _transfer(self, e, e_transfer, e2_transfer):
        e = e + torch.sum(e * e_transfer, -1, True) * e2_transfer
        e_norm = F.normalize(e, p=2, dim=-1)
        return e_norm

    def _calc(self, h, t):
        return 1.0 + torch.cosine_similarity(h, t)

    def loss(self, p_score, n_score):
        y = Variable(torch.Tensor([-1]).cuda())
        return self.criterion(p_score, n_score, y)

    def forward(self):
        h = self.ent_embeddings(self.batch_h)
        t = self.ent_embeddings(self.batch_t)
        h_transfer = self.ent_transfer(self.batch_h)
        t_transfer = self.ent_transfer(self.batch_t)
        h = self._transfer(h, h_transfer, t_transfer)
        t = self._transfer(t, t_transfer, h_transfer)
        score = self._calc(h, t)
        p_score = self.get_positive_score(score)
        n_score = self.get_negative_score(score)
        return self.loss(p_score, n_score)

    def predict(self):
        h = self.ent_embeddings(self.batch_h)
        t = self.ent_embeddings(self.batch_t)
        h_transfer = self.ent_transfer(self.batch_h)
        t_transfer = self.ent_transfer(self.batch_t)
        h = self._transfer(h, h_transfer, t_transfer)
        t = self._transfer(t, t_transfer, h_transfer)
        score = self._calc(h, t)
        return score.cpu().data.numpy()
