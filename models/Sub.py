import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model
import json

class Sub(Model):
    def __init__(self, config):
        super(Sub, self).__init__(config)
        self.ent_embeddings = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.rel_embeddings = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.ent_transfer = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.mn_embeddings = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.ht_embeddings = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.ht_transfer = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.criterion = nn.MarginRankingLoss(self.config.margin, False)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform(self.rel_embeddings.weight.data)
        nn.init.xavier_uniform(self.ent_transfer.weight.data)
        nn.init.xavier_uniform(self.mn_embeddings.weight.data)
        nn.init.xavier_uniform(self.ht_embeddings.weight.data)
        nn.init.xavier_uniform(self.ht_transfer.weight.data)

    def _transfer(self, e, e_transfer, e2_transfer):
        e = e + torch.sum(e * e_transfer, -1, True) * e2_transfer
        e_norm = F.normalize(e, p=2, dim=-1)
        return e_norm

    def _calc(self, h, t, r, m, n, h1, t1):
        return 1.0 - torch.cosine_similarity(h, t) + torch.norm(m, dim=1) - torch.norm(n, dim=1) + torch.norm(h1+r-t1, dim=1)

    def loss(self, p_score, n_score):
        y = Variable(torch.Tensor([-1]).cuda())
        return self.criterion(p_score, n_score, y)

    def forward(self):
        h = self.ent_embeddings(self.batch_h)
        t = self.ent_embeddings(self.batch_t)
        m = self.mn_embeddings(self.batch_h)
        n = self.mn_embeddings(self.batch_t)
        h1 = self.ht_embeddings(self.batch_h)
        t1 = self.ht_embeddings(self.batch_t)
        h1_transfer = self.ht_transfer(self.batch_h)
        t1_transfer = self.ht_transfer(self.batch_t)
        h1 = self._transfer(h1, h1_transfer, t1_transfer)
        t1 = self._transfer(t1, t1_transfer, h1_transfer)
        r = self.rel_embeddings(self.batch_r)
        score = self._calc(h, t, r, m, n, h1, t1)
        p_score = self.get_positive_score(score)
        n_score = self.get_negative_score(score)
        return self.loss(p_score, n_score)

    def predict(self):
        h = self.ent_embeddings(self.batch_h)
        t = self.ent_embeddings(self.batch_t)
        m = self.mn_embeddings(self.batch_h)
        n = self.mn_embeddings(self.batch_t)
        h1 = self.ht_embeddings(self.batch_h)
        t1 = self.ht_embeddings(self.batch_t)
        h1_transfer = self.ht_transfer(self.batch_h)
        t1_transfer = self.ht_transfer(self.batch_t)
        h1 = self._transfer(h1, h1_transfer, t1_transfer)
        t1 = self._transfer(t1, t1_transfer, h1_transfer)
        r = self.rel_embeddings(self.batch_r)
        score = self._calc(h, t, r, m, n, h1, t1)
        return score.cpu().data.numpy()
