import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model
import sys
import random


class Our(Model):
    def __init__(self, config):
        super(Our, self).__init__(config)
        self.ent_embeddings = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        # self.rel_embeddings = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.criterion = nn.MarginRankingLoss(self.config.margin, False)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform(self.ent_embeddings.weight.data)
        # nn.init.xavier_uniform(self.rel_embeddings.weight.data)

    def _calc(self, h, t, r, y):
        # return torch.norm(1-torch.cosine_similarity(h, t)+torch.norm(h)-torch.norm(t), self.config.p_norm, -1)
        ans = y
        # print(type(ans))
        # print(len(r))
        for i in range(len(r)):
            # a1 = np.random.random(100)
            # b1 = np.random.random(100)
            # while (np.linalg.norm(a1) > 1.0):
            #     a1 /= 2
            # while (np.linalg.norm(b1) > 1.0):
            #     b1 /= 2
            # m = np.linalg.norm(a1)
            # n = np.linalg.norm(b1)
            # print(a)
            # print(b)
            # print("\n")
            # m = random.random()
            # n = random.random()
            if r[i] == 0:  # sub
                ans[i] = 1.0 - torch.cosine_similarity(h[i], t[i], 0)
                # print(type([h[i]]))
                # print(type(1.0 - torch.cosine_similarity(h[i], t[i], 0)))
                # print("sub")
            if r[i] == 1:  # dis
                ans[i] = 1.0 + torch.cosine_similarity(h[i], t[i], 0)
                # print("dis")
        # return 1 - torch.cosine_similarity(h, t) + torch.norm(h) * torch.norm(h) - torch.norm(t) * torch.norm(t)
        # print(type(ans))
        return ans

    def loss(self, p_score, n_score):
        y = Variable(torch.Tensor([-1]).cuda())
        return self.criterion(p_score, n_score, y)

    def forward(self):
        h = self.ent_embeddings(self.batch_h)
        t = self.ent_embeddings(self.batch_t)
        r = self.batch_r
        y = self.batch_y
        score = self._calc(h, t, r, y)
        # print(type(score[0]))
        p_score = self.get_positive_score(score)
        n_score = self.get_negative_score(score)
        return self.loss(p_score, n_score)

    def predict(self):
        h = self.ent_embeddings(self.batch_h)
        t = self.ent_embeddings(self.batch_t)
        r = self.batch_r
        y = self.batch_y
        score = self._calc(h, t, r, y)
        # print(type(score))
        s = torch.from_numpy(score)
        return s.cpu().data.numpy()
