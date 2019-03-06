import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model


class Our(Model):
    def __init__(self, config):
        super(Our, self).__init__(config)
        self.ent_embeddings = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        # self.rel_embeddings = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.ent_transfer = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.rel_transfer = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.criterion = nn.MarginRankingLoss(self.config.margin, False)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform(self.ent_embeddings.weight.data)
        # nn.init.xavier_uniform(self.rel_embeddings.weight.data)
        nn.init.xavier_uniform(self.ent_transfer.weight.data)
        nn.init.xavier_uniform(self.rel_transfer.weight.data)

    def _calc(self, h, t):
        # print(torch.norm(h, dim=1))
        # print(torch.norm(h))
        # print(torch.norm(self.m, dim=1))
        # print("r")
        # print(self.batch_h)
        # print(self.batch_t)
        # print(r)
        # a = 1.0 + (r.to(torch.float32) - 0.5) * 2 * torch.cosine_similarity(h, t) + (r.to(
        #     torch.float32) - 0.5) * -2 * torch.norm(m, dim=1) - torch.norm(n, dim=1)
        # print(a)
        # print("r change")
        # print((r.to(torch.float32) - 0.5) * 2)
        # h_r_t = torch.norm(h + r - t, self.config.p_norm, -1)
        # trans_e = (self.batch_r.to(torch.float32) - 0.5) * 2 * h_r_t
        cos = 1.0 + (self.batch_r.to(torch.float32) - 0.5) * 2 * torch.cosine_similarity(h, t)
        # print(self.batch_r)
        # print((self.batch_r.to(torch.float32) - 0.5) * 2)
        # print(torch.cosine_similarity(h, t))
        return cos
        # return 1.0 + (r.to(torch.float32) - 0.5) * 2 * torch.cosine_similarity(h, t) + (
        #             r.to(torch.float32) - 0.5) * -2 * torch.norm(h, dim=1) - torch.norm(t, dim=1)
        # return 1.0 + (r.to(torch.float32) - 0.5) * 2 * torch.cosine_similarity(h, t) \
        #        + (r.to(torch.float32) - 0.5) * -2 * torch.norm(h, dim=1) * torch.norm(h, dim=1) \
        #        - torch.norm(t, dim=1) * torch.norm(t, dim=1)
        # return 1.0 + (r.to(torch.float32) - 0.5) * 2 * torch.cosine_similarity(h, t) + (
        #             r.to(torch.float32) - 0.5) * -2 * torch.norm(m) - torch.norm(n)

    def _transfer(self, e, e_transfer, r_transfer):
        e = e + torch.sum(e * e_transfer, -1, True) * r_transfer
        e_norm = F.normalize(e, p=2, dim=-1)
        return e_norm

    def loss(self, p_score, n_score):
        y = Variable(torch.Tensor([-1]).cuda())
        return self.criterion(p_score, n_score, y)

    def forward(self):
        h = self.ent_embeddings(self.batch_h)
        t = self.ent_embeddings(self.batch_t)
        # r = self.rel_embeddings(self.batch_r)
        h_transfer = self.ent_transfer(self.batch_h)
        t_transfer = self.ent_transfer(self.batch_t)
        r_transfer = self.rel_transfer(self.batch_r)
        h = self._transfer(h, h_transfer, r_transfer)
        t = self._transfer(t, t_transfer, r_transfer)
        score = self._calc(h, t)
        p_score = self.get_positive_score(score)
        n_score = self.get_negative_score(score)
        return self.loss(p_score, n_score)

    def predict(self):
        h = self.ent_embeddings(self.batch_h)
        t = self.ent_embeddings(self.batch_t)
        # r = self.rel_embeddings(self.batch_r)
        h_transfer = self.ent_transfer(self.batch_h)
        t_transfer = self.ent_transfer(self.batch_t)
        r_transfer = self.rel_transfer(self.batch_r)
        h = self._transfer(h, h_transfer, r_transfer)
        t = self._transfer(t, t_transfer, r_transfer)
        score = self._calc(h, t)
        return score.cpu().data.numpy()
