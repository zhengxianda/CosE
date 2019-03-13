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
        # ans = Variable(torch.from_numpy(np.zeros(len(t), dtype=np.float32).cuda()), requires_grad=False)
        # print(ans.size)
        # ans = ans.float()
        # ans = ans.cuda()
        # ans = Variable(ans)
        # print(ans.size())
        # print(h.size())
        # ans = 1.0 + (self.batch_r.to(torch.float32) - 1.0) * torch.cosine_similarity(h_sub, t_sub) + (
        #     self.batch_r.to(torch.float32)) * torch.cosine_similarity(h_dis, t_dis)
        # for i in range(len(ans)):
        #     # print(torch.cosine_similarity(h_sub[i], t_sub[i]))
        #     # print(h_sub[i].size())
        #     # print(t_sub[i].size())
        #     # print(torch.cosine_similarity(h_sub[i], t_sub[i], dim=0))
        #     if self.batch_r[i] == 0:  # sub
        #         ans[i] = 1.0 - torch.cosine_similarity(h_sub[i], t_sub[i], dim=0)
        #         # print(ans[i])
        #     if self.batch_r[i] == 1:  # dis
        #         ans[i] = 1.0 + torch.cosine_similarity(h_dis[i], t_dis[i], dim=0)
        #         # print(ans[i])
        # return ans
        cos = 1.0 + (self.batch_r.to(torch.float32) - 0.5) * 2 * torch.cosine_similarity(h, t)
        return cos
        # return 1.0 + (r.to(torch.float32) - 0.5) * 2 * torch.cosine_similarity(h, t) + (
        #             r.to(torch.float32) - 0.5) * -2 * torch.norm(h, dim=1) - torch.norm(t, dim=1)
        # return 1.0 + (r.to(torch.float32) - 0.5) * 2 * torch.cosine_similarity(h, t) \
        #        + (r.to(torch.float32) - 0.5) * -2 * torch.norm(h, dim=1) * torch.norm(h, dim=1) \
        #        - torch.norm(t, dim=1) * torch.norm(t, dim=1)
        # return 1.0 + (r.to(torch.float32) - 0.5) * 2 * torch.cosine_similarity(h, t) + (
        #             r.to(torch.float32) - 0.5) * -2 * torch.norm(m) - torch.norm(n)

    def loss(self, p_score, n_score):
        y = Variable(torch.Tensor([-1]).cuda())
        return self.criterion(p_score, n_score, y)

    def forward(self):
        h = self.ent_embeddings(self.batch_h)
        t = self.ent_embeddings(self.batch_t)
        # r = self.batch_r
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
        # r = self.batch_r
        h_transfer = self.ent_transfer(self.batch_h)
        t_transfer = self.ent_transfer(self.batch_t)
        h = self._transfer(h, h_transfer, t_transfer)
        t = self._transfer(t, t_transfer, h_transfer)
        score = self._calc(h, t)
        return score.cpu().data.numpy()
