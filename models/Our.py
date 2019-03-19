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
        self.sub_transfer = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.dis_transfer = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.criterion = nn.MarginRankingLoss(self.config.margin, False)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform(self.sub_transfer.weight.data)
        nn.init.xavier_uniform(self.dis_transfer.weight.data)

    def _transfer(self, e, e_transfer, e2_transfer):
        e = e + torch.sum(e * e_transfer, -1, True) * e2_transfer
        e_norm = F.normalize(e, p=2, dim=-1)
        return e_norm

    def _calc(self, h_sub, t_sub, h_dis, t_dis):
        # print(h_sub_transfer.size())
        # ans = Variable(torch.from_numpy(np.zeros(len(t), dtype=np.float32).cuda()), requires_grad=False)
        # print(ans.size)
        # ans = ans.float()
        # ans = ans.cuda()
        # ans = Variable(ans)
        # print(ans.size())
        # print(h.size())
        # print(self.batch_r.to(torch.float32))
        # ans = 1.0 + (self.batch_r.to(torch.float32) - 1.0) * torch.cosine_similarity(h_sub, t_sub) + (
        #     self.batch_r.to(torch.float32)) * torch.cosine_similarity(h_dis, t_dis)
        ans = 1.0 + (self.batch_r.to(torch.float32) - 1.0) * torch.cosine_similarity(h_sub, t_sub) + (
            self.batch_r.to(torch.float32)) * -1.0 * torch.cosine_similarity(h_dis, t_dis)
        # print(ans)
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
        return ans
        # cos = 1.0 + (self.batch_r.to(torch.float32) - 0.5) * 2 * torch.cosine_similarity(h, t)
        # return cos
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
        h_sub_transfer = self.sub_transfer(self.batch_h)
        t_sub_transfer = self.sub_transfer(self.batch_t)
        h_dis_transfer = self.dis_transfer(self.batch_h)
        t_dis_transfer = self.dis_transfer(self.batch_t)
        h_sub = self._transfer(h, h_sub_transfer, t_sub_transfer)
        t_sub = self._transfer(t, t_sub_transfer, h_sub_transfer)
        h_dis = self._transfer(h, h_dis_transfer, t_dis_transfer)
        t_dis = self._transfer(t, t_dis_transfer, h_dis_transfer)
        score = self._calc(h_sub, t_sub, h_dis, t_dis)
        p_score = self.get_positive_score(score)
        n_score = self.get_negative_score(score)
        return self.loss(p_score, n_score)

    def predict(self):
        h = self.ent_embeddings(self.batch_h)
        t = self.ent_embeddings(self.batch_t)
        h_sub_transfer = self.sub_transfer(self.batch_h)
        t_sub_transfer = self.sub_transfer(self.batch_t)
        h_dis_transfer = self.dis_transfer(self.batch_h)
        t_dis_transfer = self.dis_transfer(self.batch_t)
        h_sub = self._transfer(h, h_sub_transfer, t_sub_transfer)
        t_sub = self._transfer(t, t_sub_transfer, h_sub_transfer)
        h_dis = self._transfer(h, h_dis_transfer, t_dis_transfer)
        t_dis = self._transfer(t, t_dis_transfer, h_dis_transfer)
        score = self._calc(h_sub, t_sub, h_dis, t_dis)
        return score.cpu().data.numpy()
