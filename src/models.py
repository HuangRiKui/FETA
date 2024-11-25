import pdb
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class LCWT(nn.Module):
    def __init__(self, rule_dim, args):
        super().__init__()
        self.rule_num = rule_dim-1
        self.W = nn.Parameter(torch.randn(self.rule_num,1))
        self.W0 = nn.Parameter(torch.randn(1))

        self.W.data.clamp_(0, 1)

    def forward(self, rel_data):
        pos_data = rel_data[:, 0, :]
        neg_data = rel_data[:, 1:, :]

        pos_score = self.score_function(pos_data).squeeze(-1)
        neg_score = self.score_function(neg_data).squeeze(-1)

        return pos_score, neg_score

    def score_function(self, data):
        # score = torch.matmul(data, self.W) + self.W0
        score = torch.ones_like(data)
        score[data == 0] = 0
        score = torch.matmul(score, self.W)

        return score

class LCFT(nn.Module):
    def __init__(self, rule_dim, args):
        super().__init__()
        self.rule_num = rule_dim-1
        self.W = nn.Parameter(torch.randn(self.rule_num,1))
        self.beta = args.beta

        self.W.data.clamp_(0,1)

    def forward(self, rel_data):
        # rel_data[rel_data==0] = 1e9

        pos_data = rel_data[:,0,:]
        neg_data = rel_data[:,1:,:]

        pos_score = self.score_function(pos_data).squeeze(-1)
        neg_score = self.score_function(neg_data).squeeze(-1)

        return pos_score, neg_score

    def score_function(self, data):
        # score = torch.matmul(data, self.W) + self.W0
        # score = F.relu(data - self.T0)
        score = torch.exp(-self.beta*data)
        mask = torch.ones_like(score)
        mask[data==0] = 0
        score = score * mask
        score = torch.matmul(score, self.W)+1e-9

        return score

class LTV(nn.Module):
    def __init__(self, rule_dim, conf_tensor, args):
        super().__init__()
        self.rule_num = rule_dim-1
        self.W = conf_tensor.reshape(-1,1)
        if args.cuda:
            self.W = self.W.cuda()
        self.beta = nn.Parameter(torch.randn(1, self.rule_num))
        # self.W0 = nn.Parameter(torch.randn(1, 1))

        self.W.data.clamp_(0,1)
        # self.W0.data.clamp_(0, 1)
        self.beta.data.clamp_(min=0)

    def forward(self, rel_data):
        pos_data = rel_data[:,0,:]
        neg_data = rel_data[:,1:,:]

        pos_score = self.score_function(pos_data).squeeze(-1)
        neg_score = self.score_function(neg_data).squeeze(-1)

        return pos_score, neg_score

    def score_function(self, data):
        score = torch.exp(-self.beta*data)
        # score = torch.sigmoid(self.beta1 * data)
        mask = torch.ones_like(score)
        mask[data==0] = 0
        score = score * mask
        score = torch.matmul(score, self.W) + 1e-9

        return score

class TempValid(nn.Module):
    def __init__(self, rule_dim, args):
        super().__init__()
        self.rule_num = rule_dim-1
        self.W = nn.Parameter(torch.randn(self.rule_num,1))
        self.beta = nn.Parameter(torch.randn(1, self.rule_num))
        # self.W0 = nn.Parameter(torch.randn(1, 1))

        self.W.data.clamp_(0,1)
        # self.W0.data.clamp_(0, 1)
        self.beta.data.clamp_(min=0)

    def forward(self, rel_data):
        pos_data = rel_data[:,0,:]
        neg_data = rel_data[:,1:,:]

        pos_score = self.score_function(pos_data).squeeze(-1)
        neg_score = self.score_function(neg_data).squeeze(-1)

        return pos_score, neg_score

    def score_function(self, data):
        score = torch.exp(-self.beta*data)
        # score = torch.sigmoid(self.beta1 * data)
        mask = torch.ones_like(score)
        mask[data==0] = 0
        score = score * mask
        score = torch.matmul(score, self.W) + 1e-9

        return score

class FETA(nn.Module):
    def __init__(self, rule_dim, args):
        super().__init__()
        self.args =args
        self.rule_num = rule_dim-1
        self.W1 = nn.Parameter(torch.randn(self.rule_num, 1))
        self.W2 = nn.Parameter(torch.randn(self.rule_num, 1))
        self.W3 = nn.Parameter(torch.randn(self.rule_num, 1))
        self.C1 = nn.Parameter(torch.randn(1,self.rule_num))
        self.C2 = nn.Parameter(torch.randn(1, self.rule_num))
        self.C3 = nn.Parameter(torch.randn(1, self.rule_num))
        self.beta = nn.Parameter(torch.randn(1, self.rule_num))

        self.W1.data.clamp_(0,1)
        self.W2.data.clamp_(0, 1)
        self.W3.data.clamp_(0, 1)
        self.C1.data.clamp_(0, 1)
        self.C2.data.clamp_(0, 1)
        self.C3.data.clamp_(0, 1)
        self.beta.data.clamp_(min=0)

    def forward(self, latest_data, lf_data, sf_data):
        mask = torch.ones_like(latest_data)
        mask[latest_data == 0] = 0

        temp_score = torch.exp(-self.beta * latest_data)
        lf_score = torch.sigmoid(self.C1 * lf_data)
        temp_score = temp_score * mask

        if self.args.fluctuation != 'none':
            if self.args.fluctuation == 'fixed':
                sf_score = torch.sigmoid(self.C2*(sf_data - lf_data))
            elif self.args.fluctuation == 'dense':
                sf_score = torch.sigmoid(self.C2*(sf_data/20.0 - lf_data/200.0))
            else:
                sf_score = torch.sigmoid(self.C2 * sf_data + self.C3 * lf_data)
            sf_score = sf_score * mask


        if self.args.recency and (not self.args.frequency) and self.args.fluctuation == 'none':
            score1 = temp_score + 1e-9
            all_score = self.args.gamma*torch.matmul(score1,self.W1)
        elif (not self.args.recency) and self.args.frequency and self.args.fluctuation == 'none':
            lf_score = lf_score * mask
            score1 = lf_score + 1e-9
            all_score = self.args.gamma*torch.matmul(score1,self.W1)
        elif self.args.recency and (not self.args.frequency) and self.args.fluctuation == 'para':
            score1 = temp_score + 1e-9
            score2 = sf_score + 1e-9
            all_score = self.args.gamma*torch.matmul(score1,self.W1) + (1-self.args.gamma)*torch.matmul(score2, self.W2)
        elif (not self.args.recency) and self.args.frequency and self.args.fluctuation == 'para':
            lf_score = lf_score * mask
            score1 = lf_score + 1e-9
            score2 = sf_score + 1e-9
            all_score = self.args.gamma * torch.matmul(score1, self.W1) + (1 - self.args.gamma) * torch.matmul(score2, self.W2)
        else:
            score1 = temp_score * (1 + lf_score) + 1e-9
            score2 = sf_score + 1e-9
            all_score = self.args.gamma*torch.matmul(score1,self.W1) + (1-self.args.gamma)*torch.matmul(score2, self.W2)


        pos_score = all_score[:,0]
        neg_score = all_score[:, 1:].squeeze(-1)


        return pos_score, neg_score


class Noisy_OR(nn.Module):
    def __init__(self, rule_dim, args):
        super().__init__()
        self.rule_num = rule_dim-1
        self.W = nn.Parameter(torch.randn(1, self.rule_num))
        self.beta = nn.Parameter(torch.randn(1, self.rule_num))

        self.W.data.clamp_(0,1)
        self.beta.data.clamp_(min=0)

    def forward(self, rel_data):
        pos_data = rel_data[:,0,:]
        neg_data = rel_data[:,1:,:]

        pos_score = self.score_function(pos_data).squeeze(-1)
        neg_score = self.score_function(neg_data).squeeze(-1)

        return pos_score, neg_score

    def score_function(self, data):
        score = torch.exp(-self.beta*data)
        mask = torch.ones_like(score)
        mask[data==0] = 0
        score = score * mask
        score = 1 - torch.prod((1-score * self.W), dim=-1) + 1e-9

        return score

def ccorr(a, b):
    fft_a = torch.fft.fft(a, norm='ortho', dim=-1)
    fft_b = torch.fft.fft(b, norm='ortho', dim=-1)
    fft_b_conj = fft_b.conj()
    fft_prod = fft_a*fft_b_conj
    ccorr = torch.fft.ifft(fft_prod, dim=-1)
    return ccorr.real

def cconv(a, b):
    fft_a = torch.fft.fft(a, norm='ortho', dim=-1)
    fft_b = torch.fft.fft(b, norm='ortho', dim=-1)
    fft_prod = fft_a*fft_b
    ccorr = torch.fft.ifft(fft_prod, dim=-1)
    return ccorr.real
