import torch
import torch.nn as nn
import numpy as np
from .attention import Seq_Transformer
import torch.nn.functional as F


class TC(nn.Module):
    def __init__(self, configs, device):
        super(TC, self).__init__()
        self.num_channels = configs.final_out_channels
        self.timestep = configs.TC.timesteps
        self.Wk = nn.ModuleList([nn.Linear(configs.TC.hidden_dim, self.num_channels) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax()
        self.device = device
        
        self.projection_head = nn.Sequential(
            nn.Linear(configs.TC.hidden_dim, configs.final_out_channels // 2),
            nn.BatchNorm1d(configs.final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(configs.final_out_channels // 2, configs.final_out_channels // 4),
        )

        self.seq_transformer = Seq_Transformer(patch_size=self.num_channels, dim=configs.TC.hidden_dim, depth=4, heads=4, mlp_dim=64)

    def forward(self, features_aug1, features_aug2):
        z_aug1 = features_aug1  # features are (batch_size, #channels, seq_len)
        seq_len = z_aug1.shape[2]
        z_aug1 = z_aug1.transpose(1, 2)

        z_aug2 = features_aug2
        z_aug2 = z_aug2.transpose(1, 2)

        batch = z_aug1.shape[0]
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(self.device)  # randomly pick time stamps

        nce = 0  # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)

        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z_aug2[:, t_samples + i, :].view(batch, self.num_channels)
        forward_seq = z_aug1[:, :t_samples + 1, :]

        c_t = self.seq_transformer(forward_seq)

        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * batch * self.timestep
        return nce, self.projection_head(c_t)


class TS_SD(nn.Module):
    def __init__(self, configs, device):
        super(TS_SD, self).__init__()
        self.num_heads = 12 # to prevent reading another config file, we will hardcode this (it's a baseline exp anyway)
        self.kernel_sizes = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
        self.device = device
        self.conv_Q_encoders = nn.ModuleList([nn.Conv1d(1, 1, kernel_size=n) for n in self.kernel_sizes])
        self.conv_V_encoders = nn.ModuleList([nn.Conv1d(1, 1, kernel_size=n) for n in self.kernel_sizes])
        self.conv_K_encoders = nn.ModuleList([nn.Conv1d(1, 1, kernel_size=n) for n in self.kernel_sizes])
        self.dim = np.sqrt(1500)

    def forward(self, signal):
        heads_out = []
        signal.to(self.device)
        for Qe, Ve, Ke in zip(self.conv_Q_encoders, self.conv_V_encoders, self.conv_K_encoders):
            Q = Qe(signal)
            V = Ve(signal)
            K = Ke(signal)
            # K, Q, V of shape batch_size x 1 x feature_len (= 1498 in this case, not sure why it isn't 1500)
            score = torch.bmm(Q, K.transpose(1,2)) / self.dim)
            attn = F.softmax(score, -1)
            print(attn)
            context = torch.bmm(attn, V)
            heads_out.append(context)
            print(context.shape)
