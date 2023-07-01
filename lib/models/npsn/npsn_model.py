import torch
import torch.nn as nn
import math
from .utils import box_muller_transform

class GAT(nn.Module):
    def __init__(self, in_feat=2, out_feat=64, feat_dim=64, n_head=4, dropout=0.1, skip=True):
        super(GAT, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.feat_dim = feat_dim # modify
        self.n_head = n_head
        self.skip = skip
        self.w = nn.Parameter(torch.Tensor(n_head, in_feat, out_feat))
        self.a_src = nn.Parameter(torch.Tensor(n_head, out_feat, self.feat_dim))
        self.a_dst = nn.Parameter(torch.Tensor(n_head, out_feat, self.feat_dim))
        self.bias = nn.Parameter(torch.Tensor(out_feat))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)
        nn.init.constant_(self.bias, 0)

    def forward(self, h, mask=None):
        h_prime = h.unsqueeze(1) @ self.w
        attn_src = h_prime @ self.a_src
        attn_dst = h_prime @ self.a_dst
        attn = attn_src @ attn_dst.permute(0, 1, 3, 2)
        attn = self.leaky_relu(attn)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        attn = attn * mask if mask is not None else attn
        out = (attn @ h_prime).sum(dim=1) + self.bias
        if self.skip:
            out += h_prime.sum(dim=1)
        return out, attn

class MLP(nn.Module):
    def __init__(self, in_feat, out_feat, hid_feat=(1024, 512), activation=None, dropout=-1):
        super(MLP, self).__init__()
        dims = (in_feat, ) + hid_feat + (out_feat, )

        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

        self.activation = activation if activation is not None else lambda x: x
        self.dropout = nn.Dropout(dropout) if dropout != -1 else lambda x: x

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.activation(x)
            x = self.dropout(x)
            x = self.layers[i](x)
        return x

class NPSN(nn.Module):
    def __init__(self, hidden_dim=512, dec_steps=12, s=2, n=20, dropout=0.0, init=True):
        super(NPSN, self).__init__()
        self.s, self.n = s, n
        # self.obs_len = t_obs
        # self.input_dim = t_obs * 2
        # self.hidden_dim = self.input_dim * 1
        self.hidden_dim = hidden_dim
        self.dec_steps = dec_steps
        self.output_dim = s * n
        # self.lstm_dim = 64
        self.dropout=nn.Dropout(dropout)

        self.hidden_emb = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.goal_emb = nn.Sequential(
            nn.Linear(self.dec_steps*self.s, self.hidden_dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.output = nn.Linear(self.hidden_dim*2, self.s*self.n)

        if init:
            # 1. 可以尝试Torch官方提供的一种初始化策略 Initialization strategy by Pytorch official
            # nn.init.uniform_(self.hidden_emb[0].weight, -math.sqrt(1 / self.hidden_dim), math.sqrt(1 / self.hidden_dim))
            # nn.init.uniform_(self.hidden_emb[0].bias, -math.sqrt(1 / self.hidden_dim), math.sqrt(1 / self.hidden_dim))
            # nn.init.uniform_(self.goal_emb[0].weight, -math.sqrt(1 / self.dec_steps*self.s), math.sqrt(1 / self.dec_steps*self.s))
            # nn.init.uniform_(self.goal_emb[0].bias, -math.sqrt(1 / self.dec_steps*self.s), math.sqrt(1 / self.dec_steps*self.s))
            # nn.init.uniform_(self.output.weight, -math.sqrt(1 / self.self.hidden_dim*2), math.sqrt(1 / self.hidden_dim*2))
            # nn.init.uniform_(self.output.bias, -math.sqrt(1 / self.self.hidden_dim*2), math.sqrt(1 / self.hidden_dim*2))

            # 2. 根据场景不同使用xavier和kaiming初始化: Use Xavier and Kaiming(不如3)
            # nn.init.kaiming_normal_(self.hidden_emb[0].weight)
            # nn.init.kaiming_normal_(self.goal_emb[0].weight)
            # nn.init.kaiming_normal_(self.output.weight)

            # 3. normal_初始化也可以得到很好的结果: normal_
            nn.init.normal_(self.hidden_emb[0].weight, std=0.01)
            nn.init.normal_(self.goal_emb[0].weight, std=0.01)
            nn.init.normal_(self.output.weight, std=0.01)

    def forward(self, enc_hidden, goal_traj): # x: obs
        r'''

        Args:
            enc_hidden: [bs, 512] encoder Ht with obs-len steps
            goal_traj: [bs, 12, 2]
        Returns:
            loc: [bs, 20, 2] each element should be in [0, 1]
        '''

        # debug
        hidden_embedding = self.hidden_emb(enc_hidden)
        hidden_embedding = self.dropout(hidden_embedding)
        goal_embedding = self.goal_emb(goal_traj.view(goal_traj.size(0), -1)) # 12 * 2 -> 512
        goal_embedding = self.dropout(goal_embedding)
        feature = torch.cat((hidden_embedding, goal_embedding), dim=-1) # 1024
        # feature = goal_embedding
        output = self.output(feature) # 1024 -> 40
        # output = torch.tanh(output)
        # 后面有sigmoid，此处不能再用tanh!
        # tanh[-1, 1]常用在隐藏层激活，sigmoid[0, 1]常用在输出层激活

        return output.view(-1, self.n, self.s).sigmoid().clamp(min=0.01, max=0.99)

        # mask = self.get_scene_mask(x.size(1), seq_start_end) if seq_start_end is not None else mask
        # node = x.reshape(x.size(0), x.size(1), -1)
        # # change
        # # strategy: lstm add feat64, post
        # node, edge = self.graph_attention(node, mask)
        # lstm_input = x.reshape(-1, x.shape[2], x.shape[3]).permute(0, 2, 1) # b*n seq feature
        # lstm_output, state = self.lstm(lstm_input)
        # lstm_feature = self.lstm_linear(lstm_output)
        # lstm_feature = lstm_feature.reshape(x.shape[0], x.shape[1], x.shape[3], -1)[:, :, -1, :]
        # feature = node + lstm_feature
        # if not global_noise:
        #     # out = self.linear(node).reshape(x.size(0), x.size(1), self.n, -1)
        #     out = self.linear(feature).sigmoid() # 1, num, 40
        #     out, latent_edge = self.post_process(out)
        #     out = out.reshape(x.size(0), x.size(1), self.n, -1)
        # else:
        #     node_ = torch.zeros((node.size(0), seq_start_end.size(0), node.size(2)), device='cuda')
        #     for i, (start, end) in enumerate(seq_start_end):
        #         node_[:, i] = node[:, start:end].mean(dim=1)
        #     out = self.linear(node_).reshape(x.size(0), seq_start_end.size(0), self.n, -1)
        #
        # return out[..., 0:self.s].sigmoid().clamp(min=0.01, max=0.99)

    def get_loss(self, loc, mu, cov, gt):
        r'''
        Args:
            loc: [bs, 20, 2]
            mu: [bs, obs_len, 12, 2]
            cov: [bs, obs_len, 12, 2, 2]
            gt: [bs, 12, 2]
        Returns:
        '''
        loc_norm = box_muller_transform(loc).permute(1, 0, 2).unsqueeze(2).expand((loc.size(1),) + mu.shape)
        p_sample = mu + (torch.cholesky(cov) @ loc_norm.unsqueeze(dim=-1)).squeeze(dim=-1)

        # Distance loss
        loss_dist = (p_sample.mean(dim=2) - gt.unsqueeze(0).mean(dim=2)).norm(p=2, dim=-1).min(dim=0)[0]
        loss_dist = loss_dist.mean()

        # Discrepancy loss
        loss_disc = (loc.unsqueeze(dim=1) - loc.unsqueeze(dim=2)).norm(p=2, dim=-1)
        loss_disc = loss_disc.topk(k=2, dim=-1, largest=False, sorted=True)[0][..., 1].log().mul(-1).mean(dim=-1)
        loss_disc = loss_disc.mean()

        return loss_dist, loss_disc

# Original NPSN code please refers to https://github.com/inhwanbae/NPSN
class NPSNOriginal(nn.Module):
    def __init__(self, t_obs=8, s=2, n=20):
        super(NPSNOriginal, self).__init__()
        self.s, self.n = s, n
        self.input_dim = t_obs * 2
        self.hidden_dim = self.input_dim * 1
        self.output_dim = s * n

        self.graph_attention = GAT(self.input_dim, self.hidden_dim)
        self.linear = MLP(self.hidden_dim, self.output_dim, (16, 64), activation=nn.ReLU())

    def get_scene_mask(self, peds, seq_start_end):
        mask = torch.zeros((peds, peds), device='cuda')
        for (start, end) in seq_start_end:
            mask[start:end, start:end] = 1

    def forward(self, x, seq_start_end=None, mask=None, global_noise=False):
        mask = self.get_scene_mask(x.size(1), seq_start_end) if seq_start_end is not None else mask
        node = x.reshape(x.size(0), x.size(1), -1)
        node, edge = self.graph_attention(node, mask)

        if not global_noise:
            out = self.linear(node).reshape(x.size(0), x.size(1), self.n, -1)
        else:
            node_ = torch.zeros((node.size(0), seq_start_end.size(0), node.size(2)), device='cuda')
            for i, (start, end) in enumerate(seq_start_end):
                node_[:, i] = node[:, start:end].mean(dim=1)
            out = self.linear(node_).reshape(x.size(0), seq_start_end.size(0), self.n, -1)
        # return out[..., 0:self.s].sigmoid().clamp(min=0.01, max=0.99)
        res = out[..., 0:self.s].sigmoid().clamp(min=0.01, max=0.99)
        return res.squeeze(0)[0]

    def get_loss(self, loc, mu, cov, gt):
        r'''
        Args:
            loc: [bs, 20, 2]
            mu: [bs, obs_len, 12, 2]
            cov: [bs, obs_len, 12, 2, 2]
            gt: [bs, 12, 2]
        Returns:
        '''
        loc_norm = box_muller_transform(loc).permute(1, 0, 2).unsqueeze(2).expand((loc.size(1),) + mu.shape)
        p_sample = mu + (torch.cholesky(cov) @ loc_norm.unsqueeze(dim=-1)).squeeze(dim=-1)

        # Distance loss
        loss_dist = (p_sample.mean(dim=2) - gt.unsqueeze(0).mean(dim=2)).norm(p=2, dim=-1).min(dim=0)[0]
        loss_dist = loss_dist.mean()

        # Discrepancy loss
        loss_disc = (loc.unsqueeze(dim=1) - loc.unsqueeze(dim=2)).norm(p=2, dim=-1)
        loss_disc = loss_disc.topk(k=2, dim=-1, largest=False, sorted=True)[0][..., 1].log().mul(-1).mean(dim=-1)
        loss_disc = loss_disc.mean()

        return loss_dist, loss_disc

class NPSNOrigAndNew(nn.Module):
    def __init__(self, t_obs=8, s=2, n=20):
        super(NPSNOrigAndNew, self).__init__()
        self.s, self.n = s, n
        self.input_dim = t_obs * 2
        self.hidden_dim = self.input_dim * 1
        self.output_dim = s * n

        self.graph_attention = GAT(self.input_dim, self.hidden_dim)
        self.linear = MLP(self.hidden_dim, self.output_dim, (16, 64), activation=nn.ReLU())

    def get_scene_mask(self, peds, seq_start_end):
        mask = torch.zeros((peds, peds), device='cuda')
        for (start, end) in seq_start_end:
            mask[start:end, start:end] = 1

    def forward(self, x, seq_start_end=None, mask=None, global_noise=False):
        mask = self.get_scene_mask(x.size(1), seq_start_end) if seq_start_end is not None else mask
        node = x.reshape(x.size(0), x.size(1), -1)
        node, edge = self.graph_attention(node, mask)

        if not global_noise:
            out = self.linear(node).reshape(x.size(0), x.size(1), self.n, -1)
        else:
            node_ = torch.zeros((node.size(0), seq_start_end.size(0), node.size(2)), device='cuda')
            for i, (start, end) in enumerate(seq_start_end):
                node_[:, i] = node[:, start:end].mean(dim=1)
            out = self.linear(node_).reshape(x.size(0), seq_start_end.size(0), self.n, -1)
        # return out[..., 0:self.s].sigmoid().clamp(min=0.01, max=0.99)
        res = out[..., 0:self.s].sigmoid().clamp(min=0.01, max=0.99)
        return res.squeeze(0)[0]

    def get_loss(self, loc, mu, cov, gt):
        r'''
        Args:
            loc: [bs, 20, 2]
            mu: [bs, obs_len, 12, 2]
            cov: [bs, obs_len, 12, 2, 2]
            gt: [bs, 12, 2]
        Returns:
        '''
        loc_norm = box_muller_transform(loc).permute(1, 0, 2).unsqueeze(2).expand((loc.size(1),) + mu.shape)
        p_sample = mu + (torch.cholesky(cov) @ loc_norm.unsqueeze(dim=-1)).squeeze(dim=-1)

        # Distance loss
        loss_dist = (p_sample.mean(dim=2) - gt.unsqueeze(0).mean(dim=2)).norm(p=2, dim=-1).min(dim=0)[0]
        loss_dist = loss_dist.mean()

        # Discrepancy loss
        loss_disc = (loc.unsqueeze(dim=1) - loc.unsqueeze(dim=2)).norm(p=2, dim=-1)
        loss_disc = loss_disc.topk(k=2, dim=-1, largest=False, sorted=True)[0][..., 1].log().mul(-1).mean(dim=-1)
        loss_disc = loss_disc.mean()

        return loss_dist, loss_disc

# if __name__ == '__main__':
#     model = NPSN(t_obs=8, s=2, n=20).cuda()
#     output = model(torch.ones(size=(1, 3, 2, 8)).cuda())
#     print(output.shape)  # torch.Size([1, 3, 20, 2])
