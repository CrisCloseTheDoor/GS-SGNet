import torch
import torch.nn as nn
from .feature_extractor import build_feature_extractor
import torch.nn.functional as F
import math


class SGNet(nn.Module):
    def __init__(self, args):
        super(SGNet, self).__init__()

        self.hidden_size = args.hidden_size
        self.enc_steps = args.enc_steps
        self.dec_steps = args.dec_steps
        self.dataset = args.dataset
        self.dropout = args.dropout
        self.feature_extractor = build_feature_extractor(args)
        if self.dataset in ['JAAD','PIE']:
            self.pred_dim = 4
            self.regressor = nn.Sequential(nn.Linear(self.hidden_size, self.pred_dim),
                                                    nn.Tanh())
        elif self.dataset in ['ETH', 'HOTEL','UNIV','ZARA1', 'ZARA2']:
            self.pred_dim = 2
            self.regressor = nn.Sequential(nn.Linear(self.hidden_size, 
                                                        self.pred_dim))  
             
        self.enc_goal_attn = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                1),
                                                nn.ReLU(inplace=True))
        self.dec_goal_attn = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                1),
                                                nn.ReLU(inplace=True))

        self.enc_to_goal_hidden = nn.Sequential(nn.Linear(self.hidden_size,
                                                self.hidden_size//4),
                                                nn.ReLU(inplace=True))
        self.enc_to_dec_hidden = nn.Sequential(nn.Linear(self.hidden_size,
                                                self.hidden_size),
                                                nn.ReLU(inplace=True))


        self.goal_hidden_to_input = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    self.hidden_size//4),
                                                    nn.ReLU(inplace=True))
        self.dec_hidden_to_input = nn.Sequential(nn.Linear(self.hidden_size,
                                                    self.hidden_size),
                                                    nn.ReLU(inplace=True))
        self.goal_hidden_to_traj = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    self.hidden_size),
                                                    nn.ReLU(inplace=True))
        self.goal_to_enc = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    self.hidden_size//4),
                                                    nn.ReLU(inplace=True))
        self.goal_to_dec = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    self.hidden_size//4),
                                                    nn.ReLU(inplace=True))
        self.enc_drop = nn.Dropout(self.dropout)
        self.goal_drop = nn.Dropout(self.dropout)
        self.dec_drop = nn.Dropout(self.dropout)

        # self.traj_enc_cell = nn.GRUCell(self.hidden_size + self.hidden_size//4, self.hidden_size)
        self.traj_enc_cell = nn.GRUCell(self.hidden_size + self.hidden_size // 4 + self.hidden_size, self.hidden_size)
        self.goal_cell = nn.GRUCell(self.hidden_size//4, self.hidden_size//4)
        self.dec_cell = nn.GRUCell(self.hidden_size + self.hidden_size//4, self.hidden_size)

        self.dist_feat_extractor = nn.Sequential(
            nn.Linear(1, self.hidden_size//2),
            nn.ReLU(inplace=True)
        )
        self.social_cell = NeighborCell(input_size=args.hidden_size, hidden_size=args.hidden_size, bias=True)
        self.social_attention = MultiHeadSocialAttention(hidden_dim=args.hidden_size, dropout=self.dropout)

    def SGE(self, goal_hidden):
        goal_input = goal_hidden.new_zeros((goal_hidden.size(0), self.hidden_size//4))
        goal_traj = goal_hidden.new_zeros(goal_hidden.size(0), self.dec_steps, self.pred_dim)
        goal_list = []
        for dec_step in range(self.dec_steps):
            goal_hidden = self.goal_cell(self.goal_drop(goal_input), goal_hidden)
            goal_input = self.goal_hidden_to_input(goal_hidden)
            goal_list.append(goal_hidden)
            goal_traj_hidden = self.goal_hidden_to_traj(goal_hidden)
            # regress goal traj for loss
            goal_traj[:,dec_step,:] = self.regressor(goal_traj_hidden)
        # get goal for decoder and encoder
        goal_for_dec = [self.goal_to_dec(goal) for goal in goal_list]
        goal_for_enc = torch.stack([self.goal_to_enc(goal) for goal in goal_list],dim = 1)
        enc_attn= self.enc_goal_attn(torch.tanh(goal_for_enc)).squeeze(-1)
        enc_attn = F.softmax(enc_attn, dim =1).unsqueeze(1)
        goal_for_enc  = torch.bmm(enc_attn, goal_for_enc).squeeze(1)
        return goal_for_dec, goal_for_enc, goal_traj

    def decoder(self, dec_hidden, goal_for_dec):
        # initial trajectory tensor
        dec_traj = dec_hidden.new_zeros(dec_hidden.size(0), self.dec_steps, self.pred_dim)
        for dec_step in range(self.dec_steps):
            goal_dec_input = dec_hidden.new_zeros(dec_hidden.size(0), self.dec_steps, self.hidden_size//4)
            goal_dec_input_temp = torch.stack(goal_for_dec[dec_step:],dim=1)
            goal_dec_input[:,dec_step:,:] = goal_dec_input_temp
            dec_attn= self.dec_goal_attn(torch.tanh(goal_dec_input)).squeeze(-1)
            dec_attn = F.softmax(dec_attn, dim =1).unsqueeze(1)
            goal_dec_input  = torch.bmm(dec_attn,goal_dec_input).squeeze(1)#.view(goal_hidden.size(0), self.dec_steps, self.hidden_size//4).sum(1)
            
            
            dec_dec_input = self.dec_hidden_to_input(dec_hidden)
            dec_input = self.dec_drop(torch.cat((goal_dec_input,dec_dec_input),dim = -1))
            dec_hidden = self.dec_cell(dec_input, dec_hidden)
            # regress dec traj for loss
            dec_traj[:, dec_step, :] = self.regressor(dec_hidden)

        return dec_traj
        
    def encoder(self, traj_input, flow_input=None, start_index = 0, neighbors_data_st=None,
                        neighbors_edge_value=None, neighbors_idx_start=None, neighbors_idx_end=None):
        # initial output tensor
        all_goal_traj = traj_input.new_zeros(traj_input.size(0), self.enc_steps, self.dec_steps, self.pred_dim)
        all_dec_traj = traj_input.new_zeros(traj_input.size(0), self.enc_steps, self.dec_steps, self.pred_dim)
        # initial encoder goal with zeros
        goal_for_enc = traj_input.new_zeros((traj_input.size(0), self.hidden_size//4))
        # initial encoder hidden with zeros
        traj_enc_hidden = traj_input.new_zeros((traj_input.size(0), self.hidden_size))
        # initial social hidden with zeros
        social_ht = traj_input.new_zeros((neighbors_data_st.size(0), self.hidden_size))
        # initial social aggregated feature with zeros
        aggregated_social_feat = traj_input.new_zeros((traj_input.size(0), self.hidden_size))

        for enc_step in range(start_index, self.enc_steps):
            
            traj_enc_hidden = self.traj_enc_cell(self.enc_drop(torch.cat((traj_input[:,enc_step,:],
                                                                          goal_for_enc,
                                                                          aggregated_social_feat), 1)),
                                                 traj_enc_hidden)
            if self.dataset in ['JAAD','PIE', 'ETH', 'HOTEL','UNIV','ZARA1', 'ZARA2']:
                enc_hidden = traj_enc_hidden
            # generate hidden states for goal and decoder

            # TO DO: Neighbor
            social_ht = self.social_cell(traj_input[:,enc_step,:],
                                          neighbors_data_st[:, enc_step,:],
                                          neighbors_edge_value[:, enc_step,:],
                                          neighbors_idx_start,
                                          neighbors_idx_end,
                                          social_ht)

            aggregated_social_feat = self.social_attention(traj_enc_hidden, social_ht,
                                                         neighbors_idx_start, neighbors_idx_end)

            goal_hidden = self.enc_to_goal_hidden(enc_hidden)
            dec_hidden = self.enc_to_dec_hidden(enc_hidden)

            goal_for_dec, goal_for_enc, goal_traj = self.SGE(goal_hidden)
            dec_traj = self.decoder(dec_hidden, goal_for_dec)

            # output 
            all_goal_traj[:,enc_step,:,:] = goal_traj
            all_dec_traj[:,enc_step,:,:] = dec_traj
        
        return all_goal_traj, all_dec_traj
            

    def forward(self, inputs, start_index = 0, neighbors_data_st = None,
                        neighbors_edge_value = None, neighbors_idx_start = None, neighbors_idx_end = None):
        if self.dataset in ['JAAD','PIE']:
            traj_input = self.feature_extractor(inputs)
            all_goal_traj, all_dec_traj = self.encoder(traj_input)
            return all_goal_traj, all_dec_traj
        elif self.dataset in ['ETH', 'HOTEL','UNIV','ZARA1', 'ZARA2']:
            traj_input_temp = self.feature_extractor(inputs[:,start_index:,:])
            traj_input = traj_input_temp.new_zeros((inputs.size(0), inputs.size(1), traj_input_temp.size(-1)))
            traj_input[:, start_index:, :] = traj_input_temp
            if neighbors_edge_value is not None:
                neighbors_edge_value_temp = self.dist_feat_extractor(neighbors_edge_value)
                neighbors_edge_value_input = neighbors_edge_value_temp.new_zeros((neighbors_edge_value_temp.size(0), inputs.size(1), neighbors_edge_value_temp.size(-1)))
                neighbors_edge_value_input[:, start_index:, :] = neighbors_edge_value_temp

                neighbors_data_st_temp = self.feature_extractor(neighbors_data_st[:,start_index:,:])
                neighbors_edge_st_input = neighbors_data_st_temp.new_zeros((neighbors_data_st_temp.size(0), inputs.size(1), neighbors_data_st_temp.size(-1)))
                neighbors_edge_st_input[:, start_index:, :] = neighbors_data_st_temp
            else:
                neighbors_edge_value_input = None
                neighbors_edge_st_input = None

            all_goal_traj, all_dec_traj = self.encoder(traj_input, None, start_index, neighbors_edge_st_input,
                        neighbors_edge_value_input, neighbors_idx_start, neighbors_idx_end)
            return all_goal_traj, all_dec_traj

class NeighborCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(NeighborCell, self).__init__()

        self.dist_dim = 256
        self.rnn_input_dim = hidden_size

        self.social_emb = nn.Sequential(nn.Linear(input_size + hidden_size + self.dist_dim, self.rnn_input_dim),
                                      nn.ReLU(inplace=True)
                                        )
        self.cell = nn.GRUCell(self.rnn_input_dim, hidden_size, bias)

    def forward(self, traj_input, neighbor_t, dist, neighbors_idx_start, neighbors_idx_end,ht):
        batch_results = []
        for i in range(neighbors_idx_start.shape[0]):
            start_id = neighbors_idx_start[i]
            end_id = neighbors_idx_end[i]
            if start_id == end_id:
                continue
            neighbor_feature_one_sample = neighbor_t[start_id:end_id, :]
            neighbor_dist_one_sample = dist[start_id:end_id, :]
            target_traj_one_sample = traj_input[i].unsqueeze(0)
            ht_one_sample = ht[start_id:end_id]
            rnn_input = torch.cat((neighbor_feature_one_sample,
                                   target_traj_one_sample.repeat(neighbor_feature_one_sample.shape[0], 1),
                                   neighbor_dist_one_sample), dim=-1)
            rnn_input = self.social_emb(rnn_input)
            batch_results.append(self.cell(rnn_input, ht_one_sample))
        return torch.cat(batch_results, dim=0)

class SocialAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads=4):
        super(SocialAttention, self).__init__()
        self.feat_dim = hidden_dim
        self.n_heads = n_heads
        self.q_emb = nn.Sequential(
            nn.Linear(hidden_dim, self.feat_dim),
            nn.ReLU()
        )
        self.k_emb = nn.Sequential(
            nn.Linear(hidden_dim, self.feat_dim),
            nn.ReLU()
        )
        self.v_emb = nn.Sequential(
            nn.Linear(hidden_dim, self.feat_dim),
            nn.ReLU()
        )

        self.sfm = nn.Softmax(dim=-1)

    def forward(self, enc_hidden, social_ht, neighbors_idx_start, neighbors_idx_end):
        score_batch = []
        for i in range(neighbors_idx_start.shape[0]):
            enc_hidden_one_sample = enc_hidden[i].unsqueeze(0)
            start_id = neighbors_idx_start[i]
            end_id = neighbors_idx_end[i]
            social_ht_one_sample = social_ht[start_id:end_id, :]

            square_d = math.sqrt(self.feat_dim)
            q = self.q_emb(enc_hidden_one_sample) / square_d
            k = self.k_emb(social_ht_one_sample)
            v = self.v_emb(social_ht_one_sample)

            social_score = torch.matmul(self.sfm(torch.matmul(q, k.permute(1, 0))), v)

            score_batch.append(social_score)

        return torch.cat(score_batch, dim=0)

class MultiHeadSocialAttention(nn.Module):
    def __init__(self, hidden_dim, dropout, n_heads=4):
        super(MultiHeadSocialAttention, self).__init__()
        self.feat_dim = hidden_dim
        self.n_heads = n_heads
        self.q_emb = nn.Sequential(
            nn.Linear(hidden_dim, self.feat_dim),
            nn.ReLU()
        )
        self.k_emb = nn.Sequential(
            nn.Linear(hidden_dim, self.feat_dim),
            nn.ReLU()
        )
        self.v_emb = nn.Sequential(
            nn.Linear(hidden_dim, self.feat_dim),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(dropout)
        self.sfm = nn.Softmax(dim=-1)
        self.fc = nn.Linear(self.feat_dim, self.feat_dim)

    def forward(self, enc_hidden, social_ht, neighbors_idx_start, neighbors_idx_end):
        score_batch = []
        for i in range(neighbors_idx_start.shape[0]):
            enc_hidden_one_sample = enc_hidden[i].unsqueeze(0)
            start_id = neighbors_idx_start[i]
            end_id = neighbors_idx_end[i]
            social_ht_one_sample = social_ht[start_id:end_id, :]

            square_d = math.sqrt(self.feat_dim)
            q = self.q_emb(enc_hidden_one_sample) / square_d
            k = self.k_emb(social_ht_one_sample)
            v = self.v_emb(social_ht_one_sample)

            q = q.view(-1, self.n_heads, self.feat_dim // self.n_heads).permute(1, 0, 2)
            k = k.view(-1, self.n_heads, self.feat_dim // self.n_heads).permute(1, 0, 2)
            v = v.view(-1, self.n_heads, self.feat_dim // self.n_heads).permute(1, 0, 2)

            qk = self.sfm(torch.matmul(q, k.transpose(2, 1)))
            social_score = torch.matmul(self.dropout(qk), v)
            social_score = social_score.permute(1, 0, 2).contiguous()
            social_score = social_score.view(-1, self.n_heads * (self.feat_dim // self.n_heads))
            social_score = self.fc(social_score)

            score_batch.append(social_score)

        return torch.cat(score_batch, dim=0)
