#!/usr/bin/env python

import torch
from sympy.physics.units import speed
from torch import nn
import torch.nn.functional as F
import math


class Embedding(nn.Module):
    def __init__(self, embd_dim=5, num_pts=200):
        super().__init__()
        self.num_pts = num_pts
        self.embd_dim = embd_dim
        self.embedding = nn.Embedding(num_pts, embd_dim)

    def forward(self, batch_ids):
        pts_embedded = self.embedding(batch_ids)
        return pts_embedded


class AdaptiveTemporalConv(nn.Module):
    """创新点1: 自适应时间卷积 - 动态调整感受野"""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 简化的多尺度卷积设计
        self.conv_short = nn.Conv1d(input_dim, output_dim // 3, kernel_size=3, padding=1)
        self.conv_medium = nn.Conv1d(input_dim, output_dim // 3, kernel_size=5, padding=2)
        self.conv_long = nn.Conv1d(input_dim, output_dim - 2 * (output_dim // 3), kernel_size=7, padding=3)

        # 创新点1: 简化的自适应融合
        self.adaptive_fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, output_dim),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x_transposed = x.transpose(1, 2)

        # 检查输入维度是否匹配
        actual_input_dim = x_transposed.size(1)
        if actual_input_dim != self.input_dim:
            # 如果维度不匹配，使用线性层进行维度调整
            if not hasattr(self, 'input_proj') or self.input_proj is None:
                self.input_proj = nn.Linear(actual_input_dim, self.input_dim).to(x.device)
            x_transposed = self.input_proj(x_transposed.transpose(1, 2)).transpose(1, 2)

        # 多尺度特征提取
        short_feat = F.relu(self.conv_short(x_transposed))
        medium_feat = F.relu(self.conv_medium(x_transposed))
        long_feat = F.relu(self.conv_long(x_transposed))

        # 特征融合
        multi_scale_feat = torch.cat([short_feat, medium_feat, long_feat], dim=1)

        # 转回 (batch, seq_len, output_dim)
        multi_scale_feat = multi_scale_feat.transpose(1, 2)

        # 创新点1: 自适应融合权重
        fusion_weights = self.adaptive_fusion(multi_scale_feat)
        enhanced_feat = multi_scale_feat * fusion_weights

        # 残差连接
        if x.size(1) == enhanced_feat.size(1) and x.size(2) == enhanced_feat.size(2):
            enhanced_feat = enhanced_feat + x

        return self.dropout(enhanced_feat)


class Encoder(nn.Module):
    def __init__(self, enc_dim=60, dec_dim=30, input_dim=8,
                 embedding_layer=None, GRU_LSTM='GRU', is_bidirectional=True):
        super().__init__()
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim

        self.GRU_LSTM = GRU_LSTM
        self.is_bidirectional = is_bidirectional

        self.embedding_layer = embedding_layer
        if embedding_layer is not None:
            embd_dim = embedding_layer.embd_dim
        else:
            embd_dim = 0

        self.input_dim = input_dim + embd_dim

        # 创新点1: 自适应时间特征提取
        self.temporal_conv = AdaptiveTemporalConv(self.input_dim, self.input_dim)

        if GRU_LSTM == 'GRU':
            self.rnn = nn.GRU(self.input_dim, self.enc_dim,
                              bidirectional=is_bidirectional)
        if GRU_LSTM == 'LSTM':
            self.rnn = nn.LSTM(self.input_dim, self.enc_dim,
                               bidirectional=is_bidirectional)

        if self.is_bidirectional:
            self.fc = nn.Linear(enc_dim * 2, dec_dim)
        else:
            self.fc = nn.Linear(enc_dim, dec_dim)

    def forward(self, one_batch):
        '''
        input_batch: enc_len * size * input_dim
        '''
        batch_ids = one_batch[0]

        rnn_input = one_batch[1].permute(1, 0, 2)

        enc_len = rnn_input.size(0)

        if self.embedding_layer is not None:
            pts_embedded = self.embedding_layer(
                batch_ids).unsqueeze(0).repeat(enc_len, 1, 1)
            rnn_input = torch.cat((rnn_input, pts_embedded), dim=2)

        # 应用轻量级时间特征提取
        rnn_input = self.temporal_conv(rnn_input)

        if self.GRU_LSTM == 'GRU':
            outputs, hidden = self.rnn(rnn_input)
        elif self.GRU_LSTM == 'LSTM':
            outputs, (hidden, _) = self.rnn(rnn_input)

        if self.is_bidirectional:
            hidden = torch.tanh(
                self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=-1)))
        else:
            hidden = torch.tanh(self.fc(hidden[-1, :, :]))
        return outputs, hidden


class SpatialAttention(nn.Module):
    """轻量级空间注意力机制，专注于风机间关系"""

    def __init__(self, dec_dim, enc_dim):
        super().__init__()
        self.dec_dim = dec_dim
        self.enc_dim = enc_dim

        # 简化的注意力计算
        self.W_q = nn.Linear(dec_dim, dec_dim)
        self.W_k = nn.Linear(enc_dim, dec_dim)
        self.W_v = nn.Linear(enc_dim, dec_dim)

        # 步数感知的注意力调节
        self.step_adapter = nn.Sequential(
            nn.Linear(1, dec_dim // 4),
            nn.ReLU(),
            nn.Linear(dec_dim // 4, dec_dim)
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, decoder_hidden, encoder_outputs, step):
        # encoder_outputs: [batch_size, num_turbines, enc_dim]
        # decoder_hidden: [batch_size, dec_dim]

        batch_size, num_turbines, _ = encoder_outputs.shape

        # 步数感知的查询增强
        step_ratio = torch.tensor([step / 12.0], device=decoder_hidden.device)
        step_feature = self.step_adapter(step_ratio.unsqueeze(0))

        # 计算注意力
        Q = self.W_q(decoder_hidden + step_feature)  # [batch_size, dec_dim]
        K = self.W_k(encoder_outputs)  # [batch_size, num_turbines, dec_dim]
        V = self.W_v(encoder_outputs)  # [batch_size, num_turbines, dec_dim]

        # 注意力分数
        scores = torch.matmul(K, Q.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.dec_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        context = torch.sum(attn_weights.unsqueeze(-1) * V, dim=1)

        return context, attn_weights


class Decoder(nn.Module):
    def __init__(self, enc_dim=60, dec_dim=30, dec_input_dim=4, enc_len=48,
                 embedding_layer=None, attention_ind=False, GRU_LSTM='LSTM', is_bidirectional=True):
        super().__init__()
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.attention_ind = attention_ind

        self.GRU_LSTM = GRU_LSTM
        self.is_bidirectional = is_bidirectional

        self.embedding_layer = embedding_layer
        if embedding_layer is not None:
            embd_dim = embedding_layer.embd_dim
        else:
            embd_dim = 0
        self.dec_input_dim = dec_input_dim + embd_dim
        if GRU_LSTM == 'GRU':
            self.rnn = nn.GRU(self.dec_input_dim + dec_dim, dec_dim)  # 增加输入维度
        if GRU_LSTM == 'LSTM':
            self.rnn = nn.LSTM(self.dec_input_dim + dec_dim, dec_dim)  # 增加输入维度
        # 空间注意力机制
        self.spatial_attn = SpatialAttention(dec_dim, enc_dim)

        # 注意力融合门控
        self.attn_gate = nn.Sequential(
            nn.Linear(dec_dim * 2, dec_dim),
            nn.Sigmoid()
        )

        # 创新点: 预测步数感知的输出层
        self.step_aware_fc = nn.Sequential(
            nn.Linear(dec_dim + 1, dec_dim),  # +1 for step encoding
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(dec_dim, dec_dim // 2),
            nn.ReLU(),
            nn.Linear(dec_dim // 2, 1)
        )

    def forward(self, one_batch, encoder_outputs, hidden):
        '''
        input: size * input_dim
        batch_ids: size
        '''
        # change size to 1 * size * dim
        batch_ids = one_batch[0]
        y_ = one_batch[2].permute(1, 0, 2)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [seq_len, batch, dim] -> [batch, seq_len, dim]

        # 调整encoder_outputs为空间注意力需要的格式 [batch, turbines, features]
        encoder_spatial = encoder_outputs.mean(dim=1)  # [batch, enc_dim] (时间维度平均)
        encoder_spatial = encoder_spatial.unsqueeze(1)  # [batch, 1, enc_dim]

        rnn_input = y_[[0], :, :]
        rnn_input[rnn_input != rnn_input] = 0

        cell_state = torch.zeros_like(hidden).unsqueeze(0)

        output_list = []
        for i in range(1, 13):  # 预测1-12小时，输出12个时间步
            if self.embedding_layer is not None:
                pts_embedded = self.embedding_layer(batch_ids).unsqueeze(0)
                rnn_input = torch.cat((rnn_input, pts_embedded), dim=2)

            # 使用空间注意力机制
            spatial_context, attn_weights = self.spatial_attn(
                hidden.squeeze(0), encoder_spatial, i
            )

            # 门控机制融合注意力信息
            gate = self.attn_gate(torch.cat([hidden.squeeze(0), spatial_context], dim=1))
            enhanced_hidden = gate * hidden.squeeze(0) + (1 - gate) * spatial_context

            # 准备RNN输入，融入增强的隐藏状态
            rnn_input_combined = torch.cat([
                rnn_input.squeeze(0),
                enhanced_hidden
            ], dim=1).unsqueeze(0)

            # 确保hidden是3D张量用于RNN
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)  # [batch, dim] -> [1, batch, dim]

            if self.GRU_LSTM == 'GRU':
                output, hidden = self.rnn(rnn_input_combined, hidden)
            elif self.GRU_LSTM == 'LSTM':
                output, (hidden, cell_state) = self.rnn(
                    rnn_input_combined, (hidden, cell_state))
            assert (output == hidden).all()
            # 1 * size * dec_dim

            output = output.squeeze(0)
            hidden = hidden.squeeze(0)

            # 创新点: 预测步数感知的输出
            step_encoding = torch.tensor([i / 12.0], device=output.device).expand(output.size(0), 1)
            step_enhanced_input = torch.cat([output, step_encoding], dim=1)

            output = self.step_aware_fc(step_enhanced_input)

            # 修复：使用模运算避免索引越界，实现循环使用输入数据
            next_input_idx = i % y_.size(0)
            rnn_input = y_[next_input_idx, :, 1:]
            rnn_input = torch.cat((rnn_input, output), dim=1)
            rnn_input = rnn_input.unsqueeze(0)

            output_list.append(output)

        y_pred = torch.cat(output_list, dim=1).transpose(0, 1)

        return y_pred


class Seq2Seq(nn.Module):
    def __init__(self, enc_dim=60, dec_dim=30, input_dim=4, K=5, enc_len=48,
                 embedding_dim=5, attention_ind=False, GRU_LSTM='GRU',
                 is_bidirectional=False, n_turbines=200,
                 device=torch.device('cpu')):

        super().__init__()

        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.enc_input_dim = input_dim + K - 1
        self.dec_input_dim = input_dim
        self.device = device

        if embedding_dim > 0:
            self.embedding_layer = Embedding(embedding_dim, num_pts=n_turbines)
        else:
            self.embedding_layer = None

        self.attention_ind = attention_ind

        if GRU_LSTM == 'GRU':
            self.encoder = Encoder(enc_dim, dec_dim, self.enc_input_dim, self.embedding_layer,
                                   GRU_LSTM='GRU', is_bidirectional=is_bidirectional)
            self.decoder = Decoder(enc_dim, dec_dim, self.dec_input_dim, enc_len,
                                   self.embedding_layer, self.attention_ind, GRU_LSTM='GRU',
                                   is_bidirectional=is_bidirectional)
        if GRU_LSTM == 'LSTM':
            self.encoder = Encoder(
                enc_dim, dec_dim, self.enc_input_dim, self.embedding_layer,
                GRU_LSTM='LSTM', is_bidirectional=is_bidirectional)
            self.decoder = Decoder(enc_dim, dec_dim, self.dec_input_dim, enc_len,
                                   self.embedding_layer, self.attention_ind, GRU_LSTM='LSTM',
                                   is_bidirectional=is_bidirectional)

    def forward(self, one_batch):
        encoder_outputs, hidden = self.encoder(one_batch)

        y_pred = self.decoder(one_batch, encoder_outputs, hidden)

        return y_pred
# Test Loss: 1.4940
# Test MAE: [[0.12310838 0.151854   0.17102517 0.18638366 0.19965169 0.21050423]
#  [0.219042   0.22617795 0.23207012 0.23651524 0.24071362 0.24416558]]
# Test RMSE: [[0.16621345 0.19805447 0.21909081 0.23527359 0.24839202 0.25874259]
#  [0.26659216 0.27282744 0.27758134 0.28114744 0.28454371 0.28750632]]
# Done!

# wind speed
# Test Loss: 0.2070
# Test MAE: [[1.03208943 1.82582097 2.395397   2.76706803 3.04248201 3.25884124]
#  [3.43600583 3.58148829 3.71314723 3.83164868 3.92975232 4.01200225]]
# Test RMSE: [[1.18647464 2.38202182 3.07475847 3.5208471  3.84684561 4.10078038]
#  [4.30913619 4.48202775 4.63032103 4.76030228 4.86266578 4.94955942]]
# Done!