import torch
import torch.nn as nn
from .spd import SPDTransform, SPDTangentSpace, SPDRectified
from omegaconf import DictConfig
from ..base import BaseModel
import numpy as np
#from ..BNT import .ptdec
from .ptdec import DEC


class signal2spd(nn.Module):
    # convert signal epoch to SPD matrix
    def __init__(self):
        super().__init__()
        self.dev = torch.device('cuda')

    def forward(self, x):

        x = x.squeeze()
        mean = x.mean(axis=-1).unsqueeze(-1).repeat(1, 1, x.shape[-1])
        x = x - mean
        cov = x@x.permute(0, 2, 1)
        cov = cov.to(self.dev)
        cov = cov/(x.shape[-1]-1)
        tra = cov.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        tra = tra.view(-1, 1, 1)
        cov /= tra
        identity = torch.eye(
            cov.shape[-1], cov.shape[-1], device=self.dev).to(self.dev).repeat(x.shape[0], 1, 1)
        cov = cov+(1e-5*identity)
        return cov


# class E2R(nn.Module):
#     def __init__(self, epochs):
#         super().__init__()
#         self.epochs = epochs
#         self.signal2spd = signal2spd()

#     def patch_len(self, n, epochs):
#         list_len = []
#         base = n//epochs
#         for i in range(epochs):
#             list_len.append(base)
#         for i in range(n - base*epochs):
#             list_len[i] += 1

#         if sum(list_len) == n:
#             return list_len
#         else:
#             return ValueError('check your epochs and axis should be split again')
class E2R(nn.Module):
    def __init__(self, window_sz, stride):
        super().__init__()
        self.window_sz = window_sz
        self.stride = stride
        self.signal2spd = signal2spd()

    def patch_len(self, n, window_sz):
        list_len = []
        base = n//window_sz
        for i in range(window_sz):
            list_len.append(base)
        for i in range(n - base*window_sz):
            list_len[i] += 1

        if sum(list_len) == n:
            return list_len
        else:
            return ValueError('check your epochs and axis should be split again')
    
    def window_split(self, x):
        dim = len(x.shape) - 1
        length = x.size(dim)
        splits = []
        for slice_start in range(0, length, self.stride):
            if slice_start + self.window_sz >= length:
                splits.append(x[...,-self.window_sz:])
                break
            splits.append(x[...,slice_start:slice_start + self.window_sz])
        return splits

    def forward(self, x):
        x_list = self.window_split(x)
        for i, item in enumerate(x_list):
            x_list[i] = self.signal2spd(item)
        x = torch.stack(x_list).permute(1, 0, 2, 3)
        
        return x


class AttentionManifold(nn.Module):
    def __init__(self, in_embed_size, out_embed_size, config: DictConfig):
        super(AttentionManifold, self).__init__()

        self.config = config
        self.d_in = in_embed_size
        self.d_out = out_embed_size
        self.device = torch.device('cuda')
        self.q_trans = SPDTransform(self.d_in, self.d_out).cuda()
        self.k_trans = SPDTransform(self.d_in, self.d_out).cuda()
        self.v_trans = SPDTransform(self.d_in, self.d_out).cuda()
        

    def tensor_log(self, t):  # 4dim
        #u, s = torch.svd(t)
        #s, u = torch.linalg.eigh(t)
        u, s, v = torch.svd(t)
        s = s + 1e-6
        return u @ torch.diag_embed(torch.log(s)) @ v.permute(0, 1, 3, 2)

    def tensor_exp(self, t):  # 4dim
        # condition: t is symmetric!
        s, u = torch.linalg.eigh(t)
        r = u @ torch.diag_embed(torch.exp(s)) @ u.permute(0, 1, 3, 2)
        return r

    def log_euclidean_distance(self, A, B):
        inner_term = self.tensor_log(A) - self.tensor_log(B)
        inner_multi = inner_term @ inner_term.permute(0, 1, 3, 2)
        _, s, _ = torch.svd(inner_multi)
        s = s + 1e-6
        final = torch.sum(s, dim=-1)
        return final

    def LogEuclideanMean(self, weight, cov):
        # cov:[bs, #p, s, s]
        # weight:[bs, #p, #p]
        bs = cov.shape[0]
        num_p = cov.shape[1]
        size = cov.shape[2]
        cov = self.tensor_log(cov).view(bs, num_p, -1)
        output = weight @ cov  # [bs, #p, -1]
        output = output.view(bs, num_p, size, size)
        return self.tensor_exp(output)

    def forward(self, x, shape=None):
        if len(x.shape) == 3 and shape is not None:
            x = x.view(shape[0], shape[1], self.d_in, self.d_in)
        x = x.to(torch.float)  # patch:[b, #patch, c, c]
        q_list, k_list, v_list = [], [], []
        # calculate Q K V
        bs = x.shape[0]
        m = x.shape[1]
        x = x.reshape(bs*m, self.d_in, self.d_in)  # 48, 360, 360
        Q = self.q_trans(x).view(bs, m, self.d_out, self.d_out) # 16, 3, 180, 180
        K = self.k_trans(x).view(bs, m, self.d_out, self.d_out) # 16, 3, 180, 180
        V = self.v_trans(x).view(bs, m, self.d_out, self.d_out) # 16, 3, 180, 180

        # calculate the attention score
        Q_expand = Q.repeat(1, V.shape[1], 1, 1)  # 16, 9, 180, 180

        K_expand = K.unsqueeze(2).repeat(1, 1, V.shape[1], 1, 1) # 16, 3, 3, 180, 180
        K_expand = K_expand.view(
            K_expand.shape[0], K_expand.shape[1] * K_expand.shape[2], K_expand.shape[3], K_expand.shape[4]) # 16, 9, 180, 180

        atten_energy = self.log_euclidean_distance(
            Q_expand, K_expand).view(V.shape[0], V.shape[1], V.shape[1]) # 16, 3, 3
        # now row is c.c.
        atten_prob = nn.Softmax(
            dim=-2)(1/(1+torch.log(1 + atten_energy))).permute(0, 2, 1) # 16, 3, 3

        # calculate outputs(v_i') of attention module
        output = self.LogEuclideanMean(atten_prob, V) # 16, 3, 180, 180

        output = output.view(V.shape[0], V.shape[1], self.d_out, self.d_out) # 16, 3, 180, 180

        shape = list(output.shape[:2]) # 16, 3
        shape.append(-1)

        output = output.contiguous().view(-1, self.d_out, self.d_out) #48, 90, 90  => 16 * 90 * 270
        return output, shape


class MATT(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__()
        #num_matrics = config.dataset.timeseries_sz // config.model.window_sz + 1
        num_matrics = int(np.ceil((config.dataset.timeseries_sz - config.model.window_sz) / config.model.stride)) + 1
        
        hid_mat_dim1 = config.dataset.node_sz
        hid_mat_dim2 = hid_mat_dim1 // 2
        hid_mat_dim3 = hid_mat_dim2 // 2
        #hid_mat_dim4 = ((hid_mat_dim3 + 1) * hid_mat_dim3)

        # FE
        # bs, 1, channel, sample
        self.conv1 = nn.Conv2d(1, hid_mat_dim1, (hid_mat_dim1, 1))
        self.Bn1 = nn.BatchNorm2d(hid_mat_dim1)
        # bs, 22, 1, sample
        self.conv2 = nn.Conv2d(hid_mat_dim1, hid_mat_dim2,
                            (1, 12), padding=(0, 6))
        self.Bn2 = nn.BatchNorm2d(hid_mat_dim2)

        # E2R
        #self.ract1 = E2R(epochs=num_matrics)
        self.ract1 = E2R(config.model.window_sz, config.model.stride)
        # riemannian part
        self.att2 = AttentionManifold(hid_mat_dim2, hid_mat_dim3, config)
        self.ract2 = SPDRectified()

        # R2E
        self.tangent = SPDTangentSpace(hid_mat_dim3, False)
        self.flat = nn.Flatten()
        
        # pooling
        # output_node_num = 16
        # input_feature_size = 90
        # encoder_hidden_size = input_feature_size * num_matrics
        # encoder = nn.Sequential(
        #     nn.Linear(input_feature_size *
        #                 input_feature_size * num_matrics, encoder_hidden_size), # 270
        #     nn.LeakyReLU(),
        #     nn.Linear(encoder_hidden_size, encoder_hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(encoder_hidden_size,
        #                 input_feature_size * input_feature_size * num_matrics),
        # )
        # # 48, 90, 90    16, 90, 270 => 16, 16, 270
        # # [16, 360, 360]  => [16, 100, 360] (dec)  => [16, 100, 8] (dim_reduction) => fc 16,800
        # # cluster num = 16   hidden dim 270?
        # self.dec = DEC(cluster_number=output_node_num, hidden_dimension=input_feature_size * num_matrics, encoder=encoder,
        #                    orthogonal=config.model.orthogonal, freeze_center=config.model.freeze_center, 
        #                    project_assignment=config.model.project_assignment)

        # lstm_v2_out
        self.lstm_v2_out1 = nn.LSTM(input_size = hid_mat_dim1, hidden_size = hid_mat_dim1, batch_first = True)
        self.lstm_v2_out2 = nn.LSTM(input_size = hid_mat_dim1, hidden_size = hid_mat_dim2, batch_first = True)
        self.Bn1d1 = nn.BatchNorm1d(hid_mat_dim1)
        self.Bn1d2 = nn.BatchNorm1d(hid_mat_dim2)

        output_dim = 1 if config.dataset.regression else config.dataset.num_classes
        # fc orig
        self.linear = nn.Linear(
            hid_mat_dim3 * hid_mat_dim3 * num_matrics, output_dim, bias=True)
        # fc dec
        # self.linear = nn.Linear(
        #     output_node_num * encoder_hidden_size, output_dim, bias=True)

    # def forward(self,
    #             time_series: torch.tensor,
    #             node_feature: torch.tensor):
    #     x = time_series.unsqueeze(1)
    #     x = self.conv1(x)
    #     x = self.Bn1(x)
    #     x = self.conv2(x)
    #     x = self.Bn2(x)

    #     #torch.Size([16, 180, 1, 513])
    #     x = self.ract1(x)
    #     x, shape = self.att2(x)
    #     # [48, 90, 90]
    #     x = self.ract2(x)

    #     x = self.tangent(x)
    #     #x = x.view(shape[0], shape[1], -1)   # 16, 3, -1
    #     x = x.view(shape[0], x.shape[1], -1)
    #     # x, assignment = self.dec(x)
    #     x = self.flat(x)
    #     x = self.linear(x)
    #     return x
    
    # ver 2 lstm out
    def forward(self,
                time_series: torch.tensor,
                node_feature: torch.tensor):
        #x = time_series.unsqueeze(1)
        x = time_series
        
        x = x.permute(0,2,1)
        x, (h01, c01) = self.lstm_v2_out1(x)
        x = x.permute(0,2,1)
        x = self.Bn1d1(x)
        x = x.permute(0,2,1)
        
        x, (h02, c02) = self.lstm_v2_out2(x)
        x = x.permute(0,2,1)
        x = self.Bn1d2(x)

        x = self.ract1(x)
        x, shape = self.att2(x)
        # [48, 90, 90]
        x = self.ract2(x)

        x = self.tangent(x)
        #x = x.view(shape[0], shape[1], -1)   # 16, 3, -1
        x = x.view(shape[0], x.shape[1], -1)
        # x, assignment = self.dec(x)
        x = self.flat(x)
        x = self.linear(x)
        return x

