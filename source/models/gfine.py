
from omegaconf import DictConfig
from .base import BaseModel
from .BNT import BrainNetworkTransformer

import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime
import numpy as np
import math

class GlobalFineGrinedNN(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.config = config
        self.stride = config.model.stride
        self.window_sz = config.model.window_sz
        self.signal2spd = signal2spd()
        self.Bn1 = nn.BatchNorm1d(config.model.sizes[0])
        if config.model.mask_0:
            self.mask = nn.Parameter(torch.Tensor(config.model.sizes[0], config.model.sizes[0]).cuda(), requires_grad=True)
            nn.init.kaiming_uniform_(self.mask, a=0, mode='fan_in', nonlinearity='leaky_relu')
        else:
            self.mask = nn.Parameter(torch.ones(config.model.sizes[0], config.model.sizes[0]).cuda(), requires_grad=True)
        # Experiments have shown that LSTM is not so effective in this case
        # self.lstm_hidden_size = 512
        # self.lstm = nn.LSTM(input_size = config.model.window_sz,
        #                     hidden_size = self.lstm_hidden_size, batch_first = True)
        # BNT
        self.bnt = BrainNetworkTransformer(config).cuda()
        self.bnt2 = BrainNetworkTransformer(config).cuda()

        self.node_width = 800
        # custom Transformer
        assert not (self.node_width % config.model.num_heads)

        self.config = config
        self.dot_dim = self.node_width//config.model.num_heads # // float division

        # linear norm， normalized sample data  (not for whole population but for each sample individually) 
        self.mha_ln_h = nn.LayerNorm(self.node_width)
        self.ln = nn.LayerNorm(config.dataset.node_sz)#360)
        # a linear layer
        self.lin_QKV = nn.Linear(self.node_width, self.node_width*2)
        self.multihead_attn = nn.MultiheadAttention(self.node_width, 5, batch_first = True)

        # other

        self.flat = nn.Flatten()

        output_dim = 1 if config.dataset.regression else config.dataset.num_classes
        num_matrics = int(np.ceil((config.dataset.timeseries_sz - config.model.window_sz) / config.model.stride)) + 1

        self.pos_emb = PositionalEmbedding(800, num_matrics)

        if self.config.model.control == 1:
            self.fc = nn.Sequential(
                nn.Linear(self.node_width, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 32),
                nn.LeakyReLU(),
                nn.Linear(32, output_dim)
            )
        elif self.config.model.control == 2:
            if config.model.fc_dropout:
                self.fc = nn.Sequential(
                    nn.Linear(self.node_width * 2, 256),
                    nn.LeakyReLU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(256, 32),
                    nn.LeakyReLU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(32, output_dim)
                )
            else:
                self.fc = nn.Sequential(
                    nn.Linear(self.node_width * 2, 256),
                    nn.LeakyReLU(),
                    nn.Linear(256, 32),
                    nn.LeakyReLU(),
                    nn.Linear(32, output_dim)
                )

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
    
    def save_attention(self, x, grind):
        tmp = x.numpy()
        uuid = datetime.now().strftime("%m-%d-%H-%M-%S")
        if grind:
            #np.save(f'../../attention/{self.bt_count}_grind.npy', tmp)
            #print(f'save to ../../attention/{uuid}_{self.bt_count}_grind.npy')
            self.bt_count = self.bt_count + 1
            print('save disabled')
        else:
            #np.save(f'../../attention/{self.bt_count}_global.npy', tmp)
            #print(f'save to ../../attention/{uuid}_{self.bt_count}_global.npy')
            print('save disabled')
    
    def getMask(self):
        return self.mask

    def forward(self,
                time_seires: torch.tensor,
                node_feature: torch.tensor,
                save: bool = False):
        x = time_seires
        gnt = None
        if self.config.model.use_node:
            gnt = self.bnt2(None, node_feature)
        x = self.Bn1(x)
        g_x = self.signal2spd(x) # => 16*360*360
        g_x = self.bnt(None, g_x)
        #self.save_attention(g_x, False)
        x_list = self.window_split(x)
        # initialization for LSTM model
        #h0 = torch.zeros(1, x.shape[0], self.lstm_hidden_size).cuda()
        #c0 = torch.zeros(1, x.shape[0], self.lstm_hidden_size).cuda()
        for i, item in enumerate(x_list):
            #corr_m, (h0, c0) = self.lstm(item, (h0, c0))
            corr_m = self.signal2spd(item)
            corr_m = torch.mul(self.mask, corr_m)
            x_list[i] = self.bnt(None, corr_m).unsqueeze(1)
            
        x = torch.cat(x_list,1)
        x = self.pos_emb(x)
        x = self.mha_ln_h(x)
        g_x_n = self.mha_ln_h(g_x)
        QKV = self.lin_QKV(x)
        shp = QKV.shape
        if not self.config.model.use_node:
            att, att_weight = self.multihead_attn(g_x_n.unsqueeze(1), x, x)
        else:
            att, att_weight = self.multihead_attn(x, g_x_n.unsqueeze(1), gnt.unsqueeze(1))
        #self.save_attention(x, True)
        att = torch.squeeze(att)
        if self.config.model.control == 2:
            x = torch.cat((g_x, att),1)
        if save:
            return att_weight
        x = self.fc(x)
        return x
    
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

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEmbedding, self).__init__()
        
        # Initialize a positional encoding matrix
        positional_encoding = torch.zeros(max_len, d_model)
        
        # Create position and dimension indices
        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Compute sine and cosine based positional encodings
        positional_encoding[:, 0::2] = torch.sin(positions * div_term)
        positional_encoding[:, 1::2] = torch.cos(positions * div_term)
        
        # Add an extra dimension to the positional encoding matrix
        self.register_buffer('positional_encoding', positional_encoding.unsqueeze(0))

    def forward(self, x):
        # Add the positional encodings to the input
        x = x + self.positional_encoding
        return x
