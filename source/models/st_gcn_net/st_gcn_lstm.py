import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .utils.tgcn import ConvTemporalGraphical
from .fmri_lstm import fMRI_LSTM
from .utils.graph import Graph
import numpy as np
from scipy import stats
from omegaconf import DictConfig

import pdb


class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """
    # net = Model(1,1,None,True, device)
    # def __init__(self, in_channels, num_class, graph_args,
    #              edge_importance_weighting, device, **kwargs):
    def __init__(self, config: DictConfig):
        super().__init__()

        # original parameters are stored in config, initialize them
        in_channels = self.config.model.inchannels
        num_class = self.config.model.numclass
        graph_args = self.config.model.graphargs
        edge_importance_weighting = self.config.model.edgeimportanceweighting
        kwargs = None

        device = torch.device('cuda')

        # load graph

        # **this is the adj matrix Soham produced that computes correlation based on raw data **
        # A = np.load('../cs230/adj/adj_matrix.npy')

        # **this is the adj matrix that computes correlation based on z-score of data for all 1200 timesteps**
        A = np.load('data/adj_matrix_qingyu.npy')

        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-0.5)
        DAD = np.dot(np.dot(Dn, A), Dn)

        temp_matrix = np.zeros((1, A.shape[0], A.shape[0]))
        temp_matrix[0] = DAD
        A = torch.tensor(temp_matrix, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks (**number of layers, final output features, kernel size**)
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 1  # update temporal kernel size
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        self.st_gcn_layer = st_gcn(in_channels, 64, kernel_size, device, 1, residual=False, **kwargs0)
        self.device = device
        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.Parameter(torch.ones(self.A.size()))
            """
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
            """
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction (**number of fully connected layers**)
        self.fcn = nn.Conv2d(64, num_class, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        x, _ = self.st_gcn_layer(x, self.A * self.edge_importance)

        # forwad
        # for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
        #    x, _ = gcn(x, self.A * importance)
        """
        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        # pdb.set_trace()
        x = self.fcn(x)
        x = self.sig(x)

        x = x.view(x.size(0), -1)
        """

        return x

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)
        # pdb.set_trace()
        return output, feature


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 device,
                 stride=1,
                 dropout=0.5,
                 residual=True,
                 batch_size=64):
        super().__init__()
        print("Dropout={}".format(dropout))
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])
        self.lstm_layer = fMRI_LSTM(64, 64, 1, batch_size=batch_size)

        self.batch_size = batch_size
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        self.device = device
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = torch.mean(x, dim=3)
        x = x + res
        # print(res)
        # Tried converting to z-scores
        # x = x.data.cpu().numpy()
        # for i in range(x.shape[0]):
        #     x[i] = stats.zscore(x[i], axis = 1)
        # x = torch.from_numpy(x).float().to(self.device)

        # x = self.tcn(x) + res
        # x = x + res
        x = x.permute(0, 2, 1)
        self.lstm_layer.hidden = self.lstm_layer.init_hidden(batch_size=self.batch_size)
        x = self.lstm_layer(x)
        print(x)
        return x, A