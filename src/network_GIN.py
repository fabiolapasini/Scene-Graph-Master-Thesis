#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from src.network_util import build_mlp
from src.networks_base import BaseNetwork
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter
from typing import Optional
from torch.nn import functional, ModuleList
from torch_geometric.nn import GINConv
from torch_geometric.nn import MessagePassing


# EAN WAY TO DEAL WITH EDGES / MS / TRIPLET_GNN INPUT ###################################################

#######################################################################################


class GIN_(MessagePassing):
    def __init__(self, dim_node, dim_edge, dim_hidden, aggr='add', use_bn=True):
        super().__init__(aggr=aggr)
        self.dim_node = dim_node            # input_dim
        self.dim_edge = dim_edge            # output_dim
        self.dim_hidden = dim_hidden        # hidden_dim

        net1_layers = [dim_hidden, dim_hidden, dim_node]
        self.node_nn = build_mlp(net1_layers, batch_norm=use_bn, final_nonlinearity=False)                        # node_nn_bn

        net2_layers = [dim_node * 2 + dim_edge, dim_node + dim_edge, dim_edge]
        self.edge_nn = build_mlp(net2_layers, batch_norm=use_bn, final_nonlinearity=True)

    def forward(self, x, edge_feature, edge_index):
        xx, gcn_e = self.propagate(edge_index, x=x, edge_feature=edge_feature)
        return xx, gcn_e

    def message(self, x_i, x_j, edge_feature):
        gcn_edge_feature = self.edge_nn(torch.cat([x_i, edge_feature, x_j], dim=1))
        gcn_node_feature = torch.cat([x_i, x_j], dim=1)
        return [gcn_node_feature, gcn_edge_feature]
        # return gcn_edge_feature

    def aggregate(self, x:Tensor, index:Tensor, ptr:Optional[Tensor]=None, dim_size:Optional[int]=None) -> Tensor:
        x[0] = scatter(x[0], index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return x

#########################################################################################################


class GINnet(BaseNetwork):
    """ A sequence of scene graph convolution layers  """

    def __init__(self, num_layers, dim_node, dim_edge, dim_hidden, aggr):
        super().__init__()
        self.num_layers = num_layers
        self.gconvs = ModuleList()
        gconv_kwargs = {
            'dim_node': dim_node,
            'dim_edge': dim_edge,
            'dim_hidden': dim_hidden,
            'aggr': aggr
        }

        for _ in range(self.num_layers):
            self.gconvs.append(GIN_(**gconv_kwargs))

        # net1_leyers = [2*dim_node, dim_hidden, dim_node]
        # net1 = build_mlp(net1_leyers, batch_norm=False, final_nonlinearity=False)

        self.node_nn2 = GINConv(nn.Sequential(nn.Linear(2*dim_node, dim_hidden), nn.ReLU(),
                                            nn.Linear(dim_hidden, dim_node)), eps=0., train_eps=True)
        # self.node_nn2 = GINConv(nn.Sequential(nn.Linear(dim_node, dim_hidden), nn.ReLU(),
                                              # nn.Linear(dim_hidden, dim_node)), eps=0., train_eps=True)
        self.prop = build_mlp([dim_node, dim_node + dim_node, dim_node], batch_norm=False,
                              final_nonlinearity=False)

    def forward(self, node_feature, edge_feature, edges_indices):
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            node_feature, edge_feature = gconv(node_feature, edge_feature, edges_indices)
            # edge_feature = gconv(node_feature, edge_feature, edges_indices)
            node_feature = self.node_nn2(node_feature, edges_indices)
            node_feature = self.prop(node_feature)

            if i < (self.num_layers - 1):
                node_feature = functional.relu(node_feature)
                edge_feature = functional.relu(edge_feature)
        return node_feature, edge_feature


if __name__ == '__main__':
    num_layers = 2
    dim_node = 256
    dim_edge = 256
    dim_hidden = 512
    num_node = 3
    num_edge = 4

    model = GINnet(num_layers, dim_node=dim_node, dim_edge=dim_edge, dim_hidden=dim_hidden)
    # print("model", model)

    x = torch.rand(num_node, dim_node)
    edge_feature = torch.rand([num_edge, dim_edge], dtype=torch.float)
    edge_index = torch.randint(0, num_node, [num_edge, 2])
    edge_index = edge_index.t().contiguous()
    y = model(x, edge_feature, edge_index)
    # print("y", y)


