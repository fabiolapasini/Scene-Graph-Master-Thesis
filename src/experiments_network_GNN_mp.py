#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Some codes here are modified from SuperGluePretrainedNetwork https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py

import torch
from src.network_util import build_mlp
from src.networks_base import BaseNetwork
from torch_geometric.nn.conv import MessagePassing
from torch import Tensor
from torch_scatter import scatter
from typing import Optional
from torch.nn import functional, ModuleList
import torch.nn.functional as F


# EAN WAY TO DEAL WITH EDGES / MS / TRIPLET_GNN INPUT ###################################################
'''
class model_(torch.nn.Module):
    def __init__(self, dim_node, dim_edge, dim_hidden, use_bn):
        super().__init__()

        net1_layers = [dim_hidden, dim_hidden, dim_node]
        self.node_nn = build_mlp(net1_layers)
        net2_layers = [dim_node * 2 + dim_edge, dim_node + dim_edge, dim_edge]
        self.edge_nn = build_mlp(net2_layers, batch_norm=use_bn, final_nonlinearity=True)

    def forward(self, x, edge, value):
        xx = torch.cat([x, value], dim=1)
        xx = self.node_nn(xx)
        edge_feature = self.edge_nn(torch.cat([x, edge, value], dim=1))
        return xx, edge_feature

class GCN_(MessagePassing):
    def __init__(self, dim_node, dim_edge, dim_hidden, aggr='add', use_bn=True):
        super().__init__(aggr=aggr)
        self.dim_node = dim_node            # input_dim
        self.dim_edge = dim_edge            # output_dim
        self.dim_hidden = dim_hidden        # hidden_dim

        self.edgeatten = model_(dim_node, dim_edge, dim_hidden, use_bn)

    def forward(self, x, edge_feature, edge_index):
        gcn_x, gcn_e = self.propagate(edge_index, x=x, edge_feature=edge_feature)
        return gcn_x, gcn_e

    def message(self, x_i, x_j, edge_feature):
        xx, gcn_edge_feature = self.edgeatten(x_i, edge_feature, x_j)
        return [xx, gcn_edge_feature]

    def aggregate(self, x:Tensor, index:Tensor, ptr:Optional[Tensor]=None, dim_size:Optional[int]=None) -> Tensor:
        x[0] = scatter(x[0], index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return x
'''

#########################################################################################################


# SAME AS BEFORE.. ######################################################################################
from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv

class GCN_(MessagePassing):
    def __init__(self, dim_node, dim_edge, dim_hidden, aggr='add', use_bn=True):
        super().__init__(aggr=aggr)
        self.dim_node = dim_node            # input_dim
        self.dim_edge = dim_edge            # output_dim
        self.dim_hidden = dim_hidden        # hidden_dim

        # net1_layers = [dim_hidden, dim_hidden, dim_node]
        # self.node_nn = build_mlp(net1_layers)

        self.node_nn1 = GCNConv(dim_hidden, dim_hidden)
        self.node_nn2 = GCNConv(dim_hidden, dim_node)
        # self.node_nn1 = GINConv(dim_hidden, dim_hidden)
        # self.node_nn2 = GINConv(dim_hidden, dim_node)
        # self.node_nn1 = GATConv(dim_hidden, dim_hidden, heads=4, dropout=0.6)
        # self.node_nn2 = GATConv(dim_hidden, dim_node, heads=4, dropout=0.6)

        net2_layers = [dim_node * 2 + dim_edge, dim_node + dim_edge, dim_edge]
        self.edge_nn = build_mlp(net2_layers, batch_norm=use_bn, final_nonlinearity=True)

    def forward(self, x, edge_feature, edge_index):
        xx, gcn_e = self.propagate(edge_index, x=x, edge_feature=edge_feature)
        #vgcn_x = self.node_nn(xx)
        xx = self.node_nn1(xx, edge_index)
        gcn_x = self.node_nn2(xx, edge_index)
        return gcn_x, gcn_e

    def message(self, x_i, x_j, edge_feature):
        gcn_edge_feature = self.edge_nn(torch.cat([x_i, edge_feature, x_j], dim=1))
        gcn_node_feature = torch.cat([x_i, x_j], dim=1)
        return [gcn_node_feature, gcn_edge_feature]

    def aggregate(self, x:Tensor, index:Tensor, ptr:Optional[Tensor]=None, dim_size:Optional[int]=None) -> Tensor:
        x[0] = scatter(x[0], index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return x

#########################################################################################################


class GCNnet(BaseNetwork):
    """ A sequence of scene graph convolution layers  """
    def __init__(self, num_layers, dim_node, dim_edge, dim_hidden):
        super().__init__()
        self.num_layers = num_layers
        self.gconvs = ModuleList()
        gconv_kwargs = {
            'dim_node': dim_node,
            'dim_edge': dim_edge,
            'dim_hidden': dim_hidden,
        }
        for _ in range(self.num_layers):
            self.gconvs.append(GCN_(**gconv_kwargs))

    def forward(self, node_feature, edge_feature, edges_indices):
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            node_feature, edge_feature = gconv(node_feature, edge_feature, edges_indices)

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

    model = GCNnet(num_layers, dim_node=dim_node, dim_edge=dim_edge, dim_hidden=dim_hidden)
    # print("model", model)

    x = torch.rand(num_node, dim_node)
    edge_feature = torch.rand([num_edge, dim_edge], dtype=torch.float)
    edge_index = torch.randint(0, num_node, [num_edge, 2])
    edge_index = edge_index.t().contiguous()
    y = model(x, edge_feature, edge_index)
    # print("y", y)


