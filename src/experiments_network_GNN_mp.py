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


'''class model(torch.nn.Module):
    def __init__(self, dim_node, dim_edge, dim_hidden, use_bn):
        super().__init__()
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.d_o = dim_hidden

        net1_layers = [dim_node, dim_hidden]
        self.nn1 = build_mlp(net1_layers)

        net2_layers = [dim_node * 2 + dim_edge, dim_node + dim_edge, dim_edge]
        self.nn2 = build_mlp(net2_layers, batch_norm=use_bn, final_nonlinearity=True)

    def forward(self, x, edge, value):
        value = self.nn1(value)
        edge_feature = self.nn2(torch.cat([x, edge, value], dim=1))
        return value, edge_feature

# (G)EAN
class GCN_(MessagePassing):
    def __init__(self, dim_node, dim_edge, dim_hidden, use_bn=False):
        super().__init__(aggr='max')
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.dim_hidden = dim_hidden

        self.edgeatten = model(dim_node, dim_edge, dim_hidden, use_bn)

        net_leyer = [dim_node + dim_hidden, dim_node + dim_hidden, dim_node]
        self.nn = build_mlp(net_leyer, batch_norm=use_bn, final_nonlinearity=False)

    def forward(self, x, edge_feature, edge_index):
        x, gcn_edge_feature = self.propagate(edge_index, x=x, edge_feature=edge_feature)
        x = self.nn(x)
        return x, gcn_edge_feature

    def message(self, x_i, x_j, edge_feature):
        xx, gcn_edge_feature = self.edgeatten(x_i, edge_feature, x_j)
        return [xx, gcn_edge_feature]

    def aggregate(self, x: Tensor,
                  index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        x[0] = scatter(x[0], index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return x

    

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
        return node_feature, edge_feature'''


# EAN ################################################################################
class model_(torch.nn.Module):
    def __init__(self, dim_node, dim_edge, dim_hidden, use_bn):
        super().__init__()

        net1_layers = [dim_hidden, dim_hidden, dim_node]
        self.nn1 = build_mlp(net1_layers)
        net2_layers = [dim_node * 2 + dim_edge, dim_node + dim_edge, dim_edge]
        self.nn2 = build_mlp(net2_layers, batch_norm=use_bn, final_nonlinearity=True)

    def forward(self, x, edge, value):
        xx = torch.cat([x, value], dim=1)
        xx = self.nn1(xx)
        edge_feature = self.nn2(torch.cat([x, edge, value], dim=1))
        return xx, edge_feature


class GCN_(MessagePassing):
    """ A single layer of scene graph convolution """
    def __init__(self, dim_node, dim_edge, dim_hidden, aggr='add', use_bn=True):
        super().__init__(aggr=aggr)
        self.dim_node = dim_node            # input_dim
        self.dim_edge = dim_edge            # output_dim
        self.dim_hidden = dim_hidden        # hidden_dim

        self.edgeatten = model_(dim_node, dim_edge, dim_hidden, use_bn)

    def message(self, x_i, x_j, edge_feature):
        xx, gcn_edge_feature = self.edgeatten(x_i, edge_feature, x_j)
        return [xx, gcn_edge_feature]

    def forward(self, x, edge_feature, edge_index):
        gcn_x, gcn_e = self.propagate(edge_index, x=x, edge_feature=edge_feature)
        return x, gcn_e

    def aggregate(self, x:Tensor, index:Tensor, ptr:Optional[Tensor]=None, dim_size:Optional[int]=None) -> Tensor:
        x[0] = scatter(x[0], index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return x


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

################################################################################


if __name__ == '__main__':
    n_node = 8
    dim_node = 256
    dim_edge = 256
    dim_hidden = 256
    use_bn = False
    num_layers = 2

    num_edge = 8
    model = GCNnet(num_layers, dim_node=dim_node, dim_edge=dim_edge, dim_hidden=dim_hidden)
    print("model", model)


