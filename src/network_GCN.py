#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
'''

#########################################################################################################


# SAME AS BEFORE.. ######################################################################################
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


'''class GraphConvolution(MessagePassing):
    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super(GraphConvolution, self).__init__(aggr='add', **kwargs)
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x, edge_feature, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return norm.view(-1, 1) * x_j
        # return x_j

    def update(self, aggr_out):
        return aggr_out'''

class GraphConvolution(MessagePassing):
    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super(GraphConvolution, self).__init__(aggr='add', **kwargs)
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.root_emb = torch.nn.Embedding(1, out_channels)

    def forward(self, x, edge_feature, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)

        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr=edge_feature, norm=norm) + F.relu(x + self.root_emb.weight) * 1. / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


class GCNConv_2(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv_2, self).__init__(aggr='add')
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.root_emb = torch.nn.Embedding(1, out_channels)

    def forward(self, x, edge_attr, edge_index):
        x = self.linear(x)
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm) + F.relu(x + self.root_emb.weight) * 1. / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GCN_(MessagePassing):
    def __init__(self, dim_node, dim_edge, dim_hidden, aggr='add', use_bn=True):
        super().__init__(aggr=aggr)
        self.dim_node = dim_node            # input_dim
        self.dim_edge = dim_edge            # output_dim
        self.dim_hidden = dim_hidden        # hidden_dim

        net2_layers = [dim_node * 2 + dim_edge, dim_node + dim_edge, dim_edge]
        self.edge_nn = build_mlp(net2_layers, batch_norm=use_bn, final_nonlinearity=True)

    def forward(self, x, edge_feature, edge_index):
        xx, gcn_e = self.propagate(edge_index, x=x, edge_feature=edge_feature)
        return xx, gcn_e

    def message(self, x_i, x_j, edge_feature):
        triplet = torch.cat([x_i, edge_feature, x_j], dim=1)
        gcn_edge_feature = self.edge_nn(triplet)
        gcn_node_feature = torch.cat([x_i, x_j], dim=1)
        # gcn_node_feature = x_i                                  # for GCNConv_2
        return [gcn_node_feature, gcn_edge_feature]

    def aggregate(self, x:Tensor, index:Tensor, ptr:Optional[Tensor]=None, dim_size:Optional[int]=None) -> Tensor:
        x[0] = scatter(x[0], index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return x

'''class GCN__(MessagePassing):
    def __init__(self, dim_node, dim_edge, dim_hidden, aggr='add', use_bn=True):
        super().__init__(aggr=aggr)
        self.dim_node = dim_node            # input_dim
        self.dim_edge = dim_edge            # output_dim
        self.dim_hidden = dim_hidden        # hidden_dim

        net2_layers = [dim_node * 2 + dim_edge, dim_node + dim_edge, dim_edge]
        self.edge_nn = build_mlp(net2_layers, batch_norm=use_bn, final_nonlinearity=True)

    def forward(self, x, edge_feature, edge_index):
        cur_s_vecs = x[edge_index[0]]
        cur_o_vecs = x[edge_index[1]]
        triplet = torch.cat([cur_s_vecs, edge_feature, cur_o_vecs], dim=1)
        gcn_edge_feature = self.edge_nn(triplet)
        return gcn_edge_feature'''


#########################################################################################################


class GCNnet(BaseNetwork):
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
            self.gconvs.append(GCN_(**gconv_kwargs))
            # self.gconvs.append(GCN__(**gconv_kwargs))
            # self.gconvs.append(GCN_(**gconv_kwargs))

        # self.node_nn2 = GCNConv(dim_node, dim_node, add_self_loops=False)
        self.node_nn2 = GCNConv(2 * dim_node, dim_node, add_self_loops=False)
        # self.node_nn2 = GCNConv_2(2*dim_node, dim_node)
        # self.node_nn2 = GraphConvolution(dim_node, dim_node)

        self.prop = build_mlp([dim_node, dim_node + dim_node, dim_node], batch_norm=False, final_nonlinearity=False)


    def forward(self, node_feature, edge_feature, edges_indices):
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            node_feature, edge_feature = gconv(node_feature, edge_feature, edges_indices)
            # edge_feature = gconv(node_feature, edge_feature, edges_indices)
            # node_feature = self.node_nn2(node_feature, edges_indices)
            node_feature = self.node_nn2(node_feature, edge_feature, edges_indices)   # for GCNConv_2
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

    model = GCNnet(num_layers, dim_node=dim_node, dim_edge=dim_edge, dim_hidden=dim_hidden)
    # print("model", model)

    x = torch.rand(num_node, dim_node)
    edge_feature = torch.rand([num_edge, dim_edge], dtype=torch.float)
    edge_index = torch.randint(0, num_node, [num_edge, 2])
    edge_index = edge_index.t().contiguous()
    y = model(x, edge_feature, edge_index)
    # print("y", y)


