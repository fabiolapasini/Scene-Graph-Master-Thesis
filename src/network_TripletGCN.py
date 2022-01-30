#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional
import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter
from torch.nn import functional, ModuleList
from src.networks_base import BaseNetwork
from src.network_util import build_mlp


# TRIP ###########################################

class TripletGCN(MessagePassing):
    """ A single layer of scene graph convolution """
    def __init__(self, dim_node, dim_edge, dim_hidden, aggr='mean', use_bn=True):
        super().__init__(aggr=aggr)
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.dim_hidden = dim_hidden

        net1_layers = [dim_node*2+dim_edge, dim_hidden, dim_hidden*2+dim_edge]
        self.nn1 = build_mlp(net1_layers, batch_norm= use_bn, final_nonlinearity=True)
        net2_layers = [dim_hidden, dim_hidden, dim_node]                                            # dim hidden is 2*dim node
        self.nn2 = build_mlp(net2_layers, batch_norm= use_bn)

    def forward(self, x, edge_feature, edge_index):
        gcn_x, gcn_e = self.propagate(edge_index, x=x, edge_feature=edge_feature)
        gcn_x = x + self.nn2(gcn_x)
        # x = self.nn2(gcn_x)
        return gcn_x, gcn_e

    def message(self, x_i, x_j, edge_feature):
        x = torch.cat([x_i, edge_feature, x_j], dim=1)
        x = self.nn1(x)
        new_x_i = x[:, :self.dim_hidden]
        new_e = x[:, self.dim_hidden:(self.dim_hidden + self.dim_edge)]
        new_x_j = x[:, (self.dim_hidden + self.dim_edge):]
       # print("new new_e size: ", new_e.size())
        x = new_x_i + new_x_j
        return [x, new_e]

    def aggregate(self, x:Tensor, index:Tensor, ptr:Optional[Tensor]=None, dim_size:Optional[int]=None) -> Tensor:
        x[0] = scatter(x[0], index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return x


class TripletGCNModel(BaseNetwork):
    """ A sequence of scene graph convolution layers  """

    def __init__(self, num_layers, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.gconvs = ModuleList()

        for _ in range(self.num_layers):
            self.gconvs.append(TripletGCN(**kwargs))

    '''node_feature = obj_vecs (number of nodes, dimension of nodes)
    edge_feature = pred_vecs (number of relations, dimension of edges)
    edges_indices = edges (number of relations, 2, dtype=torch.long))'''

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

    x = torch.rand(num_node, dim_node)
    edge_feature = torch.rand([num_edge, dim_edge], dtype=torch.float)
    edge_index = torch.randint(0, num_node, [num_edge,2])
    edge_index = edge_index.t().contiguous()
    
    net = TripletGCNModel(num_layers, dim_node=dim_node, dim_edge=dim_edge, dim_hidden=dim_hidden)
    y = net(x, edge_feature, edge_index)
    # print(net)
    # print(y)
