#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Some codes here are modified from SuperGluePretrainedNetwork https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py

import torch
from network_util import build_mlp, Gen_Index, Aggre_Index
from networks_base import BaseNetwork
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch import Tensor
from torch_scatter import scatter
from typing import Optional


# EAN ################################################################################
class MultiHeadedEdgeAttention_(torch.nn.Module):
    def __init__(self, dim_node: int, dim_edge: int, dim_atten: int, use_bn=False, **kwargs):
        super().__init__()
        self.dim_node=dim_node
        self.dim_edge=dim_edge
        self.d_o = dim_atten

        net1_layers = [dim_node*2+dim_edge, dim_node+dim_edge, dim_edge]
        self.nn_edge = build_mlp(net1_layers, batch_norm=use_bn, final_nonlinearity=True)
        net2_layers = [dim_node, dim_atten]
        self.proj_value = build_mlp(net2_layers)
        
    def forward(self, query, edge, value):
        value = self.proj_value(value)
        edge_feature = self.nn_edge(torch.cat([query,edge,value],dim=1) )
        return value, edge_feature


# (G)EAN
class GraphEdgeAttenNetwork_(BaseNetwork):
    def __init__(self, dim_node, dim_edge, dim_atten, aggr= 'max', use_bn=False, flow='target_to_source',use_edge:bool=True, **kwargs):
        super().__init__()
        self.dim_node=dim_node
        self.dim_edge=dim_edge
        self.index_get = Gen_Index(flow=flow)
        self.index_aggr = Aggre_Index(aggr=aggr,flow=flow)

        self.edgeatten = MultiHeadedEdgeAttention_(
                    dim_node=dim_node,dim_edge=dim_edge, dim_atten=dim_atten,
                    use_bn=use_bn,use_edge=use_edge, **kwargs)
        self.prop = build_mlp([dim_node + dim_atten, dim_node + dim_atten, dim_node], batch_norm=use_bn, final_nonlinearity=False)

    def forward(self, x, edge_feature, edge_index):
        x_i, x_j = self.index_get(x, edge_index)
        xx, gcn_edge_feature = self.edgeatten(x_i, edge_feature, x_j)
        xx = self.index_aggr(xx, edge_index, dim_size = x.shape[0])
        xx = self.prop(torch.cat([x, xx], dim=1))
        return xx, gcn_edge_feature


class GraphEdgeAttenNetworkLayers_(BaseNetwork):
    """ A sequence of scene graph convolution layers  """

    def __init__(self, dim_node, dim_edge, dim_atten, num_layers, aggr='max', use_bn=False, flow='target_to_source', use_edge: bool = True, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.gconvs = torch.nn.ModuleList()

        for _ in range(self.num_layers):
            self.gconvs.append(GraphEdgeAttenNetwork_(dim_node, dim_edge, dim_atten, aggr,
                                                     use_bn=use_bn, flow=flow, use_edge=use_edge, **kwargs))

    def forward(self, node_feature, edge_feature, edges_indices):
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            node_feature, edge_feature = gconv(node_feature, edge_feature, edges_indices)

            if i < (self.num_layers-1) or self.num_layers==1:
                node_feature = torch.nn.functional.relu(node_feature)
                edge_feature = torch.nn.functional.relu(edge_feature)

        return node_feature, edge_feature

################################################################################


if __name__ == '__main__':
    n_node = 8
    dim_node = 256
    dim_edge = 256
    dim_hidden = 256
    num_layers = 2

    num_edge = 8
    x = torch.rand(n_node, dim_node)        # query
    edge_feature  = torch.rand(num_edge, dim_edge)
    edge_index = torch.randint(0, n_node-1, [2,num_edge])
    edge_index = edge_index.t().contiguous()

    model = GraphEdgeAttenNetworkLayers_(dim_node, dim_edge, dim_hidden, num_layers)
    print("model", model)

    '''
    model GraphEdgeAttenNetworkLayers_(
  (gconvs): ModuleList(
    (0): GraphEdgeAttenNetwork_(
      (index_get): Gen_Index()
      (index_aggr): Aggre_Index()
      (edgeatten): MultiHeadedEdgeAttention_(
        (nn_edge): Sequential(
          (0): Linear(in_features=768, out_features=512, bias=True)
          (1): ReLU()
          (2): Linear(in_features=512, out_features=256, bias=True)
          (3): ReLU()
        )
        (proj_value): Sequential(
          (0): Linear(in_features=256, out_features=256, bias=True)
        )
      )
      (prop): Sequential(
        (0): Linear(in_features=512, out_features=512, bias=True)
        (1): ReLU()
        (2): Linear(in_features=512, out_features=256, bias=True)
      )
    )
    (1): GraphEdgeAttenNetwork_(
      (index_get): Gen_Index()
      (index_aggr): Aggre_Index()
      (edgeatten): MultiHeadedEdgeAttention_(
        (nn_edge): Sequential(
          (0): Linear(in_features=768, out_features=512, bias=True)
          (1): ReLU()
          (2): Linear(in_features=512, out_features=256, bias=True)
          (3): ReLU()
        )
        (proj_value): Sequential(
          (0): Linear(in_features=256, out_features=256, bias=True)
        )
      )
      (prop): Sequential(
        (0): Linear(in_features=512, out_features=512, bias=True)
        (1): ReLU()
        (2): Linear(in_features=512, out_features=256, bias=True)
      )
    )
  )
)
'''


