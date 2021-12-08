#!/usr/bin/env python3

from typing import Optional
import torch
from src.network_util import build_mlp, MLP
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter
from src.networks_base import BaseNetwork

# MESSAGE IMPLEMENTATION OF GAT NET ####################################

# check TripletEdgeNet from shun cheng repo
# trasformer "Attention Is All You Need"

class MultiHeadedEdgeAttention(torch.nn.Module):
    def __init__(self, num_heads: int, dim_node: int, dim_edge: int, dim_atten: int, use_bn=False, attention = 'fat', use_edge:bool = True, **kwargs):
        super().__init__()
        assert dim_node % num_heads == 0
        assert dim_edge % num_heads == 0
        assert dim_atten % num_heads == 0
        self.name = 'GAT_MS'
        self.dim_node=dim_node
        self.dim_edge=dim_edge
        self.d_n = d_n = dim_node // num_heads
        self.d_e = d_e = dim_edge // num_heads
        self.d_o = d_o = dim_atten // num_heads
        self.num_heads = num_heads
        self.use_edge = use_edge
        self.nn_edge = build_mlp([dim_node*2+dim_edge,(dim_node+dim_edge),dim_edge], batch_norm= use_bn, final_nonlinearity=False)
        
        DROP_OUT_ATTEN = None
        if 'DROP_OUT_ATTEN' in kwargs:
            DROP_OUT_ATTEN = kwargs['DROP_OUT_ATTEN']
        
        self.attention = attention
        assert self.attention in ['fat']
        
        if self.attention == 'fat':
            if use_edge:
                self.nn = MLP([d_n+d_e, d_n+d_e, d_o],do_bn=use_bn,drop_out = DROP_OUT_ATTEN)
            else:
                self.nn = MLP([d_n, d_n*2, d_o],do_bn=use_bn,drop_out = DROP_OUT_ATTEN)
                
            self.proj_edge = build_mlp([dim_edge,dim_edge])
            self.proj_query = build_mlp([dim_node,dim_node])
            self.proj_value = build_mlp([dim_node,dim_atten])
        else:
            raise NotImplementedError('')

    # (node, edge, neighbour)
    def forward(self, query, edge, value):
        batch_dim = query.size(0)
        edge_feature = self.nn_edge( torch.cat([query,edge,value],dim=1) ) #.view(b, -1, 1)
        
        if self.attention == 'fat':
            value = self.proj_value(value)
            query = self.proj_query(query).view(batch_dim, self.d_n, self.num_heads)
            edge = self.proj_edge(edge).view(batch_dim, self.d_e, self.num_heads)
            if self.use_edge:
                # is value between 0 and 1 showing the weights, how much a query affects the edge!!!
                prob = self.nn(torch.cat([query,edge],dim=1))
            else:
                prob = self.nn(query)
            prob = prob.softmax(1)
            # metric mutiplication einstein multiplication
            x = torch.einsum('bm,bm->bm', prob.reshape_as(value), value)
        return x, edge_feature, prob

# (G)EAN
class GraphEdgeAttenNetwork(MessagePassing):
    def __init__(self, num_heads, dim_node, dim_edge, dim_atten, aggr= 'max', use_bn=False, attention = 'fat', use_edge:bool=True, **kwargs):
        super().__init__(aggr = aggr)
        self.name = 'edgeatten'
        self.dim_node=dim_node
        self.dim_edge=dim_edge
        
        self.attention = attention
        assert self.attention in [ 'fat']
        if self.attention == 'fat':
            self.edgeatten = MultiHeadedEdgeAttention(
                dim_node=dim_node, dim_edge=dim_edge, dim_atten=dim_atten,
                num_heads=num_heads, use_bn=use_bn, attention=attention, use_edge=use_edge, **kwargs)
            self.prop = build_mlp([dim_node+dim_atten, dim_node+dim_atten, dim_node], batch_norm= use_bn, final_nonlinearity=False)
        else:
            raise NotImplementedError('')

    def forward(self, x, edge_feature, edge_index):
        assert x.ndim == 2
        assert edge_feature.ndim == 2
        xx, gcn_edge_feature, prob = self.propagate(edge_index, x=x, edge_feature=edge_feature)
        xx = self.prop(torch.cat([x,xx],dim=1))
        return xx, gcn_edge_feature, prob

    def message(self, x_i, x_j, edge_feature):
        xx, gcn_edge_feature, prob = self.edgeatten(x_i, edge_feature, x_j)
        return [xx, gcn_edge_feature, prob]

    def aggregate(self, x:Tensor, index:Tensor, ptr:Optional[Tensor]=None, dim_size:Optional[int]=None) -> Tensor:
        x[0] = scatter(x[0], index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return x

########################################################################


class GEAN_ms(BaseNetwork):
    def __init__(self, dim_node, dim_edge, dim_atten, num_layers, num_heads=1, aggr= 'max', use_bn=False, attention = 'fat', use_edge:bool=True, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.gconvs = torch.nn.ModuleList()
        
        self.drop_out = None 
        if 'DROP_OUT_ATTEN' in kwargs:
            self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])
        
        for _ in range(self.num_layers):
            self.gconvs.append(GraphEdgeAttenNetwork(num_heads, dim_node, dim_edge, dim_atten, aggr,
                                         use_bn=use_bn ,attention=attention, use_edge=use_edge, **kwargs))

    def forward(self, node_feature, edge_feature, edges_indices):
        probs = list()
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            node_feature, edge_feature, prob = gconv(node_feature, edge_feature, edges_indices)
            
            if i < (self.num_layers-1) or self.num_layers==1:
                node_feature = torch.nn.functional.relu(node_feature)
                edge_feature = torch.nn.functional.relu(edge_feature)
                if self.drop_out:
                    node_feature = self.drop_out(node_feature)
                    edge_feature = self.drop_out(edge_feature)
                
            if prob is not None:
                probs.append(prob.cpu().detach())
            else:
                probs.append(None)
        return node_feature, edge_feature, probs



if __name__ == '__main__':
    attention='fat'

    n_node = 8
    dim_node = 256
    dim_edge = 256
    dim_atten = 256
    num_head = 1

    query = torch.rand(n_node,dim_node,1)
    edge  = torch.rand(n_node,dim_edge,1)
    # model = MultiHeadedEdgeAttention(num_head, dim_node, dim_edge,dim_atten)
    # model(query,edge,value)

    num_edge = 8
    query = torch.rand(n_node,dim_node)
    edge  = torch.rand(num_edge,dim_edge)
    edge_index = torch.randint(0, n_node-1, [2,num_edge])

    # model = EdgeAtten(num_head,dim_node,dim_edge,dim_atten)
    # model(query,edge,edge_index)
    num_layers=2
    model = GEAN_ms(dim_node, dim_edge, dim_edge, num_layers, num_heads=num_head, attention=attention)
    model(query, edge, edge_index)
