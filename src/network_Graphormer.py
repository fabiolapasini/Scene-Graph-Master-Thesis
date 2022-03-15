#!/usr/bin/env python3

from typing import Optional
import torch
from src.network_util import build_mlp, MLP, MLP_attn
from torch import Tensor, nn
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter
from src.networks_base import BaseNetwork


# MESSAGE IMPLEMENTATION OF GAT NET + GRAPHORMER BIAS ####################################
class GetAttenBias (torch.nn.Module):
    def __init__(self, num_heads: int, dim_node: int, dim_edge: int, dim_atten: int):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.dim_atten = dim_atten
        self.num_heads = num_heads
        num_virtual_tokens = 1
        self.in_degree_encoder = nn.Embedding(512, dim_atten, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(512, dim_atten, padding_idx=0)

    def forward(self, x, edge_feature, edge_index):
        attn_bias = 0
        N = x.size(0)

        # node adj matrix [N, N] bool
        adj_orig = torch.zeros([N, N], dtype=torch.bool)
        adj_orig[edge_index[0, :], edge_index[1, :]] = True
        in_degree = adj_orig.long().sum(dim=1).view(-1).to(self.device)
        out_degree = adj_orig.long().sum(dim=0).view(-1).to(self.device)

        node_feature = (
                x
                + self.in_degree_encoder(in_degree)
                + self.out_degree_encoder(out_degree)
        )

        return node_feature, attn_bias


class Graphormer_Net(BaseNetwork):
    def __init__(self, dim_node, dim_edge, dim_atten, num_layers, num_heads=1, aggr='max', use_bn=False,
                 attention='fat', use_edge: bool = True, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.gconvs = torch.nn.ModuleList()

        for _ in range(self.num_layers):
            self.gconvs.append(EncoderLayer(num_heads, dim_node, dim_edge, dim_atten, aggr,
                                                     use_bn=use_bn, attention=attention, use_edge=use_edge, **kwargs))

    def forward(self, node_feature, edge_feature, edges_indices):
        probs = list()
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            node_feature, edge_feature, prob = gconv(node_feature, edge_feature, edges_indices)

            if i < (self.num_layers - 1) or self.num_layers == 1:
                node_feature = torch.nn.functional.relu(node_feature)
                edge_feature = torch.nn.functional.relu(edge_feature)

            if prob is not None:
                probs.append(prob.cpu().detach())
            else:
                probs.append(None)
        return node_feature, edge_feature, probs


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadedEdgeAttention(torch.nn.Module):
    def __init__(self, num_heads: int, dim_node: int, dim_edge: int, dim_atten: int,
                 use_bn=False, attention='fat', use_edge: bool = True, **kwargs):
        super().__init__()
        assert dim_node % num_heads == 0
        assert dim_edge % num_heads == 0
        assert dim_atten % num_heads == 0
        self.name = 'GRAPHORMER'
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.d_n = d_n = dim_node // num_heads
        self.d_e = d_e = dim_edge // num_heads
        self.d_o = d_o = dim_atten // num_heads

        self.scaling = (self.d_o) ** -0.5

        self.num_heads = num_heads
        self.use_edge = use_edge
        self.nn_edge = build_mlp([dim_node * 2 + dim_edge, (dim_node + dim_edge), dim_edge], batch_norm=use_bn, final_nonlinearity=False)

        DROP_OUT_ATTEN = None
        if 'DROP_OUT_ATTEN' in kwargs:
            DROP_OUT_ATTEN = kwargs['DROP_OUT_ATTEN']

        self.attention = attention
        assert self.attention in ['fat']

        if self.attention == 'fat':
            self.nn = MLP([d_n + d_e, d_n + d_e, d_o], do_bn=use_bn, drop_out=DROP_OUT_ATTEN)
            # self.proj_edge = nn.Linear(dim_edge, dim_edge)
            # self.proj_query = nn.Linear(dim_node, dim_node)
            # self.proj_value = nn.Linear(dim_node, dim_atten)
            self.proj_edge = build_mlp([dim_edge, dim_edge])
            self.proj_query = build_mlp([dim_node, dim_node])
            self.proj_value = build_mlp([dim_node, dim_atten])
        else:
            raise NotImplementedError('')

    # (node, edge, neighbour)
    def forward(self, query, edge, value, attn_bias):
        batch_dim = query.size(0)
        edge_feature = self.nn_edge(torch.cat([query, edge, value], dim=1))  # .view(b, -1, 1)

        if self.attention == 'fat':
            # projecting
            value = self.proj_value(value)
            query = self.proj_query(query).view(batch_dim, self.d_n, self.num_heads) * self.scaling
            edge = self.proj_edge(edge).view(batch_dim, self.d_e, self.num_heads)

            # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V

            # Scaled Dot-Product Attention.
            attention = self.nn(torch.cat([query, edge], dim=1))

            # add the bias
            # attention = attention + attn_bias

            attention = attention.softmax(1)
            # metric mutiplication einstein multiplication
            x = torch.einsum('bm,bm->bm', attention.reshape_as(value), value)
        return x, edge_feature, attention


# (G)EAN
class EncoderLayer(MessagePassing):
    def __init__(self, num_heads, dim_node, dim_edge, dim_atten, aggr='max', use_bn=False, attention='fat',
                 use_edge: bool = True, **kwargs):
        super().__init__(aggr=aggr)
        self.name = 'edgeatten'
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.attention = attention
        self.attn_bias = 0

        self.GetAttenBias = GetAttenBias(dim_node=dim_node, dim_edge=dim_edge, dim_atten=dim_atten, num_heads=num_heads)
        self.self_attention_norm = nn.LayerNorm(dim_node)

        assert self.attention in ['fat']
        if self.attention == 'fat':
            self.edgeatten = MultiHeadedEdgeAttention(
                dim_node=dim_node, dim_edge=dim_edge, dim_atten=dim_atten,
                num_heads=num_heads, use_bn=use_bn, attention=attention,
                use_edge=use_edge, **kwargs)
            self.prop = build_mlp([1024, dim_node + dim_atten, dim_node], batch_norm=use_bn, final_nonlinearity=False)
            # self.prop = build_mlp([dim_node, dim_node + dim_atten, dim_node], batch_norm=use_bn, final_nonlinearity=False)
        else:
            raise NotImplementedError('')

        self.self_attention_dropout = nn.Dropout(0.5)
        self.ffn_norm = nn.LayerNorm(2*dim_node)
        # self.ffn_norm = nn.LayerNorm(dim_node)
        self.ffn = FeedForwardNetwork(2*dim_node, 512, 0.5)
        # self.ffn = FeedForwardNetwork(dim_node, 512, 0.5)
        self.ffn_dropout = nn.Dropout(0.5)

    def forward(self, x, edge_feature, edge_index):
        assert x.ndim == 2
        assert edge_feature.ndim == 2

        x = self.self_attention_norm(x)
        x_new, self.attn_bias = self.GetAttenBias(x, edge_feature, edge_index)

        y, gcn_edge_feature, prob = self.propagate(edge_index, x=x_new, edge_feature=edge_feature)

        y = self.self_attention_dropout(y)
        x = torch.cat([x, y], dim=1)
        # x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = torch.cat([x, y], dim=1)
        # x = x + y

        xx = self.prop(x)
        return xx, gcn_edge_feature, prob

    def message(self, x_i, x_j, edge_feature):
        xx, gcn_edge_feature, prob = self.edgeatten(x_i, edge_feature, x_j, self.attn_bias)
        return [xx, gcn_edge_feature, prob]

    def aggregate(self, x: Tensor, index: Tensor, ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        x[0] = scatter(x[0], index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return x


########################################################################



if __name__ == '__main__':
    attention = 'fat'

    n_node = 8
    dim_node = 256
    dim_edge = 256
    dim_atten = 256
    num_head = 1

    query = torch.rand(n_node, dim_node, 1)
    edge = torch.rand(n_node, dim_edge, 1)
    # model = MultiHeadedEdgeAttention(num_head, dim_node, dim_edge,dim_atten)
    # model(query,edge,value)

    num_edge = 8
    query = torch.rand(n_node, dim_node)
    edge = torch.rand(num_edge, dim_edge)
    edge_index = torch.randint(0, n_node - 1, [2, num_edge])

    # model = EdgeAtten(num_head,dim_node,dim_edge,dim_atten)
    # model(query,edge,edge_index)
    num_layers = 2
    model = Graphormer_Net(dim_node, dim_edge, dim_edge, num_layers, num_heads=num_head, attention=attention)
    model(query, edge, edge_index)
