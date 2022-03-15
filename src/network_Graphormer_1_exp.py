#!/usr/bin/env python3

from typing import Optional
import torch
from src.network_util import build_mlp, MLP, MLP_attn
from torch import Tensor, nn
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter
from src.networks_base import BaseNetwork
import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
import algos


# MESSAGE IMPLEMENTATION OF GAT NET + GRAPHORMER BIAS ####################################
class GetAttenBias (torch.nn.Module):
    def __init__(self, num_heads: int, dim_node: int, dim_edge: int, dim_atten: int):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.dim_atten = dim_atten
        self.num_heads = num_heads
        self.in_degree_encoder = nn.Embedding(512, dim_atten, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(512, dim_atten, padding_idx=0)
        self.rel_pos_encoder = nn.Embedding(512, num_heads, padding_idx=0)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)
        self.edge_encoder = nn.Embedding(dim_edge, num_heads, padding_idx=0)

    def forward(self, x, edge_feature, edge_index):
        # attn_bias = None
        N = x.size(0)
        # x = self.convert_to_single_emb(x)

        # node adj matrix [N, N] bool
        adj_orig = torch.zeros([N, N], dtype=torch.bool)
        adj_orig[edge_index[0, :], edge_index[1, :]] = True

        # edge feature here
        if len(edge_feature.size()) == 1:
            edge_feature = edge_feature[:, None]
        edge_feature1 = edge_feature.to(torch.int32)
        attn_edge_type = torch.zeros([N, N, edge_feature.size(-1)], dtype=torch.int32).to(self.device)

        attn_edge_type[edge_index[0, :], edge_index[1, :]] = self.convert_to_single_emb(edge_feature1) + 1  # [n_nodes, n_nodes, 1] for ZINC
        attn_edge_type2 = attn_edge_type.clone().detach().requires_grad_(False)
        attn_edge_type2 = attn_edge_type2.cpu().numpy()

        shortest_path_result, path = algos.floyd_warshall(adj_orig.numpy())  # [n_nodesxn_nodes, n_nodesxn_nodes]
        max_dist = np.amax(shortest_path_result)
        edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type2)
        rel_pos = torch.from_numpy((shortest_path_result)).long()
        attn_bias = torch.zeros([N , N], dtype=torch.int32).to(self.device)  # with graph token


        adj = torch.zeros([N , N ], dtype=torch.bool)
        adj[edge_index[0, :], edge_index[1, :]] = True

        in_degree = adj_orig.long().sum(dim=1).view(-1).to(self.device)
        out_degree = adj_orig.long().sum(dim=0).view(-1).to(self.device)
        edge_input = torch.from_numpy(edge_input).long().to(self.device)
        edge_input = edge_input[:, :, :20, :]


        attn_bias[:, :][rel_pos >= 20] = -99999999   #float("-inf")

        node_feature = (
                x
                + self.in_degree_encoder(in_degree)
                + self.out_degree_encoder(out_degree)
        )

        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1)
        rel_pos_bias = self.rel_pos_encoder(rel_pos.to(self.device)).permute(0, 2, 1)
        graph_attn_bias[:, :, :] = (graph_attn_bias[:, :, :] + rel_pos_bias)

        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, :] = ( graph_attn_bias[:, :, :] + t)


        # edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 2, 1)
        # graph_attn_bias[:, :, :] = graph_attn_bias[:, :, :] + edge_input

        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)
        graph_attn_bias.to(self.device)
        return node_feature, graph_attn_bias

    # this thing is not good at all!
    def convert_to_single_emb(self, x, offset=512):
        feature_num = x.size(1) if len(x.size()) > 1 else 1
        feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.int32).to(self.device)
        x = x + feature_offset
        return x


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
        self.att_size = att_size = dim_atten // num_heads
        self.d_n = dim_node // num_heads
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.scale = att_size ** -0.5
        self.num_heads = num_heads
        self.use_edge = use_edge

        self.proj_query = nn.Linear(dim_atten, num_heads * att_size)
        self.proj_key = nn.Linear(dim_atten, num_heads * att_size)
        self.proj_value = nn.Linear(dim_atten, num_heads * att_size)

        self.linear_prob = nn.Linear(dim_node, num_heads * 28)


    # (node, edge, neighbour)
    def forward(self, q, k, v, attn_bias):
        batch_dim = q.size(0)

        v = self.proj_value(v).view(batch_dim, self.d_n, self.num_heads).permute(0,2,1)         # torch.Size([28, 4, 64)
        q = self.proj_query(q).view(batch_dim, self.d_n, self.num_heads).permute(0,2,1)
        k = self.proj_key(k).view(batch_dim, self.d_n, self.num_heads).permute(0,2,1)

        k = k.transpose(1, 2)                                       # torch.Size([28, 4, 4)
        prob = torch.matmul(q, k)
        # prob = self.linear_prob(prob)

        # print("prob size: ", prob.size())                         # torch.Size([28, 4, 4])
        # attn_bias = attn_bias.view(self.d_n, batch_dim)
        # attn_bias = attn_bias.transpose(attn_bias, 0, 1)
        # print("attn_bias size: ", attn_bias.size())               # attn_bias size:  torch.Size([28, 4, 28])

        prob = prob + attn_bias.resize_(28, 4, 4)

        x = prob.softmax(1)
        x = x.matmul(v)
        x = x.view(batch_dim, self.d_n*self.num_heads)
        return x, prob


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

        self.nn_edge = build_mlp([dim_node * 2 + dim_edge, (dim_node + dim_edge), dim_edge], batch_norm=use_bn,
                                 final_nonlinearity=False)
        self.GetAttenBias = GetAttenBias(dim_node=dim_node, dim_edge=dim_edge, dim_atten=dim_atten, num_heads=num_heads)
        self.self_attention_norm = nn.LayerNorm(dim_node)

        assert self.attention in ['fat']
        if self.attention == 'fat':
            self.edgeatten = MultiHeadedEdgeAttention(
                dim_node=dim_node, dim_edge=dim_edge, dim_atten=dim_atten,
                num_heads=num_heads, use_bn=use_bn, attention=attention,
                use_edge=use_edge, **kwargs)
            self.prop = build_mlp([dim_node, dim_node + dim_atten, dim_node], batch_norm=use_bn, final_nonlinearity=False)
        else:
            raise NotImplementedError('')

        self.self_attention_dropout = nn.Dropout(0.5)
        self.ffn_norm = nn.LayerNorm(dim_node)
        self.ffn = FeedForwardNetwork(dim_node, 512, 0.5)
        self.ffn_dropout = nn.Dropout(0.5)

    def forward(self, x, edge_feature, edge_index):
        assert x.ndim == 2
        assert edge_feature.ndim == 2

        x = self.self_attention_norm(x)
        x_new, self.attn_bias = self.GetAttenBias(x, edge_feature, edge_index)

        x_i = x[edge_index[0]]
        x_j = x[edge_index[1]]
        edge_feature = self.nn_edge(torch.cat([x_i, edge_feature, x_j], dim=1))
        y, atten = self.edgeatten(x_new, x_new, x_new, self.attn_bias)

        y = self.self_attention_dropout(y)
        x = x + y
        # y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        xx = self.prop(x)
        return xx, edge_feature, atten


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
