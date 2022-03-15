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
        num_virtual_tokens = 1
        self.in_degree_encoder = nn.Embedding(512, dim_atten, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(512, dim_atten, padding_idx=0)
        self.rel_pos_encoder = nn.Embedding(512, num_heads, padding_idx=0)
        self.graph_token = nn.Embedding(num_virtual_tokens, 32)
        # self.edge_encoder = nn.Embedding(512 * 3 + 1, num_heads, padding_idx=0)
        self.edge_encoder = nn.Embedding(512 +1, num_heads, padding_idx=0)
        self.graph_token_virtual_distance = nn.Embedding(num_virtual_tokens, num_heads)

    def forward(self, x, edge_feature, edge_index):
        num_virtual_tokens = 1
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
        # print("edge edge_feature1:", edge_feature1.dtype)
        attn_edge_type = torch.zeros([N, N, edge_feature.size(-1)], dtype=torch.int32).to(self.device)
        # source = self.convert_to_single_emb(edge_feature) + 1

        # print("attn_edge_type: ")
        # print(type(attn_edge_type), attn_edge_type.dtype)
        # print("source: ")
        # print(type(source), source.dtype)
        # print("edge_feature: ")
        # print(type(edge_feature), edge_feature.dtype)

        attn_edge_type[edge_index[0, :], edge_index[1, :]] = self.convert_to_single_emb(edge_feature1) + 1  # [n_nodes, n_nodes, 1] for ZINC
        attn_edge_type2 = attn_edge_type.clone().detach().requires_grad_(False)
        attn_edge_type2 = attn_edge_type2.cpu().numpy()

        # print("edge feature:", edge_feature1.dtype)
        # print("attn_edge_type2: ", type(attn_edge_type2), attn_edge_type2.dtype)            # attn_edge_type2:  <class 'numpy.ndarray'> float32
        # print("max_dist: ", type(max_dist), max_dist.dtype)                                 # max_dist:  <class 'numpy.int32'> int32
        # print("path: ", type(path), path.dtype)                                          # path:  <class 'numpy.ndarray'> int32

        # print(algos)   # <module 'algos' from 'C:\\Users\\fabio\\Desktop\\Scene-Graph-Master-Thesis\\algos.cp38-win_amd64.pyd'>
        shortest_path_result, path = algos.floyd_warshall(adj_orig.numpy())  # [n_nodesxn_nodes, n_nodesxn_nodes]
        max_dist = np.amax(shortest_path_result)
        edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type2)
        rel_pos = torch.from_numpy((shortest_path_result)).long()
        attn_bias = torch.zeros([N + num_virtual_tokens, N + num_virtual_tokens], dtype=torch.int32).to(self.device)  # with graph token
        # print(type(attn_bias), attn_bias.size())          # <class 'torch.Tensor'> torch.Size([29, 29])
        # print(type(rel_pos), rel_pos.size())                # <class 'torch.Tensor'> torch.Size([28, 28])

        adj = torch.zeros([N + num_virtual_tokens, N + num_virtual_tokens], dtype=torch.bool)
        adj[edge_index[0, :], edge_index[1, :]] = True
        for i in range(num_virtual_tokens):
            adj[N + i, :] = True
            adj[:, N + i] = True
        # print(type(adj), adj.size())                      # <class 'torch.Tensor'> torch.Size([29, 29])

        in_degree = adj_orig.long().sum(dim=1).view(-1).to(self.device)
        out_degree = adj_orig.long().sum(dim=0).view(-1).to(self.device)
        edge_input = torch.from_numpy(edge_input).long().to(self.device)
        edge_input = edge_input[:, :, :20, :]

###################################################################

        # print("in_degrees type: ", type(in_degree), "in_degrees len: ", len(in_degree))
        # in_degrees type:  <class 'torch.Tensor'> in_degrees len:  28
        # for i in in_degree:
            # print(i)
            # print(type(i))
            # print(i.size())
        # tensor(<class 'torch.Tensor'> torch.Size([])

        attn_bias[num_virtual_tokens:, num_virtual_tokens:][rel_pos >= 20] = -99999999   #float("-inf")
        # print(attn_bias)

        # Maximum number of nodes in the batch.
        # max_node_num = max(i.size(0) for i in x)
        # print("Max num node" ,max_node_num)       # 256
        # div = max_node_num // 4
        # print("Div", div)                         # 64
        # max_node_num = 4 * div + 3
        # print("Max num node", max_node_num)       # 259
        # max_dist = max(i.size(-2) for i in edge_input)

        # print(type(edge_input), len(edge_input))            # <class 'torch.Tensor'> 28
        # for i in edge_input:
            # print(type(i), len(i))                    # <class 'torch.Tensor'> 28
            # print(i.size(0), i.size(1), i.size(2))      # 28 20 256
        # edge_input = torch.cat([self.pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_input])
        # edge_input = torch.cat(edge_input, max_node_num, max_node_num, max_dist)
        # edge_input = torch.cat([(i, max_node_num, max_node_num, max_dist) for i in edge_input])
        '''attn_bias = torch.cat(attn_bias, max_node_num + num_virtual_tokens)
        adj = torch.cat(adj, max_node_num + num_virtual_tokens)
        attn_edge_type = torch.cat(attn_edge_type, max_node_num + num_virtual_tokens)
        rel_pos = torch.cat(rel_pos, max_node_num)'''

        # x_ = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in x])
        # edge_input = torch.cat([pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_input])
        # attn_bias = torch.cat([ pad_attn_bias_unsqueeze(i, max_node_num + num_virtual_tokens) for i in attn_bias]
        # )
        # adj = torch.cat([pad_2d_bool(i, max_node_num + num_virtual_tokens) for i in adj])
        # attn_edge_type = torch.cat([ pad_edge_type_unsqueeze(i, max_node_num + num_virtual_tokens) for i in attn_edge_type]
        # )       # CUDA out of memory.
        # rel_pos = torch.cat([pad_rel_pos_unsqueeze(i, max_node_num) for i in rel_pos])
        # print(in_degree.type(), " " , in_degree.size())             # torch.cuda.LongTensor   torch.Size([28])
        # in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degree])
        # out_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in out_degree])

        # print("in_deg: ", in_degree)

        '''print(x.type, " ", x.size())
        print(x)
        <built-in method type of Tensor object at 0x0000022865E74140>   torch.Size([28, 256])
        tensor([[ 1.8597,  0.6377, -0.1120,  ..., -1.0960, -0.8911, -0.6178],
        [ 2.2728,  0.4284, -0.5702,  ..., -0.8751, -0.8852, -0.4174],
        [ 1.2081,  0.7645, -0.5365,  ..., -0.7045, -0.9640, -0.4276],
        ...,
        [ 1.2837,  0.5763, -0.4840,  ..., -0.9089, -0.6512,  0.0983],
        [ 0.9570,  0.0727, -0.7103,  ..., -0.9300, -0.8179, -0.5429],
        [ 1.5980,  0.0470, -0.1716,  ..., -1.0533, -1.0309, -0.0157]],
       device='cuda:0', grad_fn=<NativeLayerNormBackward>)'''
        node_feature = (
                x
                + self.in_degree_encoder(in_degree)
                + self.out_degree_encoder(out_degree)
        )
        '''print(node_feature.type, " ", node_feature.size())
        print(node_feature)
        <built-in method type of Tensor object at 0x0000022865EA7E80>   torch.Size([28, 256])
        tensor([[ 4.1587,  0.6539,  1.8609,  ..., -0.1431, -1.3055, -0.5006],
        [ 1.5114,  2.6222, -0.6620,  ..., -3.0142, -1.8880,  2.4716],
        [ 2.8993,  1.6384, -2.3330,  ..., -1.4606, -2.4581, -1.0281],
        ...,
        [-1.4922,  1.4572, -0.6880,  ..., -1.7961, -1.6915, -0.0109],
        [-1.6546, -0.7372, -2.6260,  ...,  0.8427, -1.9961, -1.2283],
        [ 2.3781,  0.2284, -1.0966,  ..., -0.3084, -1.4343,  1.0812]],
       device='cuda:0', grad_fn=<AddBackward0>)
        '''

    #######################################################

        graph_attn_bias = attn_bias.clone()
        # print(graph_attn_bias.size())                                                     # torch.Size([29, 29])
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1)
        # print(graph_attn_bias.size())                                                       # torch.Size([29, 4, 29])
        rel_pos_bias = self.rel_pos_encoder(rel_pos.to(self.device)).permute(0, 2, 1)
        # rel_pos_bias = self.rel_pos_encoder(rel_pos.to(self.device))
        print(type(rel_pos_bias), rel_pos_bias.size())                                      # <class 'torch.Tensor'> torch.Size([28, 28, 4])
        graph_attn_bias[num_virtual_tokens:, :, num_virtual_tokens:] = (graph_attn_bias[num_virtual_tokens:, :, num_virtual_tokens:] + rel_pos_bias)

        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, num_virtual_tokens)
        # print("1: ", t.size())                                                                 # torch.Size([1, 4, 1])
        self.graph_token_virtual_distance.weight.view(1, self.num_heads, num_virtual_tokens).unsqueeze(-2)
        graph_attn_bias[num_virtual_tokens:, :, num_virtual_tokens:] = (graph_attn_bias[num_virtual_tokens:, :, num_virtual_tokens:] + t)
        # print("2: ", graph_attn_bias.size())                                                            # torch.Size([29, 4, 29])

        # print("0: ", attn_edge_type.size())                                 # 1:  torch.Size([28, 28, 256])
        # edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 2, 1)
        # edge_input = attn_edge_type.permute(0, 2, 1)
        # print("1: ", edge_input.size())                                        # torch.Size([28, 4, 28])
        # graph_attn_bias[num_virtual_tokens:, :, num_virtual_tokens:] = (graph_attn_bias[num_virtual_tokens:, :, num_virtual_tokens:] + edge_input)
        # print("2: ", graph_attn_bias.size())
        '''graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset'''
        # print("3: ", graph_attn_bias.size())

        # return x, attn_bias

        # print("Graph_attn_bias: ", graph_attn_bias.type())                    # torch.cuda.IntTensor
        # print("Graph_attn_bias shape: ", graph_attn_bias.size())              # Graph_attn_bias shape:  torch.Size([29, 4, 29])
        # print(graph_attn_bias)
        # print("node_feature shape: ", node_feature.size())                    # node_feature shape:  torch.Size([28, 256])
        # print(node_feature)
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
        atten_bias = 0
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            node_feature, edge_feature, prob = gconv(node_feature, edge_feature, edges_indices, atten_bias)

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


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


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
            self.nn_bias = MLP_attn([29, d_o])
            self.nn_bias_ = MLP_attn([29, 420])
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
        batch_dim = query.size(0)                                                                       # batch dim = 420, number of edges
        edge_feature = self.nn_edge(torch.cat([query, edge, value], dim=1))  # .view(b, -1, 1)

        if self.attention == 'fat':
            # projecting
            value = self.proj_value(value)
            query = self.proj_query(query).view(batch_dim, self.d_n, self.num_heads) * self.scaling
            edge = self.proj_edge(edge).view(batch_dim, self.d_e, self.num_heads)

            # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V

            # Scaled Dot-Product Attention.
            attention = self.nn(torch.cat([query, edge], dim=1))
            print("attention2: ", attention.size())                   # attention:  torch.cuda.FloatTensor  attention:  torch.Size([420, 64, 4])
            print("attn_bias1: ", attn_bias.size())

            '''attn_bias = attn_bias.permute(0, 2, 1)
            # attn_bias = attn_bias.float()
            # attn_bias = self.nn_bias(attn_bias.squeeze(1))
            attn_bias = self.nn_bias(attn_bias)
            print("attn_bias: ", attn_bias.size())
            attn_bias = attn_bias.permute(1, 0, 2)
            # print("attn_bias 0: ", attn_bias.size())
            attn_bias = self.nn_bias_(attn_bias.squeeze(1))
            # print("attn_bias 1: ", attn_bias.size())
            attn_bias = attn_bias.permute(1, 0, 2)'''

            # add the bias
            attention = attention + attn_bias
            attention = torch.einsum('bm,bm->bm', attn_bias.reshape_as(attention), attention)

            '''print("attention 2: ", attention.size())
            print(attention)
            print("\n")
            print("\n")'''

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

    def forward(self, x, edge_feature, edge_index, atten_bias):
        assert x.ndim == 2
        assert edge_feature.ndim == 2

        x = self.self_attention_norm(x)
        x_new, self.attn_bias = self.GetAttenBias(x, edge_feature, edge_index)

        # print("attn_bias net 1: ", self.attn_bias.size())
        # print(self.attn_bias)
        # print("\n")
        # print("x: ", x_new.size())
        # print(x_new)

        y, gcn_edge_feature, prob = self.propagate(edge_index, x=x_new, edge_feature=edge_feature, atten_bias=self.attn_bias)

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
