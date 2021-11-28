#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional
import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing, SAGEConv, GINConv, GCNConv
from torch_scatter import scatter
from torch.nn import functional
from src.networks_base import BaseNetwork
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d


# EXPERIMENTS ###########################################


# COMMON ARCHITECTURE WITHOUT FOR LOOPS ###########################################
class TripletGCN_0(MessagePassing):
    """ A single layer of scene graph convolution """

    def __init__(self, dim_node, dim_edge, dim_hidden, aggr= 'add'):
        super().__init__(aggr=aggr)
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.dim_hidden = dim_hidden

        self.nn1 = Sequential(Linear(dim_node*2+dim_edge, dim_hidden),
                         BatchNorm1d(dim_hidden),
                         ReLU(),
                         Linear(dim_hidden, dim_hidden*2+dim_edge),
                         ReLU())

        self.nn2 = Sequential(Linear(dim_hidden, dim_hidden),
                          BatchNorm1d(dim_hidden),
                          ReLU(),
                          Linear(dim_hidden, dim_node))

        
    def forward(self, x, edge_feature, edge_index):
        gcn_x, gcn_e = self.propagate(edge_index, x=x, edge_feature=edge_feature)
        x = self.nn2(gcn_x)
        return x, gcn_e

    def message(self, x_i, x_j, edge_feature):
        x = torch.cat([x_i,edge_feature,x_j],dim=1)
        x = self.nn1(x)
        new_x_i = x[:,:self.dim_hidden]
        new_e   = x[:,self.dim_hidden:(self.dim_hidden+self.dim_edge)]
        new_x_j = x[:,(self.dim_hidden+self.dim_edge):]
        x = new_x_i + new_x_j
        return [x, new_e]
    
    def aggregate(self, x: Tensor,index: Tensor,ptr: Optional[Tensor] = None,dim_size: Optional[int] = None) -> Tensor:
        x[0] = scatter(x[0], index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr )
        return x
    

class TripletGCNModel_0(BaseNetwork):
    """ A sequence of scene graph convolution layers  """
    def __init__(self, num_layers, **kwargs):
        super().__init__()

        self.conv1 = TripletGCN_0(**kwargs)
        self.conv2 = TripletGCN_0(**kwargs)

    def forward(self, node_feature, edge_feature, edges_indices):
        node_feature, edge_feature = self.conv1(node_feature, edge_feature, edges_indices)
        node_feature = functional.relu(node_feature)
        edge_feature = functional.relu(edge_feature)
        node_feature, edge_feature = self.conv2(node_feature, edge_feature, edges_indices)
        return node_feature, edge_feature

# END #############################################################################


# J&J IMPLEMENTATION ############################################################
import torch.nn as nn

def build_mlp_2(dim_list, activation='relu', batch_norm='none', dropout=0, final_nonlinearity=True):
      layers = []
      for i in range(len(dim_list) - 1):
            dim_in = dim_list[i]
            dim_out = dim_list[i + 1]
            layers.append(nn.Linear(dim_in, dim_out))

            final_layer = (i == len(dim_list) - 2)
            if not final_layer or final_nonlinearity:
                if batch_norm == 'batch':
                    layers.append(nn.BatchNorm1d(dim_out))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leakyrelu':
                layers.append(nn.LeakyReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
      return nn.Sequential(*layers)

def _init_weights(module):
      if hasattr(module, 'weight'):
          if isinstance(module, nn.Linear):
              nn.init.kaiming_normal_(module.weight)

class GraphTripleConv(nn.Module):
    """
    A single layer of scene graph convolution.
    """

    def __init__(self, input_dim, output_dim=None, hidden_dim=512, pooling='avg', mlp_normalization='none'):
        super(GraphTripleConv, self).__init__()
        if output_dim is None:
            output_dim = input_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        assert pooling in ['sum', 'avg'], 'Invalid pooling "%s"' % pooling
        self.pooling = pooling

        # first network g : Linear(3Din -> H) & Linear(H -> 2H + Dout)
        net1_layers = [3 * input_dim, hidden_dim, 2 * hidden_dim + output_dim]
        net1_layers = [l for l in net1_layers if l is not None]
        self.net1 = build_mlp_2(net1_layers, batch_norm=mlp_normalization)
        self.net1.apply(_init_weights)

        # second network g : Linear(H -> H) & Linear(H -> Dout)
        net2_layers = [hidden_dim, hidden_dim, output_dim]
        self.net2 = build_mlp_2(net2_layers, batch_norm=mlp_normalization)
        self.net2.apply(_init_weights)

    def forward(self, obj_vecs, pred_vecs, edges):
        """
        Inputs:
        - obj_vecs: FloatTensor of shape (O, D) giving vectors for all objects
        - pred_vecs: FloatTensor of shape (T, D) giving vectors for all predicates
        - edges: LongTensor of shape (T, 2) where edges[k] = [i, j] indicates the
          presence of a triple [obj_vecs[i], pred_vecs[k], obj_vecs[j]]

        Outputs:
        - new_obj_vecs: FloatTensor of shape (O, D) giving new vectors for objects
        - new_pred_vecs: FloatTensor of shape (T, D) giving new vectors for predicates
        """
        dtype, device = obj_vecs.dtype, obj_vecs.device
        O, T = obj_vecs.size(0), pred_vecs.size(0)
        Din, H, Dout = self.input_dim, self.hidden_dim, self.output_dim

        # Break apart indices for subjects and objects; these have shape (T,)
        s_idx = edges[:, 0].contiguous()
        o_idx = edges[:, 1].contiguous()

        # Get current vectors for subjects and objects; these have shape (T, Din)
        cur_s_vecs = obj_vecs[s_idx]
        cur_o_vecs = obj_vecs[o_idx]

        # Get current vectors for triples; shape is (T, 3 * Din)
        # Pass through net1 to get new triple vecs; shape is (T, 2 * H + Dout)
        cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=1)      # CONCATENATE
        new_t_vecs = self.net1(cur_t_vecs)

        # Break apart into new s, p, and o vecs; s and o vecs have shape (T, H) and p vecs have shape (T, Dout)
        new_s_vecs = new_t_vecs[:, :H]                              # output of the function gs
        new_p_vecs = new_t_vecs[:, H:(H + Dout)]                    # output of the function gp
        new_o_vecs = new_t_vecs[:, (H + Dout):(2 * H + Dout)]       # output of the function go

        """
        the output of the gp function gives the output of the edge (Dout)
        
        the output of the gs and go give the output for the first node and the second one,
        those need to be polled (avg)
        """

        # Allocate space for pooled object vectors of shape (O, H)
        pooled_obj_vecs = torch.zeros(O, H, dtype=dtype, device=device)

        # Use scatter_add to sum vectors for objects that appear in multiple triples;
        # we first need to expand the indices to have shape (T, D)
        s_idx_exp = s_idx.view(-1, 1).expand_as(new_s_vecs)
        o_idx_exp = o_idx.view(-1, 1).expand_as(new_o_vecs)
        pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, s_idx_exp, new_s_vecs)
        pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, o_idx_exp, new_o_vecs)

        # symmetric pooling function
        if self.pooling == 'avg':
            # Figure out how many times each object has appeared, again using
            # some scatter_add trickery.
            obj_counts = torch.zeros(O, dtype=dtype, device=device)
            ones = torch.ones(T, dtype=dtype, device=device)
            obj_counts = obj_counts.scatter_add(0, s_idx, ones)
            obj_counts = obj_counts.scatter_add(0, o_idx, ones)

            # Divide the new object vectors by the number of times they
            # appeared, but first clamp at 1 to avoid dividing by zero;
            # objects that appear in no triples will have output vector 0
            # so this will not affect them.
            obj_counts = obj_counts.clamp(min=1)
            pooled_obj_vecs = pooled_obj_vecs / obj_counts.view(-1, 1)

        # Send pooled object vectors through net2 to get output object vectors,
        # of shape (O, Dout)
        new_obj_vecs = self.net2(pooled_obj_vecs)

        return new_obj_vecs, new_p_vecs


class GraphTripleConvNet(nn.Module):
    """ A sequence of scene graph convolution layers  """

    # Din -> H -> Dout
    def __init__(self, input_dim, num_layers=5, hidden_dim=512, pooling='avg', mlp_normalization='none'):
        super(GraphTripleConvNet, self).__init__()

        self.num_layers = num_layers
        self.gconvs = nn.ModuleList()
        gconv_kwargs = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'pooling': pooling,
            'mlp_normalization': mlp_normalization,
        }
        for _ in range(self.num_layers):
            self.gconvs.append(GraphTripleConv(**gconv_kwargs))

    def forward(self, obj_vecs, pred_vecs, edges):
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            obj_vecs, pred_vecs = gconv(obj_vecs, pred_vecs, edges)
        return obj_vecs, pred_vecs

#################################################################################

class TripletGCN_1(MessagePassing):
    def __init__(self, dim_node, dim_edge, dim_hidden, aggr='add'):
        super().__init__(aggr=aggr)
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.dim_hidden = dim_hidden

        self.nn1 = Sequential(Linear(dim_node * 2 + dim_edge, dim_hidden),
                              BatchNorm1d(dim_hidden),
                              ReLU(),
                              Linear(dim_hidden, dim_hidden * 2 + dim_edge),
                              BatchNorm1d(dim_hidden * 2 + dim_edge),
                              ReLU())

        self.nn2 = Sequential(Linear(dim_hidden, dim_hidden),
                              BatchNorm1d(dim_hidden),
                              ReLU(),
                              Linear(dim_hidden, dim_node))

    def forward(self, x, edge_feature, edge_index):
        gcn_x, gcn_e = self.propagate(edge_index, x=x, edge_feature=edge_feature)
        x = self.nn2(gcn_x)
        return x, gcn_e

    def message(self, x_i, x_j, edge_feature):
        x = torch.cat([x_i, edge_feature, x_j], dim=1)
        x = self.nn1(x)
        new_x_i = x[:, :self.dim_hidden]
        new_e = x[:, self.dim_hidden:(self.dim_hidden + self.dim_edge)]
        new_x_j = x[:, (self.dim_hidden + self.dim_edge):]
        x = new_x_i + new_x_j
        return [x, new_e]

    def aggregate(self, x: Tensor,
                  index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        x[0] = scatter(x[0], index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return x



class TripletGCNModel_1(BaseNetwork):
    """ A sequence of scene graph convolution layers  """

    def __init__(self, dim_node, dim_edge, dim_hidden):
        super().__init__()

        gconv_kwargs = {
            'dim_node': dim_node,
            'dim_edge': dim_edge,
            'dim_hidden': dim_hidden,
        }

        self.conv1 = TripletGCN_1(**gconv_kwargs)
        self.conv2 = TripletGCN_1(**gconv_kwargs)

    def forward(self, node_feature, edge_feature, edges_indices):
        node_feature, edge_feature = self.conv1(node_feature, edge_feature, edges_indices)
        node_feature = functional.relu(node_feature)
        edge_feature = functional.relu(edge_feature)
        node_feature, edge_feature = self.conv2(node_feature, edge_feature, edges_indices)
        return node_feature, edge_feature


##############################################
    
    
if __name__ == '__main__':
    GCN_TYPE = 2

    num_layers = 2
    dim_node = 32
    dim_edge = 64
    dim_hidden = 128
    num_node = 3
    num_edge = 4

    x = torch.rand(num_node, dim_node)
    edge_feature = torch.rand([num_edge, dim_edge], dtype=torch.float)
    edge_index = torch.randint(0, num_node, [num_edge,2])
    edge_index = edge_index.t().contiguous()

    if(GCN_TYPE == 0):
        net = TripletGCNModel_0(num_layers, dim_node=dim_node, dim_edge=dim_edge, dim_hidden=dim_hidden)
    elif (GCN_TYPE == 1):
        net = TripletGCNModel_1(dim_node=dim_node, dim_edge=dim_edge, dim_hidden=dim_hidden)
    elif (GCN_TYPE == 2):
        net = GraphTripleConvNet(input_dim=128, hidden_dim=512)

    # y = net(x, edge_feature, edge_index)
    print(net)
    # print("\n\n")
     # print(y)
    
    pass