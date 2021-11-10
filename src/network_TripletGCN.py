#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional
import torch
from torch import Tensor
from src.networks_base import BaseNetwork
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter


# is function already exist (similar) in network_util
'''from src.networks_base import mySequential'''
'''def MLP(channels: list, do_bn=True, on_last=False):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    offset = 0 if on_last else 1
    for i in range(1, n):
        layers.append(
            torch.nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-offset):
            if do_bn:
                layers.append(torch.nn.BatchNorm1d(channels[i]))
            layers.append(torch.nn.ReLU())
    return mySequential(*layers)'''


# TRIP ###########################################

def build_mlp(dim_list, activation='relu', do_bn=False, dropout=0, on_last=False):
   layers = []
   for i in range(len(dim_list) - 1):
     dim_in, dim_out = dim_list[i], dim_list[i + 1]
     layers.append(torch.nn.Linear(dim_in, dim_out))
     final_layer = (i == len(dim_list) - 2)
     if not final_layer or on_last:
       if do_bn:
         layers.append(torch.nn.BatchNorm1d(dim_out))
       if activation == 'relu':
         layers.append(torch.nn.ReLU())
       elif activation == 'leakyrelu':
         layers.append(torch.nn.LeakyReLU())
     if dropout > 0:
       layers.append(torch.nn.Dropout(p=dropout))
   return torch.nn.Sequential(*layers)


class TripletGCN(MessagePassing):
    def __init__(self, dim_node, dim_edge, dim_hidden, aggr= 'add', use_bn=True):
        super().__init__(aggr=aggr)
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.dim_hidden = dim_hidden
        self.nn1 = build_mlp([dim_node*2+dim_edge, dim_hidden, dim_hidden*2+dim_edge],
                      do_bn= use_bn, on_last=True)
        self.nn2 = build_mlp([dim_hidden,dim_hidden,dim_node],do_bn= use_bn)
        
    def forward(self, x, edge_feature, edge_index):
        gcn_x, gcn_e = self.propagate(edge_index, x=x, edge_feature=edge_feature)
        x = self.nn2(gcn_x)
        return x, gcn_e

    def message(self, x_i, x_j,edge_feature):
        x = torch.cat([x_i,edge_feature,x_j],dim=1)
        x = self.nn1(x)#.view(b,-1)
        new_x_i = x[:,:self.dim_hidden]
        new_e   = x[:,self.dim_hidden:(self.dim_hidden+self.dim_edge)]
        new_x_j = x[:,(self.dim_hidden+self.dim_edge):]
        x = new_x_i+new_x_j
        return [x, new_e]
    
    def aggregate(self, x: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        x[0] = scatter(x[0], index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return x
    

class TripletGCNModel(BaseNetwork):
    """ A sequence of scene graph convolution layers  """
    def __init__(self, num_layers, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.gconvs = torch.nn.ModuleList()
        
        for _ in range(self.num_layers):
            self.gconvs.append(TripletGCN(**kwargs))

    def forward(self, node_feature, edge_feature, edges_indices):
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            node_feature, edge_feature = gconv(node_feature, edge_feature, edges_indices)
            
            if i < (self.num_layers-1):
                node_feature = torch.nn.functional.relu(node_feature)
                edge_feature = torch.nn.functional.relu(edge_feature)
        return node_feature, edge_feature

##############################################
    
    
if __name__ == '__main__':
    num_layers = 2
    dim_node = 32
    dim_edge = 64
    dim_hidden = 128
    num_node = 3
    num_edge = 4
    heads = 2       # this value here is not used anywhere

    x = torch.rand(num_node, dim_node)
    edge_feature = torch.rand([num_edge,dim_edge],dtype=torch.float)
    edge_index =torch.randint(0, num_node, [num_edge,2])
    edge_index=edge_index.t().contiguous()
    
    net = TripletGCNModel(num_layers, dim_node=dim_node, dim_edge=dim_edge, dim_hidden=dim_hidden)
    y = net(x, edge_feature, edge_index)
    print(y)

    '''
    (tensor([[ 0.6733, -0.4026,  0.4575,  0.4051,  0.0978,  0.5187,  0.5670, -0.0362,
         -0.1185, -0.1240, -0.3725, -0.7500,  0.0821,  1.0187,  0.1053,  0.7686,
         -0.0943,  0.3001,  0.0559, -0.2874, -0.0493,  0.3142, -0.1465, -0.8636,
          0.2518, -0.2971, -0.8701,  0.2693,  0.6103, -0.6082, -0.4901,  0.0266],
        [ 0.4405, -0.5427, -0.5946, -0.0697,  0.4192,  0.8928,  0.3493, -0.0640,
         -0.4411, -0.1706,  1.0110,  0.0735,  0.4274, -0.5152, -0.3980, -0.3217,
          0.5045,  0.8948, -0.2713,  0.2933, -0.1897, -0.0940, -0.2827,  0.1170,
          0.2159,  0.3353, -0.3318, -0.0723,  0.8645, -0.1046, -0.7070,  0.5572],
        [ 0.2723, -0.1289,  0.0387,  0.1690,  0.5010,  0.3400,  0.6527, -0.1718,
         -0.3134, -0.1397,  0.8401, -0.3314,  0.0562,  0.2674,  0.0506,  0.5467,
         -0.2544, -0.9794, -0.8288,  0.1749,  0.7156, -0.1249, -1.0734, -0.2279,
          0.1542,  0.3046,  0.5362, -0.1250, -0.1474,  0.1348, -1.2873,  0.0542]],
       grad_fn=<AddmmBackward>), tensor([[0.1139, 0.0000, 0.0000, 0.0000, 1.1990, 0.0000, 0.1756, 0.0000, 0.0000,
         1.7286, 0.8120, 0.8843, 0.0000, 0.0000, 1.1144, 0.7737, 0.0000, 0.0000,
         0.4636, 0.0000, 0.3172, 0.6550, 0.0000, 0.0000, 0.0000, 1.1418, 1.1662,
         0.0000, 1.6897, 0.3387, 0.0000, 0.0000, 0.6595, 0.7158, 1.4690, 1.2957,
         0.0000, 0.0000, 1.6959, 0.0000, 0.7762, 0.0000, 0.0513, 0.0000, 0.0000,
         0.0000, 1.1559, 0.0000, 0.0000, 1.4021, 1.3250, 1.1396, 1.1520, 0.1661,
         0.0000, 0.0000, 0.7920, 0.0000, 0.0000, 1.1387, 0.0000, 0.0000, 0.0000,
         0.0000],
        [0.0000, 0.0000, 1.4577, 0.0000, 0.4386, 1.4268, 0.0000, 0.5829, 1.3426,
         0.0000, 1.0382, 0.0000, 0.0000, 1.1971, 0.8570, 1.1758, 0.6679, 0.1292,
         0.9320, 0.0000, 0.0000, 0.0000, 0.0000, 0.5100, 1.3631, 0.6745, 0.5766,
         0.6159, 0.0000, 0.0000, 0.0000, 0.2046, 0.8927, 0.0000, 0.0000, 0.6398,
         0.3633, 1.1146, 0.0000, 0.0000, 0.0000, 0.8355, 1.1582, 0.0000, 0.0000,
         0.4685, 0.0000, 0.2286, 0.0000, 0.4884, 0.6075, 0.8493, 0.8116, 0.0000,
         1.4518, 0.6831, 0.4856, 0.0000, 1.0393, 0.0000, 1.2409, 0.0000, 0.8344,
         0.0000],
        [1.4066, 1.7147, 0.0000, 1.2757, 0.0000, 0.1545, 0.7096, 0.0000, 0.0000,
         0.0000, 0.0000, 1.1057, 0.9293, 0.0000, 0.0000, 0.0000, 0.0000, 1.1910,
         0.0000, 0.0000, 0.0000, 0.0000, 0.4761, 1.0571, 0.3831, 0.0000, 0.0000,
         0.0000, 0.0000, 0.8980, 0.1731, 0.9739, 0.1076, 0.0000, 0.2647, 0.0000,
         0.5916, 0.5326, 0.0000, 0.0000, 0.0000, 0.5571, 0.0000, 1.5508, 1.1027,
         1.3539, 0.7977, 1.2818, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 1.2513, 0.4382, 0.4088, 0.7971, 0.0000, 0.0000, 0.0000, 1.0228,
         1.6631],
        [0.0000, 0.0000, 0.0061, 0.2654, 0.0000, 0.0000, 0.7964, 1.2951, 0.3638,
         0.0000, 0.0000, 0.0000, 1.0655, 0.7643, 0.0000, 0.0000, 1.1937, 0.2596,
         0.2877, 1.6744, 1.4935, 0.9978, 1.3637, 0.0514, 0.0000, 0.0000, 0.0000,
         1.1305, 0.0000, 0.4563, 1.5662, 0.4867, 0.0000, 1.1647, 0.0000, 0.0000,
         0.7599, 0.0000, 0.0000, 1.6211, 0.9592, 0.3090, 0.3760, 0.1767, 0.6366,
         0.0000, 0.0000, 0.0073, 1.7104, 0.0000, 0.0000, 0.0000, 0.0000, 1.4641,
         0.3980, 0.0000, 0.0000, 1.4499, 0.0000, 0.8075, 0.7005, 1.6765, 0.0000,
         0.0000]], grad_fn=<SliceBackward>))
    '''
    
    pass