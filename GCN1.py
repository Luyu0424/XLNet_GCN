import dgl
import dgl.function as fn
import torch
import torch.nn as nn
from torch.nn import Dropout
from typing import List
# from allennlp.modules.feedforward import FeedForward
# from allennlp.modules.layer_norm import LayerNorm
# from allennlp.nn.activations import Activation
from torch.nn import LayerNorm
import torch.nn.functional as F
from numpy import array
import pandas as pd
gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce_sum = fn.sum(msg='m', out='h')
gcn_reduce_max = fn.max(msg='m', out='h')
# gcn_reduce_u_mul_v = fn.u_mul_v('m', 'h')



class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h': h}

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature           #图的节点
        g.update_all(gcn_msg, gcn_reduce_sum)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')

class FeedForward(nn.Module):
    def __init__(self,hdim):
        super(FeedForward, self).__init__()
        self.hidden_dim=hdim
        self.linear1=nn.Linear(self.hidden_dim,self.hidden_dim)
        self.linear2=nn.Linear(self.hidden_dim,self.hidden_dim)
        self.activate=nn.ReLU()
        self.drop=nn.Dropout(p=0.2)
    
    def forward(self,input):
        output=input
        output=self.drop(self.activate(self.linear1(output)))
        output=self.drop(self.linear2(output))
        return output

        


class GCNNet(nn.Module):
    def __init__(self, hdim, nlayers: int = 2, dropout_prob: int = 0.1):
        super(GCNNet, self).__init__()
        # self.gcns = nn.ModuleList([GCN(hdim, hdim, F.relu) for i in range(nlayers)])

        self.relu=nn.ReLU()

        self._gcn_layers = []
        self._feedfoward_layers: List[FeedForward] = []
        self._layer_norm_layers: List[LayerNorm] = []
        self._feed_forward_layer_norm_layers: List[LayerNorm] = []
        feedfoward_input_dim, feedforward_hidden_dim, hidden_dim = hdim, hdim, hdim
        for i in range(nlayers):
            feedfoward = FeedForward(hdim)

            # Note: Please use `ModuleList` in new code. It provides better
            # support for running on multiple GPUs. We've kept `add_module` here
            # solely for backwards compatibility with existing serialized models.
            self.add_module(f"feedforward_{i}", feedfoward)
            self._feedfoward_layers.append(feedfoward)

            feedforward_layer_norm = LayerNorm(hdim)
            self.add_module(f"feedforward_layer_norm_{i}", feedforward_layer_norm)
            self._feed_forward_layer_norm_layers.append(feedforward_layer_norm)

            gcn = GCN(hdim, hdim, F.relu)
            self.add_module(f"gcn_{i}", gcn)
            self._gcn_layers.append(gcn)

            layer_norm = LayerNorm(hdim)
            self.add_module(f"layer_norm_{i}", layer_norm)
            self._layer_norm_layers.append(layer_norm)

            feedfoward_input_dim = hidden_dim

        self.dropout = Dropout(dropout_prob)
        self._input_dim = hdim
        self._output_dim = hdim

    def forward(self, g, features):
        output = features
        # for i, _gcn in enumerate(self.gcns):
        #     x = _gcn(g, x)
        #     feedforward = getattr(self, f"feedforward_{i}")
        #     feedforward_layer_norm = getattr(self, f"feedforward_layer_norm_{i}")
        #     layer_norm = getattr(self, f"layer_norm_{i}")
        # output = g,x
        for i in range(len(self._gcn_layers)):
            # It's necessary to use `getattr` here because the elements stored
            # in the lists are not replicated by torch.nn.parallel.replicate
            # when running on multiple GPUs. Please use `ModuleList` in new
            # code. It handles this issue transparently. We've kept `add_module`
            # (in conjunction with `getattr`) solely for backwards compatibility
            # with existing serialized models.
            gcn = getattr(self, f"gcn_{i}")
            feedforward = getattr(self, f"feedforward_{i}")
            feedforward_layer_norm = getattr(self, f"feedforward_layer_norm_{i}")
            layer_norm = getattr(self, f"layer_norm_{i}")
            cached_input = output
            # Project output of attention encoder through a feedforward
            # network and back to the input size for the next layer.
            # shape (batch_size, timesteps, input_size)
            feedforward_output = feedforward(output)
            feedforward_output = self.dropout(feedforward_output)
            if feedforward_output.size() == cached_input.size():
                # First layer might have the wrong size for highway
                # layers, so we exclude it here.
                feedforward_output = feedforward_layer_norm(feedforward_output + cached_input)
            # shape (batch_size, sequence_length, hidden_dim)
            attention_output = gcn(g,
                                   feedforward_output)
            output = layer_norm(self.dropout(attention_output) + feedforward_output)

        return output


class GraphEncoder(nn.Module):
    def __init__(self, hdim,device,nlayers=2):
        super(GraphEncoder, self).__init__()
        self.GCNNet = GCNNet(hdim, nlayers)
        self.device=device
    #
    def convert_sent_tensors_to_graphs(self, sent, sent_mask, graph):
        sent=sent.squeeze(0)
        max_sent_num, hdim = sent.shape
        effective_length = torch.sum(sent_mask, dim=1).long().tolist()


        this_sent = sent  # max_sent, hdim
        this_len = effective_length
        graph_seed = graph  # List of tuples
        G = dgl.DGLGraph().to(self.device)
        G.add_nodes(max_sent_num)
        # fc_src = [i for i in range(this_len)] * this_len
        # fc_tgt = [[i] * this_len for i in range(this_len)]
        # fc_tgt = list(itertools.chain.from_iterable(fc_tgt))
        fc_src = [x[0] for x in graph_seed]
        fc_tgt = [x[1] for x in graph_seed]
        G.add_edges(fc_src, fc_tgt)
        G.ndata['h'] = this_sent  # every node has the parameter

        return G

    def transform_sent_rep(self, sent_rep, sent_mask, graph):
        init_graphs = self.convert_sent_tensors_to_graphs(sent_rep, sent_mask, graph)

        updated_graph = self.forward(init_graphs).unsqueeze(0)


        assert updated_graph.shape == sent_rep.shape
        return updated_graph

    def forward(self, g):
        h = g.ndata['h']

        out = self.GCNNet.forward(g, features=h)
        # return g, g.ndata['h'], hg  # g is the raw graph, h is the node rep, and hg is the mean of all h
        return out


# device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model=GraphEncoder(hdim=768,device=device,nlayers=2)
# represent=torch.randn((1,4,768))
# print(represent.shape,represent)
# disco_mask=(represent[:,:,0]>=0).long()
# # print(represent[:,:,0])
# # print(disco_mask)
#
# # u=torch.tensor([1,1,1,2,4])
# # v=torch.tensor([4,2,3,4,5])
# # g=dgl.graph((u,v))
# matrix=pd.read_pickle('train_matrixs.pickle')
# # g=torch.tensor([[0,1,0,0,1],
# #                 [1,0,0,0,0],
# #                 [0,1,0,0,1],
# #                 [1,0,0,0,1],
# #                 [0,0,1,0,0]])
# g=torch.tensor(matrix[0],dtype=int)
#
#
# def create_disco_graph(disco_graph, num_of_disco: int) -> List[tuple]:
#     ########
#     # disco graph
#     dis_graph_as_list_of_tuple = []
#     # dis_graph = np.zeros((num_of_disco, num_of_disco), dtype=float)
#     for rst in disco_graph:
#         rst_src, rst_tgt = rst[0], rst[1]
#         if rst_src < num_of_disco and rst_tgt < num_of_disco:
#             # dis_graph[rst_src][rst_tgt] = 1
#             dis_graph_as_list_of_tuple.append((rst_src, rst_tgt))
#     # dis_graph = ArrayField(dis_graph, padding_value=0, dtype=np.float32)
#
#     return dis_graph_as_list_of_tuple
#
# gra=create_disco_graph(g,5)
# out=model.transform_sent_rep(represent,disco_mask,gra)
# print(out.shape,out)


