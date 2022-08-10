import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
                 dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.bias = None

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)  # dropout
        y = torch.matmul(adj, x)  # AH A:[B,N,N] x:[B,N,Din] AH-> [B,N,Din]
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)  # AHW  y:[B.N.Din] w:[B.Din.Dout] -> AHW [B,N,Dout]
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)  # 归一化 DADHW 最基础的GNN
            # print(y[0][0])
        return y


class GcnEncoderGraph(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=648, embedding_dim=512, num_layers=3,
                 pred_hidden_dims=[500], concat=True, bn=True, dropout=0.2, args=None):
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        add_self = not concat
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs = 1

        self.bias = True
        if args is not None:
            self.bias = args.bias

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
            input_dim, hidden_dim, embedding_dim, num_layers,
            add_self, normalize=True, dropout=dropout)
        """
        conv_first :input_dim -> hidden_dim
        conv_block : hidden_dim -> hidden_dim
        conv_last : hidden_dim -> embedding_dim
        """
        self.act = nn.ReLU()

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        """
        pred_model enveddubf_dim -> pred_hiddem_dims -> label_dim
        """
        self.artist_head = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims,
                                                  129, num_aggs=self.num_aggs)
        self.genre_head = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims,
                                                 11, num_aggs=self.num_aggs)
        self.style_head = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims,
                                                 27, num_aggs=self.num_aggs)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
                          normalize=False, dropout=0.0):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                               normalize_embedding=normalize, bias=self.bias)
        conv_block = nn.ModuleList(
            [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                       normalize_embedding=normalize, dropout=dropout, bias=self.bias)
             for i in range(num_layers - 2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                              normalize_embedding=normalize, bias=self.bias)
        return conv_first, conv_block, conv_last

    def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1):
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:  # 没有隐藏层
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []  # 有隐藏层
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    # 构建掩码 batch_num_nodes ->[B,1]
    def construct_mask(self, max_nodes, batch_num_nodes):
        ''' For each num_nodes in batch_num_nodes, the first num_nodes entries of the
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]

        对于 batch_num_nodes 中的每个 num_nodes，相应的列是1的，其余的都是0（要屏蔽）。
        掩码尺寸：[batch_size x max_nodes x 1]
        '''
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        # [B , max_node_num]
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2)

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1])
        return bn_module(x)

    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):

        ''' Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
        '''

        x = conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        x_all = [x]
        # out_all = []
        # out, _ = torch.max(x, dim=1)
        # out_all.append(out)
        for i in range(len(conv_block)):
            x = conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
        x = conv_last(x, adj)
        x_all.append(x)
        # x_tensor: [batch_size x num_nodes x embedding]
        x_tensor = torch.cat(x_all, dim=2)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        return x_tensor

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            self.embedding_mask = self.construct_mask(max_num_nodes,
                                                      batch_num_nodes)  # [B,max_node_num] 每一行代表一个batch的node数量 有几个node 相应行就有几列为1
        else:
            self.embedding_mask = None

        # conv
        x = self.conv_first(x, adj)  # GCN x [B,N,D] -> [B,N,2D]
        x = self.act(x)  # 激活层
        if self.bn:
            x = self.apply_bn(x)  # batchnorm
        out_all = []
        out, _ = torch.max(x, dim=1)  # [B , D]每个特征维度上选择最大的几个
        out_all.append(out)  # 激活每次gcn后的max结果
        for i in range(self.num_layers - 2):
            x = self.conv_block[i](x, adj)
            x = self.act(x)  # 激活
            if self.bn:
                x = self.apply_bn(x)  # 归一
            out, _ = torch.max(x, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(x, dim=1)
                out_all.append(out)
        x = self.conv_last(x, adj)
        # x = self.act(x)
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(x, dim=1)
            out_all.append(out)
        if self.concat:  # 如果cat的话 output结果就是三次gcn的结果进行cat
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        # 输出 torch.Size([20, 2]) [B,2]
        pre_artist = self.artist_head(output)
        pre_genre = self.genre_head(output)
        pre_style = self.style_head_head(output)
        return {'artist': pre_artist, 'genre': pre_genre, 'style': pre_style}
