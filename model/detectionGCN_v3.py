import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np

from model.base_network import BaseNetwork
from model.network import FeatureExtraction
import config as cfg


class CNN(BaseNetwork):
    def __init__(self, cnn_pretrained=True, model_type='resnet18'):
        super(CNN, self).__init__()
        self.feature_extraction = FeatureExtraction(cnn_pretrained, model_type)  # restnet18 的话 返回[512,1,1]

    def forward(self, images):
        """
        :param images: list 里面k个元素 每个元素[B,C,H,W]
        :return features [B,N,D] N结点个数 D特征维
        """
        feature_list = []
        for i in range(cfg.k_node + 1):
            feature = self.feature_extraction(images[i])  # [B , 512 ,  1 , 1]
            feature = feature.squeeze(-1).squeeze(-1)  # [B,512]
            feature = feature.unsqueeze(1)
            feature_list.append(feature)
        features = torch.cat(feature_list, dim=1)
        return features


# GCN basic operation
class GraphConv(BaseNetwork):
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
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).to(cfg.device))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).to(cfg.device))
        else:
            self.bias = None

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
            # print(y[0][0])
        return y


class GcnEncoderGraph(BaseNetwork):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_layers,
                 pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None):
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        add_self = not concat
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs = 2

        self.bias = True
        if args is not None:
            self.bias = args.bias

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
            input_dim, hidden_dim, embedding_dim, num_layers,
            add_self, normalize=True, dropout=dropout)
        self.act = nn.ReLU()
        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim

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
        # 取消结果的cat之后暂时用不到
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def construct_mask(self, max_nodes, batch_num_nodes):
        ''' For each num_nodes in batch_num_nodes, the first num_nodes entries of the
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        '''
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).to(cfg.device)

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).to(cfg.device)
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


class SoftPoolingGcnEncoder(GcnEncoderGraph,BaseNetwork):
    def __init__(self, max_num_nodes=cfg.k_node, input_dim=512, hidden_dim=648, embedding_dim=512, num_layers=3,
                 assign_hidden_dim=648, assign_ratio=0.5, assign_num_layers=-1, num_pooling=2,
                 pred_hidden_dims=[256], concat=True, bn=True, dropout=0.0, linkpred=True,
                 assign_input_dim=-1, args=None):
        '''
        Args:
            num_layers: number of gc layers before each pooling
            num_nodes: number of nodes for each graph in batch
            linkpred: flag to turn on link prediction side objective
        '''

        super(SoftPoolingGcnEncoder, self).__init__(input_dim, hidden_dim, embedding_dim,
                                                    num_layers, pred_hidden_dims=pred_hidden_dims, concat=concat,
                                                    args=args)
        add_self = not concat
        self.num_pooling = num_pooling
        self.linkpred = linkpred
        self.assign_ent = True

        # GC 输出都是embedding_dim
        self.conv_first_after_pool = nn.ModuleList()
        self.conv_block_after_pool = nn.ModuleList()
        self.conv_last_after_pool = nn.ModuleList()
        for i in range(num_pooling):
            # use self to register the modules in self.modules()
            conv_first2, conv_block2, conv_last2 = self.build_conv_layers(
                self.pred_input_dim, hidden_dim, embedding_dim, num_layers,
                add_self, normalize=True, dropout=dropout)
            self.conv_first_after_pool.append(conv_first2)
            self.conv_block_after_pool.append(conv_block2)
            self.conv_last_after_pool.append(conv_last2)

        # assignment
        assign_dims = []
        if assign_num_layers == -1:
            assign_num_layers = num_layers
        if assign_input_dim == -1:
            assign_input_dim = input_dim

        self.assign_conv_first_modules = nn.ModuleList()
        self.assign_conv_block_modules = nn.ModuleList()
        self.assign_conv_last_modules = nn.ModuleList()
        self.assign_pred_modules = nn.ModuleList()
        assign_dim = int(max_num_nodes * assign_ratio)
        for i in range(num_pooling):
            assign_dims.append(assign_dim)
            assign_conv_first, assign_conv_block, assign_conv_last = self.build_conv_layers(
                assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self,
                normalize=True)
            assign_pred_input_dim = assign_hidden_dim * (num_layers - 1) + assign_dim if concat else assign_dim
            assign_pred = self.build_pred_layers(assign_pred_input_dim, [], assign_dim, num_aggs=1)

            # next pooling layer
            assign_input_dim = self.pred_input_dim
            assign_dim = int(assign_dim * assign_ratio)

            self.assign_conv_first_modules.append(assign_conv_first)
            self.assign_conv_block_modules.append(assign_conv_block)
            self.assign_conv_last_modules.append(assign_conv_last)
            self.assign_pred_modules.append(assign_pred)

        self.artist_head = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims,
                                                  129, num_aggs=self.num_aggs)
        self.genre_head = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims,
                                                 11, num_aggs=self.num_aggs)
        self.style_head = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims,
                                                 27, num_aggs=self.num_aggs)

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        if 'assign_x' in kwargs:
            x_a = kwargs['assign_x']
        else:
            x_a = x

        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        # out_all = []

        # Z = GCN_{embed}(A,X)
        embedding_tensor = self.gcn_forward(x, adj, self.conv_first, self.conv_block, self.conv_last,
                                            embedding_mask)  # [batch_size x num_nodes x embedding_dim]
        # 存储特征
        # out, _ = torch.max(embedding_tensor, dim=1)
        # out_all.append(out)

        # if self.num_aggs == 2:
        #     out = torch.sum(embedding_tensor, dim=1)
        #     out_all.append(out)
        # 开始池化
        for i in range(self.num_pooling):
            if batch_num_nodes is not None and i == 0:
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
            else:
                embedding_mask = None
            # S = Softmax(GCN_{pool} (A,X))
            self.assign_tensor = self.gcn_forward(x_a, adj,
                                                  self.assign_conv_first_modules[i], self.assign_conv_block_modules[i],
                                                  self.assign_conv_last_modules[i],
                                                  embedding_mask)
            # [batch_size x num_nodes x next_lvl_num_nodes]
            self.assign_tensor = nn.Softmax(dim=-1)(self.assign_pred_modules[i](self.assign_tensor))
            if embedding_mask is not None:
                self.assign_tensor = self.assign_tensor * embedding_mask

            # update pooled features and adj matrix
            # X^(l+1) = S^T Z
            x = torch.matmul(torch.transpose(self.assign_tensor, 1, 2), embedding_tensor)
            # A^(l+1) =  S^T A S
            adj = torch.transpose(self.assign_tensor, 1, 2) @ adj @ self.assign_tensor
            x_a = x

            # pool后再接一层
            embedding_tensor = self.gcn_forward(x, adj,
                                                self.conv_first_after_pool[i], self.conv_block_after_pool[i],
                                                self.conv_last_after_pool[i])

            out_all = []
            out, _ = torch.max(embedding_tensor, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.mean(embedding_tensor, dim=1)
                # out = torch.sum(embedding_tensor, dim=1)
                out_all.append(out)

        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        pre_artist = self.artist_head(output)
        pre_genre = self.genre_head(output)
        pre_style = self.style_head(output)
        return {'artist': pre_artist, 'genre': pre_genre, 'style': pre_style}

    def loss(self, adj=None, batch_num_nodes=None, adj_hop=1):
        '''
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7
        max_num_nodes = adj.size()[1]
        pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2)
        tmp = pred_adj0
        pred_adj = pred_adj0
        for adj_pow in range(adj_hop - 1):
            tmp = tmp @ pred_adj0
            pred_adj = pred_adj + tmp
        pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype).to(cfg.device))
        self.link_loss = -adj * torch.log(pred_adj + eps) - (1 - adj) * torch.log(1 - pred_adj + eps)
        if batch_num_nodes is None:
            num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
            print('Warning: calculating link pred loss without masking')
        else:
            num_entries = np.sum(batch_num_nodes * batch_num_nodes)
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
            adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
            self.link_loss[(1 - adj_mask).bool()] = 0.0

        self.link_loss = torch.sum(self.link_loss) / float(num_entries)
        return self.link_loss


class detectionGCNv3(BaseNetwork):
    def __init__(self):
        super(detectionGCNv3, self).__init__()
        self.cnn = CNN()
        self.GCN = SoftPoolingGcnEncoder()
        self.GCN.init_weights()

    def forward(self, images, adj):
        x = self.cnn(images)
        out_dict = self.GCN(x, adj)
        return out_dict

    def Linkloss(self, adj):
        return self.GCN.loss(adj)
