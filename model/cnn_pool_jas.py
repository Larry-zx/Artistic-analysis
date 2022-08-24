# 为jas数据集的cnn_pool
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from model.base_network import BaseNetwork
from model.network import FeatureExtraction
import config as cfg
from torchvision import models
from model.HAM import ResBlock_HAM


class CNN(BaseNetwork):
    def __init__(self):
        super(CNN, self).__init__()
        resnet18 = models.resnet18(pretrained=True)  # 得到resnet18的模型 并且拿到预训练好的参数
        model_list = list(resnet18.children())
        self.Conv1 = nn.Sequential(*model_list[:3])  # size//2
        self.HAM1 = ResBlock_HAM(64)
        self.Conv2 = model_list[4]
        self.Conv3 = model_list[5]
        self.Conv4 = model_list[6]
        self.HAM2 = ResBlock_HAM(512)
        self.adapt_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.init()

    def init(self):
        self.HAM1.init_weights()
        self.HAM2.init_weights()

    def forward(self, images, size, boxes):
        x = self.Conv1(images)  # 112
        x1 = self.HAM1(x)  # 112
        x2 = self.Conv2(x1)  # 112
        x3 = self.Conv3(x2)  # 56
        x4 = self.Conv4(x3)  # 28
        x3_112 = F.interpolate(x3, size=x1.shape[2:])
        x4_112 = F.interpolate(x4, size=x1.shape[2:])
        features = torch.cat([x1, x2, x3_112, x4_112], dim=1)
        B, C, H, W = features.shape
        Batch_Node_list = []
        for i in range(B):
            h, w = size[i]  # 当前样本的图片的原始size
            Node_feature_list = []  # 存储结点特征
            for box in boxes[i]:  # 遍历当前样本的box
                x1, y1, x2, y2 = box
                # 坐标变换
                y1 = int(torch.floor((H / h) * y1))
                y2 = int(torch.floor((H / h) * y2))
                x1 = int(torch.floor((W / w) * x1))
                x2 = int(torch.floor((W / w) * x2))
                # 从特征图中抽出box对应内容
                img = features[i:i + 1, :, y1:y2, x1:x2]
                # Adapt_pooling后变为1xD的一维向量
                obj_feature = self.adapt_pool(img).squeeze(-1).squeeze(-1)
                # 存储结点特征
                Node_feature_list.append(obj_feature)
            Batch_Node_list.append(torch.cat(Node_feature_list, dim=0).unsqueeze(0))
        total_feature = torch.cat(Batch_Node_list, dim=0)
        return total_feature


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


class SoftPoolingGcnEncoder(GcnEncoderGraph):
    def __init__(self, class_num, max_num_nodes=cfg.k_node, input_dim=512, hidden_dim=648, embedding_dim=512,
                 num_layers=3,
                 assign_hidden_dim=648, assign_ratio=0.5, assign_num_layers=-1, num_pooling=2,
                 pred_hidden_dims=[128], concat=True, bn=True, dropout=0.0, linkpred=True,
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

        self.quality_head = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims,
                                                   1, num_aggs=self.num_aggs)
        self.beauty_head = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims,
                                                  1, num_aggs=self.num_aggs)
        self.color_head = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims,
                                                 1, num_aggs=self.num_aggs)
        self.composition_head = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims,
                                                       1, num_aggs=self.num_aggs)
        self.content_head = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims,
                                                   1, num_aggs=self.num_aggs)

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

        self.quality_pre = self.quality_head(output)
        self.beauty_pre = self.beauty_head(output)
        self.color_pre = self.color_head(output)
        self.composition_pre = self.composition_head(output)
        self.content_pre = self.content_head(output)

        return {
            'aesthetic_quality': self.quality_pre,
            'beauty': self.beauty_pre,
            'color': self.color_pre,
            'composition': self.composition_pre,
            'content': self.content_pre
        }

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


class cnn_pool_jas(BaseNetwork):
    def __init__(self):
        super(cnn_pool_jas, self).__init__()
        self.cnn = CNN()
        self.GCN = SoftPoolingGcnEncoder(class_num=1)
        self.GCN.init_weights()

    def forward(self, images, size, boxes, adj):
        x = self.cnn(images, size, boxes)
        out_dict = self.GCN(x, adj)
        return out_dict

    def Linkloss(self, adj):
        return self.GCN.loss(adj)
