import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import os
import numpy as np

from model.base_network import BaseNetwork
import config as cfg

from model.FNet2 import FNet_vit


class ViT(FNet_vit):
    def __init__(self):
        super(ViT, self).__init__()

    def forward(self, x):
        x = self.forward_features(x)
        return x


class Featurer(BaseNetwork):
    def __init__(self, cnn_pretrained=True, model_type='resnet18'):
        super(Featurer, self).__init__()
        self.feature_extraction = ViT()  # [B,768]

    def vit_load(self, weights_path=os.path.join(cfg.vit_path, "%s.pth" % ("vit"))):
        state_dict = torch.load(weights_path)
        self.feature_extraction.load_state_dict(state_dict, strict=False)
        print("「%s」参数文件加载完成" % (weights_path))

    def forward(self, images):
        """
        :param images: list 里面k个元素 每个元素[B,C,H,W]
        :return features [B,N,D] N结点个数 D特征维
        """
        feature_list = []
        for i in range(cfg.k_node + 1):
            feature = self.feature_extraction(images[i])
            feature = feature.unsqueeze(1)
            feature_list.append(feature)
        features = torch.cat(feature_list, dim=1)
        return features


from model.detectionGCN_v4 import GraphConv, GcnEncoderGraph


class SoftPoolingGcnEncoder(GcnEncoderGraph):
    def __init__(self, max_num_nodes=cfg.k_node, input_dim=512, hidden_dim=648, embedding_dim=512,
                 assign_hidden_dim=648, num_layers=3, assign_ratio=0.5, assign_num_layers=-1, num_pooling=2,
                 pred_hidden_dims=[256], concat=True, bn=True, dropout=0.0, linkpred=True,
                 assign_input_dim=-1, args=None):

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

        self.head = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims,
                                           cfg.class_num, num_aggs=self.num_aggs)

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

        embedding_tensor = self.gcn_forward(x, adj, self.conv_first, self.conv_block, self.conv_last,
                                            embedding_mask)  # [batch_size x num_nodes x embedding_dim]
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
        pre_attr = self.head(output)
        return pre_attr


class fnet_pool(BaseNetwork):
    def __init__(self):
        super(fnet_pool, self).__init__()
        self.FE = Featurer()
        self.GCN = SoftPoolingGcnEncoder(input_dim=768, hidden_dim=512, embedding_dim=512,
                                         assign_hidden_dim=512)
        self.GCN.init_weights()
        self.FE.init_weights()
        self.FE.vit_load()

    def forward(self, images, adj):
        x = self.FE(images)
        out_dict = self.GCN(x, adj)
        return out_dict
