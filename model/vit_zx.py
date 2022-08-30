from model.base_network import BaseNetwork
from model.ViT import VisionTransformer
from model.diffpooling.GCN_set2set import GcnSet2SetEncoder
from model.diffpooling.GCN_pool import SoftPoolingGcnEncoder
import torch
import config as cfg


class vit_feature(VisionTransformer):
    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x


class vit_zx(BaseNetwork):
    def __init__(self, class_num, k_neighbour=21, gcn_mode='diffpool' , pretrain=True):
        super(vit_zx, self).__init__()
        self.k = k_neighbour
        self.class_num = class_num
        self.pretrained = pretrain
        self.vit = vit_feature(img_size=224,
                               patch_size=16,
                               embed_dim=768,
                               depth=12,
                               num_heads=12,
                               representation_size=None,
                               num_classes=class_num)
        self.set_gcn(gcn_mode=gcn_mode)

    def set_gcn(self, gcn_mode):
        if gcn_mode == 'set2set':
            self.gcn = GcnSet2SetEncoder(class_num=self.class_num, input_dim=768, hidden_dim=648, embedding_dim=512,
                                         num_layers=2, )
        elif gcn_mode == 'diffpool':
            self.gcn = SoftPoolingGcnEncoder(class_num=self.class_num, max_num_nodes=self.k, input_dim=768,
                                             hidden_dim=648,
                                             embedding_dim=512, assign_hidden_dim=512, num_layers=3, assign_ratio=0.5,
                                             num_pooling=2)
        self.gcn.init_weights()
        if self.pretrained:
            self.vit_load()


    def vit_load(self):
        self.vit.load_state_dict(
            torch.load('checkpoints/finetun/vit.pth', map_location=torch.device("cpu")),
            strict=False)
        print('checkpoints/finetun/vit.pth', '加载参数完成')

    def forward(self, images):
        x = self.vit(images)
        adj = getAdj_by_knn_matrix(x, k=self.k)
        pre_attr = self.gcn(x, adj)
        return pre_attr


# 计算特征与特征直接的L2距离   余弦相似度
def pairwise_distance(x):
    with torch.no_grad():
        x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)


def getAdj_by_knn_matrix(x, k):
    # 计算特征与特征直接的L2距离
    dist = pairwise_distance(x.detach())  # [batch_size, num_points, num_points]
    # 找到距离最小的k个结点
    _, nn_idx = torch.topk(-dist, k=k)  # nn_idx.shape [B,num_points , k_node]
    # 根据nn_idx构建邻接矩阵 前k个结点adj取1其余取0
    adj = torch.zeros(size=(x.shape[0], x.shape[1], x.shape[1])).to(cfg.device)
    adj = adj.scatter(2, nn_idx, 1).to(cfg.device)
    return adj
