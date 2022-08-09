import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from model.gcn_lib.torch_vertex import obj_Grapher
from model.gcn_lib.torch_nn import act_layer
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from model.base_network import BaseNetwork


class Downsample(nn.Module):
    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x, boxes = x
        x = self.conv(x)
        return x


class Channel_change(nn.Module):
    def __init__(self, in_dim=2, out_dim=768):
        super(Channel_change, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, stride=1, padding=0, kernel_size=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x, boxes = x
        x = self.conv(x)
        return x


class Features_Extractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Features_Extractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x  # .reshape(B, C, N, 1)


class objGCN(BaseNetwork):
    def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
        super(objGCN, self).__init__()
        self.k = 5  # neighbor num (default:9)
        self.conv = 'mr'  # graph conv layer {edge, mr}
        self.act = 'gelu'  # activation layer {relu, prelu, leakyrelu, gelu, hswish}
        self.norm = 'batch'  # batch or instance normalization {batch, instance}
        self.bias = True  # bias of conv layer True or False
        self.dropout = 0.0  # dropout rate
        self.use_dilation = True  # use dilated knn or not
        self.epsilon = 0.2  # stochastic epsilon for gcn
        self.use_stochastic = False  # stochastic for gcn, True or False
        self.drop_path = drop_path_rate
        self.blocks = [2, 2, 6, 2]  # number of basic blocks in the backbone
        self.channels = [48, 96, 240, 384]  # number of channels of deep features
        self.n_classes = num_classes  # Dimension of out_channels
        self.emb_dims = 1024  # Dimension of embeddings
        self.n_blocks = sum(self.blocks)
        self.n = 20  # 结点数量
        channels = self.channels
        reduce_ratios = [1, 1, 1, 1]  # 缩小的比率
        dpr = [x.item() for x in
               torch.linspace(0, self.drop_path, self.n_blocks)]  # stochastic depth decay rule 随机深度衰减规则
        num_knn = [int(x.item()) for x in torch.linspace(self.k, self.k, self.n_blocks)]  # number of knn's k
        max_dilation = 49 // max(num_knn)

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(self.blocks)):
            if i > 0:  # i=0 时不处理
                self.backbone.append(Channel_change(channels[i - 1], channels[i]))
            for j in range(self.blocks[i]):
                self.backbone += [
                    Seq(obj_Grapher(channels[i], num_knn[idx], min(idx // 4 + 1, max_dilation), self.conv, self.act,
                                    self.norm,
                                    self.bias, self.use_stochastic, self.epsilon, reduce_ratios[i], n=self.n,
                                    drop_path=dpr[idx],
                                    relative_pos=False),
                        FFN(channels[i], channels[i] * 4, act=self.act, drop_path=dpr[idx])
                        )]
                idx += 1
        self.backbone = Seq(*self.backbone)

        self.Extractor1 = Features_Extractor(in_channels=3, out_channels=8)
        self.Extractor2 = Features_Extractor(in_channels=8, out_channels=24)
        self.Extractor3 = Features_Extractor(in_channels=24, out_channels=48)

    def make_node_feature(self, image, boxes, size):
        H, W = size
        f1 = self.Extractor1(image)
        f2 = self.Extractor2(f1)
        f3 = self.Extractor3(f2)
        f3 = F.interpolate(f3 , size=(f2.shape[2] , f2.shape[3]))
        B, D, h, w = f3.shape
        batch_feature = []
        for b in range(B):
            x_ratio = W[b] / w
            y_ratio = H[b] / h
            node_feature = []
            for box in boxes[b]:
                x1 = int(box[0] / x_ratio)
                y1 = int(box[1] / y_ratio)
                x2 = int(box[2] / x_ratio)
                y2 = int(box[3] / y_ratio)
                if(x1==x2 or y1==y2):
                    print("坐标相同")
                nf = f3[b:b+1, :, y1:y2, x1:x2]
                nf = F.adaptive_avg_pool2d(nf, 1)
                node_feature.append(nf)
            feature = torch.concat(node_feature, dim=2)
            batch_feature.append(feature)
        feature = torch.concat(batch_feature , dim=0)
        return feature

    def forward(self, images, boxes,size):
        # 对x进行预处理 得到结点特征
        x = self.make_node_feature(images, boxes,size)
        for i in range(len(self.backbone)):
            x = self.backbone[i]((x, boxes))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x
