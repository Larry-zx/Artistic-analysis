import torch

from model.network import *
from torch.nn import init
import config as cfg


class multitask_classifer(BaseNetwork):
    def __init__(self, init=True, cnn_pretrained=True, model_type='resnet18'):
        super(multitask_classifer, self).__init__()
        self.cnn = FeatureExtraction(cnn_pretrained, model_type)
        self.attr_list = cfg.Attr_list
        for attr in self.attr_list:
            exec("self.FC_{}=FC(model_type, output_dim=cfg.attr_num['{}'])".format(attr, attr))
    #     if init:
    #         self.init_FC()
    #
    # def init_FC(self):
    #     for attr in self.attr_list:
    #         self.FC_dict[attr].init_weights()
    #
    # def forward(self, image):
    #     feature = self.cnn(image)
    #     out_dict = {}
    #     for attr in self.attr_list:
    #         out_dict[attr] = self.FC_dict[attr](feature)
    #     return out_dict


class singletask_classifer(BaseNetwork):
    def __init__(self, init=True, cnn_pretrained=True, model_type='resnet50'):
        super(singletask_classifer, self).__init__()
        self.attr_list = cfg.Attr_list
        self.CNN_dict = {}
        for attr in self.attr_list:
            self.CNN_dict[attr] = FeatureExtraction(cnn_pretrained, model_type).to(cfg.device)
        self.FC_dict = {}
        for attr in self.attr_list:
            self.FC_dict[attr] = FC(model_type, output_dim=cfg.attr_num[attr]).to(cfg.device)
        if init:
            self.init_FC()

    def init_FC(self):
        for attr in self.attr_list:
            self.FC_dict[attr].init_weights()

    def forward(self, image):
        out_dict = {}
        for attr in self.attr_list:
            out_dict[attr] = self.FC_dict[attr](self.CNN_dict[attr](image))
        return out_dict


"""
transformer部分
"""
from model.ViT import *


class vit_multi(VisionTransformer, BaseNetwork):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        super(vit_multi, self).__init__()
        self.head_dict = {}
        for attr in cfg.Attr_list:
            self.head_dict[attr] = nn.Linear(self.num_features, cfg.attr_num[attr]).to(cfg.device)
        self.mlp_init()

    def mlp_init(self):
        for attr in cfg.Attr_list:
            self.kaiming_init(self.head_dict[attr])

    def kaiming_init(self, m):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        print('Network [%s] 成功初始化参数 方法「kaiming」' % (type(m).__name__))

    def forward(self, x):
        features = self.forward_features(x)
        out_dict = {}
        for attr in cfg.Attr_list:
            out_dict[attr] = self.head_dict[attr](features)
        return out_dict


class vit_single(BaseNetwork):
    def __init__(self, weights_path=None):
        super(vit_single, self).__init__()
        self.vit_dict = {}
        for attr in cfg.Attr_list:
            self.vit_dict[attr] = VisionTransformer(num_classes=cfg.attr_num[attr]).to(cfg.device)
        self.mlp_init()
        if weights_path != None:
            self.vit_load(weights_path)

    def mlp_init(self):
        for attr in cfg.Attr_list:
            self.kaiming_init(self.vit_dict[attr].head_zx)

    def kaiming_init(self, m):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        print('Network [%s] 成功初始化参数 方法「kaiming」' % (type(m).__name__))

    def vit_load(self, weights_path):
        state_dict = torch.load(weights_path)
        for attr in cfg.Attr_list:
            self.vit_dict[attr].load_state_dict(state_dict, strict=False)
        print("「%s」参数文件加载完成" % (weights_path))

    def forward(self, x):
        out_dict = {}
        for attr in cfg.Attr_list:
            out_dict[attr] = self.vit_dict[attr](x)
        return out_dict


"""
Swin transformer部分
"""
from model.Swin_ViT import *


class swin_multi(SwinTransformer, BaseNetwork):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super(swin_multi, self).__init__()
        self.head_dict = {}
        for attr in cfg.Attr_list:
            self.head_dict[attr] = nn.Linear(self.num_features, cfg.attr_num[attr]).to(cfg.device)
        self.mlp_init()

    def mlp_init(self):
        for attr in cfg.Attr_list:
            self.kaiming_init(self.head_dict[attr])

    def kaiming_init(self, m):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        print('Network [%s] 成功初始化参数 方法「kaiming」' % (type(m).__name__))

    def forward(self, x):
        # x: [B, L, C]
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x, H, W = layer(x, H, W)

        x = self.norm(x)  # [B, L, C]
        x = self.avgpool(x.transpose(1, 2))  # [B, C, 1]
        features = torch.flatten(x, 1)
        out_dict = {}
        for attr in cfg.Attr_list:
            out_dict[attr] = self.head_dict[attr](features)
        return out_dict


class swin_single(BaseNetwork):
    def __init__(self, weights_path=None):
        super(swin_single, self).__init__()
        self.vit_dict = {}
        for attr in cfg.Attr_list:
            self.vit_dict[attr] = SwinTransformer(num_classes=cfg.attr_num[attr]).to(cfg.device)
        self.mlp_init()
        if weights_path != None:
            self.vit_load(weights_path)

    def mlp_init(self):
        for attr in cfg.Attr_list:
            self.kaiming_init(self.vit_dict[attr].head_zx)

    def kaiming_init(self, m):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        print('Network [%s] 成功初始化参数 方法「kaiming」' % (type(m).__name__))

    def vit_load(self, weights_path):
        state_dict = torch.load(weights_path)
        for attr in cfg.Attr_list:
            self.vit_dict[attr].load_state_dict(state_dict, strict=False)
        print("「%s」参数文件加载完成" % (weights_path))

    def forward(self, x):
        out_dict = {}
        for attr in cfg.Attr_list:
            out_dict[attr] = self.vit_dict[attr](x)
        return out_dict



"""ViG系列"""
from model.ViG import DeepGCN as vigGCN
import torch.nn.functional as F
import torch.nn as nn


class ViG_multi(vigGCN, BaseNetwork):
    def __init__(self):
        super(ViG_multi, self).__init__()
        self.attr_list = cfg.Attr_list
        self.stem = self.stem
        self.pos_embed = self.pos_embed
        self.backbone = self.backbone
        self.artist_head = self.prediction_layer(num_classs=cfg.attr_num['artist'])
        self.genre_head = self.prediction_layer(num_classs=cfg.attr_num['genre'])
        self.style_head = self.prediction_layer(num_classs=cfg.attr_num['style'])

    def prediction_layer(self, num_classs):
        prediction = nn.Sequential(nn.Conv2d(self.channels[-1], 1024, 1, bias=True),
                                   nn.BatchNorm2d(1024),
                                   self.act_layer(self.act),
                                   nn.Dropout(0.5),
                                   nn.Conv2d(1024, num_classs, 1, bias=True))
        return prediction

    def get_feature(self, inputs):
        x = self.stem(inputs) + self.pos_embed  # [ b , C ,56 , 56 ] + [1 , C , 56 ,56] -> [b , C , 56 , 56 ]
        B, C, H, W = x.shape
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
        x = F.adaptive_avg_pool2d(x, 1)
        return x

    def forward(self, inputs):
        x = self.get_feature(inputs)
        out_dict = {}
        out_dict['artist'] = self.artist_head(x).squeeze(-1).squeeze(-1)
        out_dict['genre'] = self.genre_head(x).squeeze(-1).squeeze(-1)
        out_dict['style'] = self.style_head(x).squeeze(-1).squeeze(-1)
        return out_dict


class ViG_single(BaseNetwork):
    def __init__(self):
        super(ViG_single, self).__init__()
        self.artist_vig = vigGCN(num_class=cfg.attr_num['artist'])
        self.genre_vig = vigGCN(num_class=cfg.attr_num['genre'])
        self.style_vig = vigGCN(num_class=cfg.attr_num['style'])

    def forward(self, inputs):
        out_dict = {}
        out_dict['artist'] = self.artist_vig(inputs)
        out_dict['genre'] = self.genre_vig(inputs)
        out_dict['style'] = self.style_vig(inputs)
        return out_dict


