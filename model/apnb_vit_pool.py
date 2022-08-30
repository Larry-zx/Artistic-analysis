from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
from model.base_network import BaseNetwork
import config as cfg
from model.onepic_cnn_pool import SoftPoolingGcnEncoder
import os

import time


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(BaseNetwork):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):

        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1  # 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)  # [B, 196, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x


def vit_base_patch16_224():
    """
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=0)
    return model


def vit_path_id(node_list, w, h):
    patch = 16
    img_id_array_list = []
    for node in node_list:
        id_list = []
        x1 = node[0]
        y1 = node[1]
        x2 = node[2]
        y2 = node[3]
        new_x1 = int((x1 / w) * 224)
        new_x2 = int((x2 / w) * 224)
        new_y1 = int((y1 / h) * 224)
        new_y2 = int((y2 / h) * 224)
        x1_id = new_x1 // patch
        y1_id = new_y1 // patch
        x2_id = new_x2 // patch
        y2_id = new_y2 // patch
        for x in range(x1_id, x2_id + 1):
            for y in range(y1_id, y2_id + 1):
                node_id = (224 / patch) * y + x
                id_list.append(node_id)
        img_id_array_list.append(id_list)
    return img_id_array_list


class apnb_vit_pool(BaseNetwork):
    def __init__(self):
        super(apnb_vit_pool, self).__init__()
        self.vit = vit_base_patch16_224()
        self.GCN = SoftPoolingGcnEncoder(input_dim=768, hidden_dim=512, embedding_dim=512,
                                         assign_hidden_dim=512, class_num=cfg.class_num)
        self.vit.init_weights()
        self.GCN.init_weights()
        self.vit_load_weights()

    def vit_load_weights(self):
        device = torch.device("cpu")
        self.vit.load_state_dict(torch.load(os.path.join('checkpoints/finetun/vit.pth'), map_location=device),
                                 strict=False)
        print("checkpoints/finetun/vit.pth记载完成")

    def get_batch_node_feature(self, B, size, boxes, vit_features):
        # 全部batch的节点特征
        all_batch_node_feature_list = []
        for b in range(B):
            h, w = size[b]
            node_list = boxes[b]
            path_id_list = vit_path_id(node_list, w, h)
            # 当前节点的特征
            now_batch_node_feature_list = []
            for patch in path_id_list[:-1]:
                # 一个节点的特征
                one_node_feature_list = []
                for id in patch:
                    id = int(id)
                    # 一个patch的特征
                    tmp_feature = vit_features[b:b + 1, id + 1:id + 2, :]
                    one_node_feature_list.append(tmp_feature)
                one_node_feature = torch.cat(one_node_feature_list, dim=1)
                # 求均值
                one_node_feature = torch.mean(one_node_feature, dim=1).unsqueeze(1)
                # 保存到当前batch中
                now_batch_node_feature_list.append(one_node_feature)
            # 全图特征
            now_batch_node_feature_list.append(torch.mean(vit_features[b:b + 1, :, :], dim=1).unsqueeze(1))
            # 得到当前batch的节点特征
            now_batch_node_feature = torch.cat(now_batch_node_feature_list, dim=1)
            # 保存到所有batch中
            all_batch_node_feature_list.append(now_batch_node_feature)
        all_batch_node_feature = torch.cat(all_batch_node_feature_list, dim=0)
        return all_batch_node_feature

    def forward(self, images, size, boxes, adj):
        vit_features = self.vit(images)
        batch_node_features = self.get_batch_node_feature(images.shape[0], size, boxes, vit_features)
        out_dict = self.GCN(batch_node_features, adj)
        return out_dict
