import torch
import torch.nn as nn
import math
from torchvision import models
from model.base_network import BaseNetwork


class ChannelAttention(nn.Module):
    def __init__(self, Channel_nums):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.alpha = nn.Parameter(data=torch.FloatTensor([0.5]), requires_grad=True)
        self.beta = nn.Parameter(data=torch.FloatTensor([0.5]), requires_grad=True)
        self.gamma = 2
        self.b = 1
        self.k = self.get_kernel_num(Channel_nums)
        self.conv1d = nn.Conv1d(kernel_size=self.k, in_channels=1, out_channels=1, padding=self.k // 2)
        self.sigmoid = nn.Sigmoid()

    def get_kernel_num(self, C):  # odd|t|最近奇数
        t = math.log2(C) / self.gamma + self.b / self.gamma
        floor = math.floor(t)
        k = floor + (1 - floor % 2)
        return k

    def forward(self, x):
        F_avg = self.avg_pool(x)
        F_max = self.max_pool(x)
        F_add = 0.5 * (F_avg + F_max) + self.alpha * F_avg + self.beta * F_max
        F_add_ = F_add.squeeze(-1).permute(0, 2, 1)
        F_add_ = self.conv1d(F_add_).permute(0, 2, 1).unsqueeze(-1)
        out = self.sigmoid(F_add_)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, Channel_num):
        super(SpatialAttention, self).__init__()
        self.channel = Channel_num
        self.Lambda = 0.6  # separation rate
        self.C_im = self.get_important_channelNum(Channel_num)
        self.C_subim = Channel_num - self.C_im
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.norm_active = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Sigmoid()
        )

    def get_important_channelNum(self, C):  # even|t|最近偶数
        t = self.Lambda * C
        floor = math.floor(t)
        C_im = floor + floor % 2
        return C_im

    def get_im_subim_channels(self, C_im, M):
        _, topk = torch.topk(M, dim=1, k=C_im)
        topk_ = topk.squeeze(-1).squeeze(-1)  # [B,C_im]
        important_channels = torch.zeros_like(M.squeeze(-1).squeeze(-1))
        subimportant_channels = torch.ones_like(M.squeeze(-1).squeeze(-1))
        for i in range(M.shape[0]):
            important_channels[i][topk_[i]] = 1
            subimportant_channels[i][topk_[i]] = 0
        important_channels = important_channels.unsqueeze(-1).unsqueeze(-1)
        subimportant_channels = subimportant_channels.unsqueeze(-1).unsqueeze(-1)
        return important_channels, subimportant_channels

    def get_features(self, im_channels, subim_channels, channel_refined_feature):
        import_features = im_channels * channel_refined_feature
        subimportant_features = subim_channels * channel_refined_feature
        return import_features, subimportant_features

    def forward(self, x, M):
        important_channels, subimportant_channels = self.get_im_subim_channels(self.C_im, M)
        important_features, subimportant_features = self.get_features(important_channels, subimportant_channels, x)

        im_AvgPool = torch.mean(important_features, dim=1, keepdim=True) * (self.channel / self.C_im)
        im_MaxPool, _ = torch.max(important_features, dim=1, keepdim=True)

        subim_AvgPool = torch.mean(subimportant_features, dim=1, keepdim=True) * (self.channel / self.C_subim)
        subim_MaxPool, _ = torch.max(subimportant_features, dim=1, keepdim=True)

        im_x = torch.cat([im_AvgPool, im_MaxPool], dim=1)
        subim_x = torch.cat([subim_AvgPool, subim_MaxPool], dim=1)

        A_S1 = self.norm_active(self.conv(im_x))
        A_S2 = self.norm_active(self.conv(subim_x))

        F1 = important_features * A_S1
        F2 = subimportant_features * A_S2

        refined_feature = F1 + F2

        return refined_feature


class ResBlock_HAM(BaseNetwork):
    def __init__(self, Channel_nums):
        super(ResBlock_HAM, self).__init__()
        self.channel = Channel_nums
        self.ChannelAttention = ChannelAttention(self.channel)
        self.SpatialAttention = SpatialAttention(self.channel)
        self.relu = nn.ReLU()

    def forward(self, x_in):
        residual = x_in
        channel_attention_map = self.ChannelAttention(x_in)
        channel_refined_feature = channel_attention_map * x_in
        final_refined_feature = self.SpatialAttention(channel_refined_feature, channel_attention_map)
        out = self.relu(final_refined_feature + residual)
        return out


class FC(BaseNetwork):
    def __init__(self, class_nums):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(512, 2 * 512 + 1),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(2 * 512 + 1, 2 * class_nums + 1),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(2 * class_nums + 1, class_nums)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class resnet18_HAM(BaseNetwork):
    def __init__(self, class_nums=None):
        super(resnet18_HAM, self).__init__()
        resnet18 = models.resnet18(pretrained=True)  # 得到resnet18的模型 并且拿到预训练好的参数
        model_list = list(resnet18.children())
        self.model_pre = nn.Sequential(*model_list[:3])  # 拿到resnet18一开始的conv-bn-relu
        self.ham1 = ResBlock_HAM(64)  # 第一个HAM注意力模块
        self.model_mid = nn.Sequential(*model_list[4:8])  # 拿到resnet18中间部分的resnetBlock
        self.ham2 = ResBlock_HAM(512)  # 第二个HAM注意力模块
        self.model_tail = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # 自适应池化
        # self.fc = FC(class_nums)
        self.init_weight()

    def init_weight(self):
        self.ham1.init_weights()
        self.ham2.init_weights()
        # self.fc.init_weights()

    def forward(self, images,adj=None):
        x = self.model_pre(images)
        x = self.ham1(x)
        x = self.model_mid(x)
        x = self.ham2(x)
        x = self.model_tail(x)
        return x
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        # return x
