import json

import numpy as np
import torch
from torch.utils import data
import os
from PIL import Image
import config as cfg
import math
import random
import json


def get_random():
    return random.randrange(1, 10)


# 将图像分成九个patch
def make_patch(width, height, patch_id):
    # 对于左上角
    h_num = (patch_id - 1) // 3
    w_num = (patch_id - 1) % 3
    x1 = int(width * w_num // 3)
    y1 = int(height * h_num // 3)
    x2 = int(width * (w_num + 1) // 3)
    y2 = int(height * (h_num + 1) // 3)
    return [x1, y1, x2, y2]


def cal_IoU(box1, box2):
    x1 = box1[0]
    y1 = box1[1]
    x2 = box1[2]
    y2 = box1[3]

    x3 = box2[0]
    y3 = box2[1]
    x4 = box2[2]
    y4 = box2[3]
    assert x1 == min(x1, x2)
    assert y1 == min(y1, y2)
    assert x3 == min(x3, x4)
    assert y3 == min(y3, y4)

    # 计算C
    x_min, x_max = min(x1, x2, x3, x4), max(x1, x2, x3, x4)
    y_min, y_max = min(y1, y2, y3, y4), max(y1, y2, y3, y4)
    C = math.fabs((x_max - x_min) * (y_max - y_min))
    # A B单独
    A = math.fabs((x2 - x1) * (y2 - y1))
    B = math.fabs((x4 - x3) * (y4 - y3))
    # 计算重合面积
    wide = min(x2, x4) - max(x1, x3)
    high = min(y2, y4) - max(y1, y3)
    AB = wide * high if wide * high >= 0 else 0
    # 计算机IoU
    IoU = (A + B - AB) / C
    return IoU


class wikiArtObj(data.Dataset):
    def __init__(self, transform, mode='train'):
        super(wikiArtObj, self).__init__()
        json_file_path = os.path.join(cfg.json_path, "%s.json" % (mode))
        self.data = json.load(open(json_file_path, 'r'))
        self.transform = transform
        assert self.transform != None
        self.k_node = cfg.k_node
        assert self.k_node == 9

    def __len__(self):
        return len(self.data)

    def get_node_coord(self):
        # 判断一下obj数量够不够
        if (self.k_node <= len(self.boxes)):  # 结果数量足够就去前k个
            for i in range(self.k_node):
                crop_box = [int(self.boxes[i][0]), int(self.boxes[i][1]), int(self.boxes[i][2]), int(self.boxes[i][3])]
                self.node_Coordinate_list.append(crop_box)
        else:  # 结点数量不足 就取patch 目前用的是3x3
            for i in range(len(self.boxes)):
                crop_box = (int(self.boxes[i][0]), int(self.boxes[i][1]), int(self.boxes[i][2]), int(self.boxes[i][3]))
                self.node_Coordinate_list.append(crop_box)
            for j in range(len(self.boxes), self.k_node):
                self.node_Coordinate_list.append(make_patch(self.width, self.height, get_random()))
        # 把全图的加上
        self.node_Coordinate_list.append([0, 0, self.width, self.height])

    def get_node_img(self):
        for i in range(self.k_node):
            crop_box = self.node_Coordinate_list[i]
            node_img = self.image.crop(crop_box)
            self.node_img.append(self.transform(node_img))
        # 全图特征
        self.node_img.append(self.transform(self.image))

    # 根据 AUB/C 来制造邻接矩阵
    def make_Adj_matrix(self):
        Adj = []
        for i in range(len(self.node_Coordinate_list)):
            Iou_list = []
            for j in range(len(self.node_Coordinate_list)):
                Iou = cal_IoU(self.node_Coordinate_list[i], self.node_Coordinate_list[j])
                Iou_list.append(Iou)
            Adj.append(Iou_list)
        return torch.tensor(Adj)

    def __getitem__(self, index):
        img_path = os.path.join(cfg.img_dir, self.data[index]['image_path'])
        artist = self.data[index]['artist']
        genre = self.data[index]['genre']
        style = self.data[index]['style']
        self.boxes = self.data[index]['boxes']
        self.image = Image.open(img_path).convert("RGB")
        self.width, self.height = self.image.size
        # 结点的box坐标
        self.node_Coordinate_list = []
        self.get_node_coord()
        # 每个物体框的图片都变为224x224
        self.node_img = []
        self.get_node_img()
        # AUB/C生成的邻接矩阵
        self.adj = self.make_Adj_matrix()
        return {'labels': [artist, genre, style], 'images': self.node_img, 'adj': self.adj}

    # 返回值的情况
    """
    lables list(B个值 , B个值 ,B个值)
    image: 长度为10的list image[i] -> [B,C,H,W]
    adj： [B,num ,num]
    """


def data_balance():
    Data = json.load(open(os.path.join(cfg.json_path, 'train.json'), 'r'))
    Attr_dict = {'artist': dict(), 'genre': dict(), 'style': dict()}
    for i in range(129):
        Attr_dict['artist'][i] = 0
    for i in range(11):
        Attr_dict['genre'][i] = 0
    for i in range(27):
        Attr_dict['style'][i] = 0
    for data in Data:
        artist = data['artist']
        genre = data['genre']
        style = data['style']
        Attr_dict['artist'][artist] += 1
        Attr_dict['genre'][genre] += 1
        Attr_dict['style'][style] += 1

    artist_num_list = []
    genre_num_list = []
    style_num_list = []

    for i in range(129):
        artist_num_list.append(Attr_dict['artist'][i])
    for i in range(11):
        genre_num_list.append(Attr_dict['genre'][i])
    for i in range(27):
        style_num_list.append(Attr_dict['style'][i])

    artist_weights = [1.0 / math.sqrt(x) for x in artist_num_list]
    genre_weights = [1.0 / math.sqrt(x) for x in genre_num_list]
    style_weights = [1.0 / math.sqrt(x) for x in style_num_list]

    return artist_weights, genre_weights, style_weights


def wiki_kind_num():
    Data = json.load(open(os.path.join('..', cfg.json_path, 'test.json'), 'r'))

    Attr_dict = {'artist': dict(), 'genre': dict(), 'style': dict()}
    for i in range(129):
        Attr_dict['artist'][i] = 0
    for i in range(11):
        Attr_dict['genre'][i] = 0
    for i in range(27):
        Attr_dict['style'][i] = 0
    for data in Data:
        artist = data['artist']
        genre = data['genre']
        style = data['style']
        Attr_dict['artist'][artist] += 1
        Attr_dict['genre'][genre] += 1
        Attr_dict['style'][style] += 1

    artist_num_list = []
    genre_num_list = []
    style_num_list = []

    for i in range(129):
        artist_num_list.append(Attr_dict['artist'][i])
    for i in range(11):
        genre_num_list.append(Attr_dict['genre'][i])
    for i in range(27):
        style_num_list.append(Attr_dict['style'][i])
    nums_dict = {}
    nums_dict['artist'] = artist_num_list
    nums_dict['genre'] = genre_num_list
    nums_dict['style'] = style_num_list
    return nums_dict
