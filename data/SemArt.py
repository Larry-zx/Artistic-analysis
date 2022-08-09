import json

from torch.utils import data
import os
from PIL import Image
import ArtDataset_classifer.config as config
import math
import numpy as np

class SemArt(data.Dataset):
    def __init__(self, transform, mode='train'):
        super(SemArt, self).__init__()
        json_file_path = os.path.join(config.json_path, "%s.json" % (mode))
        self.data = json.load(open(json_file_path, 'r'))
        self.transform = transform
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = os.path.join(config.img_dir, self.data[index]['image_file'])
        image = Image.open(img_path).convert("RGB")
        if self.transform != None:
            image = self.transform(image)
        return image, self.data[index]


def data_balance():
    mode = 'train'
    path = os.path.join(config.json_path, '%s.json' % (mode))
    data = json.load(open(path, 'r'))
    attr_list = ['author', 'technique', 'material', 'type', 'school', 'timeFrame']
    num = len(data)
    attr_dict = {}
    weight_dict = {}
    for attr in attr_list:
        attr_dict[attr] = dict()
        weight_dict[attr] = []
        for i in range(num):
            a = data[i][attr]
            if a not in attr_dict[attr].keys():
                attr_dict[attr][a] = 0
            attr_dict[attr][a] += 1
        for j in range(len(attr_dict[attr].keys())):
            weight_dict[attr].append(attr_dict[attr][j])
        weight_dict[attr] = np.array(weight_dict[attr]) / num
        weight_dict[attr] = np.array([1.0/math.sqrt(x) for x in weight_dict[attr]])
    return weight_dict


def Sem_kind_num():
    mode = 'test'
    path = os.path.join('../datasets/SemArt/json', '%s.json' % (mode))
    data = json.load(open(path, 'r'))
    attr_list = ['author', 'technique', 'material', 'type', 'school', 'timeFrame']
    num = len(data)
    attr_dict = {}
    weight_dict = {}
    for attr in attr_list:
        attr_dict[attr] = dict()
        weight_dict[attr] = []
        for i in range(num):
            a = data[i][attr]
            if a not in attr_dict[attr].keys():
                attr_dict[attr][a] = 0
            attr_dict[attr][a] += 1
        for j in range(len(attr_dict[attr].keys())):
            weight_dict[attr].append(attr_dict[attr][j])

    return weight_dict
