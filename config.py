# 超参数

import argparse
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataSet', default='wikiArt', help="使用哪一个数据集")
        self.parser.add_argument('--img_dir', default='Datasets/SemArt/Images', help='图片存放的路径')
        self.parser.add_argument('--json_path', default='Datasets/SemArt/json', help='json文件的路径')

        self.parser.add_argument('--bs', default=32, type=int, help='在train集上的batchSize')

        self.parser.add_argument('--epoch', default=30, type=int, help="训练迭代次数")
        self.parser.add_argument('--lr', default=0.0001, type=float, help="学习率")
        self.parser.add_argument('--gpu', default=0, type=int, help="显卡ID")

        self.parser.add_argument('--model_type', default='resnet18', type=str, help="CNN的模型名称")
        self.parser.add_argument('--task_type', default='multi', type=str, help='multi/single')
        self.parser.add_argument('--pretrained', default=True, type=bool, help="模型是否使用预训练的参数")

        self.parser.add_argument('--model_path', default=None, type=str, help='加载模型参数文件的路径')
        self.parser.add_argument('--mode', default='train', type=str, help='train / test')

    def parse(self):
        self.initialize()
        opt = self.parser.parse_args()
        # 根据数据集名称选择图片路径以及json文件
        if opt.dataSet == 'wikiArt':
            opt.img_dir = 'datasets/wikiart'
            opt.json_path = 'datasets/wikiart/json/'

        return opt


opt = BaseOptions().parse()

# 文件路径
dataset = opt.dataSet
img_dir = opt.img_dir
json_path = opt.json_path

# 超参数
batchsize = opt.bs
test_batchsize = opt.bs
val_batchsize = opt.bs

epoch = opt.epoch
learning_rate = opt.lr
DEVICE_ID = opt.gpu

model_type = opt.model_type
task_type = opt.task_type  # multi or single
pretrained = opt.pretrained

model_path = opt.model_path
mode = opt.mode

device = torch.device("cuda:" + str(DEVICE_ID) if torch.cuda.is_available() else "cpu")
Attr_list = ['artist', 'genre', 'style']
