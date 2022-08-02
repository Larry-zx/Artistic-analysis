import json
from torchvision import transforms
import math
import time
import os


def set_transform():
    transform = []
    transform.append(transforms.Resize(256))
    transform.append(transforms.CenterCrop(250))
    transform.append(transforms.Resize(size=(224, 224)))
    transform.append(transforms.ToTensor())
    transform.append(transforms.RandomHorizontalFlip(p=0.5))
    transform.append(transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                          std=[0.5, 0.5, 0.5]))
    transform = transforms.Compose(transform)
    return transform


# 计算当前运行时间
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%d h %d m %d s' % (h, m, s)


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print("%s创建完成"%dir_path)
