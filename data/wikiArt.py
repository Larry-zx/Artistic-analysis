import json

from torch.utils import data
import os
from PIL import Image
import config as cfg
import math


class wikiArt(data.Dataset):
    def __init__(self, transform, mode='train'):
        super(wikiArt, self).__init__()
        json_file_path = os.path.join(cfg.json_path, "%s.json" % (mode))
        self.data = json.load(open(json_file_path, 'r'))
        self.transform = transform
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = os.path.join(cfg.img_dir, self.data[index][0])
        label_list = self.data[index][-1]
        artist = label_list[0]
        genre = label_list[1]
        style = label_list[2]
        image = Image.open(img_path).convert("RGB")
        if self.transform != None:
            image = self.transform(image)
        return image, artist, genre, style


def data_balance():
    Data = json.load(open(os.path.join(cfg.json_path, 'train.json'), 'r'))
    Attr_dict = {'artist': dict(), 'genre': dict(), 'style': dict()}
    for data in Data:
        label = data[1]
        artist = label[0]
        genre = label[1]
        style = label[2]
        if artist not in Attr_dict['artist'].keys():
            Attr_dict['artist'][artist] = 0
        if genre not in Attr_dict['genre'].keys():
            Attr_dict['genre'][genre] = 0
        if style not in Attr_dict['style'].keys():
            Attr_dict['style'][style] = 0

        Attr_dict['artist'][artist] += 1
        Attr_dict['genre'][genre] += 1
        Attr_dict['style'][style] += 1

    artist_num_list = []
    genre_num_list = []
    style_num_list = []

    for i in range(129):
        artist_num_list.append(Attr_dict['artist'][i])
    for i in range(11):
        genre_num_list.append(Attr_dict['genre'][129 + i])
    for i in range(27):
        style_num_list.append(Attr_dict['style'][140 + i])

    artist_weights = [1.0 / math.sqrt(x) for x in artist_num_list]
    genre_weights = [1.0 / math.sqrt(x) for x in genre_num_list]
    style_weights = [1.0 / math.sqrt(x) for x in style_num_list]
    return artist_weights, genre_weights, style_weights






