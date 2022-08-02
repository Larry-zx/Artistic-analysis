from torch.utils import data
import config as cfg
from data.wikiArt import wikiArt


def get_loader(batch_size, mode='train', transform=None):
    dataset = None
    if cfg.dataset == 'wikiArt':
        dataset = wikiArt(transform=transform, mode=mode)
    else:
        print("请输入正确的数据集名称")
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode == 'train'),
                                  drop_last=True,
                                  num_workers=1)
    return data_loader
