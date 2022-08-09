from torch.utils import data
import config as cfg
from data.wikiArt import wikiArt
from data.SemArt import SemArt
from data.wikiArtObj import  wikiArtObj
from utils.utils import transform_obj

def get_loader(batch_size, mode='train', transform=None):
    dataset = None
    if cfg.dataset == 'wikiArt':
        dataset = wikiArt(transform=transform, mode=mode)
    elif cfg.dataset =='SemArt':
        dataset = SemArt(transform=transform , mode=mode)
    elif cfg.dataset =='wikiArtObj':
        dataset = wikiArtObj(transform=transform_obj() , mode=mode)
    else:
        print("请输入正确的数据集名称")
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode == 'train'),
                                  drop_last=True,
                                  num_workers=0)
    return data_loader
