import torch

from ArtDataset_classifer.model.base_network import BaseNetwork
from ArtDataset_classifer.model.network import FeatureExtraction
import ArtDataset_classifer.config as cfg


class CNN(BaseNetwork):
    def __init__(self, cnn_pretrained=True, model_type='resnet18'):
        super(CNN, self).__init__()
        self.feature_extraction = FeatureExtraction(cnn_pretrained, model_type)  # restnet18 的话 返回[512,1,1]

    def forward(self, images):
        """
        :param images: list 里面k个元素 每个元素[B,C,H,W]
        :return features [B,N,D] N结点个数 D特征维度
        """
        feature_list = []
        for i in range(cfg.k_node + 1):
            feature = self.feature_extraction(images[i])  # [B , 512 ,  1 , 1]
            feature = feature.squeeze(-1).squeeze(-1)  # [B,512]
            feature = feature.unsqueeze(1)
            feature_list.append(feature)
        features = torch.cat(feature_list, dim=1)
        return features
