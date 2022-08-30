from model.network import FeatureExtraction, FC
from model.base_network import BaseNetwork
import config as cfg


class CNN(BaseNetwork):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = FeatureExtraction(pretrained=True, model_type='resnet18')
        self.fc = FC(output_dim=cfg.class_num)
        self.fc.init_weights()

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x
