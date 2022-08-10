import torch.nn as nn
from torchvision import models
from ArtDataset_classifer.model.base_network import BaseNetwork


# CNN模块提取特征
class FeatureExtraction(nn.Module):
    def __init__(self, pretrained, model_type="resnet50"):
        super(FeatureExtraction, self).__init__()
        # 选择网络模型
        self.model = None
        if model_type == 'resnet50':
            self.model = models.resnet50(pretrained)  # 2048, 1, 1
        elif model_type == 'resnet18':
            self.model = models.resnet18(pretrained)  # 512
        elif model_type == 'GoogLeNet':
            self.model = models.googlenet(pretrained)
        elif model_type == 'vgg16':
            self.model = models.vgg16(pretrained)  # 512, 7, 7
        elif model_type == 'densenet121':
            self.model = models.densenet121(pretrained)  # 1024, 7, 7
        elif model_type == 'resnet101':
            self.model = models.resnet101(pretrained)
        elif model_type == 'densenet201':
            self.model = models.densenet201(pretrained)  # 1920, 7, 7
        elif model_type == 'inception_v3':
            self.model = models.inception_v3(pretrained)
        elif model_type == 'alexnet':
            self.model = models.alexnet(pretrained) #9216
        else:
            print("%s输入不正确,请输入包含在代码内的网络模型" % (model_type))

        self.model = nn.Sequential(*list(self.model.children())[:-1])  # 去掉网络的最后全连接层

    def forward(self, image):
        return self.model(image)


class FC(BaseNetwork):  # 443
    def __init__(self, output_dim, in_dim=0 ,  model_type='resnet18'):
        super(FC, self).__init__()
        num1 = 0
        num2 = 0
        if model_type == 'resnet50':  # 2048
            input_dim = 2048
            num1 = 2 * input_dim + 1
            num2 = 2 * output_dim + 1
        elif model_type == 'vgg16':  # 25088
            input_dim = 25088
            num1 = 1000
            num2 = 2 * output_dim + 1
        elif model_type == 'resnet18':
            input_dim = 512
            num1 = 2 * input_dim + 1
            num2 = 2 * output_dim + 1
        elif model_type == 'alexnet':
            input_dim = 9216
            num1 = 1000
            num2 = 2 * output_dim + 1
        elif model_type == 'densenet121':
            input_dim = 1024 * 7 * 7
            num1 = 1024
            num2 = 2 * output_dim + 1
        elif model_type == 'odg':
            input_dim = in_dim
            num1 = 2*input_dim+1
            num2 = 2*output_dim+1

        self.fc = nn.Sequential(
            nn.Linear(input_dim, num1),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(num1, num2),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(num2, output_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        res = self.fc(x)
        return res
