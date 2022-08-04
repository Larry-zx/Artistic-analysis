import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from ArtDataset_classifer.config import Attr_list

model_type_list = ['resnet18', 'alexnet', 'vgg16']
task_type_list = ['multi', 'single']
dataset = 'wikiArt'
attr_list = Attr_list

def make_loss(model_type, task_type):
    loss = pd.read_csv(os.path.join( dataset ,task_type, model_type, 'loss.csv'))
    for attr in attr_list:
        attr_loss = np.array(loss[attr])
        x = np.arange(len(attr_loss))
        plt.plot(x, attr_loss)
        plt.title(attr + '_loss')
        plt.ylabel('loss')
        plt.savefig(os.path.join(dataset + '/res_img/loss', "%s-%s-%s.png" % (model_type, task_type, attr)))
        plt.show()


for model_type in model_type_list:
    for task_type in task_type_list:
        make_loss(model_type, task_type)
