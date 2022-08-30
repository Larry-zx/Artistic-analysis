import copy
import json
import os
import pandas as pd
from data.get_dataLoader import get_loader
from utils.utils import set_transform, timeSince, create_dir
from model.cnn_pool import cnn_pool
from model.FNet_pool import fnet_pool
from model.ham_pool import ham_pool
from model.vit_pool import vit_pool
import config as cfg
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
import time
import numpy as np
from data.wikiart_single import data_balance


class Classifier_Trainer(object):
    # 初始化参数
    def __init__(self):
        super(Classifier_Trainer, self).__init__()
        # 超参数
        self.batchsize = cfg.batchsize
        self.val_batchsize = cfg.val_batchsize
        self.test_batchsize = cfg.test_batchsize
        self.epoch = cfg.epoch
        self.lr = cfg.learning_rate
        # 数据加载器
        self.train_dataloader = get_loader(batch_size=self.batchsize, mode='train')
        self.val_dataloader = get_loader(batch_size=self.val_batchsize, mode='val')
        self.test_dataloader = get_loader(batch_size=self.test_batchsize, mode='test')
        # 计算设备
        self.device = torch.device("cuda:" + str(cfg.DEVICE_ID) if torch.cuda.is_available() else "cpu")
        print('计算设备:', self.device)
        # 模型
        self.mode_type = cfg.model_type
        # 任务类型
        self.task_type = cfg.task_type
        # 是否预训练
        self.pretrained = cfg.pretrained
        # 模型
        self.model = self.set_model(self.pretrained, self.mode_type)
        # 优化器 负责更新参数
        self.optimer = optim.Adam(self.model.parameters(), lr=self.lr)
        # 负责调节学习率
        # self.scheduler = self.set_lr_scheduler()
        self.scheduler = optim.lr_scheduler.StepLR(optimizer=self.optimer, step_size=3, gamma=0.5)
        # 回归任务的损失函数
        self.MSE = nn.MSELoss()
        # 开始时间
        self.startTime = 0
        # 创建文件夹
        self.create_model_dir()
        #
        self.set_weights()

    # 建立模型
    def set_model(self, pretrained, model_type):
        init = cfg.model_path == None
        model = None
        print("模型:%s" % (model_type))
        print("属性:%s" % (cfg.ATTR))
        if model_type == 'cnn_pool':
            model = cnn_pool(cfg.class_num)
        elif model_type == 'fnet_pool':
            model = fnet_pool()
        elif model_type == 'ham_pool':
            model = ham_pool(cfg.class_num)
        elif model_type == 'vit_pool':
            model = vit_pool()
        model.print_network()
        if cfg.continue_train == 1:
            model_path = os.path.join('checkpoints', cfg.dataset, "%s.pth" % (cfg.path_name))
            model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
            print("加载参数文件: {}".format(model_path))
        model = model.to(self.device)
        # print(model)
        return model

    def set_lr_scheduler(self):
        if cfg.lr_delay == 'cosine':
            optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimer, T_mult=2, T_0=5)

    def create_model_dir(self):
        name1 = os.path.join('result/', cfg.dataset, self.mode_type)
        self.result_dir = name1
        create_dir(name1)
        name2 = os.path.join('checkpoints', cfg.dataset)
        create_dir(name2)
        self.checkpoints_dir = name2

    def set_input(self, datas):
        label = datas['labels']
        images = datas['images']
        adj = datas['adj']
        label = label.to(self.device)
        adj = adj.to(self.device)
        for i in range(len(images)):
            images[i] = images[i].to(self.device)
        return label, images, adj

    def set_weights(self):
        self.attr_weights = data_balance()
        self.attr_weights = torch.FloatTensor(self.attr_weights).to(self.device)

    # 一个epoch的训练
    def train(self, epoch_id):
        self.model.train()
        for batch_idx, datas in enumerate(self.train_dataloader):
            self.optimer.zero_grad()
            label, images, adj = self.set_input(datas)
            pre_attr = self.model(images, adj)
            loss = F.cross_entropy(input=pre_attr, target=label,
                                   weight=self.attr_weights)

            # 误差回传
            loss.backward()
            # 更新参数
            self.optimer.step()
            # 记录loss
            if (batch_idx + 1) % (len(self.train_dataloader) // 32) == 0:
                # 存储loss
                self.loss_dict[cfg.ATTR].append(loss.cpu().item())
            if (batch_idx + 1) % (len(self.train_dataloader) // 3) == 0:
                # 终端输出
                print(
                    "Epoch: %d/%d , batch_idx:%d , time:%s , Loss:%.4f " % (
                        epoch_id + 1, cfg.epoch, batch_idx + 1, timeSince(self.startTime),
                        loss.data))

                # 在验证集 上验证

    def val(self):
        self.model.eval()
        ATTR = cfg.ATTR
        accuracy_dict = {ATTR: []}
        with torch.no_grad():
            for batch_idx, datas in enumerate(self.val_dataloader):
                label, images, adj = self.set_input(datas)
                pre_attr = self.model(images, adj)
                pred_list = pre_attr
                true_list = label
                for i in range(self.val_batchsize):
                    pre = np.argmax(pred_list[i].data.cpu().numpy())
                    true = true_list[i].item()
                    accuracy_dict[ATTR].append(pre == true)

        print("在验证集上的正确率")
        for attr in [ATTR]:
            attr_acc = sum(accuracy_dict[attr]) / len(accuracy_dict[attr])
            accuracy_dict[attr] = attr_acc
            print("属性「%s」: %.2f" % (attr, attr_acc * 100))
        return accuracy_dict

    # 在测试集上进行测试
    def test(self, model_path=None):
        # 如果有参数文件
        if model_path is not None:
            # 加载模型文件
            device = torch.device("cpu")
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.to(self.device)
            print("加载参数文件: {}".format(model_path))
        self.model.eval()
        ATTR = cfg.ATTR
        accuracy_dict = {ATTR: []}
        pre_dict = {ATTR: []}
        true_dict = {ATTR: []}
        with torch.no_grad():
            for batch_idx, datas in enumerate(self.test_dataloader):
                label, images, adj = self.set_input(datas)
                pre_attr = self.model(images, adj)
                pred_list = pre_attr
                true_list = label
                for i in range(self.test_batchsize):
                    # 拿到单个值
                    pre = np.argmax(pred_list[i].data.cpu().numpy())
                    true = true_list[i].item()
                    # 计算是否正确
                    accuracy_dict[ATTR].append(pre == true)
                    # 将预测结果存储
                    pre_dict[ATTR].append(pre)
                    true_dict[ATTR].append(true)

            print("在测试集上的正确率")
            for attr in [ATTR]:
                attr_acc = sum(accuracy_dict[attr]) / len(accuracy_dict[attr])
                accuracy_dict[attr] = attr_acc
                print("属性「%s」: %.2f" % (attr, attr_acc * 100), end=" ")

            # 存储正确率结果
            with open(os.path.join(self.result_dir, '%s_test_acc.json' % (cfg.ATTR)), 'w') as fp:
                json.dump(accuracy_dict, fp)
            # 存储预测值与正确值方便展示混淆矩阵
            pre_csv = pd.DataFrame(pre_dict)
            pre_csv.to_csv(os.path.join(self.result_dir, '%s_pre.csv' % (cfg.ATTR)), index=False)
            true_csv = pd.DataFrame(true_dict)
            true_csv.to_csv(os.path.join(self.result_dir, '%s_true.csv' % (cfg.ATTR)), index=False)

    def start(self, model_path=None):
        # 如果有参数文件
        if model_path is not None:
            # 加载模型文件
            self.model.load_state_dict(torch.load(model_path))
            print("加载参数文件: {}".format(model_path))
        # 记录最好的模型以及正确率
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        # loss字典
        ATTR = cfg.ATTR
        self.loss_dict = {ATTR: []}
        # 统计每个epoch的正确率weight
        epoch_acc_dict = {ATTR: []}
        self.startTime = time.time()
        print("lr:", self.optimer.param_groups[0]['lr'])
        for epoch in range(self.epoch):
            print("train............")
            self.train(epoch)
            print('val..............')
            acc_dict = self.val()
            old_lr = self.optimer.param_groups[0]['lr']
            self.scheduler.step()
            new_lr = self.optimer.param_groups[0]['lr']
            print("学习率: %s >>>> %s" % (old_lr, new_lr))
            for attr in epoch_acc_dict.keys():
                epoch_acc_dict[attr].append(acc_dict[attr])

            mAP = acc_dict[cfg.ATTR]
            # 比较正确率并保存最佳模型
            if mAP > best_acc:
                best_acc = mAP
                best_model_wts = copy.deepcopy(self.model.state_dict())
            newest_model_wts = copy.deepcopy(self.model.state_dict())

            # 每个epoch的正确率保存
            epoch_acc_csv = pd.DataFrame(epoch_acc_dict)
            epoch_acc_csv.to_csv(os.path.join(self.result_dir, '%s_val_acc.csv' % (cfg.ATTR)), index=False)
            # loss保存
            loss_csv = pd.DataFrame(self.loss_dict)
            loss_csv.to_csv(os.path.join(self.result_dir, '%s_loss.csv' % (cfg.ATTR)), index=False)
            # 模型参数保存
            model_save_path = os.path.join(self.checkpoints_dir, '%s_%s_best.pth' % (self.mode_type, cfg.ATTR))
            torch.save(best_model_wts, model_save_path)
            # 最新模型
            torch.save(newest_model_wts,
                       os.path.join(self.checkpoints_dir, '%s_%s_newest.pth' % (self.mode_type, cfg.ATTR)))

        self.test()
