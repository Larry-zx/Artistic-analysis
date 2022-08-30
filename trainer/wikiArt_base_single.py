import copy
import json
import os
import pandas as pd
from data.get_dataLoader import get_loader
from utils.utils import set_transform, timeSince, create_dir
import config as cfg
# 模型
from model.ViT import vit_base_patch16_224
from model.Swin_ViT import swin_base_patch4_window7_224_in22k
from model.CNN import CNN
from model.vit_zx import vit_zx
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
import time
import numpy as np
from data.wikiart_base_single import data_balance
import tqdm

from efficientnet_pytorch import EfficientNet


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
        self.task_type = 'single'
        # 是否预训练
        self.pretrained = cfg.pretrained
        # 创建相关文件夹
        self.create_model_dir()
        # 模型
        self.set_model(self.mode_type)
        # 优化器 负责更新参数
        self.optimer = optim.Adam(self.model.parameters(), lr=self.lr)
        # 负责调节学习率
        self.set_scheduler()
        # 回归任务的损失函数
        self.MSE = nn.MSELoss()
        # 开始时间
        self.startTime = 0
        # 创建文件夹
        self.set_weights()
        #
        if cfg.load_optimer == 1:
            self.load_weights(mode='optimer')
        if cfg.load_scheduler == 1:
            self.load_weights(mode='scheduler')

    def set_scheduler(self):
        if cfg.scheduler == 'None':
            self.scheduler = None
        elif cfg.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimer, T_mult=2, T_0=5)
        elif cfg.scheduler == 'linear':
            self.scheduler = optim.lr_scheduler.StepLR(optimizer=self.optimer, step_size=1, gamma=0.9)


    # 建立模型
    def set_model(self, model_type):
        self.model = None
        print("模型:%s" % (model_type))
        print("属性:%s" % (cfg.ATTR))
        if model_type == 'vit':
            self.model = vit_base_patch16_224(num_classes=cfg.class_num)
            self.model.load_state_dict(torch.load('checkpoints/finetun/vit.pth', map_location=torch.device("cpu")),
                                       strict=False)
            print("加载参数文件: {}".format('vit.pth'))
        elif model_type == 'cnn':
            self.model = CNN()
        elif model_type == 'vit_pool':
            self.model = vit_zx(cfg.class_num, k_neighbour=27, gcn_mode='diffpool')
        elif model_type == 'vit_set2set':
            self.model = vit_zx(cfg.class_num, k_neighbour=27, gcn_mode='set2set')
        elif model_type == 'swin':
            self.model = swin_base_patch4_window7_224_in22k(num_classes=cfg.class_num)
            self.model.load_state_dict(torch.load('checkpoints/finetun/swin.pth', map_location=torch.device("cpu")),
                                       strict=False)
        elif model_type == 'EfficientNet':
            self.model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=cfg.class_num)
        elif model_type  == 'vit_pool_nopre':
            self.model = vit_zx(cfg.class_num, k_neighbour=29, gcn_mode='diffpool',pretrain=False)
            self.model.init_weights()
        elif model_type == 'vit_pool_datapro':
            self.model = vit_zx(cfg.class_num, k_neighbour=27, gcn_mode='diffpool')
        if cfg.continue_train == 1: self.load_weights(mode='model')
        self.model = self.model.to(self.device)

    # 加载 模型/优化器/调度器 的参数
    def load_weights(self, mode='model'):
        if mode == 'optimer':
            path = os.path.join(self.checkpoints_dir, "optimer_%s.pth" % (cfg.wts_name))
            self.optimer.load_state_dict(
                torch.load(path, map_location=torch.device("cpu")))
            print('optimer加载完成{%s}' % (path))
        elif mode == 'scheduler':
            path = os.path.join(self.checkpoints_dir, "scheduler_%s.pth" % (cfg.wts_name))
            self.scheduler.load_state_dict(
                torch.load(path, map_location=torch.device("cpu")))
            print('scheduler加载完成{%s}' % (path))
        elif mode == 'model':
            model_path = os.path.join(self.checkpoints_dir, "model_%s.pth" % (cfg.wts_name))
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
            print("加载参数文件: {}".format(model_path))

    def create_model_dir(self):
        name1 = os.path.join('result/', cfg.dataset, self.task_type, self.mode_type)
        self.result_dir = name1
        create_dir(name1)
        name2 = os.path.join('checkpoints', cfg.dataset, self.mode_type, cfg.ATTR)
        create_dir(name2)
        self.checkpoints_dir = name2

    # 将数据放到cuda上
    def set_input(self, datas):
        label = datas['labels']
        images = datas['images']
        label = label.to(self.device)
        images = images.to(self.device)
        return label, images

    # 设置交叉熵的weights以缓解长尾问题
    def set_weights(self):
        self.attr_weights = data_balance()
        self.attr_weights = torch.FloatTensor(self.attr_weights).to(self.device)

    # 一个epoch的训练
    def train(self, epoch_id):
        self.model.train()
        for batch_idx, datas in tqdm.tqdm(enumerate(self.train_dataloader)):
            label, images = self.set_input(datas)
            pre_attr = self.model(images)
            loss = F.cross_entropy(input=pre_attr, target=label, weight=self.attr_weights)
            # print(loss)
            # 梯度清空
            self.optimer.zero_grad()
            # 误差回传
            loss.backward()

            # 更新参数
            self.optimer.step()
            # 记录loss
            if (batch_idx + 1) % (len(self.train_dataloader) // 32) == 0:
                self.loss_dict[cfg.ATTR].append(loss.cpu().item())  # 存储loss
            if (batch_idx + 1) % (len(self.train_dataloader) // 3) == 0:
                # 终端输出
                print("Epoch: %d/%d , batch_idx:%d , time:%s , Loss:%.8f " % (
                    epoch_id + 1, cfg.epoch, batch_idx + 1, timeSince(self.startTime), loss.data))

    def val(self):
        self.model.eval()
        ATTR = cfg.ATTR
        accuracy_dict = {ATTR: []}
        with torch.no_grad():
            for batch_idx, datas in tqdm.tqdm(enumerate(self.test_dataloader)):
                label, images = self.set_input(datas)
                pre_attr = self.model(images)
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
    def test(self, ):
        model_path = os.path.join(self.checkpoints_dir, "model_%s.pth" % (cfg.wts_name))
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        print("加载参数文件: {}".format(model_path))
        self.model.eval()
        ATTR = cfg.ATTR
        accuracy_dict = {ATTR: []}
        pre_dict = {ATTR: []}
        true_dict = {ATTR: []}
        with torch.no_grad():
            for batch_idx, datas in tqdm.tqdm(enumerate(self.test_dataloader)):
                label, images = self.set_input(datas)
                pre_attr = self.model(images)
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
        return accuracy_dict

    def start(self, model_path=None):
        # 如果有参数文件
        if model_path is not None:
            # 加载模型文件
            self.model.load_state_dict(torch.load(model_path))
            print("加载参数文件: {}".format(model_path))
        best_acc = 0.0
        # loss字典
        ATTR = cfg.ATTR
        self.loss_dict = {ATTR: []}
        # 统计每个epoch的正确率weight
        epoch_acc_dict = {ATTR: []}
        self.startTime = time.time()
        if cfg.last_lr != 0:
            self.optimer.param_groups[0]['lr'] = cfg.last_lr
        print("lr:", self.optimer.param_groups[0]['lr'])
        for epoch in range(cfg.start_epoch, self.epoch):
            # 释放cuda
            torch.cuda.synchronize(self.device)
            print("train............")
            self.train(epoch)
            print('val..............')
            acc_dict = self.val()
            old_lr = self.optimer.param_groups[0]['lr']
            if self.scheduler != None and epoch < 15:
                self.scheduler.step()
            new_lr = self.optimer.param_groups[0]['lr']
            print("学习率: %s >>>> %s" % (old_lr, new_lr))
            for attr in epoch_acc_dict.keys():
                epoch_acc_dict[attr].append(acc_dict[attr])

            mAP = acc_dict[cfg.ATTR]
            # 比较正确率并保存最佳模型
            if mAP > best_acc:
                best_acc = mAP
                self.save_checkpoints(map=mAP)

            # 每个epoch的正确率保存
            epoch_acc_csv = pd.DataFrame(epoch_acc_dict)
            epoch_acc_csv.to_csv(os.path.join(self.result_dir, '%s_val_acc.csv' % (cfg.ATTR)), index=False)
            # loss保存
            loss_csv = pd.DataFrame(self.loss_dict)
            loss_csv.to_csv(os.path.join(self.result_dir, '%s_loss.csv' % (cfg.ATTR)), index=False)

            self.save_checkpoints()

        self.test()

    def save_checkpoints(self, map=None):
        if map != None:
            # 模型参数
            torch.save(copy.deepcopy(self.model.state_dict()),
                       os.path.join(self.checkpoints_dir, 'model_%.4f.pth' % (map)))
            # 优化器
            torch.save(copy.deepcopy(self.optimer.state_dict()),
                       os.path.join(self.checkpoints_dir, 'optimer_%.4f.pth' % (map)))
            # 学习率调度器
            if self.scheduler != None:
                torch.save(copy.deepcopy(self.scheduler.state_dict()),
                           os.path.join(self.checkpoints_dir, 'scheduler_%.4f.pth' % (map)))
        else:
            torch.save(copy.deepcopy(self.model.state_dict()), os.path.join(self.checkpoints_dir, 'model_newest.pth'))
            # 优化器
            torch.save(copy.deepcopy(self.optimer.state_dict()),
                       os.path.join(self.checkpoints_dir, 'optimer_newest.pth'))
            # 学习率调度器
            if self.scheduler != None:
                torch.save(copy.deepcopy(self.scheduler.state_dict()),
                           os.path.join(self.checkpoints_dir, 'scheduler_newest.pth'))
