import copy
import json
import os
import pandas as pd
from data.get_dataLoader import get_loader
from utils.utils import set_transform, timeSince, create_dir
from model.cnn_pool_jas import cnn_pool_jas
import config as cfg
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
import time
import numpy as np
import tqdm


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
        self.task_type = 'multi'
        # 损失函数
        # 是否预训练
        self.pretrained = cfg.pretrained
        # 模型
        self.model = self.set_model(self.pretrained, self.mode_type)
        # 优化器 负责更新参数
        self.optimer = optim.Adam(self.model.parameters(), lr=self.lr)
        # 负责调节学习率
        # self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimer, T_mult=2, T_0=5)
        self.scheduler = optim.lr_scheduler.StepLR(optimizer=self.optimer, step_size=5, gamma=0.9)
        # 回归任务的损失函数
        self.MSE = nn.MSELoss()
        # 开始时间
        self.startTime = 0
        # 创建文件夹
        self.create_model_dir()
        #
        self.attr_list = ['aesthetic_quality', 'beauty', 'color', 'composition', 'content']
        #
        self.loss_dict = {attr: [] for attr in self.attr_list}

    # 建立模型
    def set_model(self, pretrained, model_type):
        init = cfg.model_path == None
        model = None
        print("模型:%s" % (model_type))
        if model_type == 'cnn_pool_jas':
            model = cnn_pool_jas()
        model.print_network()
        if cfg.continue_train == 1:
            model_path = os.path.join('checkpoints', cfg.dataset, "%s.pth" % (cfg.path_name))
            model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
            print("加载参数文件: {}".format(model_path))
        model = model.to(self.device)
        # print(model)
        return model

    def create_model_dir(self):
        name1 = os.path.join('result/', cfg.dataset, self.task_type, self.mode_type)
        self.result_dir = name1
        create_dir(name1)
        name2 = os.path.join('checkpoints', cfg.dataset)
        create_dir(name2)
        self.checkpoints_dir = name2

    def set_input(self, datas):
        label_dict = datas['labels']
        images = datas['images']
        adj = datas['adj']
        boxes = datas['boxes']
        size = datas['size']
        for attr in self.attr_list:
            label_dict[attr] = label_dict[attr].to(self.device)
        adj = adj.to(self.device)
        images = images.to(self.device)
        boxes = boxes.to(self.device)
        size = size.to(self.device)
        return label_dict, images, adj, boxes, size

    # 一个epoch的训练
    def train(self, epoch_id):
        self.model.train()
        for batch_idx, datas in tqdm.tqdm(enumerate(self.train_dataloader)):
            self.optimer.zero_grad()
            label_dict, images, adj, boxes, size = self.set_input(datas)
            pre_attr = self.model(images, size, boxes, adj)
            now_loss_dict = {}
            total_loss = 0
            for attr in self.attr_list:
                pre = pre_attr[attr].squeeze(-1)
                label = label_dict[attr]
                now_loss_dict[attr] = F.mse_loss(pre.to(torch.float32), label.to(torch.float32))
                total_loss += now_loss_dict[attr]
            # 误差回传
            total_loss.backward()
            # 更新参数
            self.optimer.step()
            # 记录loss
            if 1 == 1:
                # 存储loss
                for attr in self.attr_list:
                    self.loss_dict[attr].append(now_loss_dict[attr].cpu().item())
            if (batch_idx + 1) % (len(self.train_dataloader) // 3) == 0:
                # 终端输出
                print(
                    "Epoch: %d/%d , batch_idx:%d , time:%s " % (
                        epoch_id + 1, cfg.epoch, batch_idx + 1, timeSince(self.startTime)), end="")
                for attr in self.attr_list:
                    print(", %s:%.3f " % (attr, now_loss_dict[attr]), end="")
                print("")

    # 在验证集 上验证 评价指标 PLCC SRCC

    def val(self):
        self.model.eval()
        label_list_dict = {}
        pre_list_dict = {}
        PLCC_dict = {}
        RMSE_dict = {}
        for attr in self.attr_list:
            label_list_dict[attr] = list()
            pre_list_dict[attr] = list()
        with torch.no_grad():
            for batch_idx, datas in tqdm.tqdm(enumerate(self.val_dataloader)):
                label_dict, images, adj, boxes, size = self.set_input(datas)
                pre_dict = self.model(images, size, boxes, adj)
                for attr in self.attr_list:
                    pre = pre_dict[attr].squeeze(-1).cpu().numpy()
                    label = label_dict[attr].cpu().numpy()
                    pre_list_dict[attr] = np.concatenate((pre_list_dict[attr], pre))
                    label_list_dict[attr] = np.concatenate((label_list_dict[attr], label))

        total_RMSE = 0
        for attr in self.attr_list:
            # PLCC
            p = pre_list_dict[attr]
            s = label_list_dict[attr]
            p_mean = np.mean(p)
            s_mean = np.mean(s)
            PLCC = np.sum((s - s_mean) * (p - p_mean)) / (
                    np.sqrt(np.sum((s - s_mean) ** 2)) * np.sqrt(np.sum((p - p_mean) ** 2)))
            PLCC_dict[attr] = PLCC
            # RMSE
            RMSE = np.sqrt((1 / len(s)) * np.sum((s - p) ** 2))
            RMSE_dict[attr] = RMSE
            total_RMSE += RMSE
        RMSE_dict['ave'] = total_RMSE / 5
        return PLCC_dict, RMSE_dict

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
        label_list_dict = {}
        pre_list_dict = {}
        PLCC_dict = {}
        RMSE_dict = {}
        with torch.no_grad():
            for batch_idx, datas in enumerate(self.test_dataloader):
                label_dict, images, adj, boxes, size = self.set_input(datas)
                pre_dict = self.model(images, size, boxes, adj)
                for attr in self.attr_list:
                    pre = pre_dict[attr].squeeze(-1).cpu().numpy()
                    label = label_dict[attr].cpu().numpy()
                    pre_list_dict[attr] = np.concatenate((pre_list_dict[attr], pre))
                    label_list_dict[attr] = np.concatenate((label_list_dict[attr], label))
        total_RMSE = 0
        for attr in self.attr_list:
            # PLCC
            p = pre_list_dict[attr]
            s = label_list_dict[attr]
            p_mean = np.mean(p)
            s_mean = np.mean(s)
            PLCC = np.sum((s - s_mean) * (p - p_mean)) / (
                    np.sqrt(np.sum((s - s_mean) ** 2)) * np.sqrt(np.sum((p - p_mean) ** 2)))
            PLCC_dict[attr] = PLCC
            # RMSE
            RMSE = np.sqrt((1 / len(s)) * np.sum((s - p) ** 2))
            RMSE_dict[attr] = RMSE
            total_RMSE += RMSE
        RMSE_dict['ave'] = total_RMSE / 5
        # 预测值
        pre_csv = pd.DataFrame(pre_list_dict)
        pre_csv.to_csv(os.path.join(self.result_dir, 'pre.csv'), index=False)
        # 真实值
        true_csv = pd.DataFrame(label_list_dict)
        true_csv.to_csv(os.path.join(self.result_dir, 'true.csv'), index=False)
        # PLCC RMSE
        with open(os.path.join(self.result_dir, 'test_PLCC.json'), 'w') as fp:
            json.dump(PLCC_dict, fp)
        with open(os.path.join(self.result_dir, 'test_RMSE.json'), 'w') as fp:
            json.dump(RMSE_dict, fp)

    def start(self, model_path=None):
        # 如果有参数文件
        if model_path is not None:
            # 加载模型文件
            self.model.load_state_dict(torch.load(model_path))
            print("加载参数文件: {}".format(model_path))
        # 记录最好的模型以及正确率
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_RMSE = 100000
        # 存储每个epoch的val情况
        epoch_val_PLCC_dict = {attr: [] for attr in self.attr_list}
        epoch_val_RMSE_dict = {attr: [] for attr in self.attr_list}
        self.startTime = time.time()
        print("lr:", self.optimer.param_groups[0]['lr'])
        for epoch in range(self.epoch):
            print("###############-----train-----###############")
            self.train(epoch)
            print("###############-----valid-----###############")
            PLCC_dict, RMSE_dict = self.val()
            print("评价指标:PLCC")
            for attr in self.attr_list:
                print("%s: %.3f" % (attr, PLCC_dict[attr]), end=" ")
            print("")
            print("评价指标:RMSE")
            for attr in self.attr_list:
                print("%s: %.2f" % (attr, RMSE_dict[attr]), end=" ")
            print("%s: %.3f" % ('ave', RMSE_dict['ave']))

            old_lr = self.optimer.param_groups[0]['lr']
            self.scheduler.step()
            new_lr = self.optimer.param_groups[0]['lr']
            print("学习率: %s >>>> %s" % (old_lr, new_lr))
            for attr in self.attr_list:
                epoch_val_PLCC_dict[attr].append(PLCC_dict[attr])
                epoch_val_RMSE_dict[attr].append(RMSE_dict[attr])

            ave_RMSE = RMSE_dict['ave']
            # 比较正确率并保存最佳模型
            if ave_RMSE < best_RMSE:
                best_acc = ave_RMSE
                best_model_wts = copy.deepcopy(self.model.state_dict())
                # 模型参数保存
                model_save_path = os.path.join(self.checkpoints_dir, '%s_%.4f.pth' % (self.mode_type, ave_RMSE))
                torch.save(best_model_wts, model_save_path)
            newest_model_wts = copy.deepcopy(self.model.state_dict())
            # 每个epoch的PLCC
            epoch_val_PLCC_csv = pd.DataFrame(epoch_val_PLCC_dict)
            epoch_val_PLCC_csv.to_csv(os.path.join(self.result_dir, 'val_PLCC.csv'), index=False)
            # 每个epoch的RMSE
            epoch_val_RMSE_csv = pd.DataFrame(epoch_val_RMSE_dict)
            epoch_val_RMSE_csv.to_csv(os.path.join(self.result_dir, 'val_RMSE.csv'), index=False)
            # loss保存
            loss_csv = pd.DataFrame(self.loss_dict)
            loss_csv.to_csv(os.path.join(self.result_dir, 'loss.csv'), index=False)
            # 模型参数保存
            model_save_path = os.path.join(self.checkpoints_dir, '%s_best.pth' % (self.mode_type))
            torch.save(best_model_wts, model_save_path)
            # 最新模型
            torch.save(newest_model_wts,
                       os.path.join(self.checkpoints_dir, '%s_newest.pth' % (self.mode_type)))

        self.test()
