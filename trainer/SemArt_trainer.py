import copy
import json
import os
import pandas as pd
from data.get_dataLoader import get_loader
from utils.utils import set_transform, timeSince, create_dir
from model.classifer import multitask_classifer, singletask_classifer, vit_multi, vit_single, swin_multi, swin_single
import config as cfg
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
import time
import numpy as np
from data.SemArt import data_balance


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
        self.train_dataloader = get_loader(batch_size=self.batchsize, mode='train', transform=set_transform())
        self.val_dataloader = get_loader(batch_size=self.val_batchsize, mode='val', transform=set_transform())
        self.test_dataloader = get_loader(batch_size=self.test_batchsize, mode='test', transform=set_transform())
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
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimer, [30, 80], gamma=0.1)
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
        print("模型:%s-%s" % (model_type, self.task_type))
        if self.task_type == 'multi':
            if self.mode_type == 'vit':
                model = vit_multi()
                state_dict = torch.load(os.path.join(cfg.vit_path, "%s.pth" % (model_type)))
                model.load_state_dict(state_dict, strict=False)
                print("「%s」参数文件已经加载" % (os.path.join(cfg.vit_path, "%s.pth" % (model_type))))
            elif self.mode_type == 'swin':
                model = swin_multi()
                state_dict = torch.load(os.path.join(cfg.vit_path, "%s.pth" % (model_type)))
                model.load_state_dict(state_dict, strict=False)
                print("「%s」参数文件已经加载" % (os.path.join(cfg.vit_path, "%s.pth" % (model_type))))
            else:
                model = multitask_classifer(init, cnn_pretrained=pretrained, model_type=model_type)
        elif self.task_type == 'single':
            pass
            if self.mode_type == 'vit':
                model = vit_single(os.path.join(cfg.vit_path, "%s.pth" % (model_type)))
            if self.mode_type == 'swin':
                model = swin_single(os.path.join(cfg.vit_path, "%s.pth" % (model_type)))
            else:
                model = singletask_classifer(init, cnn_pretrained=pretrained, model_type=model_type)
        else:
            print("「task_type」输入有误 请在multi / single 中选择一个")
        model.print_network()
        model = model.to(self.device)
        return model

    def create_model_dir(self):
        name1 = os.path.join('result/', cfg.dataset, self.task_type, self.mode_type)
        self.result_dir = name1
        create_dir(name1)
        name2 = os.path.join('checkpoints', cfg.dataset, self.task_type)
        create_dir(name2)
        self.checkpoints_dir = name2

    def set_input(self, data):
        images, label_data = data
        images = images.to(self.device)
        label_data['author'] = label_data['author'].to(self.device)
        label_data['technique'] = (label_data['technique']).to(self.device)
        label_data['material'] = (label_data['material']).to(self.device)
        label_data['year'] = label_data['year'].reshape(-1, 1).to(self.device)
        label_data['type'] = (label_data['type']).to(self.device)
        label_data['school'] = (label_data['school']).to(self.device)
        label_data['timeFrame'] = (label_data['timeFrame']).to(self.device)
        return images, label_data

    def set_weights(self):
        weight_dict = data_balance()
        self.author_w = torch.FloatTensor(weight_dict['author']).to(self.device)
        self.technique_w = torch.FloatTensor(weight_dict['technique']).to(self.device)
        self.material_w = torch.FloatTensor(weight_dict['material']).to(self.device)
        self.type_w = torch.FloatTensor(weight_dict['type']).to(self.device)
        self.school_w = torch.FloatTensor(weight_dict['school']).to(self.device)
        self.timeFrame_w = torch.FloatTensor(weight_dict['timeFrame']).to(self.device)

    # 一个epoch的训练
    def train(self, epoch_id):
        self.model.train()
        for batch_idx, datas in enumerate(self.train_dataloader):
            self.optimer.zero_grad()
            images, data = self.set_input(datas)
            out_dict = self.model(images)
            author_loss = F.cross_entropy(input=out_dict['author'], target=data['author'],
                                          weight=self.author_w)
            technique_loss = F.cross_entropy(input=out_dict['technique'], target=data['technique'],
                                             weight=self.technique_w)
            material_loss = F.cross_entropy(input=out_dict['material'], target=data['material'],
                                            weight=self.material_w)
            type_loss = F.cross_entropy(input=out_dict['type'], target=data['type'],
                                        weight=self.type_w)
            school_loss = F.cross_entropy(input=out_dict['school'], target=data['school'],
                                          weight=self.school_w)
            timeFrame_loss = F.cross_entropy(input=out_dict['timeFrame'], target=data['timeFrame'],
                                             weight=self.timeFrame_w)
            # 合计loss
            total_loss = author_loss + technique_loss + material_loss + type_loss + school_loss + timeFrame_loss
            # 误差回传
            total_loss.backward()
            # 更新参数
            self.optimer.step()
            # 记录loss
            if (batch_idx + 1) % (len(self.train_dataloader) // 6) == 0:
                # 存储loss
                self.loss_dict['author'].append(author_loss.cpu().item())
                self.loss_dict['technique'].append(technique_loss.cpu().item())
                self.loss_dict['material'].append(material_loss.cpu().item())
                self.loss_dict['type'].append(type_loss.cpu().item())
                self.loss_dict['school'].append(school_loss.cpu().item())
                self.loss_dict['timeFrame'].append(timeFrame_loss.cpu().item())
                # 终端输出
                print(
                    "Epoch: %d/%d , batch_idx:%d , time:%s , author:%.2f , technique:%.2f , year:%.2f , type:%.2f , school:%.2f , timeFrame:%.2f " % (
                        epoch_id + 1, cfg.epoch, batch_idx + 1, timeSince(self.startTime),
                        author_loss.data, technique_loss.data, year_loss.data,
                        type_loss.data, school_loss.data, timeFrame_loss.data))

    # 在验证集 上验证

    def val(self):
        self.model.eval()
        accuracy_dict = {'author': [], 'technique': [], 'material': [], 'type': [], 'school': [],
                         'timeFrame': []}
        with torch.no_grad():
            for batch_idx, datas in enumerate(self.val_dataloader):
                # TODO修改
                images, data = self.set_input(datas)
                out_dict = self.model(images)
                for attr in cfg.Attr_list:
                    pred_list = out_dict[attr]
                    true_list = data[attr]
                    for i in range(self.val_batchsize):
                        pre = np.argmax(pred_list[i].data.cpu().numpy())
                        true = true_list[i].item()
                        accuracy_dict[attr].append(pre == true)

        print("在验证集上的正确率")
        average_acc = 0
        for attr in cfg.Attr_list:
            attr_acc = sum(accuracy_dict[attr]) / len(accuracy_dict[attr])
            accuracy_dict[attr] = attr_acc
            print("属性「%s」: %.2f" % (attr, attr_acc * 100), end=" ")
            average_acc += attr_acc
        print("平均正确率: %.2f" % (average_acc / len(cfg.Attr_list)))
        accuracy_dict['mAP'] = average_acc / len(cfg.Attr_list)
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
        accuracy_dict = {'author': [], 'technique': [], 'material': [], 'type': [], 'school': [],
                         'timeFrame': []}
        pre_dict = {'author': [], 'technique': [], 'material': [], 'type': [], 'school': [],
                    'timeFrame': []}
        true_dict = {'author': [], 'technique': [], 'material': [], 'type': [], 'school': [],
                     'timeFrame': []}
        with torch.no_grad():
            for batch_idx, datas in enumerate(self.test_dataloader):
                images, data = self.set_input(datas)
                out_dict = self.model(images)
                for attr in cfg.Attr_list:
                    pred_list = out_dict[attr]
                    true_list = data[attr]
                    for i in range(self.test_batchsize):
                        # 拿到单个值
                        pre = np.argmax(pred_list[i].data.cpu().numpy())
                        true = true_list[i].item()
                        # 计算是否正确
                        accuracy_dict[attr].append(pre == true)
                        # 将预测结果存储
                        pre_dict[attr].append(pre)
                        true_dict[attr].append(true)

            print("在测试集上的正确率")
            average_acc = 0
            for attr in cfg.Attr_list:
                attr_acc = sum(accuracy_dict[attr]) / len(accuracy_dict[attr])
                accuracy_dict[attr] = attr_acc
                print("属性「%s」: %.2f" % (attr, attr_acc * 100), end=" ")
                average_acc += attr_acc
            print("平均正确率: %.2f" % (average_acc / len(cfg.Attr_list)))
            accuracy_dict['mAP'] = average_acc / len(cfg.Attr_list)

            # 存储正确率结果
            with open(os.path.join(self.result_dir, 'test_acc.json'), 'w') as fp:
                json.dump(accuracy_dict, fp)
            # 存储预测值与正确值方便展示混淆矩阵
            pre_csv = pd.DataFrame(pre_dict)
            pre_csv.to_csv(os.path.join(self.result_dir, 'pre.csv'), index=False)
            true_csv = pd.DataFrame(true_dict)
            true_csv.to_csv(os.path.join(self.result_dir, 'true.csv'), index=False)

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
        self.loss_dict = {'author': [], 'technique': [], 'material': [], 'type': [], 'school': [],
                          'timeFrame': []}
        # 统计每个epoch的正确率weight
        epoch_acc_dict = {'author': [], 'technique': [], 'material': [], 'type': [], 'school': [],
                          'timeFrame': [], 'mAP': []}
        self.startTime = time.time()
        for epoch in range(self.epoch):
            print("train............")
            self.train(epoch)
            print('val..............')
            acc_dict = self.val()

            for attr in epoch_acc_dict.keys():
                epoch_acc_dict[attr].append(acc_dict[attr])

            mAP = acc_dict['mAP']
            # 比较正确率并保存最佳模型
            if mAP > best_acc:
                best_acc = mAP
                best_model_wts = copy.deepcopy(self.model.state_dict())

            # 每个epoch的正确率保存
            epoch_acc_csv = pd.DataFrame(epoch_acc_dict)
            epoch_acc_csv.to_csv(os.path.join(self.result_dir, 'val_acc.csv'), index=False)
            # loss保存
            loss_csv = pd.DataFrame(self.loss_dict)
            loss_csv.to_csv(os.path.join(self.result_dir, 'loss.csv'), index=False)
            # 模型参数保存
            model_save_path = os.path.join(self.checkpoints_dir, '%s.pth' % (self.mode_type))
            torch.save(best_model_wts, model_save_path)

        self.test()
