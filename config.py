#!/usr/bin/env python
# encoding: utf-8
# @author: 有莘不殁 （Lei Wang）
# @license: MIT
# @contact: lwangshark2013@gmail.com
# @software: pycharm
# @file: config.py
# @time: 2021/3/27 下午1:05
# @desc:
import torch

class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_name = 'TextCNN'
        self.train_path = dataset + '/data/train.txt'  # 训练集
        self.dev_path = dataset + '/data/dev.txt'  # 验证集
        self.test_path = dataset + '/data/test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]  # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'  # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.save_pt_path = dataset + '/saved_dict/' + self.model_name + '.pt'  # 保存为pt
        self.save_fm_path =  dataset + '/saved_dict/' + self.model_name + '.pth'  # 保存为pt
        self.log_path = dataset + '/log/' + self.model_name
        # self.embedding_pretrained = torch.tensor(
        #     np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
        #     if embedding != 'random' else None                                       # 预训练词向量
        # self.embedding_pretrained = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.dropout = 0.5  # 随机失活
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 20  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率
        # self.embed = self.embedding_pretrained.size(1)\
        #     if self.embedding_pretrained is not None else 300           # 字向量维度
        self.embed = 300
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256  # 卷积核数量(channels数)