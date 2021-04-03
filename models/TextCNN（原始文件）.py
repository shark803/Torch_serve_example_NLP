# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import Config
'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # if config.embedding_pretrained is not None:
        #     self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        # else:
        #     self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        # self.convs = nn.ModuleList(
        #     [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        # torchscript 不支持 列表表达式这样的迭代器
        self.conv_f2 = nn.Conv2d(1, config.num_filters, (2, config.embed))
        self.conv_f3 = nn.Conv2d(1, config.num_filters, (3, config.embed))
        self.conv_f4 = nn.Conv2d(1, config.num_filters, (4, config.embed))
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        pool_size = int(x.size(2))
        x = F.max_pool1d(x, pool_size).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)
        # print(x[0].shape)
        out = out.unsqueeze(1)
        # out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out_f2 = self.conv_and_pool(out, self.conv_f2)
        out_f3 = self.conv_and_pool(out, self.conv_f3)
        out_f4 = self.conv_and_pool(out, self.conv_f4)
        out = torch.cat([out_f2, out_f3, out_f4], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
