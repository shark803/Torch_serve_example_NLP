#!/usr/bin/env python
# encoding: utf-8
# @author: 有莘不殁 （Lei Wang）
# @license: MIT
# @contact: lwangshark2013@gmail.com
# @software: pycharm
# @file: inference.py
# @time: 2021/3/16 下午5:40
# @desc: 直接加载模型进行预测，支持离线文件和单句调试， api式调用

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from importlib import import_module
from sklearn import metrics
from datetime import timedelta
import argparse
from config import Config

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN')
parser.add_argument('--type', type=str, required=True, help='choose the type of inference : Sent, File')

args = parser.parse_args()

UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
vocab_dir = "./THUCNews/data/vocab.pkl"
model_path = "./THUCNews/saved_dict/TextCNN.pt"
vocab = pkl.load(open(vocab_dir, 'rb'))
print(len(vocab))

batch_size = 128  # mini-batch大小
pad_size = 32  # 最好和训练集一致，根据文本特点选择该超参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_inputs_from_file(filepath, pad_size=32):
    tokenizer = lambda x: [y for y in x]
    contents = []
    with open(filepath, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            words_line = []
            token = tokenizer(lin)
            seq_len = len(token)
            if pad_size:
                if len(token) < pad_size:
                    token.extend([PAD] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            # word to id
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))  # dict.get(key, default) 第二个参数为key不存在的缺省值
            contents.append((words_line, int(0), seq_len))
    return contents  # [([...], 0), ([...], 1), ...]


def get_inputs(sents, pad_size=32):
    tokenizer = lambda x: [y for y in x]
    contents = []
    for sent in tqdm(sents):
        lin = sent.strip()
        if not lin:
            raise Exception('The input sentence is None !!!')
        words_line = []
        token = tokenizer(lin)
        seq_len = len(token)
        if pad_size:
            if len(token) < pad_size:
                token.extend([PAD] * (pad_size - len(token)))
            else:
                token = token[:pad_size]
                seq_len = pad_size
        # word to id
        for word in token:
            words_line.append(vocab.get(word, vocab.get(UNK)))  # dict.get(key, default) 第二个参数为key不存在的缺省值
        contents.append((words_line, int(0), seq_len))
    return contents  # [([...], 0), ([...], 1), ...]


def get_input(sent, pad_size=32):
    tokenizer = lambda x: [y for y in x]
    contents = []
    lin = sent.strip()
    if not lin:
        raise Exception('The input sentence is None !!!')
    words_line = []
    token = tokenizer(lin)
    seq_len = len(token)
    if pad_size:
        if len(token) < pad_size:
            token.extend([PAD] * (pad_size - len(token)))
        else:
            token = token[:pad_size]
            seq_len = pad_size
    # word to id
    for word in token:
        words_line.append(vocab.get(word, vocab.get(UNK)))  # dict.get(key, default) 第二个参数为key不存在的缺省值
    contents.append((words_line, int(0), seq_len))
    return contents  # [([...], 0), ([...], 1), ...]


class InferDataIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if self.n_batches == 0:
            self.residue = True
        elif len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        # pad前的长度(超过pad_size的设为pad_size)
        # seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return x, y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_infer_iterator(dataset, batch_size, device):
    iter = InferDataIterater(dataset, batch_size, device)
    return iter


def infer4file(model, infer_iter, dataset):
    # test
    model.eval()

    with open(dataset + "/data/label.txt", "w", encoding="utf-8") as fout:
        for texts, _ in infer_iter:
            outputs = model(texts)
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            for label in predic:
                fout.write(str(label) + "\n")


def infer4sent(model, infer_iter, dataset):
    # test
    model.eval()

    for texts, _ in infer_iter:
        outputs = model(texts)
        predic = torch.max(outputs.data, 1)[1].cpu().numpy()
        return predic[0]


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random

    # embedding = 'random'
    model_name = args.model  # TextCNN
    test_type = args.type  # Sent, Batch, File
    config = Config(dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    #
    model = torch.load(config.save_fm_path)
    model.eval()  # 固化如dropout 这样的层
    model = model.to(device)
    from utils import get_time_dif

    id2class = {}
    with open(dataset + "/data/class.txt", "r", encoding="utf-8") as fdic:
        for index, class_name in enumerate(fdic):
            id2class[index] = class_name.strip()

    start_time = time.time()
    if test_type == "File":
        infer_data = get_inputs_from_file(dataset + "/data/infer.txt")
        infer_iter = build_infer_iterator(infer_data, batch_size, device)
        infer4file(model, infer_iter, dataset)
    elif test_type == "Sent":
        sent = input("Input your sentence, exit by input the string '&&' :")
        while sent != "&&":
            infer_data = get_input(sent)
            infer_iter = build_infer_iterator(infer_data, batch_size, device)
            label = infer4sent(model, infer_iter, dataset)
            print(str(label) + ": " + id2class[label])
            sent = input("Input your sentence, exit by input the string '&&'：")
    else:
        raise Exception('You must specify the type of inference !!!')
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
