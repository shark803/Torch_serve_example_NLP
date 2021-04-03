#!/usr/bin/env python
# encoding: utf-8
# @author: 有莘不殁 （Lei Wang）
# @license: MIT
# @contact: lwangshark2013@gmail.com
# @software: pycharm
# @file: app.py.py
# @time: 2021/3/19 下午4:02
# @desc: RESTFUL API based on Flask

import io
import json


from flask import Flask, jsonify, request
import torch
import pickle as pkl
import numpy as np

# from inference import get_inputs, build_infer_iterator
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
vocab_dir = "./THUCNews/data/vocab.pkl"
model_path = "./THUCNews/saved_dict/TextCNN.pth"
vocab = pkl.load(open(vocab_dir, 'rb'))

batch_size = 128  # mini-batch大小
pad_size = 32  # 最好和训练集一致，根据文本特点选择该超参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)
# imagenet_class_index = json.load(open('<PATH/TO/.json/FILE>/imagenet_class_index.json'))
# model = models.densenet121(pretrained=True)
# model.eval()
dataset = 'THUCNews'  # 数据集

# 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random

# embedding = 'random'
model_name = "TextCNN" # TextCNN
test_type = "Sent"  # Sent, Batch, File

batch_size = 128

id2class = {}
with open(dataset + "/data/class.txt", "r", encoding="utf-8") as fdic:
    for index, class_name in enumerate(fdic):
        id2class[index] = class_name.strip()

model = torch.load(model_path)  # 加载 pt格式的模型
model.eval()  # 固化如dropout 这样的层
model = model.to(device)

def get_inputs(sents, pad_size=32):
    tokenizer = lambda x: [y for y in x]
    contents = []
    for lin in sents:
        lin = lin.strip()
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

def get_prediction(infer_iter):
    predicts = []

    labels_all = np.array([], dtype=int)

    for texts, _ in infer_iter:
        outputs = model(texts)
        labels = torch.max(outputs.data, 1)[1].cpu().numpy()
        labels_all = np.append(labels_all, labels)

    labellist = labels_all.tolist()
    class_names = [id2class[label] for label in labellist]

    return labellist, class_names


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        texts = request.get_json()['inputs']
        print(texts)
        infer_data = get_inputs(texts)
        infer_iter = build_infer_iterator(infer_data, batch_size, device)
        label_list, class_name = get_prediction(infer_iter)
        return jsonify({'class_id': label_list, 'class_name': class_name})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)

