#!/usr/bin/env python
# encoding: utf-8
# @author: 有莘不殁 （Lei Wang）
# @license: MIT
# @contact: lwangshark2013@gmail.com
# @software: pycharm
# @file: TextCNN_handle.py
# @time: 2021/3/25 下午6:08
# @desc: ModelHandler defines a custom model handler for TextCNN

import logging
import torch
import torch.nn.functional as F
from torchtext.data.utils import ngrams_iterator
import pickle as pkl
import os
from ts.torch_handler.base_handler import BaseHandler

id2class = {0:"finance",1:"realty",2:"stocks",3:"edu",4:"sci",5:"society",6:"politics",7:"sports",8:"game",9:"entertainment"}

UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
vocab_dir = "vocab.pkl"
vocab = pkl.load(open(vocab_dir, 'rb'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128  # mini-batch大小
pad_size = 32  # 最好和训练集一致，根据文本特点选择该超参数


logger = logging.getLogger(__name__)

class TextCNN_handle(BaseHandler):
    """
    A custom model handler implementation.
    """

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

    def build_infer_iterator(self, dataset, batch_size, device):
        iter = self.InferDataIterater(dataset, batch_size, device)
        return iter

    def __init__(self):
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0
        self.device = device

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        super().initialize(context)
        self.initialized = True
        #  load the model, refer 'custom handler class' above for details
        # self.manifest = context.manifest
        #
        # properties = context.system_properties
        # model_dir = properties.get("model_dir")
        # self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        #
        # # Read model serialize/pt file
        # serialized_file = self.manifest['model']['serializedFile']
        # model_pt_path = os.path.join(model_dir, serialized_file)
        # if not os.path.isfile(model_pt_path):
        #     raise RuntimeError("Missing the model.pt file")
        #
        # model = torch.load(model_pt_path)
        # model.eval()  # 固化如dropout 这样的层
        # self.model = model.to(device)

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and make it inference ready
        # print(data)
        # logger.info("data is ")
        # logger.info(data.keys())
        inputs= data[0]['body']['input']
        for key in data:
            logger.info(key)
        #data = inputs.get("data") or inputs.get("body")
        preprocessed_data = inputs



        tokenizer = lambda x: [y for y in x]
        contents = []
        for lin in preprocessed_data:
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
        infer_iter = self.build_infer_iterator(contents, batch_size, device)
        return infer_iter  # [([...], 0), ([...], 1), ...]

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        model_output = []
        for texts, _ in model_input:
            out_puts = self.model.forward(texts)
            logger.info("the output is :")
            logger.info(out_puts)
            predics = torch.max(out_puts.data, 1)[1].cpu().numpy().tolist()
            for i in predics:
                model_output.append(id2class[i])
            res = ";".join(model_output)

        return [res]

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        postprocess_output = inference_output

        return postprocess_output

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)