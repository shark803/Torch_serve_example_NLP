#!/usr/bin/env python
# encoding: utf-8
# @author: 有莘不殁 （Lei Wang）
# @license: MIT
# @contact: lwangshark2013@gmail.com
# @software: pycharm
# @file: model_convert.py
# @time: 2021/3/16 下午3:58
# @desc: 将pytorch pt 格式的模型轉換成 onnx 格式

import torch
import torch.onnx as onnx
from importlib import import_module
import argparse
import onnx as ox
from onnx_tf.backend import prepare
from config import Config

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN')
args = parser.parse_args()
dataset = 'THUCNews'  # 数据集
config = Config(dataset)
model_name = "TextCNN"
x = import_module('models.' + model_name)


def convert2onnx(save_dict, output_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # loaded_model = torch.load(save_dict)
    global x
    model = x.Model(config).to(config.device)
    loaded_model = model.load_state_dict(torch.load(save_dict))

    loaded_model.eval()
    batch_size = 1  # 批处理大小
    input_shape = (32,)  # 输入数据,改成自己的输入shape
    loaded_model.eval()
    # dynamic_axes = {'input': {0: 'batch_size', 1: 'sentence_length'},'output': {0: 'batch_size'}}
    # Export the model

    x = torch.randint(1, 300, (batch_size, 32)).to(device)  # 生成张量
    sm = torch.jit.trace(loaded_model,x)
    sm.save("STextCNN.pt")
    # print(x.shape)
    # print(x)
    # X = X.to(device)
    export_onnx_file = output_path
    torch.onnx.export(loaded_model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      export_onnx_file,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}}
                      )

def onnx2pb(onnx_input_path, pb_output_path):
    onnx_model = ox.load(onnx_input_path)  # load onnx model
    tf_exp = prepare(onnx_model)  # prepare tf representation
    tf_exp.export_graph(pb_output_path)  # export the model


if __name__ == "__main__":
    dataset = 'THUCNews'  # 数据集
    x = import_module('models.' + "TextCNN")
    config = x.Config(dataset)
    model_name = args.model  # TextCNN
    output_onnx_path = dataset + "/saved_onnx/" + model_name + ".onnx"
    convert2onnx(config.save_pt_path, output_onnx_path)
    onnx_model = ox.load(output_onnx_path)
    print(ox.checker.check_model(onnx_model))
    print(ox.helper.printable_graph(onnx_model.graph))


    pb_output_path = dataset + "/saved_pb/" + model_name + ".pb"


    onnx2pb(output_onnx_path, pb_output_path)



