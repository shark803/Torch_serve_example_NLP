# Torch_serve_example_NLP
TextClassification using TextCNN; Generating pt model; Inference； Deploying



# 源码修改于Chinese-Text-Classification-Pytorch 
###（https://github.com/649453932/Chinese-Text-Classification-Pytorch.git）
## 感谢原作者提供的优雅的模型代码

[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)
MIT License

中文文本分类，TextCNN，基于pytorch，增加预测、服务化与torch serve

## 介绍
模型介绍、数据流动过程：[我的博客](https://zhuanlan.zhihu.com/p/73176084)  

数据以字为单位输入模型，预训练词向量使用 [搜狗新闻 Word+Character 300d](https://github.com/Embedding/Chinese-Word-Vectors)，[点这里下载](https://pan.baidu.com/s/14k-9jsspp43ZhMxqPmsWMQ)  

## 环境
python 3.8  
pytorch 1.7.1  
tqdm  
sklearn  
tensorboardX

## 中文数据集
[THUCNews](http://thuctc.thunlp.org/)中抽取了20万条新闻标题，已上传至github，文本长度在20到30之间。一共10个类别，每类2万条。

类别：财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐。

数据集划分：

数据集|数据量
--|--
训练集|18万
验证集|1万
测试集|1万


### 更换自己的数据集
 - 如果用字，按照我数据集的格式来格式化你的数据。  
 - 如果用词，提前分好词，词之间用空格隔开，`python run.py --model TextCNN --word True`  
 - 使用预训练词向量：utils.py的main函数可以提取词表对应的预训练词向量。  



## 使用说明
```
# 训练并测试：
# TextCNN
python run.py --model TextCNN


### 参数
模型都在models目录下，超参定义和模型定义在同一文件中。  


## 对应论文
[1] Convolutional Neural Networks for Sentence Classification  

## 模型预测

python inference.py --model TextCNN --type [File or Sent]
File 类型支持离线预测，Sent类型单句实时交互，输入&& 退出

## Flask 服务化
python app.py (生产环境推荐uwsgi)

post:
curl -H 'Content-Type:application/json' -d '{"input":["易建联一分一个篮板,雄鹿输湖人","特朗普下台，拜登当选美国总统"]}' http://127.0.0.1:5005/predict

## torch serve

torch 需要增加自定义handle TextCNN_handle 继承 BaseHandle
启动镜像： 
docker run --rm -it --gpus all -p 8080:8080 -p 8081:8081 --name mar -v $(pwd)/model-store:/home/model-server/model-store -v $(pwd)/examples:/home/model-s

模型和相关文件打包：
torch-model-archiver --model-name TextCNN --version 1.0 --model-file /home/model-server/examples/TextCNN.py --serialized-file /home/model-server/model-store/TextCNN.pt --export-path /home/model-server/model-store --handler /home/model-server/examples/TextCNN_handle.py --extra-files /home/model-server/examples/TextCNN_handle.py,/home/model-server/examples/vocab.pkl

停止当前服务：
torch --stop

部署打包文件服务
torchserve --start --ncs --model-store model-store --models TextCNN.mar

注：生产环境可以考虑以下方式:


sudo docker run --rm --shm-size=1g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -p8080:8080 \
        -p8081:8081 \
        -p8082:8082 \
        -p7070:7070 \
        -p7071:7071 \
        --mount type=bind,source=$(pwd)/model-store,target=/tmp/models pytorch/torchserve:latest-gpu torchserve --model-store=/tmp/models

