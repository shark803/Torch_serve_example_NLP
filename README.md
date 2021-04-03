# Torch_serve_example_NLP

## [原始文件模型来源于] Chinese-Text-Classification-Pytorch
## [原作者文件地址] git@github.com:649453932/Chinese-Text-Classification-Pytorch.git
## [原作者博客] (https://zhuanlan.zhihu.com/p/73176084)


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

## 使用说明
```
# 训练并测试：
# TextCNN
python run.py --model TextCNN

# 直接预测
基于文件  ./infer4file.sh

基于单句调试： python inference.py --model TextCNN --type Sent

# 基于web 
python app.py  端口 5005
curl -H 'Content-Type:application/json' -d '{"input":["易建联一分一个篮板,雄鹿输湖人","特朗普下台，拜登当选美国总统"]}' http://127.0.0.1:5005/predict

启动 serve 的 docker
docker run --rm -it --gpus all -p 8080:8080 -p 8081:8081 --name mar -v $(pwd)/model-store:/home/model-server/model-store -v $(pwd)/examples:/home/model-server/examples  pytorch/torchserve:latest-gpu

torch-model-archiver --model-name TextCNN --version 1.0 --model-file /home/model-server/examples/TextCNN.py --serialized-file /home/model-server/model-store/TextCNN.pt --export-path /home/model-server/model-store --handler /home/model-server/examples/TextCNN_handle.py --extra-files /home/model-server/examples/TextCNN_handle.py,/home/model-server/examples/vocab.pkl


关闭  torch-model --stop

torchserve --start --ncs --model-store model-store --models TextCNN.mar



sudo docker run --rm --shm-size=1g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -p8080:8080 \
        -p8081:8081 \
        -p8082:8082 \
        -p7070:7070 \
        -p7071:7071 \
        --mount type=bind,source=$(pwd)/model-store,target=/tmp/models pytorch/torchserve:latest-gpu torchserve --model-store=/tmp/models


curl http://localhost:8080/ping

curl -H 'Content-Type:application/json' -d '{"input":["易建联一分一个篮板,雄鹿输湖人","特朗普下台，拜登当选美国总统"]}' http://127.0.0.1:8080/predictions/TextCNN


### 参数
模型都在models目录下，超参定义和模型定义在同一文件中。  


## 对应论文
[1] Convolutional Neural Networks for Sentence Classification  

