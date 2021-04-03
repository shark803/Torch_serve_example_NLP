sudo docker run --rm --shm-size=1g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -p8080:8080 \
        -p8081:8081 \
        -p8082:8082 \
        -p7070:7070 \
        -p7071:7071 \
        --mount type=bind,source=/home/shark803/other_projs/TextClassification_Pytorch_deployingTest/THUCNews/saved_dict,target=/tmp/models pytorch/torchserve:latest-gpu torchserve --model-store=/tmp/models 
        
        


