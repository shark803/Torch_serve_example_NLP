# 在docker容器里启动tensorflow serving，把8501(rest)端口映射出来
MODEL_PATH="$PWD/deploying"
CONFIG_DIR="$PWD/config.d"
IMAGE=tensorflow/serving:2.3.0-gpu
echo "111111" | sudo -S docker run -t --rm --gpus all --name tf-serving-gpu -p 8501:8501 \
    -v "${MODEL_PATH}:/models" \
    -v "${CONFIG_DIR}:/etc/tf_serving_conf.d" \
    -e TF_CPP_MIN_VLOG_LEVEL=2 \
    -e NVIDIA_VISIBLE_DEVICES=0 \
    tensorflow/serving:2.3.0-gpu \
    "--model_config_file=/etc/tf_serving_conf.d/config"
    "--enable_batching=true" \
    "--batching_parameters_file=/etc/tf_serving_conf.d/batching_config" &
