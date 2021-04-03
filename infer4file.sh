# 对文件批量预测，待预测文件放在 $dataset/data目录下
# 命令行参数 直接传入文件名称  ./infer4file filename
# 最终预测结果保存为csv格式
export dataset='THUCNews'
python inference.py --model TextCNN --type File
paste -d "," $dataset/data/$1 $dataset/data/label.txt > $dataset/data/infer_result.csv
rm $dataset/data/label.txt


