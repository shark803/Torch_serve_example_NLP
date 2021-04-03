import os



with open("test.txt","r",encoding="utf-8") as fin, open("infer.txt","w",encoding="utf-8") as fout:
    for line in fin:
         res = line.strip().split("\t")
         if len(res) == 2 :
             fout.write(res[0] + "\n")

