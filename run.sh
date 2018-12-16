#!/usr/bin/env bash
str=$"/n"
name=$"superguts"

python3 prepocess.py

CUDA_VISIBLE_DEVICES=2 nohup python3 -u main.py --fold 0  --name $name > ${name}fold0.txt  &
echo -e $str
CUDA_VISIBLE_DEVICES=2 nohup python3 -u main.py --fold 1  --name $name > ${name}fold1.txt  &
echo -e $str
CUDA_VISIBLE_DEVICES=2 nohup python3 -u main.py --fold 2  --name $name > ${name}fold2.txt  &
echo -e $str


CUDA_VISIBLE_DEVICES=3 nohup python3 -u main.py --fold 3  --name $name > ${name}fold3.txt  &
echo -e $str
CUDA_VISIBLE_DEVICES=3 nohup python3 -u main.py --fold 4  --name $name > ${name}fold4.txt  &
echo -e $str


nohup python3 -u merge_fold.py --name $name > log.txt  &
echo -e $str
