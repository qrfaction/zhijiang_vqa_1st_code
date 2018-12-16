#!/usr/bin/env bash

python2 get_IFrame.py

python2 ./bottom-up-attention-master/tools/generate_tsv.py --gpu 0 \
	--cfg ./bottom-up-attention-master/experiments/cfgs/faster_rcnn_end2end_resnet.yml \
    --def ./bottom-up-attention-master/models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt \
    --net ./bottom-up-attention-master/data/faster_rcnn_models/resnet101_faster_rcnn_final_iter_320000.caffemodel
