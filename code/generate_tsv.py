# -*- coding: utf-8 -*
# !/usr/bin/env python
import argparse
import sys

import os
import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect, _get_blobs
from fast_rcnn.nms_wrapper import nms
from multiprocessing import Manager
import multiprocessing as mp
import caffe
import pprint
import numpy as np
import cv2

import json
from glob import glob
from tqdm import tqdm



# Example:
"""
python2 ./bottom-up-attention/tools/generate_tsv.py --gpu 0 --cfg ./bottom-up-attention/experiments/cfgs/faster_rcnn_end2end_resnet.yml \
--def ./bottom-up-attention/models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt \
--net ./bottom-up-attention/data/faster_rcnn_models/resnet101_faster_rcnn_final_iter_320000.caffemodel

python2 generate_tsv.py --gpu 0 --cfg ./bottom-up-attention/experiments/cfgs/faster_rcnn_end2end_resnet.yml \
--def ./bottom-up-attention/models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt \
--net ./bottom-up-attention/data/faster_rcnn_models/resnet101_faster_rcnn_final_iter_320000.caffemodel
"""


def load_image_ids():
    video_p = '../data/'
    video_files = glob(video_p+'DatasetA/train/*.mp4') + \
                  glob(video_p+'DatasetA/test/*.mp4') + \
                  glob(video_p+'DatasetB/train/*.mp4') + \
                  glob(video_p+'DatasetB/test/*.mp4')

    return video_files


def sampling(msg_queue,files,need_frame):
    def read_videos(file, need_frame=40):
        v_id = file.split('/')[-1][:-4]
        vc = cv2.VideoCapture(file)

        rval = False
        if vc.isOpened():
            rval, frame = vc.read()
        else:
            print(file, 'error')
            exit(-1)

        frame_flow = []

        while rval:
            frame_flow.append(frame)
            rval, frame = vc.read()

        vc.release()

        num_frame = len(frame_flow)
        if need_frame > num_frame:
            return np.array(frame_flow)


        num_I = len(I_Frame_idx[v_id])
        v_flow = []
        seq = []

        I_idx = []
        if num_I>need_frame:
            for j in np.linspace(0,num_I,need_frame, False):
                v_flow.append(frame_flow[I_Frame_idx[v_id][int(j)]])
        else:
            for i in range(num_I):
                res_need = need_frame - len(v_flow)
                ave_frame = (res_need + num_I - i -1) // (num_I-i)
                if num_I-1>i:
                    max_size = len(frame_flow) - I_Frame_idx[v_id][i+1]
                    if ave_frame*(num_I-i-1) > max_size :
                        ave_frame = ave_frame*(num_I-i)-max_size

                start = I_Frame_idx[v_id][i]
                if i >= num_I - 1:
                    end = len(frame_flow)
                else:
                    end = I_Frame_idx[v_id][i + 1]

                last_j = -1
                I_idx.append(len(v_flow))
                for j in np.linspace(start, end, ave_frame, False):
                    if int(j) == last_j:
                        continue
                    v_flow.append(frame_flow[int(j)])
                    last_j = int(j)
                    seq.append(int(j))

                if len(v_flow) >= need_frame:
                    break
        if len(v_flow) != need_frame:
            print(v_flow)
        # if len(I_idx) != len(I_Frame_idx[v_id]):
        #     print(I_idx,I_Frame_idx[v_id],len(I_idx),len(I_Frame_idx[v_id]))
        assert len(v_flow) == need_frame
        # assert len(I_idx) == len(I_Frame_idx[v_id])
        return np.array(v_flow),I_idx

    for f in files:
        video_id = f.split('/')[-1][:-4]
        frame,I_idx = read_videos(f,need_frame=need_frame)
        msg_queue.put((frame,video_id,I_idx))


def get_detections_from_im(net, msg_q, outfile,num_videos,conf_thresh=0.2,attr_thresh = 0.1,part=None):

    if part is not None:
        attr_p = '../data/video_attr'+str(part)+'.json'
        I_idx_p = '../data/I_idx'+str(part)+'.json'
    else:
        attr_p = '../data/video_attr.json'
        I_idx_p = '../data/I_idx.json'

    if os.path.exists(attr_p):
        with open(attr_p,'r') as f:
            videos_attr_data = json.loads(f.read())
    else:
        videos_attr_data = {}

    if os.path.exists(I_idx_p):
        with open(I_idx_p, 'r') as f:
            I_idx_data = json.loads(f.read())
    else:
        I_idx_data = {}

    for num_v in tqdm(range(num_videos)):
        frames, v_id,I_idx = msg_q.get()
        if v_id in videos_attr_data:
            continue
        I_idx_data[v_id] = I_idx
        features = []
        video_attr = []
        print(v_id)
        for i,im in tqdm(enumerate(frames),total=len(frames)):

            frame_attr = []
            scores, boxes, attr_scores, rel_scores = im_detect(net, im)
            # Keep the original boxes, don't worry about the regresssion bbox outputs
            rois = net.blobs['rois'].data.copy()
            # unscale back to raw image space
            blobs, im_scales = _get_blobs(im, None)

            cls_boxes = rois[:, 1:5] / im_scales[0]
            cls_prob = net.blobs['cls_prob'].data
            pool5 = net.blobs['pool5_flat'].data
            attr_prob = net.blobs['attr_prob'].data

            # Keep only the best detections
            max_conf = np.zeros((rois.shape[0]))
            for cls_ind in range(1, cls_prob.shape[1]):
                cls_scores = scores[:, cls_ind]
                dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
                keep = np.array(nms(dets, cfg.TEST.NMS))
                max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

            keep_boxes = np.where(max_conf >= conf_thresh)[0]
            if len(keep_boxes) < MIN_BOXES:
                keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
            elif len(keep_boxes) > MAX_BOXES:
                keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]


            objects = np.argmax(cls_prob[keep_boxes][:, 1:], axis=1)
            attr = np.argmax(attr_prob[keep_boxes][:, 1:], axis=1)
            attr_conf = np.max(attr_prob[keep_boxes][:, 1:], axis=1)

            for i in range(len(keep_boxes)):
                cls = classes[objects[i] + 1]
                if attr_conf[i] > attr_thresh:
                    cls = attributes[attr[i] + 1] + " " + cls
                frame_attr.append(cls)

            video_attr.append(frame_attr)
            features.append(pool5[keep_boxes])
        videos_attr_data[v_id] = video_attr

        if num_v % 100 == 0:
            with open(attr_p, 'w') as f:
                f.write(json.dumps(videos_attr_data, indent=4, separators=(',', ': ')))
            with open(I_idx_p, 'w') as f:
                f.write(json.dumps(I_idx_data, indent=4, separators=(',', ': ')))
        np.save(outfile + v_id, np.array(features))
    with open(attr_p, 'w') as f:
        f.write(json.dumps(videos_attr_data, indent=4, separators=(',', ': ')))
    with open(I_idx_p, 'w') as f:
        f.write(json.dumps(I_idx_data, indent=4, separators=(',', ': ')))



def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id(s) to use',
                        default='0', type=str)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to use',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def generate_tsv(net, files, outfile):

    num_sampler = 6
    pool = mp.Pool(num_sampler + 1)
    msg_q = Manager().Queue(maxsize=num_sampler * 2)
    results = []
    batch_size = 1 + len(files) // num_sampler

    for i in range(num_sampler):
        results.append(pool.apply_async(sampling, args=(msg_q, files[i * batch_size:(i + 1) * batch_size],40)))

    pool.close()
    get_detections_from_im(net, msg_q,outfile,len(files),part=None)
    pool.join()


if __name__ == '__main__':

    model_path = "./bottom-up-attention-master/data/faster_rcnn_models/"
    if os.path.exists(model_path+"resnet101_faster_rcnn_final_iter_320000.caffemodel") == False:
        print('Please put the resnet101_faster_rcnn_final_iter_320000.caffemodel in the '+model_path)
        exit(0)

    args = parse_args()

    print('Called with args:')
    print(args)

    FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']

    # Settings for the number of features per image. To re-create pretrained features with 36 features
    # per image, set both values to 36.
    MIN_BOXES = 36
    MAX_BOXES = 36

    I_Frame_idx = "./info/I_Frame_idx.json"
    with open(I_Frame_idx, 'r') as f:
        I_Frame_idx = json.loads(f.read())

    data_path = './bottom-up-attention-master/data/genome/1600-400-20'

    # Load classes
    classes = ['__background__']
    with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
        for object in f.readlines():
            classes.append(object.split(',')[0].lower().strip())

    # Load attributes
    attributes = ['__no_attribute__']
    with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
        for att in f.readlines():
            attributes.append(att.split(',')[0].lower().strip())

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)


    gpu_id = args.gpu_id
    gpu_list = gpu_id.split(',')
    gpus = [int(i) for i in gpu_list]

    print('Using config:')
    pprint.pprint(cfg)
    assert cfg.TEST.HAS_RPN

    caffe.init_log()
    caffe.log('Using devices %s' % str(gpus))


    caffe.set_mode_gpu()

    caffe.set_device(gpus[0])
    net = caffe.Net(args.prototxt, caffe.TEST, weights=args.caffemodel)



    attr_p = '../data/video_attr.json'
    video_p = '../data/'
    image_ids = glob(video_p+'test/*.mp4')



    if os.path.exists(attr_p):
        with open(attr_p,'r') as f:
            videos_attr_data = json.loads(f.read())
        image_ids = [file for file in image_ids if file.split('/')[-1][:-4] not in videos_attr_data]
    print(len(image_ids))

    outfile ='../data/rcnn/'
    if os.path.exists(outfile) == False:
        os.makedirs(outfile)
    generate_tsv(net,image_ids, outfile)
