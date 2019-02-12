#!/usr/bin/env bash

pip2 install -r ./requirement/requirements2.txt

add-apt-repository ppa:kirillshkrogalev/ffmpeg-next
apt-get update
apt-get install ffmpeg


cd ./bottom-up-attention-master/lib
make
cd ..

apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
apt-get install --no-install-recommends libboost-all-dev
apt-get install libopenblas-dev
apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev

cd ./caffe/

make clean
make all -j32
make test -j32
make runtest -j32
make pycaffe -j32

cd ./python/
basepath=$(cd `dirname $0`; pwd)
echo export PYTHONPATH=$basepath: >> ~/.bashrc

pip2 install -r ./requirements.txt
