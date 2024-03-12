FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

ENV TZ=Asia/Seoul

RUN apt-get -y update
RUN apt-get install -y wget

RUN apt-get install -y gcc
RUN apt-get install -y g++
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y tesseract-ocr tesseract-ocr-kor
RUN apt-get install -y x11-apps
RUN apt-get install -y git

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

RUN conda init bash

RUN conda create -n gtsam_env python=3.9 -y
SHELL ["conda", "run", "-n", "gtsam_env", "/bin/bash", "-c"]

RUN apt-get install -y build-essential
RUN apt-get install -y libssl-dev

RUN wget https://cmake.org/files/v3.20/cmake-3.20.5.tar.gz

RUN tar xvzf cmake-3.20.5.tar.gz
RUN cd cmake-3.20.5 && ./bootstrap && make && make install

RUN apt-get install -y libboost-all-dev libmetis-dev

RUN pip install gtsam_quadrics --no-binary :all:
RUN pip install quadricslam
RUN pip install pudb
RUN pip uninstall matplotlit
RUN pip install matplotlib==3.4.0
RUN pip install opencv-contrib-python

RUN conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
RUN conda install nvidia/label/cuda-11.6.2::libcusolver-dev

RUN cd && mkdir workspace && cd workspace
RUN git clone https://github.com/facebookresearch/detectron2.git
RUN python -m pip install -e detectron2

RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE

RUN apt-get install -y lsb-release
RUN apt-get install -y software-properties-common

RUN add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u

RUN apt-get install -y librealsense2-dkms

RUN apt-get install -y librealsense2-utils

RUN pip install pyrealsense2

