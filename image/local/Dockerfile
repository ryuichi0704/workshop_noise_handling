FROM nvidia/cuda:10.1-cudnn7-devel

# set working directory
ARG USER_NAME
ENV HOME=/home/${USER_NAME}
WORKDIR $HOME

# install basic dependencies
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        htop \
        git \
        wget \
        tmux \
        cmake \
        gcc \
        g++ \
        build-essential \
        ca-certificates \
        software-properties-common \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# install python3.7
RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.7 \
    && wget -O ./get-pip.py https://bootstrap.pypa.io/get-pip.py \
    && python3.7 ./get-pip.py \
    && ln -s /usr/bin/python3.7 /usr/local/bin/python3 \
    && ln -s /usr/bin/python3.7 /usr/local/bin/python


# install common python packages
ADD ./requirements.txt /tmp
RUN pip install pip setuptools -U \
    && pip install -r /tmp/requirements.txt

