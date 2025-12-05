FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 \
    python3-build \
    python3-pip \
    python3-venv \
    cmake \
    git && \
    apt-get autoremove --purge -y && \
    apt-get autoclean -y && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/* && \
    ln -sn /usr/local/cuda/include /usr/local/cuda/extras/CUPTI/include
