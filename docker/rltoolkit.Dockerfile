FROM  nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get upgrade -y

# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

RUN apt-get install --no-install-recommends -y \
    software-properties-common \
    vim \
    python3-pip\
    tmux \
    git 




# Added updated mesa drivers for integration with cpu - https://github.com/ros2/rviz/issues/948#issuecomment-1428979499
RUN add-apt-repository ppa:kisak/kisak-mesa && \
    apt-get update && apt-get upgrade -y &&\
    apt-get install libxcb-cursor0 -y && \
    apt-get install ffmpeg python3-opengl -y && \
    apt-get install -y cmake

RUN pip3 install matplotlib PyQt5 dill pandas pyqtgraph

RUN add-apt-repository universe

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata


RUN  pip3 install --no-cache-dir Cython


# install dependencies for mujoco

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1-mesa-dev \
    libosmesa6-dev \
    libglfw3 \
    libglfw3-dev \
    libglew-dev \
    patchelf \
    pkg-config \
    libx11-dev \
    libxrandr-dev \
    libxi-dev \
    libxxf86vm-dev \
    libxinerama-dev \
    libxkbcommon-dev \
    libwayland-dev \
    wayland-protocols \
    libxcursor-dev \
    && rm -rf /var/lib/apt/lists/*


#install jax
RUN pip3 install  "jax[cuda12]" 

#install torch
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126


# Copy workspace files
ENV WORKSPACE_PATH=/root/workspace
COPY workspace/ $WORKSPACE_PATH/src/




# install mujoco
WORKDIR /root/workspace/workspace/external/mujoco
RUN cmake -B build -S . \
    && cmake --build build -j$(nproc) --config Release \
    && cmake --install build


# python3 bindings for mujoco and mujoco-jax
RUN pip3 install mujoco  mujoco-mjx

ENV MUJOCOMEN_PATH=$WORKSPACE_PATH/src/external/mujoco_menagerie
# Set shell to bash
SHELL ["/bin/bash", "-c"]

# Final cleanup
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Set default shell to bash
CMD ["/bin/bash"]
