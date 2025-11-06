# Dockerfile for DiskANN + dependencies

FROM ubuntu:24.04   # 选择 22.04 保证 libomp-dev 可用，也可改成 24.04 并使用 libomp-dev

# 设置环境变量避免交互
ENV DEBIAN_FRONTEND=noninteractive

# 基本工具和依赖
RUN apt-get update && apt-get install -y \
    software-properties-common \
    git \
    make \
    cmake \
    g++ \
    libaio-dev \
    libgoogle-perftools-dev \
    libunwind-dev \
    clang-format \
    libboost-dev \
    libboost-program-options-dev \
    libmkl-full-dev \
    libcpprest-dev \
    python3.13 \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# 编译 DiskANN
WORKDIR /app/DiskANN
RUN mkdir -p build
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_GNU_OMP=ON
RUN cmake --build build -- -j$(nproc)

# 设置默认工作目录
WORKDIR /app
