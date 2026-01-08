# 单阶段构建Dockerfile（优化版）
FROM dustynv/torchvision:0.21.0-r36.4.0-cu128

# 环境变量配置
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_NO_CACHE_DIR=1
# Orin compute capability
ENV TORCH_CUDA_ARCH_LIST="8.7"  

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev build-essential wget cmake ninja-build \
    python3-venv \
    libgl1 libgl1-mesa-glx libglu1-mesa \
    libglib2.0-0 \
    libsm6 \
    libxrender1 libxrender-dev \
    libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /workspace

# 安装 mmcv-full
COPY mmcv-2.1.0.tar.gz /workspace/
RUN tar -xzf mmcv-2.1.0.tar.gz && \
    cd mmcv-2.1.0 && \
    MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.7" pip install -v . && \
    cd /workspace && rm -rf mmcv-2.1.0 mmcv-2.1.0.tar.gz && \
    # 检查 mmcv-full 是否安装成功且 CUDA ops 可用
    python3 -c "import mmcv; from mmcv.ops import modulated_deform_conv; print('✅ mmcv-full 安装成功且 CUDA ops 可用')"

# 安装 MMagic 和其他依赖
RUN pip install --no-cache-dir openmim && \
    mim install --no-cache-dir mmengine mmagic && \
    pip install --no-cache-dir \
        numpy==1.26.4 \
        opencv-python==4.9.0.80 \
        opencv-python-headless==4.9.0.80 \
        huggingface-hub==0.19.4 \
        diffusers==0.24.0 \
        transformers==4.35.2 \
        pillow \
        albumentations \
        albucore

# 安装 Flask 和相关 Web 依赖
RUN pip install --no-cache-dir --upgrade --ignore-installed \
    flask \
    werkzeug \
    requests

# ------- system deps for video & cv -------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*   


# 设置 Torch Hub 默认路径
ENV TORCH_HOME=/workspace/models/torch
# 创建模型目录并下载权重
RUN mkdir -p /workspace/models/torch/hub/checkpoints && \
    # SPyNet 放到 torch hub 默认目录
    wget -q -O /workspace/models/torch/hub/checkpoints/spynet_20210409-c6c1bd09.pth https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth && \
    # BasicVSR++ 仍然放在 /workspace/models
    mkdir -p /workspace/models && \
    wget -q -O /workspace/models/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth

# 拷贝脚本
# 环境验证和依赖检查
COPY check_env.py ./
# 视频超分辨率核心处理器 & HTTP服务端入口
COPY video_sr.py \
     video_sr_server.py \
     video_sr_server_withoutTime.py ./
# API接口自动化测试
COPY test_video_sr_api.py \
     test_video_sr_api_all.py \
     test_video_sr_api_display.py \
     test_vsr_api_withTime.py ./
# 拷贝 PSNR & 预估时间 计算脚本
COPY psnr_calculator.py ./
COPY time_calculator.py ./

# 数据集
COPY test_videos ./test_videos

# 默认进入 bash
CMD ["/bin/bash"]
