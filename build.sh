#!/bin/bash
set -e

# 镜像名称与标签 小写
IMAGE_NAME=video-sr-server:latest

# 构建镜像（使用本地Dockerfile）
docker build -t ${IMAGE_NAME} -f Dockerfile \
    .

echo "✅ 镜像构建完成: ${IMAGE_NAME}"
