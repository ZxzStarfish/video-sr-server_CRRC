# video-sr-server_CRRC

## 项目简介
本模块是"轨道交通车辆智能运维边云协同系统"中"视频传输优化平台模块"的核心组件，基于深度学习模型 BasicVSR++ 实现低清视频的超分辨率增强。主要解决车载视频压缩传输后分辨率下降、细节缺失的问题，通过 ×4 放大与质量重建，提升视频 PSNR 指标和主观视觉效果，支撑后续人眼回顾和事件分析等场景。


## 技术栈
- 核心模型：BasicVSR++（基于 MMagic 框架）
- 部署环境：Docker + NVIDIA Orin（ARM64 架构）
- 接口类型：RESTful API
- 支持格式：MP4 视频文件

## 快速部署
### 环境依赖
- 硬件：NVIDIA Orin 平台（支持 GPU 加速）
- 软件：Docker >= 20.10，NVIDIA Container Toolkit

### 部署步骤
1. **部署 Docker 镜像**（镜像大小约 7.9GB）
   基础镜像 [dustynv/torchvision:0.21.0-r36.4.0-cu128](https://hub.docker.com/layers/dustynv/torchvision/0.21.0-r36.4.0-cu128)
   ```bash
   # 从压缩包加载镜像到本地
   docker load -i video-sr-server:latest.tar.gz

   # 或在相关代码目录（videoSR/10_09torchvision）下重新构建镜像
   # 运行该指令前确保当前环境中有镜像dustynv/torchvision:0.21.0-r36.4.0-cu128
   sudo chmod +x build.sh
   sudo ./build.sh
   ```

3. **启动服务容器**
   ```bash
   # 挂载数据目录，映射端口 6001（可自定义）
   sudo docker run -d --rm --gpus all \
     -p 6001:6001 \
     -v $(pwd)/data:/workspace/data \
     --name <容器名> \
     video-sr-server:latest python3 video_sr_server_withoutTime.py
   ```

4. **验证服务状态**
   ```bash
   # 检查容器是否运行
   docker ps | grep <容器名>

   # 查看服务日志
   docker logs <容器名>
   ```

## 使用指南
### 两种使用方式
| 使用方式 | 适用场景 | 操作入口 |
|----------|----------|----------|
| Web 界面 | 快速测试、手动操作、画质对比 | 浏览器访问 `http://<服务器地址>:6001` |
| API 接口 | 系统集成、批量处理、自动化流程 | 调用下方 RESTful API |


### API 接口使用
#### 接口说明（v3 版本，推荐使用）
> 注：所有接口返回结果需通过「查询任务进度接口」获取最终结果

##### 1. 上传单视频接口（upload_video）
- URL: `http://<服务器地址>:6001/api/upload_video`
- 方法: POST
- 描述: 上传单个 MP4 视频进行超分推理
- 请求参数：
  | 字段        | 类型  | 必填 | 说明                                  |
  |-------------|-------|------|---------------------------------------|
  | file        | file  | 是   | 待处理的 MP4 视频文件                 |
  | max_seq_len | int   | 否   | 模型最大序列长度（10-50），默认 10    |
- 返回示例：
  ```json
  {
    "code": 200,
    "task_id": "a1b2c3d4-xxxx-xxxx-xxxx-xxxxxxxx",
    "message": "Upload successful, processing started"
  }
  ```

##### 2. 上传对比视频接口（upload_video_display）
- URL: `http://<服务器地址>:6001/api/upload_video_display`
- 方法: POST
- 描述: 上传低清视频和高清参考视频，超分后计算 PSNR
- 请求参数：
  | 字段          | 类型  | 必填 | 说明                                  |
  |---------------|-------|------|---------------------------------------|
  | low_res_video | file  | 是   | 低分辨率视频（待超分）                |
  | gt_video      | file  | 是   | 高分辨率参考视频（Ground Truth）      |
  | max_seq_len   | int   | 否   | 模型最大序列长度（10-50），默认 10    |
- 返回示例：
  ```json
  {
    "code": 200,
    "task_id": "a1b2c3d4-xxxx-xxxx-xxxx-xxxxxxxx",
    "message": "Upload successful, processing started"
  }
  ```

##### 3. 查询任务进度接口（progress）
- URL: `http://<服务器地址>:6001/api/progress/<task_id>`
- 方法: GET
- 描述: 轮询任务进度，任务完成后返回结果
- 请求参数：
  | 字段    | 类型   | 必填 | 说明                          |
  |---------|--------|------|-------------------------------|
  | task_id | string | 是   | 上传接口返回的任务 ID         |
- 返回示例（任务完成）：
  ```json
  {
    "code": 200,
    "progress": 100,
    "status": "done",
    "result": {
      "file_url": "http://<服务器地址>:6001/uploads/output/a1b2c3_output.mp4",
      "gt_video_info": {
        "size": 4711804,
        "duration": 4.5,
        "resolution": "1920x1080"
      },
      "low_res_video_info": {
        "size": 655559,
        "duration": 4.5,
        "resolution": "480x270"
      },
      "sr_video_info": {
        "size": 15098422,
        "duration": 4.32,
        "resolution": "1920x1080"
      },
      "low_res_psnr": 20.70,
      "sr_psnr": 22.02
    }
  }
  ```

##### 4. 获取处理后视频文件
- URL: `http://<服务器地址>:6001/uploads/output/<filename>`
- 方法: GET
- 描述: 通过查询接口返回的 file_url 直接下载视频

### 测试命令
#### 1. 测试 API 接口
```bash
# 测试单个视频超分
sudo docker exec -i <容器名> python3 < test_video_sr_api_all.py

# 测试对比视频超分（含 PSNR 计算）
sudo docker exec -i <容器名> python3 < test_video_sr_api_display.py
```

#### 2. 直接运行超分脚本（容器内）
```bash
# 进入容器
sudo docker exec -it <容器名> bash

# 运行超分脚本
python3 video_sr_fast.py --input ./data/input/test9.mp4 --output ./data/output/output9_fast.mp4 --max_seq_len 10
```


## 版本说明
| 版本 | 主要变更 |
|------|----------|
| v3（最新） | 新增任务进度查询接口，优化异步处理流程 |
| v2 | 基础 API 功能，支持单视频和对比视频处理 |
```

