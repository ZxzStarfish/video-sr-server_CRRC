import os
import cv2
import numpy as np
from glob import glob

def read_video_frames(video_path_or_folder):
    """读取视频帧，如果是文件夹则按顺序读取图片"""
    if os.path.isdir(video_path_or_folder):
        # 读取文件夹内所有图片
        img_files = sorted(
            glob(os.path.join(video_path_or_folder, '*')),
            key=lambda x: os.path.basename(x)
        )
        frames = [cv2.imread(f) for f in img_files]
    else:
        # 读取视频
        cap = cv2.VideoCapture(video_path_or_folder)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
    return frames

def resize_frame(frame, target_size):
    """调整帧大小到 target_size (width, height)"""
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_CUBIC)

def rgb2y_channel(frame):
    """转换 BGR 图像到 Y 通道"""
    frame = frame.astype(np.float32)
    return 0.257 * frame[:, :, 2] + 0.504 * frame[:, :, 1] + 0.098 * frame[:, :, 0] + 16

def compute_psnr(ref_frame, target_frame):
    """计算单帧 PSNR (Y 通道)"""
    ref_y = rgb2y_channel(ref_frame)
    target_y = rgb2y_channel(target_frame)
    mse = np.mean((ref_y - target_y) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255**2 / mse)

def calculate_psnr(gt_path, target_path):
    """
    计算视频或帧序列的平均 PSNR
    自动处理分辨率差异
    """
    gt_frames = read_video_frames(gt_path)
    target_frames = read_video_frames(target_path)

    num_frames = min(len(gt_frames), len(target_frames))
    if num_frames == 0:
        raise ValueError("无法读取视频帧")

    # 对齐分辨率
    gt_h, gt_w = gt_frames[0].shape[:2]
    target_h, target_w = target_frames[0].shape[:2]

    if (gt_w, gt_h) != (target_w, target_h):
        # 下采样或上采样 target 到 GT 尺寸
        target_frames = [resize_frame(f, (gt_w, gt_h)) for f in target_frames]

    psnr_list = []
    for i in range(num_frames):
        psnr_val = compute_psnr(gt_frames[i], target_frames[i])
        psnr_list.append(psnr_val)

    avg_psnr = np.mean(psnr_list)
    return avg_psnr



# 示例用法
# gt_video = "D:\LAB/temp\RealTimeSR_framework\myTest/myData/GT/test9.mp4"
# lr_video = "D:\LAB/temp\RealTimeSR_framework\myTest/myData/input/test9_270p.mp4"
# sr_video = "D:\LAB/temp\RealTimeSR_framework\myTest/myData/input/test9_270p.mp4"
# print("LR vs GT PSNR:", calculate_psnr(gt_video, lr_video))
# print("SR vs GT PSNR:", calculate_psnr(gt_video, sr_video))
