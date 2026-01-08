# -*- coding: utf-8 -*-
# 计算两段视频的 PSNR（默认 Y 通道），可直接在 PyCharm 运行
import os
import cv2
import numpy as np
from pathlib import Path
from statistics import mean, median

# ========= 配置 =========
REF_PATH  = Path("test/test9.mp4")        # 参照视频（ground-truth）
DIST_PATH = Path("test/test9.mp4")       # 待评视频（重建/压缩/超分后的）
USE_Y_CHANNEL = True                     # True: 用 Y(亮度) 通道；False: 用 RGB
MAX_FRAMES = None                        # 限制最多比较多少帧；None 表示比较到 min(两者总帧数)
# =======================

def psnr_from_arrays(a: np.ndarray, b: np.ndarray) -> float:
    """计算单帧 PSNR (dB)。输入为同尺寸的灰度或三通道图像。"""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-10:
        return 99.0  # 近似无损
    PIX_MAX = 255.0
    return 10.0 * np.log10((PIX_MAX * PIX_MAX) / mse)

def read_frame(cap):
    ok, frame = cap.read()
    if not ok:
        return None
    return frame

def to_eval_mat(frame_bgr, use_y=True):
    """BGR -> Y (单通道) 或保留 BGR。"""
    if use_y:
        ycbcr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        return ycbcr[:, :, 0]
    else:
        return frame_bgr

def main():
    if not REF_PATH.exists() or not DIST_PATH.exists():
        raise FileNotFoundError(f"找不到视频：\n  REF : {REF_PATH}\n  DIST: {DIST_PATH}")

    cap_ref  = cv2.VideoCapture(str(REF_PATH))
    cap_dist = cv2.VideoCapture(str(DIST_PATH))

    if not cap_ref.isOpened() or not cap_dist.isOpened():
        raise RuntimeError("无法打开其中一个视频，请检查路径/编码格式。")

    # 获取分辨率、帧数
    w_ref  = int(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_ref  = int(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_ref  = int(cap_ref.get(cv2.CAP_PROP_FRAME_COUNT))

    w_dst  = int(cap_dist.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_dst  = int(cap_dist.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_dst  = int(cap_dist.get(cv2.CAP_PROP_FRAME_COUNT))

    n_cmp = min(n_ref, n_dst) if MAX_FRAMES is None else min(MAX_FRAMES, n_ref, n_dst)

    print("===== PSNR 评测 =====")
    print(f"参照视频: {REF_PATH}   尺寸: {w_ref}x{h_ref}  帧数: {n_ref}")
    print(f"待评视频: {DIST_PATH}  尺寸: {w_dst}x{h_dst}  帧数: {n_dst}")
    print(f"对齐帧数: {n_cmp}  （超出部分将忽略）")
    print(f"评测通道: {'Y(亮度)' if USE_Y_CHANNEL else 'RGB'}")
    print("---------------------")

    psnrs = []
    i = 0
    while i < n_cmp:
        f_ref  = read_frame(cap_ref)
        f_dist = read_frame(cap_dist)
        if f_ref is None or f_dist is None:
            break

        # 尺寸不一致时，将待评视频缩放到参照尺寸
        if (f_dist.shape[1], f_dist.shape[0]) != (w_ref, h_ref):
            f_dist = cv2.resize(f_dist, (w_ref, h_ref), interpolation=cv2.INTER_CUBIC)

        A = to_eval_mat(f_ref,  USE_Y_CHANNEL)
        B = to_eval_mat(f_dist, USE_Y_CHANNEL)

        # 如果用 Y 通道，则 A/B 为单通道；否则为 3 通道，函数一样适用
        psnr = psnr_from_arrays(A, B)
        psnrs.append(psnr)

        i += 1

    cap_ref.release()
    cap_dist.release()

    if not psnrs:
        raise RuntimeError("未计算到任何帧的 PSNR，可能视频无法解码或长度为 0。")

    print(f"帧数: {len(psnrs)}")
    print(f"PSNR 平均值:  {mean(psnrs):.3f} dB")
    print(f"PSNR 中位数: {median(psnrs):.3f} dB")
    print(f"PSNR 最小值:  {min(psnrs):.3f} dB")
    print(f"PSNR 最大值:  {max(psnrs):.3f} dB")

if __name__ == "__main__":
    main()
