import cv2
import numpy as np
from pathlib import Path
import time

def psnr_from_arrays(a: np.ndarray, b: np.ndarray) -> float:
    """计算单帧 PSNR (dB)。输入为同尺寸的灰度或三通道图像。"""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-10:
        return 99.0  # 近似无损
    PIX_MAX = 255.0
    return 10.0 * np.log10((PIX_MAX * PIX_MAX) / mse)

def to_eval_mat(frame_bgr, use_y=True):
    """BGR -> Y (单通道) 或保留 BGR。"""
    if use_y:
        ycbcr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        return ycbcr[:, :, 0]
    else:
        return frame_bgr

def read_frame(cap):
    ok, frame = cap.read()
    if not ok:
        return None
    return frame

def calculate_psnr(ref_path: Path, dist_path: Path, use_y_channel: bool = True, max_frames: int = None):
    """
    计算通过时间对齐并缩放分辨率的视频PSNR值。

    参数:
    - ref_path: 参照视频路径（ground-truth）
    - dist_path: 待评视频路径（重建/压缩/超分后的）
    - use_y_channel: 是否使用Y(亮度)通道，True表示使用Y通道，False表示使用RGB
    - max_frames: 限制最多比较多少帧，None表示比较到两个视频最小帧数为止

    返回:
    - psnr: 计算出的PSNR值
    """
    cap_ref = cv2.VideoCapture(str(ref_path))
    cap_dist = cv2.VideoCapture(str(dist_path))

    if not cap_ref.isOpened() or not cap_dist.isOpened():
        raise RuntimeError("无法打开其中一个视频，请检查路径/编码格式。")

    # 获取分辨率、帧数、帧率
    w_ref = int(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_ref = int(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_ref = int(cap_ref.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_ref = cap_ref.get(cv2.CAP_PROP_FPS)

    w_dst = int(cap_dist.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_dst = int(cap_dist.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_dst = int(cap_dist.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_dist = cap_dist.get(cv2.CAP_PROP_FPS)

    n_cmp = min(n_ref, n_dst) if max_frames is None else min(max_frames, n_ref, n_dst)

    print(f"参照视频: {ref_path} 尺寸: {w_ref}x{h_ref} 帧数: {n_ref}")
    print(f"待评视频: {dist_path} 尺寸: {w_dst}x{h_dst} 帧数: {n_dst}")
    print(f"对齐帧数: {n_cmp}")
    print(f"评测通道: {'Y(亮度)' if use_y_channel else 'RGB'}")

    psnrs = []
    timestamps_ref = []
    timestamps_dist = []

    # 获取时间戳
    for i in range(n_ref):
        timestamps_ref.append(i / fps_ref)

    for i in range(n_dst):
        timestamps_dist.append(i / fps_dist)

    i = 0
    while i < n_cmp:
        # 根据时间戳对齐帧
        timestamp_ref = timestamps_ref[i]
        closest_index = min(range(len(timestamps_dist)), key=lambda j: abs(timestamps_dist[j] - timestamp_ref))
        timestamp_dist = timestamps_dist[closest_index]

        # 设置视频流位置并读取帧
        cap_ref.set(cv2.CAP_PROP_POS_FRAMES, i)
        cap_dist.set(cv2.CAP_PROP_POS_FRAMES, closest_index)

        f_ref = read_frame(cap_ref)
        f_dist = read_frame(cap_dist)

        if f_ref is None or f_dist is None:
            break

        # 尺寸不一致时，将待评视频缩放到参照视频的分辨率
        if (f_dist.shape[1], f_dist.shape[0]) != (w_ref, h_ref):
            f_dist = cv2.resize(f_dist, (w_ref, h_ref), interpolation=cv2.INTER_CUBIC)

        # 转换为评测矩阵（Y通道或RGB）
        A = to_eval_mat(f_ref, use_y_channel)
        B = to_eval_mat(f_dist, use_y_channel)

        # 计算PSNR
        psnr = psnr_from_arrays(A, B)
        psnrs.append(psnr)

        i += 1

    cap_ref.release()
    cap_dist.release()

    if not psnrs:
        raise RuntimeError("未计算到任何帧的PSNR，可能视频无法解码或长度为 0。")

    print(f"PSNR 平均值:  {np.mean(psnrs):.3f} dB")
    print(f"PSNR 中位数: {np.median(psnrs):.3f} dB")
    print(f"PSNR 最小值:  {min(psnrs):.3f} dB")
    print(f"PSNR 最大值:  {max(psnrs):.3f} dB")

    return np.mean(psnrs)

if __name__ == "__main__":
    REF_PATH = Path("data/test9.mp4")  # 参照视频路径
    DIST_PATH = Path("data/output9_fast.mp4") # 待评视频路径
    USE_Y_CHANNEL = True  # 是否使用Y通道
    MAX_FRAMES = None  # 最大帧数限制，None表示不限制

    calculate_psnr(REF_PATH, DIST_PATH, USE_Y_CHANNEL, MAX_FRAMES)
