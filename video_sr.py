import os
import torch
import cv2
import numpy as np
import argparse
from mmengine import mkdir_or_exist
from mmagic.apis import MMagicInferencer


def SR(video_dir, result_video_dir, device, max_seq_len):
    """
    视频超分辨率增强（BasicVSR++）

    参数:
        video_dir: 输入视频路径
        result_video_dir: 输出视频路径
        device: 'cpu' 或 'cuda'
        max_seq_len: 模型一次处理的帧数（值越大占用GPU越多）
    """
    print("  创建推理器实例...")
    checkpoint_file = '/workspace/models/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth'
    # 创建推理器实例
    editor = MMagicInferencer(
        model_name='basicvsr_pp',
        device=device,
        model_ckpt=checkpoint_file
    )
    # 设置最大序列长度
    editor.inferencer.inferencer.extra_parameters['max_seq_len'] = max_seq_len

    # 推理前清理显存
    torch.cuda.empty_cache()
    print("  执行视频超分辨率推理...")
    editor.infer(video=video_dir, result_out_dir=result_video_dir)
    # 推理后清理显存
    torch.cuda.empty_cache()
    print(f"✅ 推理完成，输出保存至: {result_video_dir}")



def main():
    parser = argparse.ArgumentParser(description='Video Super-Resolution with PSNR Calculation')
    parser.add_argument('--input', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, required=True, help='Path to output video')
    parser.add_argument('--max_seq_len', type=int, default=10, help='Max sequence length for BasicVSR++')

    args = parser.parse_args()

    video_dir = args.input
    result_video_dir = args.output
    max_seq_len = args.max_seq_len

    # 自动检测设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")


    # 检查原始视频是否存在
    if not os.path.exists(video_dir):
        print(f"❌原始视频文件不存在: {video_dir}")
        exit(1)

    # 创建输出目录
    mkdir_or_exist(os.path.dirname(result_video_dir))

    # 超分辨率增强
    print("开始视频超分辨率处理...")
    SR(video_dir, result_video_dir, device, max_seq_len)

    print(f"\n✅处理完成! 增强后的视频保存到: {result_video_dir}")


if __name__ == "__main__":
    main()