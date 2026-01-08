import os
import cv2
import glob
import shutil
import tempfile
import subprocess
import numpy as np
import torch
import argparse
from mmengine import mkdir_or_exist
from mmagic.apis import MMagicInferencer

# ================== 全局参数（不改 main） ==================
OUT_SIZE = None   # None 表示保持 ×4 原尺寸  (1920, 1080)
H264_CRF = 18
H264_PRESET = "medium"

def _ensure_dir(d: str):
    if d:
        os.makedirs(d, exist_ok=True)

def _first_video_in_dir(d):
    exts = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm")
    for pat in exts:
        files = sorted(glob.glob(os.path.join(d, pat)))
        if files:
            return files[0]
    return None

def _ffprobe_is_h264(path: str) -> bool:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name", "-of", "default=nw=1", path
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        return False
    out = p.stdout.decode(errors="ignore").lower()
    return "codec_name=h264" in out

def _ffmpeg_encode_h264(src_path, dst_path, fps=None, out_size=None, crf=18, preset="medium"):
    vf = []
    if out_size is not None:
        w, h = out_size
        vf.append(f"scale={w}:{h}:flags=bicubic")
    vf_arg = ["-vf", ",".join(vf)] if vf else []
    fps_arg = ["-r", str(int(round(fps)))] if fps else []
    cmd = ["ffmpeg", "-y", "-i", src_path] + fps_arg + [
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", str(crf), "-preset", preset, "-an"
    ] + vf_arg + [dst_path]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg encode failed")

def _ffmpeg_mux_frames_to_h264(frames_pattern, dst_path, fps=30, out_size=None, crf=18, preset="medium"):
    vf = []
    if out_size is not None:
        w, h = out_size
        vf.append(f"scale={w}:{h}:flags=bicubic")
    vf_arg = ["-vf", ",".join(vf)] if vf else []
    cmd = [
        "ffmpeg", "-y", "-framerate", str(int(round(fps))), "-i", frames_pattern,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", str(crf), "-preset", preset, "-an"
    ] + vf_arg + [dst_path]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg failed to mux frames")

def SR(video_dir, result_video_dir, device, max_seq_len=10):
    """
    视频超分辨率增强（BasicVSR++ ×4）→ 输出 H.264（libx264, yuv420p）。
    采用“一次性调用 infer(video=原视频)”的稳妥方式；随后把输出统一转为 H.264。
    """
    print("  创建推理器实例...")
    checkpoint_file = '/workspace/models/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth'
    editor = MMagicInferencer(
        model_name='basicvsr_pp',
        device=device,
        model_ckpt=checkpoint_file
    )

    # 速度相关（安全失败）：FP16 / channels_last
    try:
        torch.backends.cudnn.benchmark = True
        torch.set_grad_enabled(False)
        model = getattr(editor, 'model', None) or getattr(editor.inferencer, 'model', None)
        if model is not None:
            model.eval()
            try:
                model.to(memory_format=torch.channels_last)
            except Exception:
                pass
            if device == "cuda":
                try:
                    model.half()
                except Exception:
                    pass
    except Exception as e:
        print(f"⚠️ 跳过模型优化：{e}")

    # 读取输入 fps
    cap = cv2.VideoCapture(video_dir)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开输入视频: {video_dir}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    # 设置 max_seq_len（兼容路径）
    for path_try in [
        ("inferencer", "inferencer", "extra_parameters"),
        ("inferencer", "extra_parameters"),
    ]:
        try:
            obj = editor
            for name in path_try:
                obj = getattr(obj, name)
            if isinstance(obj, dict):
                obj['max_seq_len'] = int(max_seq_len)
        except Exception:
            continue

    # 让 MMagic 推理到临时目录
    tmp_dir = tempfile.mkdtemp(prefix="mmagic_sr_out_")
    try:
        torch.cuda.empty_cache()
        print("  执行视频超分辨率推理（整段）...")
        editor.infer(video=video_dir, result_out_dir=tmp_dir)
        torch.cuda.empty_cache()

        gen_video = _first_video_in_dir(tmp_dir)
        if gen_video is not None:
            need_scale = OUT_SIZE is not None
            is_h264 = _ffprobe_is_h264(gen_video)
            if is_h264 and not need_scale:
                # 直接封装/拷贝
                _ensure_dir(os.path.dirname(result_video_dir))
                subprocess.run(["ffmpeg","-y","-i",gen_video,"-c:v","copy","-c:a","copy",result_video_dir],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                _ffmpeg_encode_h264(gen_video, result_video_dir, fps=fps,
                                    out_size=OUT_SIZE, crf=H264_CRF, preset=H264_PRESET)
            print(f"✅ 推理完成，H.264 已输出: {result_video_dir}")
            return

        # 试帧序列
        pngs = sorted(glob.glob(os.path.join(tmp_dir, "*.png")))
        if not pngs:
            sub_pngs = sorted(glob.glob(os.path.join(tmp_dir, "*", "*.png")))
            if sub_pngs:
                pngs = sub_pngs
        if pngs:
            # 统一编号
            frames_dir = os.path.dirname(pngs[0])
            # 检查是否为 %08d 连续命名；否则重排
            ordered = True
            for i, pth in enumerate(pngs):
                name = os.path.basename(pth).split('.')[0]
                if name != f"{i:08d}":
                    ordered = False; break
            if not ordered:
                frames_dir2 = tempfile.mkdtemp(prefix="sr_frames_")
                for i, pth in enumerate(pngs):
                    shutil.copy2(pth, os.path.join(frames_dir2, f"{i:08d}.png"))
                frames_dir = frames_dir2
            pattern = os.path.join(frames_dir, "%08d.png")
            _ffmpeg_mux_frames_to_h264(pattern, result_video_dir, fps=fps,
                                       out_size=OUT_SIZE, crf=H264_CRF, preset=H264_PRESET)
            print(f"✅ 推理完成，H.264 已输出: {result_video_dir}")
            return

        raise FileNotFoundError("No generated video or frames found in temporary directory.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

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