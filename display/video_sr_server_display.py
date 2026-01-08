from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import datetime
import torch
from mmengine import mkdir_or_exist
from mmagic.apis import MMagicInferencer
import cv2
import numpy as np
import subprocess, tempfile, glob, shutil
from psnr_calculator import calculate_psnr  # 导入PSNR计算函数

app = Flask(__name__)

# 上传文件的存储路径
UPLOAD_FOLDER = 'uploads'
INPUT_DIR = os.path.join(UPLOAD_FOLDER, 'input')
OUTPUT_DIR = os.path.join(UPLOAD_FOLDER, 'output')
ALLOWED_EXTENSIONS = {'mp4'}
PORT = 6001

# 确保目录存在
for path in [UPLOAD_FOLDER, INPUT_DIR, OUTPUT_DIR]:
    mkdir_or_exist(path)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _first_video_in_dir(d):
    exts = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm")
    for pat in exts:
        files = sorted(glob.glob(os.path.join(d, pat)))
        if files:
            return files[0]
    return None

def _ffmpeg_h264_encode(src_path, dst_path, crf=18, preset="medium"):
    # 固定为 H.264 + yuv420p；若需要更小体积可把 crf 调大（如 22~23）
    cmd = [
        "ffmpeg", "-y", "-i", src_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", str(crf), "-preset", preset,
        "-c:a", "copy"  # 有音频则直接拷贝；无音频 FFmpeg 会忽略
    ]
    # 输出路径最后
    cmd.append(dst_path)
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed: {completed.stderr.decode(errors='ignore')[:500]}")

def video_sr(input_path, output_path, max_seq_len=10):
    """运行 BasicVSR++，并把结果固定编码为 H.264 写到 output_path(.mp4)。"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 1) 先放到一个临时目录，由 MMagic 自己输出
    tmp_dir = tempfile.mkdtemp(prefix="mmagic_sr_", dir=OUTPUT_DIR)
    print(f"临时输出目录: {tmp_dir}")

    checkpoint_file = '/workspace/models/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth'

    editor = MMagicInferencer(
        model_name='basicvsr_pp',
        device=device,
        model_ckpt=checkpoint_file
    )

    # 让推理器短序列工作，避免显存暴涨
    editor.inferencer.inferencer.extra_parameters['max_seq_len'] = max_seq_len

    # 2) 推理到临时目录
    torch.cuda.empty_cache()
    # 启用混合精度加速
    with torch.autocast(device_type=device, dtype=torch.float16 if device == "cuda" else torch.float32):
        editor.infer(video=input_path, result_out_dir=tmp_dir)
    torch.cuda.empty_cache()

    # 3) 在临时目录里找生成的视频文件（不同版本可能导出 mp4/avi）
    gen_video = _first_video_in_dir(tmp_dir)
    if gen_video is None:
        # 某些版本可能导出为帧序列，此时可以考虑用 ffmpeg 把帧再封装为视频
        # 这里做一次兜底：若存在 png/jpg 帧，可按帧名序列封装
        pngs = sorted(glob.glob(os.path.join(tmp_dir, "*.png")))
        jpgs = sorted(glob.glob(os.path.join(tmp_dir, "*.jpg")))
        if pngs or jpgs:
            pattern = os.path.join(tmp_dir, "%08d.png") if pngs else os.path.join(tmp_dir, "%08d.jpg")
            # 这里假定导出为 30fps，如需更准确可从输入视频里读 fps 再传给 -r
            cmd = [
                "ffmpeg", "-y", "-framerate", "30", "-i", pattern,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "18", "-preset", "medium", output_path
            ]
            completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if completed.returncode != 0:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise RuntimeError("failed to mux frames into H.264 video")
            # 清理临时目录
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return

        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise FileNotFoundError("No generated video found in temporary directory.")

    # 4) 用 ffmpeg 固定转码为 H.264（libx264），输出到指定的 output_path（.mp4）
    _ffmpeg_h264_encode(gen_video, output_path, crf=18, preset="medium")

    # 5) 清理临时目录
    shutil.rmtree(tmp_dir, ignore_errors=True)

    
@app.route('/api/upload_video_display', methods=['POST'])
def upload_video_display():
    try:
        # 获取上传文件
        if 'gt_video' not in request.files or 'low_res_video' not in request.files:
            return jsonify({"code": 400, "message": "No video part"}), 400

        gt_video = request.files['gt_video']
        low_res_video = request.files['low_res_video']
        
        if gt_video.filename == '' or low_res_video.filename == '':
            return jsonify({"code": 400, "message": "No selected video"}), 400
        
        # 保存视频文件
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        gt_video_filename = f"gt_{timestamp}.mp4"
        low_res_video_filename = f"low_res_{timestamp}.mp4"
        
        gt_video_path = os.path.join(INPUT_DIR, gt_video_filename)
        low_res_video_path = os.path.join(INPUT_DIR, low_res_video_filename)

        gt_video.save(gt_video_path)
        low_res_video.save(low_res_video_path)
        
        # 获取max_seq_len参数
        max_seq_len = int(request.form.get('max_seq_len', 10))

        # 计算PSNR
        psnr_low_res = calculate_psnr(gt_video_path, low_res_video_path, use_y_channel=True, max_frames=max_seq_len)
        
        # 超分辨率推理过程
        output_filename = f"sr_output_{timestamp}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        video_sr(gt_video_path, output_path, max_seq_len=max_seq_len)
        
        # 计算超分辨率视频的PSNR
        psnr_sr = calculate_psnr(gt_video_path, output_path, use_y_channel=True, max_frames=max_seq_len)

        # 获取视频信息
        def get_video_info(video_path):
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            cap.release()
            return {
                "size": os.path.getsize(video_path),
                "duration": duration,
                "resolution": f"{width}x{height}"
            }

        # 获取GT视频、低清视频、超分视频信息
        gt_video_info = get_video_info(gt_video_path)
        low_res_video_info = get_video_info(low_res_video_path)
        sr_video_info = get_video_info(output_path)

        # 返回结果
        file_url = f"http://{request.host}/uploads/output/{output_filename}"
        return jsonify({
            "code": 200,
            "message": "Upload and processing successful",
            "gt_video_info": gt_video_info,
            "low_res_video_info": low_res_video_info,
            "low_res_psnr": psnr_low_res,
            "sr_video_info": sr_video_info,
            "sr_psnr": psnr_sr,
            "file_url": file_url
        })

    except Exception as e:
        return jsonify({"code": 500, "message": f"Server error: {str(e)}"}), 500


@app.route('/uploads/output/<path:filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)
