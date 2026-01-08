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


def frames_to_video(frame_folder, output_path, fps=30, codec='libx264', crf=18, preset='medium'):
    """
    将帧序列合并成视频（高质量编码）
    
    Args:
        frame_folder (str): 帧序列所在文件夹，帧文件名应按顺序命名
        output_path (str): 输出视频路径（例如 output.mp4）
        fps (int): 帧率
        codec (str): FFmpeg 编码器，默认 libx264
        crf (int): 压缩率，0 无损，18 视觉无损质量较高
        preset (str): 编码预设，slow/medium/fast 等，slow 更高质量
    """
    # 确保帧按顺序
    frames = sorted(glob.glob(os.path.join(frame_folder, '*')))
    if len(frames) == 0:
        raise ValueError("帧序列为空")

    # 临时生成帧列表 txt
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        for frame in frames:
            f.write(f"file '{os.path.abspath(frame)}'\n")
        list_file = f.name

    # FFmpeg 命令
    cmd = [
        'ffmpeg',
        '-y',  # 覆盖输出
        '-r', str(fps),
        '-f', 'concat',
        '-safe', '0',
        '-i', list_file,
        '-c:v', codec,
        '-crf', str(crf),
        '-preset', preset,
        '-pix_fmt', 'yuv420p',  # 保证兼容性
        output_path
    ]
    
    subprocess.run(cmd, check=True)
    
    # 删除临时文件
    os.remove(list_file)
    print(f"视频已生成: {output_path}")


def video_sr(input_path, output_path, max_seq_len=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

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
        editor.infer(video=input_path, result_out_dir=output_path)
    torch.cuda.empty_cache()


@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    try:
        if 'file' not in request.files:
            return jsonify({"code": 400, "message": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"code": 400, "message": "No selected file"}), 400

        if not allowed_file(file.filename):
            return jsonify({"code": 400, "message": "Invalid file type, only MP4 is allowed"}), 400

        max_seq_len = int(request.form.get('max_seq_len', 10))

        # 保存输入文件
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        input_filename = f"input_{timestamp}.mp4"
        output_filename = f"output_{timestamp}"
        output_h264_filename = f"output_h264_{timestamp}.mp4"
        input_path = os.path.join(INPUT_DIR, input_filename)
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        output_path_h264 = os.path.join(OUTPUT_DIR, output_h264_filename)
        file.save(input_path)

        # 执行超分辨率推理
        video_sr(input_path, output_path, max_seq_len=max_seq_len)
        # 然后进行转码，保持帧率不变
        cap = cv2.VideoCapture(input_path)
        frames_to_video(output_path, output_path_h264, fps = cap.get(cv2.CAP_PROP_FPS))

        # 返回结果
        file_url = f"http://{request.host}/uploads/output/{output_h264_filename}"
        return jsonify({
            "code": 200,
            "message": "Upload and processing successful",
            "file_url": file_url
        })

    except Exception as e:
        return jsonify({"code": 500, "message": f"Server error: {str(e)}"}), 500

# 新增------用于展示GT视频、低清视频和超分视频的信息及PSNR
@app.route('/api/upload_video_display', methods=['POST'])
def upload_video_display():
    try:
        print("开始上传视频...")  # 1. 开始上传文件

        # 获取上传文件
        if 'gt_video' not in request.files or 'low_res_video' not in request.files:
            print("缺少视频部分")
            return jsonify({"code": 400, "message": "No video part"}), 400

        gt_video = request.files['gt_video']
        low_res_video = request.files['low_res_video']

        if gt_video.filename == '' or low_res_video.filename == '':
            print("视频文件未选择")
            return jsonify({"code": 400, "message": "No selected video"}), 400

        # 保存视频文件
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        gt_video_filename = f"gt_{timestamp}.mp4"
        low_res_video_filename = f"low_res_{timestamp}.mp4"

        gt_video_path = os.path.join(INPUT_DIR, gt_video_filename)
        low_res_video_path = os.path.join(INPUT_DIR, low_res_video_filename)

        gt_video.save(gt_video_path)
        low_res_video.save(low_res_video_path)

        print(f"上传视频成功: {gt_video_filename}, {low_res_video_filename}")  # 2. 输出上传视频的文件名

        # 获取max_seq_len参数
        max_seq_len = int(request.form.get('max_seq_len', 10))

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

        # 获取GT视频和低清视频的信息
        print("获取视频信息...")
        gt_video_info = get_video_info(gt_video_path)
        low_res_video_info = get_video_info(low_res_video_path)

        print(f"GT视频信息: {gt_video_info}")  # 3. 输出GT视频信息
        print(f"低清视频信息: {low_res_video_info}")  # 4. 输出低清视频信息

        # 计算低清视频的PSNR
        print("计算低清视频PSNR...")
        low_res_psnr = calculate_psnr(gt_video_path, low_res_video_path)
        print(f"低清视频PSNR: {low_res_psnr}")  # 5. 输出低清视频的PSNR


        # 执行低清视频的超分辨率推理过程
        print("执行超分辨率推理过程...")
        output_filename = f"sr_output_{timestamp}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        output_h264_filename = f"output_h264_{timestamp}.mp4"
        output_path_h264 = os.path.join(OUTPUT_DIR, output_h264_filename)

        # 先进行超分处理，获取超分后的视频路径
        video_sr(low_res_video_path, output_path, max_seq_len=max_seq_len)
        print(f"超分处理完成: {output_filename}")

        # 然后进行转码，保持帧率不变
        cap = cv2.VideoCapture(low_res_video_path)
        frames_to_video(output_path, output_path_h264, fps = cap.get(cv2.CAP_PROP_FPS))
        print(f"视频转码完成: {output_h264_filename}")

        # 计算超分辨率视频的信息和文件链接
        sr_video_info = get_video_info(output_path_h264)
        file_url = f"http://{request.host}/uploads/output/{output_h264_filename}"

        # 计算超分辨率视频的PSNR
        sr_psnr = calculate_psnr(gt_video_path, output_path)
        print(f"超分辨率视频PSNR: {sr_psnr}")


        response_data = {
            "code": 200,
            "message": "Video info and PSNR calculation successful",
            "gt_video_info": gt_video_info,
            "low_res_video_info": low_res_video_info,
            "low_res_psnr": low_res_psnr,
            "sr_video_info": sr_video_info,
            "sr_psnr": sr_psnr,
            "file_url": file_url
        }
        return jsonify(response_data)

    except Exception as e:
        print(f"服务器错误: {str(e)}")
        return jsonify({"code": 500, "message": f"Server error: {str(e)}"}), 500
# 新增------

@app.route('/uploads/output/<path:filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)
