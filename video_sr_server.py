from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import datetime
import threading
import time
import uuid
import torch
from mmengine import mkdir_or_exist
from mmagic.apis import MMagicInferencer
import cv2
import numpy as np
import subprocess, tempfile, glob, shutil
from psnr_calculator import calculate_psnr

app = Flask(__name__)

# 上传文件路径
UPLOAD_FOLDER = 'uploads'
INPUT_DIR = os.path.join(UPLOAD_FOLDER, 'input')
OUTPUT_DIR = os.path.join(UPLOAD_FOLDER, 'output')
ALLOWED_EXTENSIONS = {'mp4'}
PORT = 6001

for path in [UPLOAD_FOLDER, INPUT_DIR, OUTPUT_DIR]:
    mkdir_or_exist(path)

# --- 任务进度存储 ---
task_progress = {}  # task_id: {"progress": 0, "status": "pending", "result": None}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- 保持不改动 ---
def frames_to_video(frame_folder, output_path, fps=30, codec='libx264', crf=18, preset='medium'):
    frames = sorted(glob.glob(os.path.join(frame_folder, '*')))
    if len(frames) == 0:
        raise ValueError("帧序列为空")
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        for frame in frames:
            f.write(f"file '{os.path.abspath(frame)}'\n")
        list_file = f.name
    cmd = [
        'ffmpeg', '-y', '-r', str(fps), '-f', 'concat', '-safe', '0', '-i', list_file,
        '-c:v', codec, '-crf', str(crf), '-preset', preset, '-pix_fmt', 'yuv420p',
        output_path
    ]
    subprocess.run(cmd, check=True)
    os.remove(list_file)

def video_sr(input_path, output_path, max_seq_len=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_file = '/workspace/models/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth'
    editor = MMagicInferencer(
        model_name='basicvsr_pp',
        device=device,
        model_ckpt=checkpoint_file
    )
    # editor = MMagicInferencer(model_name='basicvsr_pp',device=device)
    editor.inferencer.inferencer.extra_parameters['max_seq_len'] = max_seq_len
    torch.cuda.empty_cache()
    with torch.autocast(device_type=device, dtype=torch.float16 if device == "cuda" else torch.float32):
        editor.infer(video=input_path, result_out_dir=output_path)
    torch.cuda.empty_cache()

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

# --- 预估时间计算函数 ---
# 经验参数（可根据实际测试数据调整）
def estimate_sr_time(width, height, frame_count):
    """
    估算视频超分处理时间
    Args:
        width (int): 视频宽度
        height (int): 视频高度
        frame_count (int): 视频帧数
    Returns:
        float: 预计总耗时（秒）
    """
    T_init = 11.07 # 固定时间开销，单位s
    k = 1.161108e-05 # 参数，单位s/像素/帧
    pixels = width * height
    T = T_init + k * pixels * frame_count
    return T

# --- 后台任务通用函数 ---
def process_video_task(task_id, input_path, max_seq_len=10, is_display=False, gt_video_path=None, host="127.0.0.1:"+str(PORT)):
    try:
        task_progress[task_id]["progress"] = 0
        task_progress[task_id]["status"] = "uploaded"

        # 预处理阶段
        task_progress[task_id]["progress"] = 5
        task_progress[task_id]["status"] = "preprocessing"
        # 估算超分处理时间
        cap = cv2.VideoCapture(input_path)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release() 
        estimated_time = estimate_sr_time(width, height, frame_count)

        # video_sr 推理
        task_progress[task_id]["status"] = "sr_inference"
        output_folder = os.path.join(OUTPUT_DIR, f"sr_{task_id}")
        os.makedirs(output_folder, exist_ok=True)
        # 模拟推理进度函数
        def simulate_sr_progress(task_id, start=5, end=88, duration_estimate=30, interval=1):
            steps = int(duration_estimate / interval)
            progress = start
            increment = (end - start) / steps
            for _ in range(steps):
                time.sleep(interval)
                progress += increment
                # 只更新比当前值更大的进度
                task_progress[task_id]["progress"] = max(task_progress[task_id]["progress"], min(int(progress), end))
        # 启动模拟线程
        sim_thread = threading.Thread(target=simulate_sr_progress, args=(task_id,), kwargs={"duration_estimate": estimated_time})
        sim_thread.start()
        # 执行真实模型推理
        video_sr(input_path, output_folder, max_seq_len=max_seq_len)
        # 推理完成
        task_progress[task_id]["progress"] = 90
        sim_thread.join()  # 确保模拟线程结束

        # frames_to_video 转码
        task_progress[task_id]["status"] = "merging_video"
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        output_h264_path = os.path.join(OUTPUT_DIR, f"{task_id}_output.mp4")
        frames_to_video(output_folder, output_h264_path, fps=fps)
        if is_display and gt_video_path:
            task_progress[task_id]["progress"] = 95
            task_progress[task_id]["status"] = "calculating_psnr"
        else:
            task_progress[task_id]["progress"] = 100
            task_progress[task_id]["status"] = "done"

        # 构造结果
        result = {
            "file_url": f"http://{host}/uploads/output/{os.path.basename(output_h264_path)}",
        }

        # 如果是 upload_video_display，还返回视频信息和 PSNR
        if is_display and gt_video_path:
            gt_info = get_video_info(gt_video_path)
            low_res_info = get_video_info(input_path)
            sr_info = get_video_info(output_h264_path) # 不是 output_folder
            low_res_psnr = calculate_psnr(gt_video_path, input_path)
            sr_psnr = calculate_psnr(gt_video_path, output_folder)

            task_progress[task_id]["progress"] = 100
            task_progress[task_id]["status"] = "done"
            result.update({
                "gt_video_info": gt_info,
                "low_res_video_info": low_res_info,
                "sr_video_info": sr_info,
                "low_res_psnr": low_res_psnr,
                "sr_psnr": sr_psnr
            })

        task_progress[task_id]["result"] = result

    except Exception as e:
        task_progress[task_id]["status"] = f"error: {str(e)}"

# --- upload_video 接口 ---
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
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        input_filename = f"input_{timestamp}.mp4"
        input_path = os.path.join(INPUT_DIR, input_filename)
        file.save(input_path)

        # --- 启动后台线程 ---
        task_id = str(uuid.uuid4())
        task_progress[task_id] = {"progress": 0, "status": "pending", "result": None}
        host = request.host  # 获取host在主线程中
        threading.Thread(target=process_video_task, args=(task_id, input_path, max_seq_len), kwargs={"host": host}).start()

        return jsonify({"code": 200, "task_id": task_id, "message": "Upload successful, processing started"})

    except Exception as e:
        return jsonify({"code": 500, "message": f"Server error: {str(e)}"}), 500

# --- upload_video_display 接口 ---
@app.route('/api/upload_video_display', methods=['POST'])
def upload_video_display():
    try:
        if 'gt_video' not in request.files or 'low_res_video' not in request.files:
            return jsonify({"code": 400, "message": "No video part"}), 400
        gt_video = request.files['gt_video']
        low_res_video = request.files['low_res_video']
        if gt_video.filename == '' or low_res_video.filename == '':
            return jsonify({"code": 400, "message": "No selected video"}), 400

        max_seq_len = int(request.form.get('max_seq_len', 10))
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        gt_video_path = os.path.join(INPUT_DIR, f"gt_{timestamp}.mp4")
        low_res_video_path = os.path.join(INPUT_DIR, f"low_res_{timestamp}.mp4")
        gt_video.save(gt_video_path)
        low_res_video.save(low_res_video_path)

        # --- 启动后台线程 ---
        task_id = str(uuid.uuid4())
        task_progress[task_id] = {"progress": 0, "status": "pending", "result": None}
        host = request.host  # 获取host在主线程中
        threading.Thread(target=process_video_task, args=(task_id, low_res_video_path, max_seq_len, True, gt_video_path), kwargs={"host": host}).start()

        return jsonify({"code": 200, "task_id": task_id, "message": "Upload successful, processing started"})

    except Exception as e:
        return jsonify({"code": 500, "message": f"Server error: {str(e)}"}), 500

# --- 进度查询接口 ---
@app.route('/api/progress/<task_id>', methods=['GET'])
def get_progress(task_id):
    if task_id not in task_progress:
        return jsonify({"code": 404, "message": "Task not found"}), 404
    return jsonify({
        "code": 200,
        "progress": round(task_progress[task_id]["progress"], 2),
        "status": task_progress[task_id]["status"],
        "result": task_progress[task_id].get("result")
    })

@app.route('/uploads/output/<path:filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)
