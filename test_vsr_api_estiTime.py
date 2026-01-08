import requests
import time
import os
import cv2

# =========================
# 配置
# =========================
BASE_URL = "http://localhost:6001"
INPUT_DIR = "test_videos_estiTime"   # 本地测试视频文件夹
MAX_SEQ_LEN = 10
VIDEO_EXTS = ('.mp4', '.mov', '.avi', '.mkv')

# =========================
# 工具函数
# =========================
def get_video_info(video_path):
    """获取视频基础信息"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    duration = frame_count / fps if fps > 0 else 0
    file_size = os.path.getsize(video_path)  # bytes
    bitrate = (file_size * 8) / duration if duration > 0 else 0  # bps

    return {
        "file": os.path.basename(video_path),
        "size_MB": round(file_size / 1024 / 1024, 2),
        "resolution": f"{width}x{height}",
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "duration_sec": round(duration, 2),
        "bitrate_kbps": round(bitrate / 1000, 2)
    }


def poll_progress(task_id, interval=10):
    """轮询进度，直到完成或报错"""
    url = f"{BASE_URL}/api/progress/{task_id}"
    last_progress = None

    while True:
        r = requests.get(url)
        if r.status_code != 200:
            print(f"查询进度失败: {r.text}")
            return None

        data = r.json()
        progress = round(data['progress'], 2)
        status = data['status']

        if progress != last_progress:
            # print(f"进度: {progress}% 状态: {status}")
            last_progress = progress

        if progress >= 100 or 'error' in status.lower():
            return data.get('result')

        time.sleep(interval)


# =========================
# 测试 upload_video
# =========================
def test_upload_video(video_path):
    print("\n==============================")
    print(f"开始处理视频: {os.path.basename(video_path)}")

    # 处理前信息
    info = get_video_info(video_path)
    print("【原始视频信息】")
    for k, v in info.items():
        print(f"  {k}: {v}")

    files = {'file': open(video_path, 'rb')}
    data = {'max_seq_len': MAX_SEQ_LEN}

    start_time = time.time()
    r = requests.post(f"{BASE_URL}/api/upload_video", files=files, data=data)
    files['file'].close()

    if r.status_code != 200:
        print(f"接口返回错误: {r.text}")
        return

    task_id = r.json()['task_id']
    print(f"任务ID: {task_id}")

    # 等待完成
    result = poll_progress(task_id)

    end_time = time.time()
    elapsed = round(end_time - start_time, 2)

    print("【超分处理完成】")
    print(f"  超分耗时: {elapsed} 秒")
    if result and 'file_url' in result:
        print(f"  输出文件: {result['file_url']}")

    return {
        "video_info": info,
        "sr_time_sec": elapsed,
        "output": result
    }


# =========================
# 主流程：处理整个文件夹
# =========================
if __name__ == "__main__":
    results = []

    for name in sorted(os.listdir(INPUT_DIR)):
        if name.lower().endswith(VIDEO_EXTS):
            video_path = os.path.join(INPUT_DIR, name)
            res = test_upload_video(video_path)
            if res:
                results.append(res)

    print("\n==============================")
    print("所有视频处理完成，总结：")
    for r in results:
        info = r["video_info"]
        print(
            f"{info['file']} | "
            f"{info['resolution']} | "
            f"{info['fps']}fps | "
            f"{info['duration_sec']}s | "
            f"{r['sr_time_sec']}s"
        )


