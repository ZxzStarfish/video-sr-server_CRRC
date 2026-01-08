import requests
import time
import os

# =========================
# 配置
# =========================
BASE_URL = "http://localhost:6001"
INPUT_DIR = "test_videos"  # 本地测试视频文件夹
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# 测试视频
UPLOAD_VIDEO_FILE = os.path.join(INPUT_DIR, "test9_270p.mp4")
GT_VIDEO_FILE = os.path.join(INPUT_DIR, "test9_gt.mp4")
LOW_RES_VIDEO_FILE = os.path.join(INPUT_DIR, "test9_270p.mp4")

MAX_SEQ_LEN = 10  # 可调

# =========================
# 工具函数
# =========================
def poll_progress(task_id, interval=5, max_repeat=1):
    """轮询进度，避免重复打印相同进度超过 max_repeat 次"""
    url = f"{BASE_URL}/api/progress/{task_id}"
    last_progress = None
    repeat_count = 0

    while True:
        r = requests.get(url)
        if r.status_code != 200:
            print(f"查询进度失败: {r.text}")
            break
        data = r.json()
        status = data['status']
        progress = round(data['progress'], 2)

        # 控制重复输出，只看进度
        if progress != last_progress:
            print(f"进度: {progress}% 状态: {status}")
            last_progress = progress
            repeat_count = 1
        else:
            repeat_count += 1
            if repeat_count <= max_repeat:
                print(f"进度: {progress}% 状态: {status} (重复 {repeat_count})")

        # 完成或报错结束轮询
        if progress >= 100 or 'error' in status.lower():
            return data.get('result')
        time.sleep(interval)


def download_file(file_url, save_dir=DOWNLOAD_DIR):
    local_filename = os.path.join(save_dir, os.path.basename(file_url))
    with requests.get(file_url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"文件已下载: {local_filename}")
    return local_filename

# =========================
# 测试 /api/upload_video
# =========================
def test_upload_video():
    print("=== 测试 /api/upload_video ===")
    files = {'file': open(UPLOAD_VIDEO_FILE, 'rb')}
    data = {'max_seq_len': MAX_SEQ_LEN}
    r = requests.post(f"{BASE_URL}/api/upload_video", files=files, data=data)
    files['file'].close()
    if r.status_code != 200:
        print(f"接口返回错误: {r.text}")
        return
    resp = r.json()
    task_id = resp['task_id']
    print(f"任务ID: {task_id}")

    # 轮询进度
    result = poll_progress(task_id)
    if result and 'file_url' in result:
        print("处理后file: ", result.get('file_url'))
        # download_file(result['file_url'])
    print("=== /api/upload_video 测试完成 ===\n")

# =========================
# 测试 /api/upload_video_display
# =========================
def test_upload_video_display():
    print("=== 测试 /api/upload_video_display ===")
    files = {
        'gt_video': open(GT_VIDEO_FILE, 'rb'),
        'low_res_video': open(LOW_RES_VIDEO_FILE, 'rb')
    }
    data = {'max_seq_len': MAX_SEQ_LEN}
    r = requests.post(f"{BASE_URL}/api/upload_video_display", files=files, data=data)
    files['gt_video'].close()
    files['low_res_video'].close()
    if r.status_code != 200:
        print(f"接口返回错误: {r.text}")
        return
    resp = r.json()
    task_id = resp['task_id']
    print(f"任务ID: {task_id}")

    # 轮询进度
    result = poll_progress(task_id)
    if result and 'file_url' in result:
        # download_file(result['file_url'])
        print("GT视频信息:", result.get('gt_video_info'))
        print("低清视频信息:", result.get('low_res_video_info'))
        print("低清PSNR:", result.get('low_res_psnr'))
        print("超分视频信息:", result.get('sr_video_info'))
        print("超分PSNR:", result.get('sr_psnr'))
        print("处理后file: ", result.get('file_url'))
    print("=== /api/upload_video_display 测试完成 ===\n")

# =========================
# 主函数
# =========================
if __name__ == "__main__":
    # test_upload_video()
    test_upload_video_display()
