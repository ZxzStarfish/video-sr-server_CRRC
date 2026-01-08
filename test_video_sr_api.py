import requests
import os
import sys
import json

# 默认 Flask 服务地址
API_URL = "http://localhost:6001/api/upload_video"

def test_video_sr_api(video_path, max_seq_len=10):
    """
    测试 Flask 后端视频超分辨率接口

    参数:
        video_path (str): 待上传的 MP4 文件路径
        max_seq_len (int): 模型一次处理的帧数
    """
    if not os.path.exists(video_path):
        print(f"❌ 文件不存在: {video_path}")
        sys.exit(1)

    # 发送 POST 请求
    files = {'file': open(video_path, 'rb')}
    data = {'max_seq_len': str(max_seq_len)}

    print(f"📤 上传视频: {video_path}")
    print(f"➡️  目标接口: {API_URL}")

    try:
        response = requests.post(API_URL, files=files, data=data, timeout=1800)
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败: {e}")
        sys.exit(1)
    finally:
        files['file'].close()

    # 打印响应结果
    print(f"📥 状态码: {response.status_code}")

    try:
        result = response.json()
        print("🧾 响应内容:")
        print(json.dumps(result, indent=4, ensure_ascii=False))
    except ValueError:
        print("❌ 无法解析服务器返回的 JSON:")
        print(response.text)
        sys.exit(1)

    # 判断处理是否成功
    if response.status_code == 200 and result.get("code") == 200:
        print("✅ 测试成功！")
        print(f"处理后视频下载地址: {result['file_url']}")
    else:
        print("⚠️ 接口返回错误，请检查日志。")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python test_video_sr_api.py <video_path> [max_seq_len]")
        sys.exit(1)

    video_path = sys.argv[1]
    max_seq_len = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    test_video_sr_api(video_path, max_seq_len)
