import requests
import os
import sys
import json

# 默认 Flask 服务地址
API_URL = "http://localhost:6001/api/upload_video_display"

def test_video_sr_api(video_path, low_res_video_path, max_seq_len=10):
    """
    测试 Flask 后端视频超分辨率接口

    参数:
        video_path (str): GT视频（原始高清视频）路径
        low_res_video_path (str): 低清晰度视频路径
        max_seq_len (int): 模型一次处理的帧数
    """
    if not os.path.exists(video_path) or not os.path.exists(low_res_video_path):
        print(f"❌ 文件不存在: {video_path} 或 {low_res_video_path}")
        sys.exit(1)

    # 发送 POST 请求
    files = {
        'gt_video': open(video_path, 'rb'),
        'low_res_video': open(low_res_video_path, 'rb')
    }
    data = {'max_seq_len': str(max_seq_len)}

    print(f"📤 上传视频: {video_path}, {low_res_video_path}")
    print(f"➡️  目标接口: {API_URL}")

    try:
        # 不再使用 stream=True，直接发送完整请求并接收响应
        response = requests.post(API_URL, files=files, data=data, timeout=1800)
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败: {e}")
        sys.exit(1)
    finally:
        files['gt_video'].close()
        files['low_res_video'].close()

    # 打印响应状态码
    print(f"📥 状态码: {response.status_code}")

    try:
        # 直接解析完整的 JSON 响应
        result = response.json()
        print("🧾 响应内容:")
        print(json.dumps(result, indent=4, ensure_ascii=False))

        # 判断返回是否成功（视频信息和PSNR）
        if result.get("code") == 200:
            print("✅ 第一部分处理成功！")
            print(f"GT视频信息: {result['gt_video_info']}")
            print(f"低清视频信息: {result['low_res_video_info']}")
            print(f"低清视频PSNR: {result['low_res_psnr']}")

            # 获取超分辨率视频的信息、PSNR 和下载链接
            sr_video_info = result.get("sr_video_info")
            file_url = result.get("file_url")

            # 输出获取到的超分辨率视频信息和下载链接
            print("🧾 超分辨率视频信息:")
            print(json.dumps(sr_video_info, indent=4, ensure_ascii=False))
            print(f"超分辨率视频下载地址: {file_url}")

            # 再次获取并打印超分辨率视频的PSNR
            sr_psnr = result.get("sr_psnr")
            print(f"超分辨率视频PSNR: {sr_psnr}")
        else:
            print("⚠️ 接口返回错误，请检查日志。")
    except ValueError:
        print("❌ 无法解析服务器返回的 JSON:")
        print(response.text)
        sys.exit(1)


if __name__ == "__main__":
    gt_video_path = "/workspace/data/display/test9.mp4"
    low_res_video_path = "/workspace/data/display/test9_270p.mp4"
    max_seq_len = 10

    if not os.path.exists(gt_video_path) or not os.path.exists(low_res_video_path):
        print(f"❌ 测试文件不存在")
    else:
        test_video_sr_api(gt_video_path, low_res_video_path, max_seq_len)
