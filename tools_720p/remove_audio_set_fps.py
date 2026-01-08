import subprocess
from pathlib import Path

# === 配置 ===
input_path = Path("test_all_bili.mp4")     # 输入视频
output_path = input_path.with_name("test_all.mp4")  # 输出文件

def process_video(in_path, out_path):
    if not in_path.exists():
        raise FileNotFoundError(f"❌ 找不到文件: {in_path}")

    print(f"🎬 处理视频: {in_path}")
    print(f"➡️  输出文件: {out_path}")

    # ffmpeg 命令说明：
    # -an                去掉音频
    # -r 24              设置帧率为 24fps
    # -c:v libx264       使用 H.264 编码（兼容性好）
    # -crf 23            轻度压缩，质量较好
    # -preset medium     压缩速度与质量平衡
    # -pix_fmt yuv420p   确保播放器兼容
    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_path),
        "-an",                  # 去掉音频
        "-r", "24",             # 降帧到 24fps
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        str(out_path)
    ]

    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"✅ 处理完成，输出文件已保存到: {out_path}")
    else:
        print("❌ 处理失败，请检查 ffmpeg 是否安装。")

if __name__ == "__main__":
    process_video(input_path, output_path)
