# tools/split_to_test.py
import os
import subprocess
from pathlib import Path

INPUT = Path("test_all.mp4")     # 输入视频
OUT_DIR = Path("test")           # 输出目录
START_NUMBER = 1                       # 从 test0.mp4 开始编号
SEGMENT_SEC = 10                       # 每段 30s

def _check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except Exception as e:
        raise RuntimeError("未检测到 ffmpeg，请先安装并加入 PATH。") from e

def split_video():
    if not INPUT.exists():
        raise FileNotFoundError(f"找不到输入文件：{INPUT.resolve()}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 输出命名：tools/test/test0.mp4、test1.mp4、…
    out_pattern = str(OUT_DIR / "test%d.mp4")

    # 说明：
    # -c copy       不重编码（快/无损），切点会贴近关键帧
    # -map 0        保留所有流（音频/字幕若有）
    # -f segment    分段复用器
    # -segment_time 每段时长
    # -reset_timestamps 1 每段从 00:00:00 计时
    # -start_number 起始编号
    cmd = [
        "ffmpeg", "-y",
        "-i", str(INPUT),
        "-c", "copy",
        "-map", "0",
        "-f", "segment",
        "-segment_time", str(SEGMENT_SEC),
        "-reset_timestamps", "1",
        "-start_number", str(START_NUMBER),
        out_pattern
    ]

    print(f"🎬 输入：{INPUT.resolve()}")
    print(f"📁 输出目录：{OUT_DIR.resolve()}")
    print(f"⏱️  每段：{SEGMENT_SEC}s ；文件名：test{START_NUMBER}.mp4 起")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg 切片失败，请检查输入文件或 ffmpeg 安装。")
    print("✅ 切分完成！")

if __name__ == "__main__":
    _check_ffmpeg()
    split_video()
