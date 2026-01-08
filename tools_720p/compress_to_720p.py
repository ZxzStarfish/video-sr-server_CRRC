# tools/make_x1p5_to_720p.py
import os
import glob
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "test"    # 输入目录（1080p）
DST_DIR = SCRIPT_DIR / "input"   # 输出目录（720p）

# 编码参数：画质/体积平衡，可按需调
CRF = 36            # 18~22 常用；18 更清晰更大
PRESET = "veryslow"     # ultrafast ... veryslow
PIX_FMT = "yuv420p" # 提高播放器兼容性

def _check_ffmpeg():
    for exe in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run([exe, "-version"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except Exception as e:
            raise RuntimeError(f"未检测到 {exe}，请先安装并加入 PATH。") from e

def downsample_x1p5_to_720p():
    _check_ffmpeg()
    DST_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(SRC_DIR / "test9.mp4")))
    if not files:
        print(f"❌ 未找到输入视频：{SRC_DIR}/test9.mp4")
        return

    print(f"🔧 发现 {len(files)} 个文件，开始 ×1.5 降采样到 1280×720（bicubic）…\n")
    for src in files:
        src_path = Path(src)
        dst_path = DST_DIR / src_path.name  # 输出保持同名

        # 方式一：显式指定目标分辨率（推荐）
        # -vf scale=1280:720:flags=bicubic
        # 方式二：按比例缩放：iw/1.5:ih/1.5（更通用）
        cmd = [
            "ffmpeg", "-y",
            "-i", str(src_path),
            "-vf", "scale=1280:720:flags=bicubic",
            "-c:v", "libx264",
            "-crf", str(CRF),
            "-preset", PRESET,
            "-pix_fmt", PIX_FMT,
            "-an",          # 音频不重编码"-c:a", "copy"；若想去掉音频，改为 "-an";
            "-movflags", "+faststart",
            str(dst_path)
        ]

        print(f"➡️  {src_path.name}  ->  {dst_path.name}")
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"❌ 处理失败：{src_path.name}")
            continue

        try:
            sz_mb = os.path.getsize(dst_path) / (1024 * 1024)
            print(f"   ✅ 完成，输出大小：{sz_mb:.2f} MB\n")
        except OSError:
            print("   ✅ 完成\n")

    print("🎉 全部完成！720p 文件已输出到 input/ 目录。")

if __name__ == "__main__":
    downsample_x1p5_to_720p()
