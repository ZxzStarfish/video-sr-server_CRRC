# tools/make_lr_bicubic_x4_h264.py
import os
import glob
import subprocess
from pathlib import Path

# 以脚本所在目录为基准
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "test"    # 输入 1080p 视频目录
DST_DIR = SCRIPT_DIR / "input36"   # 输出 270p（×4降采样）目录

# ============= 质量模式（选一种） =============
# 模式A：近无损（文件较大，推荐做配对评测/训练用）
# CRF = 18               # 18~20 近无损；想更极致可设 0（无损，体积更大）
# PRESET = "slow"        # 压缩效率：ultrafast ... veryslow
# 模式B：体积更小（想更小可增大 CRF，比如 22~24）
CRF = 28            # 18~22 常用；18 更清晰更大
PRESET = "slow"     # ultrafast ... veryslow
# =============================================

PIX_FMT = "yuv420p"    # 统一像素格式，提升兼容性

def _check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        subprocess.run(["ffprobe", "-version"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except Exception as e:
        raise RuntimeError("未检测到 ffmpeg/ffprobe，请先安装并加入 PATH。") from e

def make_lr_bicubic_x4():
    _check_ffmpeg()
    DST_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(SRC_DIR / "test9.mp4")))
    if not files:
        print(f"❌ 未找到输入视频：{SRC_DIR}/test9.mp4")
        return

    print(f"🔧 发现 {len(files)} 个文件，开始生成 LR（双三次 ×4 降采样，保持 H.264）…\n")
    for src in files:
        src_path = Path(src)
        dst_path = DST_DIR / src_path.name  # 输出保持同名

        # 关键点：
        # - scale=iw/4:ih/4:flags=bicubic  -> 双三次 ×4 降采样到 270p
        # - 仍用 H.264（libx264）编码；音频直接复制
        # - 不改变帧率，沿用源时间戳（若要固定可加 -r 25）
        cmd = [
            "ffmpeg", "-y",
            "-i", str(src_path),
            "-vf", "scale=iw/4:ih/4:flags=bicubic",
            "-c:v", "libx264",
            "-crf", str(CRF),
            "-preset", PRESET,
            "-pix_fmt", PIX_FMT,
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(dst_path)
        ]
        print(f"➡️  {src_path.name}  ->  {dst_path.name}")
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"❌ 生成失败：{src_path.name}")
        else:
            try:
                sz_mb = os.path.getsize(dst_path) / (1024 * 1024)
                print(f"   ✅ 完成，输出大小：{sz_mb:.2f} MB\n")
            except OSError:
                print("   ✅ 完成\n")

    print("🎉 全部生成完成！LR 已输出到 input/ 目录。")

if __name__ == "__main__":
    make_lr_bicubic_x4()
