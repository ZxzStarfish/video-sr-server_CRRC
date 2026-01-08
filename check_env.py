import torch
import torchvision
import sys

print("====================================================")
print("=                PyTorch & GPU 环境检查             =")
print("====================================================")
print("基础镜像: dustynv/torchvision:0.21.0-r36.4.0-cu128 (ARM64架构)")
print("====================================================")

print(f"Python版本: {sys.version.split()[0]}")
print(f"PyTorch版本: {torch.__version__}")
print(f"TorchVision版本: {torchvision.__version__}")
# 查看CUDA是否可用及版本
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU设备名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU设备数量: {torch.cuda.device_count()}")
    # 简单的GPU计算测试
    print("\n🔄 执行简单的GPU计算测试...")
    x = torch.rand(5, 3).cuda()
    y = torch.rand(5, 3).cuda()
    result = x + y
    print(f"✅ GPU计算测试通过! 结果形状: {result.shape}")
else:
    print("⚠️ 未检测到可用的GPU加速")

# 查看cuDNN版本和状态
print(f"cuDNN版本: {torch.backends.cudnn.version()}")
print(f"是否启用cuDNN: {torch.backends.cudnn.enabled}")

print("\n====================================================")
print("=                  镜像依赖信息                      =")
print("====================================================")

# 显示关键依赖的版本信息
try:
    import numpy as np
    print(f"  - NumPy {np.__version__} (保持在1.x版本，适配PyTorch 2.4.0)")
except:
    print("  - ⚠️ NumPy未安装或版本不兼容")

try:
    import cv2
    print(f"  - OpenCV {cv2.__version__} (适配PyTorch 2.4.0)")
except:
    print("  - ⚠️ OpenCV未安装或版本不兼容")

try:
    import mmengine
    import mmcv
    print(f"  - mmengine: {mmengine.__version__}")
except:
    print("  - ⚠️ mmengine未安装或导入失败")
print("\n=== MMCV-FULL 检查 ===")
try:
    import mmcv
    print(f"  - mmcv version: {mmcv.__version__}")
    from mmcv.ops import modulated_deform_conv
    print("  - modulated_deform_conv CUDA ops: ✅ 可用")
except ImportError as e:
    print("  - ⚠️ mmcv 未安装或导入失败:", e)
except Exception as e:
    print("  - ⚠️ mmcv CUDA ops 测试失败:", e)
try:
    import mmagic
    print(f"  - MMagic: mmagic {mmagic.__version__}")
except:
    print("  - ⚠️ MMagic未安装或导入失败")


print("\n====================================================")
print("=                  环境检查完毕                      =")
print("====================================================")