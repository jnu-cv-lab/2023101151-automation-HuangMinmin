import cv2
import numpy as np
import os

# ===================== 1. 配置参数 =====================
# 测试图片路径（和之前一致，放在src目录或修改路径）
IMAGE_PATH = "test.jpg"
# 下采样比例（4:2:0 是最常用的标准，也可改为4:2:2等）
DOWNSAMPLE_RATIO = 2
# 插值方法：cv2.INTER_LINEAR（双线性）、cv2.INTER_CUBIC（双三次）、cv2.INTER_NEAREST（最近邻）
INTERPOLATION_METHOD = cv2.INTER_LINEAR

# ===================== 2. PSNR 计算函数 =====================
def calculate_psnr(original, reconstructed):
    """
    计算原始图像与重建图像的 PSNR (峰值信噪比)
    PSNR 越高，图像质量越好
    """
    # 确保两张图尺寸、数据类型一致
    assert original.shape == reconstructed.shape, "图像尺寸不一致"
    mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')  # 完全无失真
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# ===================== 3. 核心处理流程 =====================
def ycbcr_downsample_reconstruct(image_path, downsample_ratio=2, interpolation=cv2.INTER_LINEAR):
    # ---------------- 步骤1：读取彩色图像 ----------------
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"❌ 错误：无法读取图片 {image_path}")
        return None, None
    
    # 转换为 YCbCr 色彩空间（OpenCV 用 YCrCb 命名，通道顺序为 Y, Cr, Cb）
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(img_ycrcb)  # 拆分 Y、Cr、Cb 三个通道
    
    h, w = Y.shape  # 获取原始尺寸
    print(f"✅ 原始图像尺寸：{h}×{w}")

    # ---------------- 步骤2：对 Cb、Cr 通道进行下采样 ----------------
    # 下采样尺寸：宽高均除以 downsample_ratio
    new_h, new_w = h // downsample_ratio, w // downsample_ratio
    Cb_downsampled = cv2.resize(Cb, (new_w, new_h), interpolation=cv2.INTER_AREA)  # 下采样用INTER_AREA最优
    Cr_downsampled = cv2.resize(Cr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    print(f"✅ Cb/Cr 下采样完成，尺寸：{new_h}×{new_w}")

    # ---------------- 步骤3：用插值方法恢复到原尺寸 ----------------
    Cb_upsampled = cv2.resize(Cb_downsampled, (w, h), interpolation=interpolation)
    Cr_upsampled = cv2.resize(Cr_downsampled, (w, h), interpolation=interpolation)
    print(f"✅ Cb/Cr 插值恢复完成，尺寸：{h}×{w}")

    # ---------------- 步骤4：与原 Y 通道重建图像 ----------------
    # 合并 Y（原始）、Cr（恢复后）、Cb（恢复后）
    img_ycrcb_recon = cv2.merge((Y, Cr_upsampled, Cb_upsampled))
    # 转回 RGB 显示（OpenCV 转回BGR，再转RGB用于可视化）
    img_bgr_recon = cv2.cvtColor(img_ycrcb_recon, cv2.COLOR_YCrCb2BGR)
    img_rgb_original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb_recon = cv2.cvtColor(img_bgr_recon, cv2.COLOR_BGR2RGB)

    # ---------------- 步骤5：计算 PSNR，分析质量影响 ----------------
    psnr_value = calculate_psnr(img_bgr, img_bgr_recon)
    print(f"\n📊 重建图像 PSNR: {psnr_value:.2f} dB")
    if psnr_value > 30:
        print("✅ 图像质量优秀（PSNR > 30dB，人眼几乎无感知差异）")
    elif 20 < psnr_value <= 30:
        print("⚠️ 图像质量良好（PSNR 20-30dB，有轻微可感知失真）")
    else:
        print("❌ 图像质量较差（PSNR < 20dB，失真明显）")

    # ---------------- 步骤6：保存结果 ----------------
    output_dir = "ycbcr_experiment"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存原图、重建图、各通道
    cv2.imwrite(os.path.join(output_dir, "original.jpg"), img_bgr)
    cv2.imwrite(os.path.join(output_dir, "reconstructed.jpg"), img_bgr_recon)
    cv2.imwrite(os.path.join(output_dir, "Y_channel.jpg"), Y)
    cv2.imwrite(os.path.join(output_dir, "Cb_channel.jpg"), Cb)
    cv2.imwrite(os.path.join(output_dir, "Cr_channel.jpg"), Cr)
    cv2.imwrite(os.path.join(output_dir, "Cb_downsampled.jpg"), Cb_downsampled)
    cv2.imwrite(os.path.join(output_dir, "Cr_downsampled.jpg"), Cr_downsampled)
    cv2.imwrite(os.path.join(output_dir, "Cb_upsampled.jpg"), Cb_upsampled)
    cv2.imwrite(os.path.join(output_dir, "Cr_upsampled.jpg"), Cr_upsampled)

    print(f"\n🎉 所有结果已保存到 {output_dir} 文件夹")
    return img_rgb_original, img_rgb_recon, psnr_value

# ===================== 4. 主函数执行 =====================
if __name__ == "__main__":
    print(f"📌 OpenCV 版本：{cv2.__version__}")
    print("="*60)
    print(f"⚙️ 实验配置：下采样比例 {DOWNSAMPLE_RATIO}:1，插值方法：{INTERPOLATION_METHOD}")
    print("="*60)
    
    # 执行实验
    original, recon, psnr = ycbcr_downsample_reconstruct(
        IMAGE_PATH, 
        DOWNSAMPLE_RATIO, 
        INTERPOLATION_METHOD
    )
    
    # （可选）可视化对比（如果需要弹窗显示，取消注释）
    # if original is not None and recon is not None:
    #     combined = np.hstack((original, recon))
    #     cv2.imshow("Original vs Reconstructed", combined)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()