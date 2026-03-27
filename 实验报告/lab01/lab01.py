import cv2
import numpy as np
import os

# ===================== 1. 路径配置 =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(SCRIPT_DIR, "test.jpg")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "experiment1_output")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ===================== 任务1：读取图片 =====================
print("="*60)
print("📌 任务1：读取测试图片")
img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    print(f"❌ 错误：无法读取图片 {IMAGE_PATH}")
    exit()
print("✅ 图片读取成功！")

# ===================== 任务2：输出图像基本信息 =====================
print("\n" + "="*60)
print("📌 任务2：输出图像基本信息")
h, w, c = img_bgr.shape
dtype = img_bgr.dtype
print(f"📏 图像尺寸：宽度 {w} × 高度 {h}")
print(f"🎨 图像通道数：{c} 通道（BGR彩色）")
print(f"🔢 像素数据类型：{dtype}")
print(f"📊 总像素数：{h * w}")

# ===================== 任务3：保存原图（替代弹窗显示，满足实验要求） =====================
print("\n" + "="*60)
print("📌 任务3：保存原图")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
cv2.imwrite(os.path.join(OUTPUT_DIR, "original_show.png"), img_rgb)
print("✅ 原图已保存到输出目录，可直接打开查看")

# ===================== 任务4：转换灰度图并保存 =====================
print("\n" + "="*60)
print("📌 任务4：转换为灰度图并保存")
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(OUTPUT_DIR, "grayscale.jpg"), img_gray)
cv2.imwrite(os.path.join(OUTPUT_DIR, "grayscale_show.png"), img_gray)
print("✅ 灰度图转换完成并保存")

# ===================== 任务5：保存灰度图（已在任务4完成） =====================
print("\n" + "="*60)
print("📌 任务5：灰度图保存完成")

# ===================== 任务6：NumPy 操作 =====================
print("\n" + "="*60)
print("📌 任务6：NumPy 图像操作")
center_h, center_w = h // 2, w // 2
center_pixel = img_bgr[center_h, center_w]
print(f"🔍 中心像素BGR值：{center_pixel} | RGB值：{center_pixel[::-1]}")

crop_h, crop_w = h // 4, w // 4
img_crop = img_bgr[:crop_h, :crop_w]
crop_path = os.path.join(OUTPUT_DIR, "top_left_crop.jpg")
cv2.imwrite(crop_path, img_crop)
print(f"✅ 左上角裁剪完成，保存路径：{crop_path}")

# ===================== 最终总结 =====================
print("\n" + "="*60)
print("🎉 实验一所有任务执行完成！")
print(f"📂 所有结果已保存到：{OUTPUT_DIR}")
print("="*60)