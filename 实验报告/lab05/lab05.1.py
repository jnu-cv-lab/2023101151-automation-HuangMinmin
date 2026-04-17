import cv2
import numpy as np
import os

# 解决WSL显示问题
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# ===================== 1. 读取图片 =====================
img = cv2.imread("image.png")
if img is None:
    print("错误：请确保 test.png 在代码同一目录下！")
    exit()

h, w = img.shape[:2]
print(f"图片尺寸: 宽={w}, 高={h}")

# ===================== 2. 相似变换：自己构造矩阵 =====================

theta = np.radians(20)  # 20度
s = 0.7               # 缩放
tx, ty = 40,10      # 平移

similar_mat = np.array([
    [s * np.cos(theta), -s * np.sin(theta), tx],
    [s * np.sin(theta),  s * np.cos(theta), ty]
], dtype=np.float32)

img_similar = cv2.warpAffine(img, similar_mat, (w, h))

# ===================== 3. 仿射变换：自己直接给矩阵 =====================
# 仿射变换 2x3 矩阵
affine_mat = np.array([
    [0.8,  0.3,  30],
    [0.3,  0.8,  40]
], dtype=np.float32)

img_affine = cv2.warpAffine(img, affine_mat, (w, h))

# ===================== 4. 透视变换：自己直接给 3x3 矩阵 =====================
# 透视变换是 3x3 矩阵，手动构造
persp_mat = np.array([
    [1,    0.1,   20],
    [0.05, 1,     30],
    [0.0005, 0.0008, 1]
], dtype=np.float32)

img_perspective = cv2.warpPerspective(img, persp_mat, (w, h))

# ===================== 标注 =====================
def add_label(image, label):
    return cv2.putText(image.copy(), label, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

img1 = add_label(img, "Original")
img2 = add_label(img_similar, "Similarity")
img3 = add_label(img_affine, "Affine")
img4 = add_label(img_perspective, "Perspective")

top = np.hstack([img1, img2])
bot = np.hstack([img3, img4])
final = np.vstack([top, bot])

cv2.imwrite("all_results.png", final)
print("✅ 全部自己构造变换矩阵完成！结果已保存为 all_results.png")