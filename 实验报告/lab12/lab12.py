import cv2
import numpy as np
import glob
import os

# ----------------棋盘参数 9×6内角点----------------
chessboard_w = 9
chessboard_h = 6
square_size =20

# 输出文件夹
output_dir = "./experiment12_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 生成棋盘世界三维坐标
objp = np.zeros((chessboard_w * chessboard_h, 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_w, 0:chessboard_h].T.reshape(-1, 2) * square_size

obj_points = []
img_points = []
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 读取图片
img_paths = glob.glob("./testimagess4/*.jpg")
if len(img_paths) == 0:
    raise Exception("testimagess文件夹中没有找到jpg图片，请检查图片后缀和存放位置！")
print(f"一共读取 {len(img_paths)} 张标定图片")
img_size = None

# 遍历检测角点
for idx, path in enumerate(img_paths):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_size is None:
        img_size = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(gray, (chessboard_w, chessboard_h), None)
    if ret:
        corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners_sub)
        draw_img = cv2.drawChessboardCorners(img, (chessboard_w, chessboard_h), corners_sub, ret)
        cv2.imwrite(os.path.join(output_dir, f"corner_detect_{idx}.jpg"), draw_img)
        print(f"图片{idx+1}：角点检测成功")
    else:
        print(f"图片{idx+1}：反光/模糊导致角点检测失败，自动舍弃")

# 标定
rms_error, K, D, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)

# 打印结果
print("=" * 60)
print(f"全局重投影RMS误差：{rms_error:.4f} 像素")
print("=" * 60)
print("相机内参矩阵 K：\n", K)
print("=" * 60)
print("畸变系数 D = [k1, k2, p1, p2, k3]：\n", D.ravel())
print("=" * 60)
print(f"图像分辨率 (宽, 高) = {img_size}")

# 去畸变，仅用OpenCV保存两张图，不使用matplotlib
test_img = cv2.imread(img_paths[0])
h, w = test_img.shape[:2]
new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
map_x, map_y = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), cv2.CV_32FC1)
undist_img = cv2.remap(test_img, map_x, map_y, cv2.INTER_LINEAR)

# 保存图片到输出文件夹
cv2.imwrite(os.path.join(output_dir, "原始测试图.jpg"), test_img)
cv2.imwrite(os.path.join(output_dir, "去畸变校正图.jpg"), undist_img)

print(f"\n所有输出图片已保存至文件夹：{output_dir}")