import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# 强制使用 TkAgg 后端
plt.switch_backend('TkAgg')

# 1. 读取图片
img = cv2.imread("test.jpg")
if img is None:
    print("错误：未找到 test.jpg 图片！")
    exit()
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
points = []

# 2. Matplotlib 弹窗选点 + 实时画标记
fig, ax = plt.subplots(figsize=(10, 12))
ax.imshow(img_rgb)
#按顺序点击：左上->右上->右下->左下
ax.set_title("Follow LU->RU->RD->LD", fontsize=14, color='red')
ax.axis("on")

def on_click(event):
    if event.xdata and event.ydata:
        x, y = int(event.xdata), int(event.ydata)
        points.append((x, y))
        
        # ===================== 【选点标记：画绿色圆点】 =====================
        ax.plot(x, y, 'o', markersize=10, color='lime', markeredgecolor='black', markeredgewidth=2)
        
        # ===================== 【画连线】 =====================
        if len(points) >= 2:
            prev_x, prev_y = points[-2]
            ax.plot([prev_x, x], [prev_y, y], linewidth=2, color='blue')

        print(f"已选点 {len(points)}: ({x}, {y})")
        
        # 选满4个点自动关闭
        if len(points) == 4:
            plt.close()
    
    fig.canvas.draw()

# 绑定点击
fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()

if len(points) != 4:
    print("错误：请选够4个点！")
    exit()

# 3. 手动构建透视变换矩阵
src = np.array(points, dtype=np.float32)
w_out, h_out = 800, 1120  # 标准A4比例，保证四个角完整显示
dst = np.array([
    [0, 0],
    [w_out-1, 0],
    [w_out-1, h_out-1],
    [0, h_out-1]
], dtype=np.float32)

# 构建方程组求解矩阵参数
A = []
for i in range(4):
    x1, y1 = src[i]
    x2, y2 = dst[i]
    A.append([x1, y1, 1, 0, 0, 0, -x1*x2, -y1*x2])
    A.append([0, 0, 0, x1, y1, 1, -x1*y2, -y1*y2])

A = np.array(A, dtype=np.float64)
B = dst.reshape(8, 1)
M_param = np.linalg.solve(A, B)
M = np.append(M_param, 1).reshape(3, 3)

# 4. 使用 cv2 进行透视校正
corrected = cv2.warpPerspective(img, M, (w_out, h_out))
cv2.imwrite("corrected_result.jpg", corrected)

print("\n✅ 校正完成！已保存为 corrected_result.jpg")