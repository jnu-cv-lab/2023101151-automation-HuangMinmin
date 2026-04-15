import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

# ====================== 1. 生成测试图像 ======================
img_size = 512
x = np.linspace(-1, 1, img_size)
y = np.linspace(-1, 1, img_size)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
chirp = np.sin(2 * np.pi * (5 * R + 15 * R**2))
chirp = cv2.normalize(chirp, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
img = chirp

# ====================== 2. FFT 95% 能量最高频率 ======================
def block_fft_fmax(img, block_size=32, energy_ratio=0.95):
    H, W = img.shape
    h_blocks = H // block_size
    w_blocks = W // block_size
    fmax_map = np.zeros((h_blocks, w_blocks))

    for i in range(h_blocks):
        for j in range(w_blocks):
            y0 = i * block_size
            x0 = j * block_size
            block = img[y0:y0+block_size, x0:x0+block_size].astype(np.float32)

            fft = np.fft.fft2(block)
            mag = np.abs(fft)
            energy = mag ** 2
            total_E = np.sum(energy)

            fx = np.fft.fftfreq(block_size, 1)
            fy = np.fft.fftfreq(block_size, 1)
            f = np.sqrt(fx[None, :]**2 + fy[:, None]**2)

            f_flat = f.ravel()
            e_flat = energy.ravel()
            idx = np.argsort(f_flat)
            f_sort = f_flat[idx]
            e_sort = e_flat[idx]
            cum = np.cumsum(e_sort)

            mask = cum <= total_E * energy_ratio
            f_valid = f_sort[mask]
            fmax = f_valid[-1] if len(f_valid) > 0 else 0
            fmax_map[i, j] = fmax

    return fmax_map

# ====================== 3. 老师要求：空域梯度法 ======================
def block_gradient_fmax(img, block_size=32):
    H, W = img.shape
    h_blocks = H // block_size
    w_blocks = W // block_size
    grad_map = np.zeros((h_blocks, w_blocks))

    for i in range(h_blocks):
        for j in range(w_blocks):
            y0 = i * block_size
            x0 = j * block_size
            block = img[y0:y0+block_size, x0:x0+block_size].astype(np.float32)

            gx = cv2.Sobel(block, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(block, cv2.CV_32F, 0, 1, ksize=3)

            grad_squared_mean = np.mean(gx**2 + gy**2)
            var = np.var(block)

            if var < 1e-8:
                grad_map[i, j] = 0
                continue

            f_rms_sq = grad_squared_mean / (4 * np.pi**2 * var)
            f_rms = np.sqrt(f_rms_sq)
            grad_map[i, j] = f_rms

    gmin = grad_map.min()
    gmax = grad_map.max()
    if gmax - gmin > 1e-8:
        grad_map = (grad_map - gmin) / (gmax - gmin) * 0.5

    return grad_map

# ====================== 4. 执行计算 ======================
block_size = 32
fmax_fft = block_fft_fmax(img, block_size)
fmax_grad = block_gradient_fmax(img, block_size)
error = np.abs(fmax_fft - fmax_grad)
mean_error = np.mean(error)

# ====================== 5. 绘图 ======================
plt.figure(figsize=(18, 10))

plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(fmax_fft, cmap='jet')
plt.title('f_max by FFT (95% energy)')
plt.colorbar(fraction=0.04)
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(fmax_grad, cmap='jet')
plt.title('f_max by Gradient')
plt.colorbar(fraction=0.04)
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(error, cmap='jet')
plt.title(f'Abs Error | Mean = {mean_error:.3f}')
plt.colorbar(fraction=0.04)
plt.axis('off')

plt.subplot(2, 3, 5)
plt.scatter(fmax_fft.flatten(), fmax_grad.flatten(), s=10, alpha=0.7)
plt.plot([0, 0.5], [0, 0.5], 'r--', linewidth=2)
plt.xlabel('FFT f_max')
plt.ylabel('Gradient f_max')
plt.title('Consistency')

plt.tight_layout()
plt.savefig('fft_vs_gradient.png', dpi=300, bbox_inches='tight')
plt.close()

# ====================== 输出 ======================
print("✅ 运行完成！图片已保存为：fft_vs_gradient.png")
print(f"📊 平均误差：{mean_error:.3f}")