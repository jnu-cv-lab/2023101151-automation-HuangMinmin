import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
# ====================== 全局设置 ======================
output_dir = "experiment4_output"
os.makedirs(output_dir, exist_ok=True)
plt.switch_backend('Agg')  
# ====================== 实验参数 ======================
img_size = 512
downsample_ratio = 4
sigma_opt = 0.45 * downsample_ratio  # 理论最优 σ
sigma_list = [0.5, 1.0, 2.0, 4.0]   # 用于对比的 σ
# ====================== 生成测试图 ======================
# 棋盘格
chessboard = np.zeros((img_size, img_size), dtype=np.uint8)
block_size = 32
for i in range(0, img_size, block_size * 2):
    for j in range(0, img_size, block_size * 2):
        chessboard[i:i+block_size, j:j+block_size] = 255
        chessboard[i+block_size:i+block_size*2, j+block_size:j+block_size*2] = 255
# Chirp 图
x = np.linspace(-1, 1, img_size)
y = np.linspace(-1, 1, img_size)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
chirp = np.sin(2 * np.pi * (5 * R + 20 * R**2))
chirp = cv2.normalize(chirp, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# ====================== FFT 工具函数 ======================
def compute_fft_spectrum(img):
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    return 20 * np.log(np.abs(fft_shift) + 1e-8)
# ====================== 1. 基础对比：原图 / 直接下采样 / 高斯下采样 ======================
# 直接下采样
chess_direct = cv2.resize(chessboard, (img_size//4, img_size//4), interpolation=cv2.INTER_NEAREST)
chirp_direct = cv2.resize(chirp, (img_size//4, img_size//4), interpolation=cv2.INTER_NEAREST)
# 高斯滤波 + 下采样
ksize = int(6 * sigma_opt + 1)
ksize = ksize if ksize % 2 == 1 else ksize + 1
chess_blur = cv2.GaussianBlur(chessboard, (ksize, ksize), sigma_opt)
chess_blur_down = cv2.resize(chess_blur, (img_size//4, img_size//4), interpolation=cv2.INTER_NEAREST)
chirp_blur = cv2.GaussianBlur(chirp, (ksize, ksize), sigma_opt)
chirp_blur_down = cv2.resize(chirp_blur, (img_size//4, img_size//4), interpolation=cv2.INTER_NEAREST)
# 计算 FFT
fft_chess_original = compute_fft_spectrum(chessboard)
fft_chess_direct = compute_fft_spectrum(chess_direct)
fft_chess_gaussian = compute_fft_spectrum(chess_blur_down)

fft_chirp_original = compute_fft_spectrum(chirp)
fft_chirp_direct = compute_fft_spectrum(chirp_direct)
fft_chirp_gaussian = compute_fft_spectrum(chirp_blur_down)
# 画图：时域对比
plt.figure(figsize=(20, 12))
plt.subplot(231), plt.imshow(chessboard, cmap='gray'), plt.title('Chessboard Original'), plt.axis('off')
plt.subplot(232), plt.imshow(chess_direct, cmap='gray'), plt.title('Direct Downsample (Aliasing)'), plt.axis('off')
plt.subplot(233), plt.imshow(chess_blur_down, cmap='gray'), plt.title('Gaussian + Downsample'), plt.axis('off')
plt.subplot(234), plt.imshow(chirp, cmap='gray'), plt.title('Chirp Original'), plt.axis('off')
plt.subplot(235), plt.imshow(chirp_direct, cmap='gray'), plt.title('Direct Downsample (Aliasing)'), plt.axis('off')
plt.subplot(236), plt.imshow(chirp_blur_down, cmap='gray'), plt.title('Gaussian + Downsample'), plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'full_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
# 画图：FFT 频谱
plt.figure(figsize=(18, 8))
plt.subplot(231), plt.imshow(fft_chess_original, cmap='gray'), plt.title('Chessboard Original FFT'), plt.axis('off')
plt.subplot(232), plt.imshow(fft_chess_direct, cmap='gray'), plt.title('Chessboard Direct FFT (Aliased)'), plt.axis('off')
plt.subplot(233), plt.imshow(fft_chess_gaussian, cmap='gray'), plt.title('Chessboard Gaussian FFT (Clean)'), plt.axis('off')
plt.subplot(234), plt.imshow(fft_chirp_original, cmap='gray'), plt.title('Chirp Original FFT'), plt.axis('off')
plt.subplot(235), plt.imshow(fft_chirp_direct, cmap='gray'), plt.title('Chirp Direct FFT (Aliased)'), plt.axis('off')
plt.subplot(236), plt.imshow(fft_chirp_gaussian, cmap='gray'), plt.title('Chirp Gaussian FFT (Clean)'), plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fft_spectrum_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
# ====================== 2. σ 对比实验 ======================
direct_down = cv2.resize(chirp, (img_size//4, img_size//4), interpolation=cv2.INTER_NEAREST)
results = []
for sigma in sigma_list:
    k = int(6 * sigma + 1)
    k = k if k % 2 == 1 else k + 1
    blur_img = cv2.GaussianBlur(chirp, (k, k), sigma)
    down_img = cv2.resize(blur_img, (img_size//4, img_size//4), interpolation=cv2.INTER_NEAREST)
    results.append((sigma, down_img))
# 画图：σ 时域对比
plt.figure(figsize=(16, 8))
plt.subplot(231), plt.imshow(chirp, cmap='gray'), plt.title('Original Chirp'), plt.axis('off')
plt.subplot(232), plt.imshow(direct_down, cmap='gray'), plt.title('Direct Downsample (Aliasing)'), plt.axis('off')
for idx, (sigma, img) in enumerate(results):
    plt.subplot(2, 3, idx+3)
    plt.imshow(img, cmap='gray')
    plt.title(f'σ = {sigma:.1f}')
    plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'sigma_validation_chirp.png'), dpi=300, bbox_inches='tight')
plt.close()
# 画图：σ FFT 对比
fft_direct = compute_fft_spectrum(direct_down)
fft_results = [compute_fft_spectrum(img) for (s, img) in results]
plt.figure(figsize=(16, 8))
plt.subplot(231), plt.imshow(compute_fft_spectrum(chirp), cmap='gray'), plt.title('Original Chirp FFT'), plt.axis('off')
plt.subplot(232), plt.imshow(fft_direct, cmap='gray'), plt.title('Direct FFT (Aliased)'), plt.axis('off')
for idx, fimg in enumerate(fft_results):
    plt.subplot(2, 3, idx+3)
    plt.imshow(fimg, cmap='gray')
    plt.title(f'FFT σ = {sigma_list[idx]:.1f}')
    plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'sigma_fft_chirp.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✅ 基础实验与 σ 对比实验完成")
# ====================== 3. 自适应下采样（梯度驱动局部 M / σ） ======================
M_fixed = 4
sigma_fixed = 0.45 * M_fixed
gray = chirp.astype(np.float32)
h, w = gray.shape
# 梯度计算 → 局部 M 估计
gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
grad_mag = np.sqrt(gx**2 + gy**2)
# 归一化梯度 → 映射为局部下采样倍数 M_local
grad_norm = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)
M_local = 2 + 3 * (1 - grad_norm)  # M ∈ [2,5]
# 局部 σ：σ_local = 0.45 * M_local
sigma_local = 0.45 * M_local
# 自适应高斯滤波（带 padding，修复边缘窗口问题）
max_ksize = int(6 * sigma_local.max() + 1)
max_ksize = max_ksize + 1 if max_ksize % 2 == 0 else max_ksize
pad = max_ksize // 2
gray_pad = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REFLECT)
out_adaptive = np.zeros_like(gray, dtype=np.float32)
for i in range(h):
    for j in range(w):
        s = sigma_local[i, j]
        k = int(6 * s + 1)
        k = k + 1 if k % 2 == 0 else k
        k = min(k, max_ksize)
        r = k // 2
        patch = gray_pad[i+pad - r : i+pad + r + 1, j+pad - r : j+pad + r + 1]
        gk = cv2.getGaussianKernel(k, s)
        gk2D = gk @ gk.T
        filtered = (patch * gk2D).sum()
        out_adaptive[i, j] = filtered
# 自适应下采样
out_adaptive = cv2.resize(out_adaptive, (w//M_fixed, h//M_fixed), interpolation=cv2.INTER_LINEAR)
# 固定 σ 下采样（对比）
k_fixed = int(6 * sigma_fixed + 1)
k_fixed = k_fixed + 1 if k_fixed % 2 == 0 else k_fixed
blur_fixed = cv2.GaussianBlur(gray, (k_fixed, k_fixed), sigma_fixed)
out_fixed = cv2.resize(blur_fixed, (w//M_fixed, h//M_fixed), interpolation=cv2.INTER_LINEAR)
# 原图下采样（参考 Ground Truth）
out_gt = cv2.resize(gray, (w//M_fixed, h//M_fixed), interpolation=cv2.INTER_LINEAR)
# 误差图
error_fixed = np.abs(out_fixed - out_gt)
error_adaptive = np.abs(out_adaptive - out_gt)
# 画图展示
plt.figure(figsize=(20, 12))
plt.subplot(2, 3, 1), plt.imshow(chirp, cmap='gray'), plt.title('Original'), plt.axis('off')
plt.subplot(2, 3, 2), plt.imshow(out_fixed, cmap='gray'), plt.title(f'Fixed σ={sigma_fixed:.1f}'), plt.axis('off')
plt.subplot(2, 3, 3), plt.imshow(out_adaptive, cmap='gray'), plt.title('Adaptive σ (by gradient)'), plt.axis('off')
plt.subplot(2, 3, 4), plt.imshow(M_local, cmap='jet'), plt.title('Local M map'), plt.axis('off')
plt.subplot(2, 3, 5), plt.imshow(error_fixed, cmap='jet'), plt.title('Fixed'), plt.axis('off')
plt.subplot(2, 3, 6), plt.imshow(error_adaptive, cmap='jet'), plt.title('Adaptive'), plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'adaptive_downsample.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✅ 自适应下采样实验完成")
print("🎉 所有实验图已保存到 experiment4_output 文件夹")