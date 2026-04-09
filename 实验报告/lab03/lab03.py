import cv2
import numpy as np
import os

#创建文件夹
BASE_DIR = "output"
ORIG_DIR = os.path.join(BASE_DIR, "original")
DOWN_DIR = os.path.join(BASE_DIR, "downsampled")
REST_DIR = os.path.join(BASE_DIR, "restored")
SPEC_DIR = os.path.join(BASE_DIR, "spectrum")
DCT_DIR = os.path.join(BASE_DIR, "dct")  # DCT 专用文件夹

os.makedirs(ORIG_DIR, exist_ok=True)
os.makedirs(DOWN_DIR, exist_ok=True)
os.makedirs(REST_DIR, exist_ok=True)
os.makedirs(SPEC_DIR, exist_ok=True)
os.makedirs(DCT_DIR, exist_ok=True)

# 统一保存函数
def save_image(path, img):
    cv2.imwrite(path, img)

#1.读取图片
img = cv2.imread("test2.png")
if img is None:
    print("❌ 错误：找不到图片 test2.png")
    exit()

#转为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
original_h, original_w = gray.shape
save_image(os.path.join(ORIG_DIR, "original_gray.png"), gray)

# 2. 下采样
# 1/2 采样
down_half_direct = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
blur_half = cv2.GaussianBlur(gray, (3, 3), 0)
down_half_gaussian = cv2.resize(blur_half, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
# 1/4 采样
down_quarter_direct = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
blur_quarter = cv2.GaussianBlur(gray, (5, 5), 0)
down_quarter_gaussian = cv2.resize(blur_quarter, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
# 保存缩小图
save_image(os.path.join(DOWN_DIR, "down_half_direct.png"), down_half_direct)
save_image(os.path.join(DOWN_DIR, "down_half_gaussian.png"), down_half_gaussian)
save_image(os.path.join(DOWN_DIR, "down_quarter_direct.png"), down_quarter_direct)
save_image(os.path.join(DOWN_DIR, "down_quarter_gaussian.png"), down_quarter_gaussian)

# 3. 图像恢复
# ======================
# 最近邻
restore_half_nearest = cv2.resize(down_half_direct, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
restore_half_gaussian_nearest = cv2.resize(down_half_gaussian, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
restore_quarter_nearest = cv2.resize(down_quarter_direct, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
restore_quarter_gaussian_nearest = cv2.resize(down_quarter_gaussian, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

# 双线性
restore_half_linear = cv2.resize(down_half_direct, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
restore_half_gaussian_linear = cv2.resize(down_half_gaussian, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
restore_quarter_linear = cv2.resize(down_quarter_direct, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
restore_quarter_gaussian_linear = cv2.resize(down_quarter_gaussian, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

# 双三次
restore_half_cubic = cv2.resize(down_half_direct, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
restore_half_gaussian_cubic = cv2.resize(down_half_gaussian, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
restore_quarter_cubic = cv2.resize(down_quarter_direct, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
restore_quarter_gaussian_cubic = cv2.resize(down_quarter_gaussian, (original_w, original_h), interpolation=cv2.INTER_CUBIC)

# 保存恢复图
restored_files = [
    ("restore_half_nearest.png", restore_half_nearest),
    ("restore_half_gaussian_nearest.png", restore_half_gaussian_nearest),
    ("restore_quarter_nearest.png", restore_quarter_nearest),
    ("restore_quarter_gaussian_nearest.png", restore_quarter_gaussian_nearest),
    ("restore_half_linear.png", restore_half_linear),
    ("restore_half_gaussian_linear.png", restore_half_gaussian_linear),
    ("restore_quarter_linear.png", restore_quarter_linear),
    ("restore_quarter_gaussian_linear.png", restore_quarter_gaussian_linear),
    ("restore_half_cubic.png", restore_half_cubic),
    ("restore_half_gaussian_cubic.png", restore_half_gaussian_cubic),
    ("restore_quarter_cubic.png", restore_quarter_cubic),
    ("restore_quarter_gaussian_cubic.png", restore_quarter_gaussian_cubic),
]

for name, img_data in restored_files:
    save_image(os.path.join(REST_DIR, name), img_data)

# ======================
# 4. 计算 MSE & PSNR
# ======================
def calc_mse_psnr(original, restored):
    if original.shape != restored.shape:
        restored = cv2.resize(restored, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR)
    mse = np.mean((original - restored) ** 2)
    if mse == 0:
        return 0, 100
    max_pix = 255.0
    psnr = 20 * np.log10(max_pix / np.sqrt(mse))
    return round(mse, 3), round(psnr, 3)

mse1, psnr1 = calc_mse_psnr(gray, restore_half_nearest)
mse2, psnr2 = calc_mse_psnr(gray, restore_half_linear)
mse3, psnr3 = calc_mse_psnr(gray, restore_half_cubic)

mse4, psnr4 = calc_mse_psnr(gray, restore_half_gaussian_nearest)
mse5, psnr5 = calc_mse_psnr(gray, restore_half_gaussian_linear)
mse6, psnr6 = calc_mse_psnr(gray, restore_half_gaussian_cubic)

mse7, psnr7 = calc_mse_psnr(gray, restore_quarter_nearest)
mse8, psnr8 = calc_mse_psnr(gray, restore_quarter_linear)
mse9, psnr9 = calc_mse_psnr(gray, restore_quarter_cubic)

mse10, psnr10 = calc_mse_psnr(gray, restore_quarter_gaussian_nearest)
mse11, psnr11 = calc_mse_psnr(gray, restore_quarter_gaussian_linear)
mse12, psnr12 = calc_mse_psnr(gray, restore_quarter_gaussian_cubic)

# ======================
# ✅ 傅里叶频谱（全部居中 + 对数）
# ======================
def fourier_spectrum(img):
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    magnitude = 20 * np.log(np.abs(fft_shift) + 1e-8)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return magnitude

# 频谱计算
spec_original = fourier_spectrum(gray)
spec_half_direct = fourier_spectrum(down_half_direct)
spec_half_gaussian = fourier_spectrum(down_half_gaussian)
spec_quarter_direct = fourier_spectrum(down_quarter_direct)
spec_quarter_gaussian = fourier_spectrum(down_quarter_gaussian)

spec_restore_half_linear = fourier_spectrum(restore_half_linear)
spec_restore_half_gaussian_linear = fourier_spectrum(restore_half_gaussian_linear)
spec_restore_quarter_linear = fourier_spectrum(restore_quarter_linear)
spec_restore_quarter_gaussian_linear = fourier_spectrum(restore_quarter_gaussian_linear)

# 保存频谱
save_image(os.path.join(SPEC_DIR, "spectrum_original.png"), spec_original)
save_image(os.path.join(SPEC_DIR, "spectrum_down_half_direct.png"), spec_half_direct)
save_image(os.path.join(SPEC_DIR, "spectrum_down_half_gaussian.png"), spec_half_gaussian)
save_image(os.path.join(SPEC_DIR, "spectrum_down_quarter_direct.png"), spec_quarter_direct)
save_image(os.path.join(SPEC_DIR, "spectrum_down_quarter_gaussian.png"), spec_quarter_gaussian)
save_image(os.path.join(SPEC_DIR, "spectrum_restore_half_linear.png"), spec_restore_half_linear)
save_image(os.path.join(SPEC_DIR, "spectrum_restore_half_gaussian_linear.png"), spec_restore_half_gaussian_linear)
save_image(os.path.join(SPEC_DIR, "spectrum_restore_quarter_linear.png"), spec_restore_quarter_linear)
save_image(os.path.join(SPEC_DIR, "spectrum_restore_quarter_gaussian_linear.png"), spec_restore_quarter_gaussian_linear)

# ======================
# ✅ DCT 分析（已补全：最近邻 + 双线性 + 双三次）
# ======================
def dct_process(img):
    img_float = img.astype(np.float32)
    dct = cv2.dct(img_float)
    
    dct_abs = np.abs(dct)
    dct_log = 20 * np.log(dct_abs + 1e-8)
    dct_show = cv2.normalize(dct_log, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    h, w = dct.shape
    low_h, low_w = h // 4, w // 4  # 左上角 1/16 低频区域
    total_energy = np.sum(dct_abs ** 2)
    low_energy = np.sum(dct_abs[:low_h, :low_w] ** 2)
    ratio = low_energy / total_energy if total_energy != 0 else 0
    return dct_show, round(ratio * 100, 2)

# 原图 DCT
dct_original, ratio_original = dct_process(gray)
save_image(os.path.join(DCT_DIR, "dct_original.png"), dct_original)

# ========== 最近邻插值 DCT（你要的部分已补全）==========
dct_half_nearest, ratio_half_nearest = dct_process(restore_half_nearest)
dct_half_gaussian_nearest, ratio_half_gaussian_nearest = dct_process(restore_half_gaussian_nearest)
dct_quarter_nearest, ratio_quarter_nearest = dct_process(restore_quarter_nearest)
dct_quarter_gaussian_nearest, ratio_quarter_gaussian_nearest = dct_process(restore_quarter_gaussian_nearest)

# 双线性 DCT
dct_half_linear, ratio_half_linear = dct_process(restore_half_linear)
dct_half_gaussian_linear, ratio_half_gaussian_linear = dct_process(restore_half_gaussian_linear)
dct_quarter_linear, ratio_quarter_linear = dct_process(restore_quarter_linear)
dct_quarter_gaussian_linear, ratio_quarter_gaussian_linear = dct_process(restore_quarter_gaussian_linear)

# 双三次 DCT
dct_half_cubic, ratio_half_cubic = dct_process(restore_half_cubic)
dct_half_gaussian_cubic, ratio_half_gaussian_cubic = dct_process(restore_half_gaussian_cubic)
dct_quarter_cubic, ratio_quarter_cubic = dct_process(restore_quarter_cubic)
dct_quarter_gaussian_cubic, ratio_quarter_gaussian_cubic = dct_process(restore_quarter_gaussian_cubic)

# 保存所有 DCT 图片（包含最近邻）
save_image(os.path.join(DCT_DIR, "dct_restore_half_nearest.png"), dct_half_nearest)
save_image(os.path.join(DCT_DIR, "dct_restore_half_gaussian_nearest.png"), dct_half_gaussian_nearest)
save_image(os.path.join(DCT_DIR, "dct_restore_quarter_nearest.png"), dct_quarter_nearest)
save_image(os.path.join(DCT_DIR, "dct_restore_quarter_gaussian_nearest.png"), dct_quarter_gaussian_nearest)

save_image(os.path.join(DCT_DIR, "dct_restore_half_linear.png"), dct_half_linear)
save_image(os.path.join(DCT_DIR, "dct_restore_half_gaussian_linear.png"), dct_half_gaussian_linear)
save_image(os.path.join(DCT_DIR, "dct_restore_quarter_linear.png"), dct_quarter_linear)
save_image(os.path.join(DCT_DIR, "dct_restore_quarter_gaussian_linear.png"), dct_quarter_gaussian_linear)

save_image(os.path.join(DCT_DIR, "dct_restore_half_cubic.png"), dct_half_cubic)
save_image(os.path.join(DCT_DIR, "dct_restore_half_gaussian_cubic.png"), dct_half_gaussian_cubic)
save_image(os.path.join(DCT_DIR, "dct_restore_quarter_cubic.png"), dct_quarter_cubic)
save_image(os.path.join(DCT_DIR, "dct_restore_quarter_gaussian_cubic.png"), dct_quarter_gaussian_cubic)

# ======================
# 输出 MSE & PSNR
# ======================
print("\n====== 空间域对比指标（1/2 直接缩小恢复）======")
print(f"最近邻插值   MSE = {mse1}  |  PSNR = {psnr1} dB")
print(f"双线性插值   MSE = {mse2}  |  PSNR = {psnr2} dB")
print(f"双三次插值   MSE = {mse3}  |  PSNR = {psnr3} dB")

print("\n====== 空间域对比指标（1/2 高斯缩小恢复）======")
print(f"最近邻插值   MSE = {mse4}  |  PSNR = {psnr4} dB")
print(f"双线性插值   MSE = {mse5}  |  PSNR = {psnr5} dB")
print(f"双三次插值   MSE = {mse6}  |  PSNR = {psnr6} dB")

print("\n====== 空间域对比指标（1/4 直接缩小恢复）======")
print(f"最近邻插值   MSE = {mse7}  |  PSNR = {psnr7} dB")
print(f"双线性插值   MSE = {mse8}  |  PSNR = {psnr8} dB")
print(f"双三次插值   MSE = {mse9}  |  PSNR = {psnr9} dB")

print("\n====== 空间域对比指标（1/4 高斯缩小恢复）======")
print(f"最近邻插值   MSE = {mse10}  |  PSNR = {psnr10} dB")
print(f"双线性插值   MSE = {mse11}  |  PSNR = {psnr11} dB")
print(f"双三次插值   MSE = {mse12}  |  PSNR = {psnr12} dB")

# ======================
# ✅ 完整 DCT 低频能量占比（最近邻 + 双线性 + 双三次）
# ======================
print("\n==================================================")
print("📊 DCT 低频能量占比（左上角 1/16 区域）")
print("==================================================")
print(f"原图                    : {ratio_original} %")
print()
print(f"1/2 直接 + 最近邻      : {ratio_half_nearest} %")
print(f"1/2 直接 + 双线性      : {ratio_half_linear} %")
print(f"1/2 直接 + 双三次      : {ratio_half_cubic} %")
print()
print(f"1/2 高斯 + 最近邻      : {ratio_half_gaussian_nearest} %")
print(f"1/2 高斯 + 双线性      : {ratio_half_gaussian_linear} %")
print(f"1/2 高斯 + 双三次      : {ratio_half_gaussian_cubic} %")
print()
print(f"1/4 直接 + 最近邻      : {ratio_quarter_nearest} %")
print(f"1/4 直接 + 双线性      : {ratio_quarter_linear} %")
print(f"1/4 直接 + 双三次      : {ratio_quarter_cubic} %")
print()
print(f"1/4 高斯 + 最近邻      : {ratio_quarter_gaussian_nearest} %")
print(f"1/4 高斯 + 双线性      : {ratio_quarter_gaussian_linear} %")
print(f"1/4 高斯 + 双三次      : {ratio_quarter_gaussian_cubic} %")

print("\n✅ 实验完成！文件已分类保存")
print(f"📂 原图：{ORIG_DIR}")
print(f"📂 缩小图：{DOWN_DIR}")
print(f"📂 恢复图：{REST_DIR}")
print(f"📂 频谱图：{SPEC_DIR}")
print(f"📂 DCT 图：{DCT_DIR}")