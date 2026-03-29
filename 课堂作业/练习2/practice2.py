import cv2
import numpy as np
import matplotlib
# 强制无界面绘图，彻底解决 Qt 报错！
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# -------------------------- 1. 自定义实现模块 --------------------------
def my_histogram_equalization(img_gray):
    h, w = img_gray.shape
    hist = np.bincount(img_gray.flatten(), minlength=256)
    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0].min()
    equalized = ((cdf[img_gray] - cdf_min) * 255 / (h * w - cdf_min)).astype(np.uint8)
    return equalized

def my_mean_filter(img_gray, ksize=3):
    kernel = np.ones((ksize, ksize), np.float32) / (ksize*ksize)
    return cv2.filter2D(img_gray, -1, kernel)

# -------------------------- 2. 增强算法（已修复参数名） --------------------------
def clahe_enhancement(img_gray, clip_limit=2.0, grid_size=(8,8)):
    # 修复：参数名完全对应，gridSize 改为 grid_size
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img_gray)

def gaussian_filter(img_gray, ksize=3, sigma=0):
    return cv2.GaussianBlur(img_gray, (ksize, ksize), sigma)

def median_filter(img_gray, ksize=3):
    return cv2.medianBlur(img_gray, ksize)

def laplacian_sharpen(img_gray, ksize=3):
    lap = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=ksize)
    return cv2.convertScaleAbs(img_gray - lap)

# -------------------------- 3. 组合方法 --------------------------
def filter_then_equalize(img_gray):
    img = my_mean_filter(img_gray)
    return my_histogram_equalization(img)

def equalize_then_filter(img_gray):
    img = my_histogram_equalization(img_gray)
    return my_mean_filter(img)

# -------------------------- 4. 指标计算 --------------------------
def calculate_metrics(original, processed):
    psnr_val = psnr(original, processed)
    ssim_val = ssim(original, processed, data_range=255)
    return {"PSNR(dB)": round(psnr_val, 2), "SSIM": round(ssim_val, 4)}

# -------------------------- 5. 绘图保存（绝对不报错） --------------------------
def plot_results(original, processed_list, titles, save_path, img_name):
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(5 * (len(processed_list)+1), 10))
    
    # 原图
    plt.subplot(2, len(processed_list)+1, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Original")
    plt.axis("off")
    
    plt.subplot(2, len(processed_list)+1, len(processed_list)+2)
    plt.hist(original.ravel(), 256, [0,256])
    plt.title("Hist")

    # 处理后图
    for i, (img, title) in enumerate(zip(processed_list, titles)):
        plt.subplot(2, len(processed_list)+1, i+2)
        plt.imshow(img, cmap="gray")
        plt.title(title)
        plt.axis("off")
        
        plt.subplot(2, len(processed_list)+1, i+2 + len(processed_list)+1)
        plt.hist(img.ravel(), 256, [0,256])

    plt.tight_layout()
    save_file = os.path.join(save_path, f"{img_name}_{titles[0]}.png")
    plt.savefig(save_file, dpi=200)
    plt.close()
    print(f"✅ 图片已保存：{save_file}")

# -------------------------- 6. 主程序（路径 100% 适配你的文件夹） --------------------------
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "test_images")
    output_dir = os.path.join(os.path.dirname(current_dir), "experiment2_output")

    img_paths = [
        os.path.join(img_dir, "low_contrast_face.jpg"),
        os.path.join(img_dir, "noisy_document.jpg"),
        os.path.join(img_dir, "dark_landscape.jpg")
    ]

    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️  跳过：{img_path}")
            continue

        print(f"✅ 正在处理：{os.path.basename(img_path)}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_name = os.path.basename(img_path).split(".")[0]

        # 均衡类（只保留自定义全局均衡）
        eq1 = my_histogram_equalization(gray)
        cl1 = clahe_enhancement(gray, clip_limit=2.0)
        cl2 = clahe_enhancement(gray, clip_limit=4.0)

        # 滤波类
        m1 = my_mean_filter(gray)
        m2 = my_mean_filter(gray, 5)
        g1 = gaussian_filter(gray)
        g2 = gaussian_filter(gray, 5)
        md1 = median_filter(gray)
        md2 = median_filter(gray, 5)

        # 锐化 + 组合
        sharp = laplacian_sharpen(gray)
        fe = filter_then_equalize(gray)
        ef = equalize_then_filter(gray)

        # ===================== 全算法指标输出（替换你原来的） =====================
        print("\n======", img_name, "全指标结果======")
        print("自定义全局均衡 :", calculate_metrics(gray, eq1))
        print("CLAHE 2.0       :", calculate_metrics(gray, cl1))
        print("CLAHE 4.0       :", calculate_metrics(gray, cl2))
        print("均值滤波 3×3    :", calculate_metrics(gray, m1))
        print("均值滤波 5×5    :", calculate_metrics(gray, m2))
        print("高斯滤波 3×3    :", calculate_metrics(gray, g1))
        print("高斯滤波 5×5    :", calculate_metrics(gray, g2))
        print("中值滤波 3×3    :", calculate_metrics(gray, md1))
        print("中值滤波 5×5    :", calculate_metrics(gray, md2))
        print("拉普拉斯锐化    :", calculate_metrics(gray, sharp))
        print("滤波→均衡       :", calculate_metrics(gray, fe))
        print("均衡→滤波       :", calculate_metrics(gray, ef))

        # 保存图片
        plot_results(gray, [eq1, cl1, cl2], ["MyHist", "CLAHE2", "CLAHE4"], output_dir, img_name)
        plot_results(gray, [m1, m2, g1, g2, md1, md2], ["Mean3","Mean5","Gauss3","Gauss5","Med3","Med5"], output_dir, img_name)
        plot_results(gray, [sharp, fe, ef], ["Sharpen", "Filter->Eq", "Eq->Filter"], output_dir, img_name)

    print("\n🎉 全部运行完成！图片已保存到 experiment2_output")