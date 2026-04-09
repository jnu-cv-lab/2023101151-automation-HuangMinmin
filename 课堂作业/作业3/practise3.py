import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===================== 1. Read Image =====================
img = cv2.imread("lena.png")
if len(img.shape) == 3:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    img_gray = img

row = img_gray.shape[0] // 2
x = img_gray[row, :].astype(np.float32)
N = len(x)

# ===================== 2. Two Extension Sequences =====================
x_periodic = np.concatenate([x, x])                # DFT Periodic Extension
x_even_sym = np.concatenate([x, x[::-1]])          # DCT Even Symmetric Extension

# ===================== Print Sequences =====================
print("="*60)
print("原始序列（前20个点）:\n", np.round(x[:20]))
print("\nDFT 周期延拓序列（前40个点）:\n", np.round(x_periodic[:40]))
print("\nDCT 偶对称延拓序列（前40个点）:\n", np.round(x_even_sym[:40]))
print("="*60)

# ===================== 3. Compute DFT / DCT =====================
dft = np.fft.fft(x)
dft_amp = np.abs(dft)
dct = cv2.dct(x.reshape(-1, 1))
dct = dct.flatten()
dct_amp = np.abs(dct)

# ===================== 4. Energy Concentration =====================
def energy_ratio(spec, ratio):
    total = np.sum(spec ** 2)
    k = int(len(spec) * ratio)
    topk = np.sum(spec[:k] ** 2)
    return (topk / total) * 100

er_dft10 = energy_ratio(dft_amp, 0.1)
er_dct10 = energy_ratio(dct_amp, 0.1)

# ===================== Plot Extended Sequences =====================
plt.figure(figsize=(12, 6))
plt.subplot(2,1,1)
plt.plot(x_periodic, 'r', linewidth=1, label="DFT Periodically Extended Sequence")
plt.axvline(N, color='k', linestyle='--', label="Original Boundary")
plt.title("DFT Periodically Extended Sequence", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(2,1,2)
plt.plot(x_even_sym, 'b', linewidth=1, label="DCT Even-Symmetrically Extended Sequence")
plt.axvline(N, color='k', linestyle='--', label="Original Boundary")
plt.title("DCT Even-Symmetrically Extended Sequence", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("extension_compare.png", dpi=150)
plt.close()

# ===================== Plot Spectra =====================
plt.figure(figsize=(12,6))
plt.subplot(211)
plt.plot(dft_amp, 'r')
plt.title("DFT Amplitude Spectrum")
plt.ylabel("Amplitude")

plt.subplot(212)
plt.plot(dct_amp, 'b')
plt.title("DCT Amplitude Spectrum")
plt.xlabel("Frequency Index")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.savefig("spectrum_compare.png")
plt.close()

# ===================== Output Results =====================
print("\n前10%系数能量占比:")
print(f"DFT: {er_dft10:.2f}%")
print(f"DCT: {er_dct10:.2f}%")