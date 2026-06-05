import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ===================== 1. 实现 Sinusoidal Position Encoding =====================
class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # 位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算分母项
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        # 偶数维度用 sin，奇数维度用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为缓冲区（不参与训练）
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        返回: x + positional encoding
        """
        seq_len = x.size(1)
        # 直接叠加位置编码（E+pos 核心方式）
        return x + self.pe[:seq_len, :]

# ===================== 2. 实现二维向量旋转（RoPE 基础） =====================
def rotate_2d(x: torch.Tensor, theta: float):
    """
    二维向量旋转公式：
    [x1'] = [cosθ, -sinθ] [x1]
    [x2']   [sinθ,  cosθ] [x2]
    x: [batch, ..., 2]  二维向量
    theta: 旋转角度
    """
    x1, x2 = x[..., 0], x[..., 1]
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    # 旋转计算
    x1_rot = x1 * cos_t - x2 * sin_t
    x2_rot = x1 * sin_t + x2 * cos_t
    return torch.stack([x1_rot, x2_rot], dim=-1)

# ===================== 3. 实现高维 RoPE (已修复维度错误) =====================
class RoPE(nn.Module):
    def __init__(self, d_model: int, base: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.base = base
        # 预计算旋转角度 θ_i = base^(-2i/d_model)
        self.inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))

    def forward(self, x: torch.Tensor, seq_len: int = None):
        """
        x: [batch, head, seq_len, d_k]  注意力的 Q/K 矩阵
        返回: 旋转后的 Q/K
        """
        if seq_len is None:
            seq_len = x.size(2)
        
        # 位置序列 0,1,2,...,seq_len-1
        position = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        # 计算每个位置的角度：pos * θ_i
        freqs = torch.outer(position, self.inv_freq) # [seq_len, d//2]

        # 关键修复：把 cos/sin 每个维度重复一次，匹配向量分组
        cos = torch.repeat_interleave(freqs.cos(), 2, dim=-1)  # [seq_len, d]
        sin = torch.repeat_interleave(freqs.sin(), 2, dim=-1)
        
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1,1,seq_len,d]
        sin = sin.unsqueeze(0).unsqueeze(0)

        # RoPE 旋转公式
        x1, x2 = x[..., 0::2], x[..., 1::2]  # 两两分组
        
        # 维度完全匹配
        x_rot = torch.cat([
            x1 * cos[..., 0::2] - x2 * sin[..., 0::2],
            x1 * sin[..., 0::2] + x2 * cos[..., 0::2]
        ], dim=-1)
        
        return x_rot

# ===================== 4. 对比 E+pos 和 RoPE 的输入方式 =====================
def compare_input_style():
    d_model = 16
    batch_size = 2
    seq_len = 4
    
    # 随机输入向量
    x = torch.randn(batch_size, seq_len, d_model)
    
    # === Sinusoidal PE (E+pos)：直接叠加 ===
    spe = SinusoidalPE(d_model)
    out_spe = spe(x)
    print("="*50)
    print("Sinusoidal PE (E+pos) 输入方式：x + pos_encoding")
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out_spe.shape}")
    
    # === RoPE：先拆分 Q/K，再旋转 ===
    # 模拟注意力 Q/K: [batch, head, seq_len, d_k]
    q = torch.randn(batch_size, 2, seq_len, d_model//2)
    k = torch.randn(batch_size, 2, seq_len, d_model//2)
    rope = RoPE(d_model//2)
    q_rot = rope(q)
    k_rot = rope(k)
    print("\nRoPE 输入方式：对 Q/K 进行旋转，不直接叠加")
    print(f"Q 形状: {q.shape} -> 旋转后 Q 形状: {q_rot.shape}")
    print("="*50)

# ===================== 5. 验证 RoPE 相对位置性质（核心实验） =====================
def verify_rope_relative_position():
    """
    实验：RoPE 满足 f(q,m)·f(k,n) = f(q,m-n)·f(k,0)
    即：点积只和相对位置有关，与绝对位置无关
    """
    d_k = 16
    rope = RoPE(d_k)
    
    # 随机 query 和 key 向量
    q = torch.randn(1, 1, 1, d_k)
    k = torch.randn(1, 1, 1, d_k)
    
    # 设定位置：m=5, n=2 → 相对位置 3
    m, n = 5, 2
    rel_pos = m - n
    
    # 构造位置序列
    q_pos = torch.zeros(1,1,6,d_k)  # 位置 0~5
    k_pos = torch.zeros(1,1,3,d_k)  # 位置 0~2
    q_pos[...,m,:] = q
    k_pos[...,n,:] = k
    
    # RoPE 旋转
    q_rot_m = rope(q_pos)[...,m,:]
    k_rot_n = rope(k_pos)[...,n,:]
    dot1 = (q_rot_m * k_rot_n).sum()  # 绝对位置点积
    
    # 相对位置验证：m-n 和 0
    q_rel = torch.zeros(1,1,4,d_k)
    k_rel = torch.zeros(1,1,1,d_k)
    q_rel[...,rel_pos,:] = q
    k_rel[...,0,:] = k
    q_rot_rel = rope(q_rel)[...,rel_pos,:]
    k_rot_0 = rope(k_rel)[...,0,:]
    dot2 = (q_rot_rel * k_rot_0).sum()  # 相对位置点积
    
    print("\nRoPE 相对位置性质验证：")
    print(f"绝对位置点积 f(q,{m})·f(k,{n}) = {dot1.item():.6f}")
    print(f"相对位置点积 f(q,{rel_pos})·f(k,0) = {dot2.item():.6f}")
    print(f"差值: {abs(dot1.item() - dot2.item()):.2e} → 几乎相等！")
    print("="*50)

# ===================== 运行所有实验 =====================
if __name__ == "__main__":
    compare_input_style()
    verify_rope_relative_position()