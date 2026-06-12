import torch
import torch.nn as nn

class SkeletonTransformer(nn.Module):
    def __init__(
        self,
        input_dim=132,        # 每帧特征维度
        target_frames=30,     # 序列长度
        d_model=128,          # Transformer维度
        nhead=4,              # 多头注意力数
        num_layers=2,         # Encoder层数
        dim_feedforward=256,  # FFN维度
        num_classes=6,        # 分类类别数
        dropout=0.1
    ):
        super().__init__()
        self.target_frames = target_frames
        # 1. 特征映射: 132 -> 128
        self.linear_emb = nn.Linear(input_dim, d_model)
        # 2. 可学习位置编码
        self.pos_emb = nn.Parameter(torch.randn(1, target_frames, d_model))
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # [B, T, D] 格式
            activation="relu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 4. 分类头 (全局平均池化 + MLP)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: [Batch, 30, 132]
        B, T, _ = x.shape
        # 特征映射
        x = self.linear_emb(x)  # [B, 30, 128]
        # 叠加位置编码
        x = x + self.pos_emb[:, :T, :]
        # Transformer编码
        x = self.transformer_encoder(x)  # [B, 30, 128]
        # 全局平均池化
        x = torch.mean(x, dim=1)  # [B, 128]
        # 分类输出
        x = self.dropout(x)
        logits = self.classifier(x)  # [B, 6]
        return logits

# 测试模型维度
if __name__ == "__main__":
    model = SkeletonTransformer()
    test_input = torch.randn(2, 30, 132)  # batch=2, 30帧, 132维
    out = model(test_input)
    print(f"输入shape: {test_input.shape} | 输出logits shape: {out.shape}")