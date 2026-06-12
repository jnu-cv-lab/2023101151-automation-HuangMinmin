import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json
from model import SkeletonTransformer

# ===================== 配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data"
MODEL_PATH = f"{DATA_DIR}/badminton_transformer.pth"
BATCH_SIZE = 16

# 加载标签映射
with open(f"{DATA_DIR}/label_map.json", "r", encoding="utf-8") as f:
    data = json.load(f)
id2label = data["id2label"]
# 严格按标签数字 0~5 顺序生成类别名称列表
CLASS_NAMES = [id2label[str(i)] for i in range(6)]

# 数据集类（同训练）
class BadmintonDataset(Dataset):
    def __init__(self, data, label):
        self.data = torch.from_numpy(data).float()
        self.label = torch.from_numpy(label).long()
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

def test():
    # 加载测试数据
    X_test = np.load(f"{DATA_DIR}/X_test.npy")
    y_test = np.load(f"{DATA_DIR}/y_test.npy")
    test_set = BadmintonDataset(X_test, y_test)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # 加载模型
    model = SkeletonTransformer().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_pred = []
    all_true = []

    with torch.no_grad():
        for batch_data, batch_label in test_loader:
            batch_data = batch_data.to(DEVICE)
            logits = model(batch_data)
            _, pred = torch.max(logits, dim=1)
            all_pred.extend(pred.cpu().numpy())
            all_true.extend(batch_label.numpy())

    # 计算指标
    acc = accuracy_score(all_true, all_pred)
    cm = confusion_matrix(all_true, all_pred)
    report = classification_report(all_true, all_pred, target_names=CLASS_NAMES)

    print("=" * 60)
    print(f"测试集准确率: {acc:.4f}")
    print("=" * 60)
    print("混淆矩阵:")
    print(cm)
    print("=" * 60)
    print("分类报告:")
    print(report)

if __name__ == "__main__":
    test()