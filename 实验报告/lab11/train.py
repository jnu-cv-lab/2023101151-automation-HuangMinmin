import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import SkeletonTransformer
import matplotlib.pyplot as plt

# ===================== 全局配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data"
BATCH_SIZE = 16
EPOCHS = 40
LR = 1e-3

# ===================== 自定义数据集 =====================
class BadmintonDataset(Dataset):
    def __init__(self, data, label):
        self.data = torch.from_numpy(data).float()
        self.label = torch.from_numpy(label).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

# ===================== 加载数据 =====================
def load_data():
    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    X_test = np.load(f"{DATA_DIR}/X_test.npy")
    y_test = np.load(f"{DATA_DIR}/y_test.npy")

    train_set = BadmintonDataset(X_train, y_train)
    test_set = BadmintonDataset(X_test, y_test)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader

# ===================== 训练主函数 =====================
def train():
    train_loader, test_loader = load_data()
    model = SkeletonTransformer().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_loss_list = []
    train_acc_list = []

    print(f"使用设备: {DEVICE}")
    print("开始训练......")

    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_data, batch_label in train_loader:
            batch_data = batch_data.to(DEVICE)
            batch_label = batch_label.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_data)
            loss = criterion(logits, batch_label)

            # 反向传播 + 参数更新
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, pred = torch.max(logits, dim=1)
            correct += (pred == batch_label).sum().item()
            total += batch_label.size(0)

        # 统计指标
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        train_loss_list.append(epoch_loss)
        train_acc_list.append(epoch_acc)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "./data/badminton_transformer.pth")
    print("模型已保存为 badminton_transformer.pth")

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1,2,1)
    plt.plot(train_loss_list, label="Train Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(train_acc_list, label="Train Acc")
    plt.title("Training Accuracy")
    plt.legend()
    plt.savefig("./data/train_curve.png")
    plt.show()

if __name__ == "__main__":
    train()