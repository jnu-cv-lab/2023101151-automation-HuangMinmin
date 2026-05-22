import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 保存路径
save_dir = "experiment9_output"
os.makedirs(save_dir, exist_ok=True)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# 数据预处理与加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

batch_size = 64
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ======================
# 任务1：复用原有基础CNN模型
# ======================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 通用训练评估函数
def train_eval(model, opt_name, lr, epochs=10):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    # 三种优化器
    if opt_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif opt_name == "SGD_Momentum":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, pred = torch.max(out, 1)
            total += label.size(0)
            correct += (pred == label).sum().item()
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total

        # 验证集
        model.eval()
        val_loss, val_corr, val_tot = 0, 0, 0
        with torch.no_grad():
            for data, label in val_loader:
                data, label = data.to(device), label.to(device)
                out = model(data)
                val_loss += criterion(out, label).item()
                _, pred = torch.max(out, 1)
                val_tot += label.size(0)
                val_corr += (pred == label).sum().item()
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_corr / val_tot

        train_loss_list.append(avg_train_loss)
        val_loss_list.append(avg_val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        print(f"Epoch{epoch+1} | TrainLoss:{avg_train_loss:.4f} ValLoss:{avg_val_loss:.4f} | TrainAcc:{train_acc:.2f}% ValAcc:{val_acc:.2f}%")

    # 测试集准确率
    test_corr, test_tot = 0, 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            out = model(data)
            _, pred = torch.max(out, 1)
            test_tot += label.size(0)
            test_corr += (pred == label).sum().item()
    test_acc = 100 * test_corr / test_tot

    return train_loss_list, val_loss_list, train_acc_list, val_acc_list, test_acc, model

print("===== 任务1：复用模型重新训练 =====")
base_model = CNN()
tl, vl, ta, va, test_acc, trained_model = train_eval(base_model, "Adam", 0.001, epochs=10)
print(f"✅ 任务1 测试集准确率: {test_acc:.2f}%")
# ======================
# 任务2：三种优化器对比
# ======================
print("\n===== 任务2：优化器对比实验 =====")
opt_list = ["SGD", "SGD_Momentum", "Adam"]
opt_result = {}
for opt in opt_list:
    print(f"\n当前优化器：{opt}")
    temp_model = CNN()
    res = train_eval(temp_model, opt, 0.001, epochs=10)
    opt_result[opt] = res
    print(f"✅ {opt} 测试集准确率: {res[4]:.2f}%") 
# 绘制优化器对比曲线
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
for name, res in opt_result.items():
    plt.plot(range(1,11), res[0], label=f"{name}_TrainLoss")
    plt.plot(range(1,11), res[1], label=f"{name}_ValLoss")
plt.title("Optimizer Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.subplot(1,2,2)
for name, res in opt_result.items():
    plt.plot(range(1,11), res[2], label=f"{name}_TrainAcc")
    plt.plot(range(1,11), res[3], label=f"{name}_ValAcc")
plt.title("Optimizer Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy(%)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "optimizer_compare.png"), dpi=150)
plt.close()

# ======================
# 任务3：Adam不同学习率对比
# ======================
print("\n===== 任务3：学习率对比实验 =====")
lr_list = [0.1, 0.01, 0.001]
lr_result = {}
for lr in lr_list:
    print(f"\n当前学习率：{lr}")
    temp_model = CNN()
    res = train_eval(temp_model, "Adam", lr, epochs=10)
    lr_result[lr] = res

# 绘制学习率曲线
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
for lr, res in lr_result.items():
    plt.plot(range(1,11), res[0], label=f"lr={lr} TrainLoss")
    plt.plot(range(1,11), res[1], label=f"lr={lr} ValLoss")
plt.title("Learning Rate Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.subplot(1,2,2)
for lr, res in lr_result.items():
    plt.plot(range(1,11), res[2], label=f"lr={lr} TrainAcc")
    plt.plot(range(1,11), res[3], label=f"lr={lr} ValAcc")
plt.title("Learning Rate Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy(%)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "lr_compare.png"), dpi=150)
plt.close()

# ======================
# 任务4：第一层卷积核可视化
# ======================
print("\n===== 任务4：卷积核可视化 =====")
conv1_weight = trained_model.conv1.weight.cpu().detach().numpy()
plt.figure(figsize=(10,4))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(conv1_weight[i,0,:,:], cmap="gray")
    plt.title(f"Kernel {i+1}")
    plt.axis("off")
plt.suptitle("First Layer Convolution Kernels")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "conv_kernel.png"), dpi=150)
plt.close()

# ======================
# 任务5：特征图可视化
# ======================
print("\n===== 任务5：特征图可视化 =====")
sample_img, _ = test_dataset[0]
sample_input = sample_img.unsqueeze(0).to(device)
# 提取第一层卷积输出
feat_map = trained_model.relu(trained_model.conv1(sample_input))
feat_map = feat_map.cpu().detach().numpy()[0]

plt.figure(figsize=(10,4))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(feat_map[i], cmap="gray")
    plt.title(f"FeatureMap {i+1}")
    plt.axis("off")
plt.suptitle("First Layer Feature Maps")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "feature_map.png"), dpi=150)
plt.close()

# ======================
# 任务6：错误分类样本展示
# ======================
print("\n===== 任务6：错误样本分析 =====")
error_imgs, error_true, error_pred = [], [], []
trained_model.eval()
with torch.no_grad():
    for data, label in test_loader:
        data, label = data.to(device), label.to(device)
        out = trained_model(data)
        _, pred = torch.max(out,1)
        for idx in range(len(label)):
            if pred[idx] != label[idx]:
                error_imgs.append(data[idx].cpu().squeeze())
                error_true.append(label[idx].item())
                error_pred.append(pred[idx].item())
            if len(error_imgs) >=8:
                break
        if len(error_imgs)>=8:
            break

plt.figure(figsize=(10,4))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(error_imgs[i], cmap="gray")
    plt.title(f"True:{error_true[i]}\nPred:{error_pred[i]}")
    plt.axis("off")
plt.suptitle("Misclassified Samples")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "error_samples.png"), dpi=150)
plt.close()

# ======================
# 任务7：混淆矩阵绘制
# ======================
print("\n===== 任务7：混淆矩阵 =====")
all_pred, all_label = [], []
with torch.no_grad():
    for data, label in test_loader:
        data = data.to(device)
        out = trained_model(data)
        _, pred = torch.max(out,1)
        all_pred.extend(pred.cpu().numpy())
        all_label.extend(label.numpy())

cm = confusion_matrix(all_label, all_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predict Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
plt.close()


print("\n✅ 全部7项实验任务完成，结果图片已保存至 experiment9_output 文件夹")