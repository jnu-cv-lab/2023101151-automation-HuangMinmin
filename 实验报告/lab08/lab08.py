import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt

save_dir = "experiment8_output"
os.makedirs(save_dir, exist_ok=True)

# ==========================
# 任务1：环境准备
# ==========================
print("===== 任务1：环境测试 =====")
print("PyTorch 版本:", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("是否支持GPU:", torch.cuda.is_available())
print("使用设备:", device)
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print("张量运算 a + b =", a + b)
print("环境测试成功 ✅")

# ==========================
# 任务2：加载 MNIST 数据集
# ==========================
print("\n===== 任务2：加载数据集 =====")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
classes = [str(i) for i in range(10)]

plt.figure(figsize=(10, 4))
for i in range(8):
    img, label = train_dataset[i]
    plt.subplot(2, 4, i+1)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f'True:{classes[label]}')
    plt.axis('off')
plt.suptitle('Training Samples')
plt.savefig(os.path.join(save_dir, "training_samples.png"), dpi=150, bbox_inches='tight')
plt.close()
print("训练集样本已保存")

# ==========================
# 任务3：基础 CNN 模型
# ==========================
print("\n===== 任务3：基础CNN模型 =====")
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

model = CNN().to(device)
print(model)

# ==========================
# 任务4+5：训练 + 验证
# ==========================
print("\n===== 任务4+5：训练与验证 =====")
epochs = 10
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []

for epoch in range(epochs):
    model.train()
    train_loss = train_correct = train_total = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        train_total += target.size(0)
        train_correct += (pred == target).sum().item()

    avg_train_loss = train_loss / len(train_loader)
    train_acc = 100 * train_correct / train_total

    model.eval()
    val_loss = val_correct = val_total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            val_loss += criterion(outputs, target).item()
            _, pred = torch.max(outputs, 1)
            val_total += target.size(0)
            val_correct += (pred == target).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total

    train_loss_history.append(avg_train_loss)
    train_acc_history.append(train_acc)
    val_loss_history.append(avg_val_loss)
    val_acc_history.append(val_acc)

    print(f"Epoch {epoch+1} | 训练损失:{avg_train_loss:.4f} 训练准确率:{train_acc:.2f}% | 验证损失:{avg_val_loss:.4f} 验证准确率:{val_acc:.2f}%")

# ==========================
# 任务6：测试模型
# ==========================
print("\n===== 任务6：测试模型 =====")
model.eval()
test_loss = test_correct = test_total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        test_loss += criterion(outputs, target).item()
        _, pred = torch.max(outputs, 1)
        test_total += target.size(0)
        test_correct += (pred == target).sum().item()

base_test_acc = 100 * test_correct / test_total
base_test_loss = test_loss / len(test_loader)
print(f"基础模型测试准确率: {base_test_acc:.2f}%")

plt.figure(figsize=(12, 6))
data, target = next(iter(test_loader))
data, target = data.to(device), target.to(device)
_, pred = torch.max(model(data), 1)
for i in range(8):
    img = data[i].cpu().squeeze()
    plt.subplot(2,4,i+1)
    plt.imshow(img, cmap='gray')
    plt.title(f'True:{classes[target[i]]}\nPred:{classes[pred[i]]}')
    plt.axis('off')
plt.suptitle('Test Predictions')
plt.savefig(os.path.join(save_dir, "test_predictions.png"), dpi=150, bbox_inches='tight')
plt.close()

# ==========================
# 任务7：曲线
# ==========================
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(range(1,11), train_loss_history, label='Train Loss')
plt.plot(range(1,11), val_loss_history, label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.subplot(1,2,2)
plt.plot(range(1,11), train_acc_history, label='Train Acc')
plt.plot(range(1,11), val_acc_history, label='Val Acc')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
plt.close()

# ==========================
# 进阶任务 1：改进模型
# ==========================
print("\n===== 进阶任务1：改进网络结构 =====")

class CNN_Improved(nn.Module):
    def __init__(self):
        super(CNN_Improved, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*3*3, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1,64*3*3)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model_improved = CNN_Improved().to(device)
print("改进模型结构:")
print(model_improved)

optimizer_imp = optim.Adam(model_improved.parameters(), lr=0.001)

for epoch in range(10):
    model_improved.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer_imp.zero_grad()
        loss = criterion(model_improved(data), target)
        loss.backward()
        optimizer_imp.step()

model_improved.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        _, pred = torch.max(model_improved(data), 1)
        total += target.size(0)
        correct += (pred == target).sum().item()
imp_acc = 100 * correct / total

print(f"改进模型测试准确率: {imp_acc:.2f}%")
print(f"基础模型: {base_test_acc:.2f}% → 改进模型: {imp_acc:.2f}%")

# ==========================
# 进阶任务2：优化器对比
# ==========================
print("\n===== 进阶任务2：优化器比较 =====")

def run_optimizer(optim_name, lr):
    m = CNN().to(device)
    opt = optim.SGD(m.parameters(), lr=lr, momentum=0.9) if optim_name=="SGD" else optim.Adam(m.parameters(), lr=lr)
    for e in range(10):
        m.train()
        for d,t in train_loader:
            d,t = d.to(device),t.to(device)
            opt.zero_grad()
            criterion(m(d),t).backward()
            opt.step()
    m.eval()
    c=0
    tot=0
    with torch.no_grad():
        for d,t in test_loader:
            d,t = d.to(device),t.to(device)
            c+=(torch.max(m(d),1)[1]==t).sum().item()
            tot+=t.size(0)
    return 100*c/tot

sgd_acc = run_optimizer("SGD", 0.01)
adam_acc = run_optimizer("Adam", 0.001)

print("Optimizer\tLearning Rate\tTest Accuracy")
print(f"SGD\t\t0.01\t\t{sgd_acc:.2f}%")
print(f"Adam\t\t0.001\t\t{adam_acc:.2f}%")

# ==========================
# 进阶任务3：MNIST vs CIFAR-10
# ==========================
print("\n===== 进阶任务3：CIFAR-10 测试 =====")

transform_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
testset_cifar = datasets.CIFAR10('./data', train=False, download=True, transform=transform_cifar)
testloader_cifar = DataLoader(testset_cifar, batch_size=100, shuffle=False)

class CIFAR_CNN(nn.Module):
    def __init__(self):
        super(CIFAR_CNN,self).__init__()
        self.conv1 = nn.Conv2d(3,32,3,padding=1)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.pool=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(64*8*8,256)
        self.fc2=nn.Linear(256,10)
        self.relu=nn.ReLU()
    def forward(self,x):
        x=self.pool(self.relu(self.conv1(x)))
        x=self.pool(self.relu(self.conv2(x)))
        x=x.view(-1,64*8*8)
        x=self.relu(self.fc1(x))
        x=self.fc2(x)
        return x

cifar_model = CIFAR_CNN().to(device)
opt_cifar = optim.Adam(cifar_model.parameters(), lr=0.001)

trainset_cifar = datasets.CIFAR10('./data',train=True,download=True,transform=transform_cifar)
# 为了提速，只取训练集的1/10数据
from torch.utils.data import Subset
subset_indices = list(range(0, len(trainset_cifar), 10))
trainset_cifar_small = Subset(trainset_cifar, subset_indices)
trainloader_cifar = DataLoader(trainset_cifar_small, batch_size=64, shuffle=True)

# 为了提速，只训练3个epoch
for epoch in range(3):
    cifar_model.train()
    for d,t in trainloader_cifar:
        d,t = d.to(device),t.to(device)
        opt_cifar.zero_grad()
        criterion(cifar_model(d),t).backward()
        opt_cifar.step()

cifar_model.eval()
cc=0
ctot=0
with torch.no_grad():
    for d,t in testloader_cifar:
        d,t = d.to(device),t.to(device)
        cc+=(torch.max(cifar_model(d),1)[1]==t).sum().item()
        ctot+=t.size(0)
cifar_acc = 100*cc/ctot
print("CIFAR-10 测试准确率:", cifar_acc)

print("\n✅ 所有进阶任务完成！")
print("所有结果保存在 experiment8_output/")
