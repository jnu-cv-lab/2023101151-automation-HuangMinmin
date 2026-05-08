# ===================== 导入需要的库 =====================
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# 自动创建输出文件夹
os.makedirs("experiment7_output", exist_ok=True)

# 导入 6 个分类模型
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ===================== 任务1：数据准备 =====================
print("="*50)
print("任务1：数据准备")
print("="*50)

digits = datasets.load_digits()
images = digits.images
data = digits.data
labels = digits.target

print(f"图像总数量：{images.shape[0]}")
print(f"每张图像大小：{images.shape[1]} × {images.shape[2]}")
print(f"所有类别标签：{np.unique(labels)}")

# 图片：
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(images[i], cmap="gray")
    plt.title(f"{labels[i]}")
    plt.axis("off")
plt.suptitle("First 10 Sample Images & True Labels")
plt.savefig("experiment7_output/01_sample_images.png", dpi=150, bbox_inches='tight')
plt.close()

# ===================== 任务2：数据划分 =====================
print("\n" + "="*50)
print("任务2：数据划分")
print("="*50)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.25, random_state=42
)

print(f"训练集样本数：{X_train.shape[0]}")
print(f"测试集样本数：{X_test.shape[0]}")
print("\n训练集用途：用于模型学习数据规律，训练模型参数")
print("测试集用途：模拟新数据，评估模型泛化能力，不参与训练")

# ===================== 任务3：特征表示 =====================
print("\n" + "="*50)
print("任务3：特征表示")
print("="*50)

print("1. 8×8图像 → 64维向量：")
print("   把 8行×8列 的像素矩阵，按行依次拼接成 1×64 的一维向量")
print("2. 传统机器学习需要特征转换的原因：")
print("   传统模型（KNN/SVM/逻辑回归等）只能接收向量输入，不能直接处理矩阵")
print("3. 原始像素作为特征的优缺点：")
print("   优点：简单直接、无需复杂特征工程、信息完整保留")
print("   局限：高维稀疏、缺乏全局特征、对旋转/形变敏感")

# ===================== 任务4：模型训练 =====================
print("\n" + "="*50)
print("任务4：模型训练 & 计算准确率")
print("="*50)

models = {
    "KNN": KNeighborsClassifier(),
    "朴素贝叶斯": GaussianNB(),
    "逻辑回归": LogisticRegression(max_iter=10000),
    "SVM": SVC(),
    "决策树": DecisionTreeClassifier(),
    "随机森林": RandomForestClassifier()
}

accuracy_results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy_results[name] = round(acc, 4)
    print(f"{name} 测试准确率： {acc:.4f}")

# ===================== 任务5：结果比较 =====================
print("\n" + "="*50)
print("任务5：模型准确率表格")
print("="*50)

print("模型                  测试准确率")
print("-" * 40)
for name, acc in accuracy_results.items():
    print(f"{name:<22} {acc:.4f}")

max_model = max(accuracy_results, key=accuracy_results.get)
min_model = min(accuracy_results, key=accuracy_results.get)

print(f"\n准确率最高：{max_model} ({accuracy_results[max_model]})")
print(f"准确率最低：{min_model} ({accuracy_results[min_model]})")

# ===================== 任务6：错误样本分析 =====================
print("\n" + "="*50)
print("任务6：错误样本分析（选择SVM）")
print("="*50)

best_model = models["SVM"]
y_pred = best_model.predict(X_test)

# 混淆矩阵：
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("SVM Confusion Matrix")
plt.colorbar()
plt.xticks(range(10), range(10))
plt.yticks(range(10), range(10))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                 color='white' if cm[i, j] > thresh else 'black')
plt.savefig("experiment7_output/02_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()

# 错误样本图：
error_indices = np.where(y_pred != y_test)[0]
plt.figure(figsize=(12, 4))
for i, idx in enumerate(error_indices[:8]):
    plt.subplot(1, 8, i+1)
    img = X_test[idx].reshape(8,8)
    plt.imshow(img, cmap='gray')
    plt.title(f"T:{y_test[idx]}\nP:{y_pred[idx]}")  # 英文简写
    plt.axis('off')
plt.suptitle("Misclassified Samples (True → Predicted)")
plt.savefig("experiment7_output/03_error_samples.png", dpi=150, bbox_inches='tight')
plt.close()

print("\n所有图片已保存到 experiment7_output 文件夹！")