# 实验一：图片读取与处理
## 1. 实验目标
完成图片读取、信息输出、显示原图、灰度转换、保存灰度图、NumPy 简单操作等任务，严格区分“显示灰度图”和“保存灰度图”的核心逻辑。

## 2. 环境说明
- 运行环境：WSL2 (Ubuntu 22.04)
- Python 版本：3.10+
- 依赖库：numpy、matplotlib、opencv-python

## 3. 安装依赖
```bash
# 激活虚拟环境（如果有）
source .venv/bin/activate

# 安装所需 Python 库
pip install numpy matplotlib opencv-python

# WSL2 系统依赖（如需弹窗显示，可选）
sudo apt update
sudo apt install -y python3-tk libxcb-xinerama0 libqt5gui5
```
## 4. 核心功能说明
任务 1：图片读取
实现逻辑：使用 cv2.imread() 读取指定路径的测试图片，包含文件存在性校验
核心目标：确保图片正常加载

任务 2：图像信息输出
实现逻辑：提取图像的宽度、高度、通道数、数据类型、总像素数并打印
核心目标：展示图像基础属性

任务 3：显示原图
实现逻辑：转换 OpenCV BGR 格式为 Matplotlib RGB 格式，绘图美化后保存（含 show 标识）
核心目标：直观展示原始彩色图像

任务 4：灰度转换 + 显示灰度图
实现逻辑：用 cv2.cvtColor() 转灰度，Matplotlib 绘图美化后保存（含 show 标识）
核心目标：完成灰度转换并展示结果

任务 5：保存灰度图
实现逻辑：直接用 cv2.imwrite() 保存灰度图（含 saved 标识），无显示相关逻辑
核心目标：独立留存灰度图文件

任务 6：NumPy 图像操作
实现逻辑：提取中心像素值、裁剪左上角 1/4 区域并保存
核心目标：验证 NumPy 处理图像的能力

## 5. 文件说明
### 5.1 输入文件

lab01.py：主程序文件，包含所有实验任务的完整实现逻辑，可直接运行

test.jpg：测试图片文件（需放在代码同级目录），作为实验的输入数据源

### 5.2 输出文件（存储路径：experiment1_output/）

task3：original_show.png：任务 3，PNG，原图的显示文件，Matplotlib 绘图美化，用于查看原始图像

task4：grayscale.png：任务 4，PNG，灰度图的显示文件，Matplotlib 绘图美化，用于查看灰度转换结果

task5：grayscale_show.png：任务 5，JPG，灰度图的保存文件，OpenCV 原生保存，无美化，用于后续复用

task6_top_left_crop.jpg：任务 6，JPG，原图左上角 1/4 区域的裁剪文件，验证 NumPy 切片操作的效果
## 6. 代码运行
### 6.1 bash运行进入代码目录
cd cv-course/lab01/src

### 6.2 运行主程序
python lab01.py
## 7.实验程序及结果在3月22日提交至另一仓库其链接如下
### https://github.com/jnu-cv-lab/2023101151-automation-HuangMinmin-lab01
