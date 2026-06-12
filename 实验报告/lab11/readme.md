# 实验11 基于人体骨架序列与Transformer Encoder的羽毛球击球动作识别
## 项目概述
本项目为计算机视觉课程实验，实现**基于人体骨架序列与Transformer Encoder的羽毛球击球动作识别**。完整覆盖数据集预处理、时序分类模型训练、模型性能评估、单视频样本推理与可视化全流程，完成6类羽毛球击球动作的识别任务。

## 代码文件说明
```
├── preprocess.py    # 数据预处理脚本
├── model.py          # 模型定义
├── train.py          # 训练脚本
├── test.py           # 测试脚本
└── inference.py      # 推理可视化脚本
              

```

## 运行环境与依赖
### 环境
Python 3.10+，推荐使用虚拟环境隔离项目依赖，可使用CPU或NVIDIA CUDA GPU加速。

### 依赖库
```
torch
opencv-python
mediapipe
numpy
scikit-learn
matplotlib
```
安装方式：
```bash
pip install torch opencv-python mediapipe numpy scikit-learn matplotlib
```

## 执行步骤
1. **数据预处理**

将原始羽毛球视频数据集放置在`archive/`目录下，运行：
```bash
python src/preprocess.py
```
自动提取骨架关键点、将所有视频统一重采样为30帧、归一化处理，按8:2分层划分训练集与测试集，结果保存至`data/`。

2. **模型训练**
```bash
python src/train.py
```
完成模型训练，训练结束后自动保存模型权重，生成训练曲线图片。

3. **模型性能测试评估**
```bash
python src/test.py
```
加载权重文件，在独立测试集上评估模型，输出准确率、混淆矩阵、分类报告。

4. **单样本推理与可视化**
修改`inference.py`内`TEST_VIDEO`为目标视频路径，运行：
```bash
python src/inference.py
```
控制台输出预测动作类别与置信度，同时在`data/`生成可视化视频。

## 实验结果说明
1. 模型在测试集上整体准确率约48.81%，属于细粒度羽毛球动作识别的基线结果；受动作相似度高、数据集规模、模型容量等因素影响，性能存在提升空间。
2. 训练曲线显示模型整体收敛，训练损失持续下降、训练准确率整体上升，后期存在小幅振荡。
3. 单样本推理可正常输出预测结果，可视化视频可直观展示MediaPipe人体姿态提取效果。

## 输出文件夹说明（data/）
运行各脚本后生成的所有输出文件均存放于`data/`目录，各文件作用如下：
```
experiment11_output/
├── badminton_transformer.pth  # 训练完成的模型权重文件
├── label_map.json              # 类别名称与数字标签的双向映射配置
├── X_train.npy / y_train.npy  # 训练集骨架特征、对应标签
├── X_test.npy / y_test.npy    # 测试集骨架特征、对应标签
├── train_curve.png             # 训练过程损失&准确率变化曲线
├── test_output.png             # 测试结果输出截图
├── inference_output.png        # 推理运行日志截图
└── infer_vis.mp4               # 推理可视化结果视频，含人体骨架绘制、预测类别与置信度标注
```

## 注意事项
1. 本实验全流程固定使用**30帧**作为输入序列长度，修改帧数会导致模型权重加载失败。
2. MediaPipe运行时输出的英文警告、日志为底层依赖库信息，不会影响本项目功能与实验结果。
3. 原始视频数据集体积较大，提交时可根据要求选择性上传。