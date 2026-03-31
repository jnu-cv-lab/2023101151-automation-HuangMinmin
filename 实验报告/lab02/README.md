# 实验二：C++OpenCV 基础图像处理实验
## 实验信息
- 实验：实验二
- 学院：国际能源学院
- 年级专业：23级自动化
- 学号：2023101151
- 姓名：黄敏敏


## 实验内容
基于 C++ 和 OpenCV 实现图像读取、显示、灰度转换、像素读取、图像裁剪与保存等基础图像处理功能。

## 环境配置
- C++ 开发环境
- OpenCV 4.x 及以上
- 支持 C++11 及以上编译器

## 实验详情
### 1. 图像读取
通过命令行参数传入图片路径，读取彩色图像并校验读取结果，避免路径错误导致程序异常。
```cpp
// 读取彩色图像，校验路径与图像有效性
Mat img = imread(argv[1], IMREAD_COLOR);
if (img.empty()) {
    cout << "无法读取图片，请检查路径！" << endl;
    return -1;
}
```
### 2.输出图像宽度、高度、通道数等信息
```cpp
// 输出图像关键信息
cout << "宽度: " << img.cols << " 像素" << endl;
cout << "高度: " << img.rows << " 像素" << endl;
cout << "通道数: " << img.channels() << endl;
```
### 3.显示读取的彩色原图
```cpp
// 显示彩色原图，设置窗口大小
namedWindow("原图", WINDOW_NORMAL);
imshow("原图", img);
resizeWindow("原图", 600, 400);
```
### 4.使用OpenCV内置函数，将BGR彩色图像转换为单通道灰度图像。
```cpp
// 彩色图像转灰度图像
Mat gray_img;
cvtColor(img, gray_img, COLOR_BGR2GRAY);
// 显示灰度图，设置窗口大小
namedWindow("灰度图", WINDOW_NORMAL);
imshow("灰度图", gray_img);
resizeWindow("灰度图", 600, 400);
```
### 5.保存灰度图
```cpp
// 保存灰度图到本地
imwrite("gray_output.jpg", gray_img);
cout << "✅ 灰度图已保存" << endl;
```
### 6.读取指定像素值
```cpp
// 读取(100,100)坐标的BGR像素值并输出
Vec3b pixel = img.at<Vec3b>(100, 100);
cout << "像素(100,100) BGR: " << (int)pixel[0] << " " << (int)pixel[1] << " " << (int)pixel[2] << endl;
```
### 7.裁剪图像左上角区域并保存
```cpp
// 裁剪左上角1/4区域并保存
Rect roi(0, 0, img.cols/4, img.rows/4);
Mat cropped = img(roi);
imwrite("cropped_corner.jpg", cropped);
cout << "✅ 裁剪图已保存" << endl;
```

## 编译与运行
1. 将代码保存为clab01.cpp
2. 链接 OpenCV 库进行编译
3. 运行命令：

## 文件说明
- test.jpg： Opencv读取图片
- clab01.cpp:  源代码
- originalshow.png: 原图显示图片
- grayshow.png:灰度显示图片
- gray_output.jpg：灰度图像保存图片
- cropped_corner.jpg：裁剪后的图像

## 注意事项
- 图片路径必须正确
- 运行时需传入图片参数
- OpenCV 默认图像通道为 BGR
- waitKey(0) 保证窗口正常显示