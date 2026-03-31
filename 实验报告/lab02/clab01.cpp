#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    // -------------------------- 任务1: 读取测试图片 --------------------------
    if (argc != 2) {
        cout << "用法: " << argv[0] << " <图片路径>" << endl;
        return -1;
    }
    Mat img = imread(argv[1], IMREAD_COLOR);
    if (img.empty()) {
        cout << "无法读取图片，请检查路径！" << endl;
        return -1;
    }

    // -------------------------- 任务2: 输出图像基本信息 --------------------------
    cout << "=== 图像基本信息 ===" << endl;
    cout << "宽度 (Width): " << img.cols << " 像素" << endl;
    cout << "高度 (Height): " << img.rows << " 像素" << endl;
    cout << "通道数 (Channels): " << img.channels() << endl;
    cout << "数据类型 (Type): CV_8UC3" << endl;
    cout << "=====================" << endl;

    // -------------------------- 弹窗 显示原图 --------------------------
    namedWindow("原图", WINDOW_NORMAL);
    imshow("原图", img);  // 弹窗！
    resizeWindow("原图", 600, 400);    // 宽600，高400

    // -------------------------- 转换灰度图 --------------------------
    Mat gray_img;
    cvtColor(img, gray_img, COLOR_BGR2GRAY);

    // -------------------------- 弹窗 显示灰度图 --------------------------
    namedWindow("灰度图", WINDOW_NORMAL);
    imshow("灰度图", gray_img);  // 弹窗！
    resizeWindow("灰度图", 600, 400);  // 宽600，高400
    // -------------------------- 保存文件 --------------------------
    imwrite("gray_output.jpg", gray_img);
    cout << "✅ 灰度图已保存" << endl;

    // 输出像素
    Vec3b pixel = img.at<Vec3b>(100, 100);
    cout << "像素(100,100) BGR: " << (int)pixel[0] << " " << (int)pixel[1] << " " << (int)pixel[2] << endl;

    // 裁剪
    Rect roi(0, 0, img.cols/4, img.rows/4);
    Mat cropped = img(roi);
    imwrite("cropped_corner.jpg", cropped);
    cout << "✅ 裁剪图已保存" << endl;

    // -------------------------- 等待窗口关闭 --------------------------
    waitKey(0);  // 
    destroyAllWindows();

    cout << "🎉 全部完成！" << endl;
    return 0;
}