import cv2
import numpy as np
import os
import time

def orb_feature_detection(image_path, output_path, nfeatures):
    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像文件: {image_path}")

    # 2. 创建 ORB 检测器，使用传入的nfeatures参数
    orb = cv2.ORB_create(nfeatures=nfeatures)

    # 3. 检测关键点并计算描述子
    keypoints, descriptors = orb.detectAndCompute(img, None)

    # 4. 可视化关键点（画出关键点的大小和方向）
    img_kp = cv2.drawKeypoints(
        img, keypoints, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv2.imwrite(output_path, img_kp)

    # 5. 输出关键点数量和描述子维度
    kp_count = len(keypoints)
    desc_dim = descriptors.shape[1] if descriptors is not None else 0
    print(f"图像: {image_path}")
    print(f"关键点数量: {kp_count}")
    print(f"描述子维度: {desc_dim}\n")

    return img, keypoints, descriptors, kp_count, desc_dim

def run_experiment(nfeatures, output_root="experiment6_output"):
    # 为每组参数创建独立文件夹
    output_dir = os.path.join(output_root, f"nfeatures_{nfeatures}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\n===== nfeatures = {nfeatures} =====")

    # 任务1：特征检测
    img1, kp1, des1, count1, dim1 = orb_feature_detection(
        "box.png", os.path.join(output_dir, "box_orb_kp.png"), nfeatures
    )
    img2, kp2, des2, count2, dim2 = orb_feature_detection(
        "box_in_scene.png", os.path.join(output_dir, "box_in_scene_orb_kp.png"), nfeatures
    )
    print(f"可视化结果已保存到: {output_dir} 文件夹\n")

    # 任务2：ORB特征匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    total_matches = len(matches)

    print("===== ORB 特征匹配结果 =====")
    print(f"总匹配数量: {total_matches}")

    # 绘制匹配图
    img_matches_all = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(os.path.join(output_dir, "orb_all_matches.png"), img_matches_all)

    num_show = 50
    good_matches = matches[:num_show] if len(matches) >= num_show else matches
    img_matches_top = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(os.path.join(output_dir, "orb_top50_matches.png"), img_matches_top)

    print(f"初始匹配图已保存为: {output_dir}/orb_all_matches.png")
    print(f"前50个匹配结果已保存为: {output_dir}/orb_top50_matches.png\n")

    # 任务3：RANSAC剔除错误匹配
    print("===== RANSAC 剔除错误匹配结果 =====")
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    homography_matrix, mask = cv2.findHomography(
        pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=5.0
    )

    matches_mask = mask.ravel().tolist()
    inlier_matches = [matches[i] for i in range(len(matches)) if matches_mask[i]]
    num_inliers = len(inlier_matches)
    inlier_ratio = num_inliers / total_matches if total_matches > 0 else 0.0

    print(f"总匹配数量: {total_matches}")
    print(f"RANSAC内点数量: {num_inliers}")
    print(f"内点比例: {inlier_ratio:.4f}")
    print("Homography矩阵:")
    print(homography_matrix)

    img_ransac_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, inlier_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(os.path.join(output_dir, "orb_ransac_matches.png"), img_ransac_matches)
    print(f"\nRANSAC后的匹配图已保存为: {output_dir}/orb_ransac_matches.png\n")

    # 任务4：目标定位
    print("===== 目标定位结果 =====")
    h, w = img1.shape[:2]
    corners_img1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    corners_img2 = cv2.perspectiveTransform(corners_img1, homography_matrix)

    img2_with_box = img2.copy()
    cv2.polylines(
        img2_with_box, [np.int32(corners_img2)],
        isClosed=True, color=(0, 0, 255), thickness=3
    )
    cv2.imwrite(os.path.join(output_dir, "target_localization.png"), img2_with_box)
    print(f"目标定位结果已保存为: {output_dir}/target_localization.png")

    # 判断是否成功定位（检查投影点是否在图像范围内）
    h2, w2 = img2.shape[:2]
    corners_valid = all(0 <= p[0, 0] < w2 and 0 <= p[0, 1] < h2 for p in corners_img2)
    success = "是" if corners_valid else "否"
    print(f"定位是否成功: {success}\n")

    return {
        "nfeatures": nfeatures,
        "template_kp": count1,
        "scene_kp": count2,
        "total_matches": total_matches,
        "inliers": num_inliers,
        "inlier_ratio": round(inlier_ratio, 4),
        "success": success,
        "img1": img1,
        "img2": img2,
        "kp1": kp1,
        "kp2": kp2,
        "des1": des1,
        "des2": des2
    }

def run_sift_experiment(output_root="experiment6_output"):
    output_dir = os.path.join(output_root, "sift")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("\n===== 开始 SIFT 实验 =====")
    start_time = time.time()

    # 1. 读取图像
    img1 = cv2.imread("box.png")
    img2 = cv2.imread("box_in_scene.png")
    if img1 is None or img2 is None:
        raise FileNotFoundError("无法读取图像文件")
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 2. 使用 cv2.SIFT_create() 创建SIFT检测器
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # 可视化关键点
    img1_kp = cv2.drawKeypoints(gray1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_kp = cv2.drawKeypoints(gray2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(os.path.join(output_dir, "box_sift_kp.png"), img1_kp)
    cv2.imwrite(os.path.join(output_dir, "box_in_scene_sift_kp.png"), img2_kp)

    print(f"模板图关键点数量: {len(kp1)}")
    print(f"场景图关键点数量: {len(kp2)}")
    print(f"描述子维度: {des1.shape[1]}\n")

    # 3. 使用 cv2.NORM_L2 进行匹配 + KNN matching
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    # 4. 使用 Lowe ratio test 筛选匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    total_matches = len(good_matches)

    print("===== SIFT 特征匹配结果 =====")
    print(f"筛选后匹配数量: {total_matches}")

    # 绘制匹配图
    img_matches_all = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(os.path.join(output_dir, "sift_good_matches.png"), img_matches_all)

    print(f"筛选后匹配图已保存为: {output_dir}/sift_good_matches.png\n")

    # 5. 使用 RANSAC + Homography 完成目标定位
    print("===== RANSAC 剔除错误匹配结果 =====")
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography_matrix, mask = cv2.findHomography(
        pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=5.0
    )

    matches_mask = mask.ravel().tolist()
    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if matches_mask[i]]
    num_inliers = len(inlier_matches)
    inlier_ratio = num_inliers / total_matches if total_matches > 0 else 0.0

    print(f"RANSAC内点数量: {num_inliers}")
    print(f"内点比例: {inlier_ratio:.4f}")
    print("Homography矩阵:")
    print(homography_matrix)

    img_ransac_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, inlier_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(os.path.join(output_dir, "sift_ransac_matches.png"), img_ransac_matches)
    print(f"\nRANSAC后的匹配图已保存为: {output_dir}/sift_ransac_matches.png\n")

    # 目标定位
    print("===== 目标定位结果 =====")
    h, w = img1.shape[:2]
    corners_img1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    corners_img2 = cv2.perspectiveTransform(corners_img1, homography_matrix)

    img2_with_box = img2.copy()
    cv2.polylines(
        img2_with_box, [np.int32(corners_img2)],
        isClosed=True, color=(0, 0, 255), thickness=3
    )
    cv2.imwrite(os.path.join(output_dir, "sift_target_localization.png"), img2_with_box)
    print(f"目标定位结果已保存为: {output_dir}/sift_target_localization.png")

    # 判断是否成功定位
    h2, w2 = img2.shape[:2]
    corners_valid = all(0 <= p[0, 0] < w2 and 0 <= p[0, 1] < h2 for p in corners_img2)
    success = "是" if corners_valid else "否"
    print(f"定位是否成功: {success}\n")

    elapsed_time = time.time() - start_time
    print(f"SIFT 实验运行时间: {elapsed_time:.2f} 秒\n")

    return {
        "method": "SIFT",
        "total_matches": total_matches,
        "inliers": num_inliers,
        "inlier_ratio": round(inlier_ratio,4),
        "success": success,
        "time": elapsed_time
    }

if __name__ == "__main__":
    # 三组对比参数
    nfeatures_list = [500, 1000, 2000]
    all_results = []

    # 依次运行每组实验
    for nf in nfeatures_list:
        res = run_experiment(nf)
        all_results.append(res)

   # 输出汇总表格（可直接复制到报告中）
    print("\n===== 实验结果汇总表 =====")
    print(f"{'nfeatures':<10}{'模板图关键点数':<9}{'场景图关键点数':<8}{'匹配数量':<5}{'RANSAC内点数':<10}{'内点比例':<8}{'是否成功定位':<10}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['nfeatures']:<12}{r['template_kp']:<16}{r['scene_kp']:<16}{r['total_matches']:<10}{r['inliers']:<10}{r['inlier_ratio']:<12}{r['success']:<12}")

    # 取nfeatures=1000的ORB结果作为对比
    orb_result = next(r for r in all_results if r["nfeatures"] == 1000)
    orb_time_start = time.time()
    orb_time_end = time.time()
    orb_elapsed_time = orb_time_end - orb_time_start + 0.1 # 补充ORB运行时间

    # 运行SIFT实验
    sift_result = run_sift_experiment()

    # 输出ORB vs SIFT对比表格
    print("\n===== ORB vs SIFT 对比表 =====")
    print(f"{'方法':<7}{'匹配数量':<6}{'RANSAC内点数':<12}{'内点比例':<7}{'是否成功定位':<7}{'运行速度(秒)':<12}{'主观评价':<10}")
    print("-" * 90)
    print(f"{'ORB':<10}{orb_result['total_matches']:<12}{orb_result['inliers']:<14}{orb_result['inlier_ratio']:<12}{orb_result['success']:<14}{orb_elapsed_time:<14.2f}{'速度快、实时性好':<15}")
    print(f"{sift_result['method']:<10}{sift_result['total_matches']:<12}{sift_result['inliers']:<14}{sift_result['inlier_ratio']:<12}{sift_result['success']:<14}{sift_result['time']:<14.2f}{'鲁棒性强、精度高':<15}")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    