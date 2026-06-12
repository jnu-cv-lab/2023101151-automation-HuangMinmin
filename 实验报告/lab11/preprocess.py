import os
import json
import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ===================== 配置参数（与实验文档一致）=====================
DATA_ROOT = "./archive"       # 数据集根目录
SAVE_DIR = "./data"           # 数据保存目录
TARGET_FRAMES = 30            # 统一帧数 T=30
TEST_RATIO = 0.2              # 测试集比例
FRAME_DIM = 132               # 单帧特征维度：33关键点 * 4(x,y,z,visibility)

# 羽毛球6类动作映射（与Kaggle数据集对应）
LABEL_LIST = [
    "forehand_drive",
    "forehand_lift",
    "forehand_net_shot",
    "forehand_clear",
    "backhand_drive",
    "backhand_net_shot"
]
label2id = {name: idx for idx, name in enumerate(LABEL_LIST)}
id2label = {idx: name for idx, name in enumerate(LABEL_LIST)}

# ===================== 初始化MediaPipe Pose =====================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,   # 视频模式（非静态图片）
    model_complexity=1,        # 平衡速度与精度
    smooth_landmarks=True,     # 关键点平滑
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ===================== 工具函数 =====================
def extract_pose_from_frame(frame: np.ndarray) -> np.ndarray:
    """单帧提取33个人体关键点，输出132维向量"""
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # 初始化空关键点（未检测到人时补0）
    landmarks_flat = np.zeros(FRAME_DIM, dtype=np.float32)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        for i, lm in enumerate(landmarks):
            # 每个关键点：x,y,z,visibility 4个特征
            landmarks_flat[i*4 + 0] = lm.x
            landmarks_flat[i*4 + 1] = lm.y
            landmarks_flat[i*4 + 2] = lm.z
            landmarks_flat[i*4 + 3] = lm.visibility
    return landmarks_flat

def resample_frames(frames_list: list, target_num: int) -> list:
    """将变长帧列表均匀重采样为固定帧数"""
    total = len(frames_list)
    if total <= 0:
        return [np.zeros(FRAME_DIM)] * target_num
    # 均匀采样索引
    indices = np.linspace(0, total - 1, target_num, dtype=int)
    return [frames_list[i] for i in indices]

def normalize_skeleton(seq: np.ndarray) -> np.ndarray:
    """骨架归一化：以左右髋部为原点，肩宽做尺度归一化（实验要求）"""
    # 关键点索引：左髋23，右髋24，左肩11，右肩12
    left_hip = seq[:, 23*4 : 23*4+2]   # x,y
    right_hip = seq[:, 24*4 : 24*4+2]
    left_shoulder = seq[:, 11*4 : 11*4+2]
    right_shoulder = seq[:, 12*4 : 12*4+2]

    # 髋部中心为原点
    hip_center = (left_hip + right_hip) / 2.0
    seq[:, ::4] -= hip_center[:, 0:1]    # 所有x坐标偏移
    seq[:, 1::4] -= hip_center[:, 1:2]  # 所有y坐标偏移

    # 肩宽做尺度归一化
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=1, keepdims=True)
    shoulder_width[shoulder_width < 1e-6] = 1.0  # 防止除0
    seq[:, ::4] /= shoulder_width
    seq[:, 1::4] /= shoulder_width
    return seq

def process_single_video(video_path: str) -> np.ndarray:
    """处理单个视频：逐帧提取关键点 -> 重采样 -> 归一化"""
    cap = cv2.VideoCapture(video_path)
    frame_features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        feat = extract_pose_from_frame(frame)
        frame_features.append(feat)
    cap.release()

    # 重采样为固定30帧
    resampled = resample_frames(frame_features, TARGET_FRAMES)
    seq_arr = np.array(resampled, dtype=np.float32)  # [30, 132]
    # 骨架归一化
    seq_arr = normalize_skeleton(seq_arr)
    return seq_arr

# ===================== 主预处理流程 =====================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    all_sequences = []
    all_labels = []

    # 遍历每个类别文件夹
    for label_name in LABEL_LIST:
        class_dir = os.path.join(DATA_ROOT, label_name)
        if not os.path.exists(class_dir):
            print(f"警告：类别文件夹 {class_dir} 不存在，跳过")
            continue
        label_id = label2id[label_name]
        video_files = [f for f in os.listdir(class_dir)
                       if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
        print(f"\n正在处理类别 [{label_name}]，视频总数：{len(video_files)}")

        # 遍历当前类别所有视频
        for vid_file in tqdm(video_files, desc=label_name):
            vid_path = os.path.join(class_dir, vid_file)
            try:
                seq = process_single_video(vid_path)
                all_sequences.append(seq)
                all_labels.append(label_id)
            except Exception as e:
                print(f"视频 {vid_file} 处理失败：{str(e)}")

    # 转换为numpy数组
    X = np.array(all_sequences, dtype=np.float32)  # [N, 30, 132]
    y = np.array(all_labels, dtype=np.int64)      # [N]
    print(f"\n全部数据总形状：X={X.shape}, y={y.shape}")

    # 划分训练集/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=42, stratify=y
    )
    print(f"训练集：{X_train.shape} | 测试集：{X_test.shape}")

    # 保存npy文件
    np.save(os.path.join(SAVE_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(SAVE_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(SAVE_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(SAVE_DIR, "y_test.npy"), y_test)

    # 保存类别映射json
    with open(os.path.join(SAVE_DIR, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

    print("\n✅ 预处理完成！数据已保存至 ./data 目录")

if __name__ == "__main__":
    main()