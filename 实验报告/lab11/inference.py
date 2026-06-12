import os
import cv2
import mediapipe as mp
import numpy as np
import torch
import json
from model import SkeletonTransformer

# ===================== 配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data"
MODEL_PATH = f"{DATA_DIR}/badminton_transformer.pth"
INFER_FRAMES = 30

# 加载标签映射
with open(f"{DATA_DIR}/label_map.json", "r", encoding="utf-8") as f:
    data = json.load(f)
id2label = data["id2label"]

# 初始化MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

# 全局仅加载一次模型
model = SkeletonTransformer(input_dim=132, target_frames=INFER_FRAMES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ===================== 工具函数 =====================
def extract_pose_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_list = []       # 保存原始BGR帧，用于可视化
    keypoint_list = []    # 保存关键点特征向量
    landmark_raw_list = []# 保存MediaPipe原始landmark对象，用于绘制骨架
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        frame_list.append(frame.copy())

        frame_keypoints = np.zeros(33 * 4, dtype=np.float32)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            landmark_raw_list.append(results.pose_landmarks)
            for idx, lm in enumerate(landmarks):
                frame_keypoints[idx*4] = lm.x
                frame_keypoints[idx*4+1] = lm.y
                frame_keypoints[idx*4+2] = lm.z
                frame_keypoints[idx*4+3] = lm.visibility
        else:
            landmark_raw_list.append(None)
        keypoint_list.append(frame_keypoints)
    cap.release()
    return np.array(keypoint_list), frame_list, landmark_raw_list

def resample_frames(frames_array, target_len):
    origin_len = len(frames_array)
    if origin_len == 0:
        return np.zeros((target_len, 132), dtype=np.float32)
    sample_indices = np.linspace(0, origin_len - 1, target_len, dtype=int)
    return frames_array[sample_indices]

def resample_aux_list(aux_list, target_len):
    """对帧/原始landmark列表做同样重采样"""
    origin_len = len(aux_list)
    if origin_len == 0:
        return []
    sample_indices = np.linspace(0, origin_len - 1, target_len, dtype=int)
    return [aux_list[i] for i in sample_indices]

def normalize_skeleton(seq):
    left_hip = seq[:, 23*4 : 23*4+2]
    right_hip = seq[:, 24*4 : 24*4+2]
    hip_center = (left_hip + right_hip) / 2.0
    left_shoulder = seq[:, 11*4 : 11*4+2]
    right_shoulder = seq[:, 12*4 : 12*4+2]
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=1, keepdims=True)
    shoulder_width[shoulder_width < 1e-6] = 1e-6
    for i in range(33):
        seq[:, i*4] = (seq[:, i*4] - hip_center[:, 0]) / shoulder_width[:, 0]
        seq[:, i*4+1] = (seq[:, i*4+1] - hip_center[:, 1]) / shoulder_width[:, 0]
    return seq

# ===================== 推理主函数（带可视化） =====================
def inference(video_path, out_video_path="vis_result.mp4"):
    if not os.path.exists(video_path):
        print(f"错误：视频 {video_path} 不存在")
        return

    # 1. 提取关键点、原始帧、原始landmark
    keypoint_seq, orig_frames, raw_landmarks = extract_pose_from_video(video_path)
    # 2. 重采样到固定帧数，同时对齐帧和landmark
    fixed_seq = resample_frames(keypoint_seq, INFER_FRAMES)
    vis_frames = resample_aux_list(orig_frames, INFER_FRAMES)
    vis_landmarks = resample_aux_list(raw_landmarks, INFER_FRAMES)
    fixed_seq = normalize_skeleton(fixed_seq)

    # 3. 模型推理
    input_tensor = torch.from_numpy(fixed_seq).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(input_tensor)
        prob = torch.softmax(logits, dim=1)
        conf, pred_label = torch.max(prob, dim=1)
    pred_class = id2label[str(pred_label.item())]
    confidence = conf.item()
    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {confidence:.2f}")

    # 4. 可视化：绘制骨架 + 推理结果，保存视频
    if len(vis_frames) > 0:
        h, w = vis_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_video_path, fourcc, 15, (w, h))

        txt = f"Class: {pred_class} | Conf: {confidence:.2f}"
        for frame, land in zip(vis_frames, vis_landmarks):
            draw_frame = frame.copy()
            # 绘制人体骨架
            if land is not None:
                mp_drawing.draw_landmarks(
                    draw_frame,
                    land,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing_styles.get_default_pose_landmarks_style()
                )
            # 绘制文本
            cv2.putText(draw_frame, txt, (10, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            out.write(draw_frame)
        out.release()
        print(f"可视化结果视频已保存至: {out_video_path}")

if __name__ == "__main__":
    TEST_VIDEO = "./archive/forehand_clear/005.mp4"
    inference(TEST_VIDEO, out_video_path="./infer_vis.mp4")