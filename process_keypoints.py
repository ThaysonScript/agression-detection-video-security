import os
import cv2
import numpy as np
from ultralytics import YOLO

# Configurações
# model = YOLO("/content/yolo11n-pose.pt").to('cuda')  # Ou 'yolov8l-pose.pt' para maior precisão
model = YOLO("/content/yolo11x-pose.pt").to('cuda')  # Ou 'yolov8l-pose.pt' para maior precisão
input_base_folder = "/content/drive/MyDrive/frames_processados/anomaly_frames_pt_1/abuse/1_frame_per_second_extract_abuse_anomaly"
output_base_folder = "/content/drive/MyDrive/keypoints_processados_yolo11x/anomaly_frames_pt_1/abuse/1_frame_per_second_extract_abuse_anomaly"
min_confidence = 0.25  # Ajuste conforme necessário

def process_frames():
    for video_folder in os.listdir(input_base_folder):
        frames_folder = os.path.join(input_base_folder, video_folder)
        output_folder = os.path.join(output_base_folder, video_folder)
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"\nProcessando: {video_folder}...")
        
        i = 0
        for frame_file in sorted(os.listdir(frames_folder)):
            if not frame_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            frame_path = os.path.join(frames_folder, frame_file)
            frame = cv2.imread(frame_path)
            
            # Detecta todas as pessoas no frame
            results = model(frame, verbose=False)
            if results[0].keypoints is None:
                continue
            
            all_people = results[0].keypoints.data.cpu().numpy()
            
            # Salva cada pessoa individualmente
            saved_count = 0
            for person_id, person_kps in enumerate(all_people):
                if person_kps.shape != (17, 3):
                    continue  # ignora keypoints com shape inválido

                if person_kps[:, 2].mean() < min_confidence:
                    continue
                
                output_path = os.path.join(
                    output_folder,
                    f"{os.path.splitext(frame_file)[0]}_p{person_id}.npy"
                )
                np.save(output_path, person_kps)
                saved_count += 1

            i += 1
            
            if i <= 50:
                print(f"Frame {frame_file}: {len(all_people)} pessoas → {saved_count} salvas")


if __name__ == "__main__":
    process_frames()
    print("\n✅ Todas as pessoas de todos os frames foram processadas!")
