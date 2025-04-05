import os
import cv2
import numpy as np
from ultralytics import YOLO

# 1. Carrega o modelo YOLO Pose
model = YOLO("yolo11n-pose.pt")  # Substitua por 'yolov8l-pose.pt' para maior precisão

# 2. Define pastas de entrada (frames) e saída (keypoints)
input_base_folder = "/content/drive/MyDrive/frames_processados/anomaly_videos_pt_1/abuse/1_frame_per_second_extract_abuse_anomaly"  # Pasta com as subpastas de vídeos
output_base_folder = "/content/keypoints"  # Pasta onde os keypoints serão salvos

# 3. Processa cada subpasta (cada vídeo)
for video_folder in os.listdir(input_base_folder):
    frames_folder = os.path.join(input_base_folder, video_folder)
    output_folder = os.path.join(output_base_folder, video_folder)
    
    # Cria a pasta de saída se não existir
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Processando: {video_folder}...")
    
    # 4. Processa cada frame na subpasta
    for frame_file in sorted(os.listdir(frames_folder)):
        if not frame_file.endswith('.png'):
            continue
        
        # Lê o frame
        frame_path = os.path.join(frames_folder, frame_file)
        frame = cv2.imread(frame_path)
        
        # Executa o YOLO Pose
        results = model(frame, verbose=False)
        
        # Pega os keypoints das pessoas com maior confiança
        all_keypoints = results[0].keypoints.data.cpu().numpy()  # Shape: [n_pessoas, 17, 3]
        if len(all_keypoints) > 0:
            confiancas = all_keypoints[:, :, 2].mean(axis=1)  # Média de confiança por pessoa
            keypoints = all_keypoints[np.argmax(confiancas)]  # Pega a mais confiável

            
            # Salva os keypoints com o mesmo nome do frame (.npy)
            output_path = os.path.join(output_folder, os.path.splitext(frame_file)[0] + '.npy')
            np.save(output_path, keypoints)

print("Processamento concluído!")
