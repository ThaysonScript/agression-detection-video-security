import cv2
import os

def extract_frames(video_path, output_folder, frame_rate=1):
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"FPS inválido ou não detectado no vídeo: {video_path}")
        cap.release()
        return

    target_interval = 1.0 / frame_rate
    next_capture = 0.0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
        current_time = frame_pos / fps

        if current_time >= next_capture:
            frame_path = os.path.join(output_folder, f"frame_{saved_count:04d}.png")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
            next_capture += target_interval

    cap.release()
    print(f"{saved_count} frames salvos em {output_folder}")

def process_videos_folder(input_folder, output_base_folder, frame_rate=1):
    # Lista todos os arquivos na pasta de entrada
    video_files = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        print(f"Nenhum vídeo encontrado em: {input_folder}")
        return

    # Processa cada vídeo
    for video_file in video_files:
        video_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(input_folder, video_file)
        output_folder = os.path.join(output_base_folder, video_name)
        
        print(f"\nProcessando: {video_file}")
        extract_frames(video_path, output_folder, frame_rate)

# Exemplo de uso:
input_folder = "/content/drive/MyDrive/Testing_Normal_Videos_Anomaly"
output_base_folder = "/content/extracted_frames"
frame_rate = 1  # Altere para a taxa desejada

process_videos_folder(input_folder, output_base_folder, frame_rate)
