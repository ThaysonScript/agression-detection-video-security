import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
import streamlit as st
from youtube_dl import YoutubeDL

# ----------------------------
# 1. Pré-processamento de Vídeo
# ----------------------------
def process_video(video_path: str, frame_rate: int = 30) -> list:
    """Lê o vídeo e retorna frames na taxa especificada."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames[::frame_rate]

def detect_poses(frame: np.ndarray) -> np.ndarray:
    """Detecta keypoints humanos com YOLO Pose."""
    model = YOLO('yolov8n-pose.pt')
    results = model(frame)
    if len(results[0].keypoints) == 0:
        return np.zeros((17, 3))  # Caso não detecte pessoas
    return results[0].keypoints.data[0].numpy()  # Keypoints da primeira pessoa

def extract_features(keypoints: np.ndarray) -> np.ndarray:
    """Extrai características absolutas e relativas dos keypoints."""
    # Exemplo: coordenadas da cabeça + distância mãos-cabeça
    head = keypoints[0][:2]
    left_hand = keypoints[9][:2]
    right_hand = keypoints[10][:2]
    dist_left = np.linalg.norm(head - left_hand)
    dist_right = np.linalg.norm(head - right_hand)
    return np.concatenate([head, [dist_left, dist_right]])

def build_sequences(features: list, timesteps: int = 10) -> np.ndarray:
    """Constrói sequências temporais para a RNN."""
    sequences = []
    for i in range(len(features) - timesteps):
        sequences.append(features[i:i+timesteps])
    return np.array(sequences)

# ----------------------------
# 2. Modelo PyTorch (LSTM/GRU)
# ----------------------------
class EventClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        return self.fc(out[:, -1, :])  # Último timestep

# ----------------------------
# 3. Dataset e Treinamento
# ----------------------------
class VideoDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# ----------------------------
# 4. Interface Web (Streamlit)
# ----------------------------
def main():
    st.title("Detector de Eventos em Vídeos")
    url = st.text_input("Cole a URL do YouTube:")
    
    if url:
        # Baixar vídeo
        with YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            video_path = info['url']
        
        # Processar vídeo
        frames = process_video(video_path, frame_rate=15)
        features = [extract_features(detect_poses(frame)) for frame in frames]
        sequences = build_sequences(features, timesteps=10)
        
        # Carregar modelo treinado (exemplo)
        model = EventClassifier(input_size=4, hidden_size=64, num_classes=2)
        model.load_state_dict(torch.load('model.pth'))
        model.eval()
        
        # Classificar
        with torch.no_grad():
            inputs = torch.tensor(sequences, dtype=torch.float32)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
        
        st.write("Resultado:", "Evento Detectado" if predictions[0] == 1 else "Normal")

# ----------------------------
# Execução Principal
# ----------------------------
if __name__ == "__main__":
    # Exemplo de treinamento (ajuste com seus dados)
    # sequences = np.random.rand(100, 10, 4)  # (amostras, timesteps, features)
    # labels = np.random.randint(0, 2, 100)
    # dataset = VideoDataset(sequences, labels)
    # train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    # model = EventClassifier(input_size=4, hidden_size=64, num_classes=2)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # train_model(model, train_loader, criterion, optimizer, epochs=10)
    # torch.save(model.state_dict(), 'model.pth')
    
    # Executar interface web
    main()
