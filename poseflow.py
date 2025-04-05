import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from joblib import Parallel, delayed

class KeypointProcessor:
    def __init__(self, frame_size=(264, 264)):
        self.frame_size = np.array(frame_size)
        self.joint_pairs = [
            (5, 6), (11, 12), (7, 9), (8, 10),  # Ombros, quadris, braços
            (13, 15), (14, 16)                  # Pernas
        ]
        self.angle_triplets = [
            (5, 7, 9), (6, 8, 10),              # Ângulos dos braços
            (11, 13, 15), (12, 14, 16)          # Ângulos das pernas
        ]

    def process_frame(self, kps):
        """Processa um frame e retorna features normalizadas"""
        # Normalização
        kps = kps.copy()
        kps[:, 0] /= self.frame_size[0]  # X
        kps[:, 1] /= self.frame_size[1]  # Y
        
        # Features básicas
        features = {
            'coords': kps[:, :2].flatten(),  # 34D (17 keypoints * 2)
            'distances': [np.linalg.norm(kps[i] - kps[j]) 
                         for i, j in self.joint_pairs],  # 6D
            'angles': []
        }
        
        # Cálculo de ângulos
        for a, b, c in self.angle_triplets:
            ba = kps[a] - kps[b]
            bc = kps[c] - kps[b]
            cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
            features['angles'].append(np.arccos(cosine))  # 4D
        
        return np.concatenate([
            features['coords'],
            features['distances'],
            features['angles']
        ])

def process_video(video_dir, output_base, seq_length=10):
    processor = KeypointProcessor()
    frames = sorted(video_dir.glob("*.npy"))
    
    # Processa todos os frames válidos
    features = []
    for frame_path in frames:
        try:
            kps = np.load(frame_path)
            if kps.shape == (17, 3):  # Verifica formato
                features.append(processor.process_frame(kps))
        except Exception as e:
            print(f"Erro no frame {frame_path.name}: {str(e)}")
    
    # Gera sequências temporais
    sequences = []
    for i in range(len(features) - seq_length + 1):
        sequences.append(features[i:i+seq_length])
    
    # Cria pasta de saída
    output_dir = output_base / video_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Salva as sequências
    for i, seq in enumerate(sequences):
        np.save(output_dir / f"seq_{i:04d}.npy", np.array(seq))
    
    return len(sequences)

def main():
    # Configurações (ajuste esses caminhos)
    input_dir = Path("/content/drive/MyDrive/keypoints_frames_processados/anomaly_frames_pt_1/abuse/1_frame_per_second_extract_abuse_anomaly_keypoints")
    output_dir = Path("/content/features_processed")
    seq_length = 10  # Janela temporal (em frames)
    
    # Encontra todos os diretórios de vídeo
    video_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    print(f"🔍 Encontrados {len(video_dirs)} vídeos para processar")
    
    # Processamento paralelo
    results = Parallel(n_jobs=4)(
        delayed(process_video)(video_dir, output_dir, seq_length)
        for video_dir in tqdm(video_dirs, desc="Processando vídeos")
    )
    
    # Gera relatório
    metadata = {
        "total_videos": len(video_dirs),
        "total_sequences": sum(results),
        "feature_dim": 34 + 6 + 4,  # coords + distances + angles
        "frame_size": [264, 264],
        "sequence_length": seq_length
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Processamento concluído!")
    print(f"   - Vídeos processados: {len(video_dirs)}")
    print(f"   - Sequências geradas: {sum(results)}")
    print(f"   - Features por frame: {metadata['feature_dim']}D")
    print(f"   - Saída salva em: {output_dir}")

if __name__ == "__main__":
    main()
