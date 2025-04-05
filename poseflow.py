import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import json

class FeatureExtractor:
    def __init__(self, frame_size=(264, 264)):
        self.frame_size = np.array(frame_size)
        self.joint_pairs = [
            (5, 6),    # Ombros
            (11, 12),  # Quadris
            (7, 9),    # Braço esquerdo
            (8, 10),   # Braço direito
            (13, 15),  # Perna esquerda
            (14, 16)   # Perna direita
        ]
        self.angle_triplets = [
            (5, 7, 9),  # Braço esquerdo
            (6, 8, 10), # Braço direito
            (11, 13, 15), # Perna esquerda
            (12, 14, 16)  # Perna direita
        ]

    def _normalize_coords(self, keypoints):
        return keypoints[:, :2] / self.frame_size

    def _calculate_distances(self, keypoints):
        return [np.linalg.norm(keypoints[i] - keypoints[j]) 
                for i, j in self.joint_pairs]

    def _calculate_angles(self, keypoints):
        angles = []
        for a, b, c in self.angle_triplets:
            ba = keypoints[a] - keypoints[b]
            bc = keypoints[c] - keypoints[b]
            cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
            angles.append(np.arccos(cosine))
        return angles

    def extract_features(self, current_kps, prev_kps=None):
        features = {}
        
        # Coordenadas normalizadas
        features['coords'] = self._normalize_coords(current_kps).flatten()
        
        # Distâncias relativas
        features['distances'] = self._calculate_distances(current_kps)
        
        # Ângulos das articulações
        features['angles'] = self._calculate_angles(current_kps)
        
        # Movimento temporal
        if prev_kps is not None:
            features['velocity'] = (current_kps[:, :2] - prev_kps[:, :2]).flatten()
            features['acceleration'] = features['velocity'] - self.prev_velocity
            
        return np.concatenate([v for v in features.values()])

def process_single_video(video_path, output_dir, seq_length=30):
    extractor = FeatureExtractor()
    frames = sorted(Path(video_path).glob('*.npy'))
    
    all_features = []
    prev_kps = None
    
    for frame_file in frames:
        try:
            kps = np.load(frame_file)
            if kps.shape != (17, 3):
                continue
                
            features = extractor.extract_features(kps, prev_kps)
            all_features.append(features)
            prev_kps = kps
        except Exception as e:
            print(f"Error processing {frame_file}: {str(e)}")
    
    # Cria sequências temporais
    sequences = []
    for i in range(len(all_features) - seq_length + 1):
        sequences.append(all_features[i:i+seq_length])
    
    # Salva resultados
    video_name = video_path.name
    output_path = output_dir / video_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    for idx, seq in enumerate(sequences):
        np.save(output_path / f"seq_{idx:04d}.npy", seq)
    
    return len(sequences)

def main():
    config = {
        'input_dir': Path("/content/drive/MyDrive/keypoints_frames_processados"),
        'output_dir': Path("/content/drive/MyDrive/feature_vectors"),
        'sequence_length': 30,
        'n_jobs': -1  # Usa todos os cores
    }
    
    # Encontra todos os vídeos
    video_paths = [p for p in config['input_dir'].rglob('*') if p.is_dir() and 'Abuse' in p.name]
    
    # Processamento paralelo
    results = Parallel(n_jobs=config['n_jobs'])(
        delayed(process_single_video)(
            video_path,
            config['output_dir'],
            config['sequence_length']
        ) for video_path in tqdm(video_paths, desc="Processing Videos")
    )
    
    # Salva metadados
    metadata = {
        'feature_dim': 17*2 + len(FeatureExtractor().joint_pairs) + len(FeatureExtractor().angle_triplets),
        'total_sequences': sum(results),
        'config': config
    }
    
    with open(config['output_dir'] / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    main()
