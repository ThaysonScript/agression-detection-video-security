import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

num = '0040'

# Configurações (substitua pelos seus caminhos reais)
keypoint_path = f"/content/drive/MyDrive/keypoints_frames_processados/anomaly_frames_pt_1/abuse/1_frame_per_second_extract_abuse_anomaly_keypoints/Abuse001_x264/frame_{num}.npy"
image_path = f"/content/drive/MyDrive/frames_processados/anomaly_frames_pt_1/abuse/1_frame_per_second_extract_abuse_anomaly/Abuse001_x264/frame_{num}.png"  # Substitua pelo caminho real da imagem

output_path = f"/content/abuse_detection{num}"
frame_size = (264, 264)

# Verifica e carrega os keypoints
try:
    keypoints = np.load(keypoint_path)
    if keypoints.size == 0:
        raise ValueError("Array de keypoints vazio")
    if keypoints.shape != (17, 3):
        raise ValueError(f"Formato inválido: {keypoints.shape} (esperado: (17, 3))")
except Exception as e:
    print(f"Erro ao carregar keypoints: {str(e)}")
    keypoints = np.zeros((17, 3))  # Array vazio como fallback

# Carrega a imagem (se existir)
try:
    img = plt.imread(image_path) if Path(image_path).exists() else None
except Exception as e:
    print(f"Erro ao carregar imagem: {str(e)}")
    img = None

# Mapeamento COCO (17 keypoints)
COCO_KEYPOINTS = [
    "nariz", "olho_esq", "olho_dir", "orelha_esq", "orelha_dir",
    "ombro_esq", "ombro_dir", "cotovelo_esq", "cotovelo_dir",
    "pulso_esq", "pulso_dir", "quadril_esq", "quadril_dir",
    "joelho_esq", "joelho_dir", "tornozelo_esq", "tornozelo_dir"
]

# Cores para visualização
COLORS = [
    'red', 'blue', 'green', 'cyan', 'magenta', 
    'yellow', 'black', 'orange', 'purple',
    'brown', 'pink', 'gray', 'olive',
    'navy', 'teal', 'coral', 'gold'
]

# Conexões entre keypoints (esqueleto)
connections = [
    (5, 7), (7, 9),   # Braço esquerdo
    (6, 8), (8, 10),   # Braço direito
    (11, 13), (13, 15), # Perna esquerda
    (12, 14), (14, 16), # Perna direita
    (5, 6), (11, 12)    # Ombros e quadris
]

# Cria a figura
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Subplot 1: Imagem original
if img is not None:
    ax1.imshow(img)
    ax1.set_title("Frame Original")
else:
    ax1.text(0.5, 0.5, "Imagem não disponível", ha='center', va='center')
    ax1.set_title("Frame Original (Não disponível)")
ax1.axis('off')

# Subplot 2: Keypoints
ax2.add_patch(patches.Rectangle((0, 0), frame_size[0], frame_size[1], 
             linewidth=1, edgecolor='black', facecolor='white'))

# Verifica se há keypoints válidos
if keypoints.size > 0 and keypoints.shape == (17, 3):
    # Plota keypoints
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.3:
            ax2.scatter(x, y, color=COLORS[i], s=100, label=f"{COCO_KEYPOINTS[i]} (conf: {conf:.2f})")
            ax2.text(x, y, f"{i}", color='white', fontsize=8, ha='center', va='center')
    
    # Plota conexões
    for (i, j) in connections:
        if keypoints[i, 2] > 0.3 and keypoints[j, 2] > 0.3:
            ax2.plot(
                [keypoints[i, 0], keypoints[j, 0]],
                [keypoints[i, 1], keypoints[j, 1]],
                color='gray', linestyle='-', linewidth=2, alpha=0.5
            )
    ax2.set_title("Keypoints Detectados")
else:
    ax2.text(0.5, 0.5, "Keypoints inválidos ou não detectados", ha='center', va='center')
    ax2.set_title("Keypoints (Não disponíveis)")

ax2.set_xlim(0, frame_size[0])
ax2.set_ylim(frame_size[1], 0)
ax2.grid(True, alpha=0.3)

# Legenda
handles, labels = ax2.get_legend_handles_labels()
if handles:  # Só adiciona legenda se houver keypoints
    fig.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()

# Salva a figura
try:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Visualização salva em: {output_path}")
except Exception as e:
    print(f"Erro ao salvar: {str(e)}")

plt.show()
