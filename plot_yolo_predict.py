import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

# === CONFIGURAÇÕES ===
FRAMES_DIR = Path("/content/drive/MyDrive/frames_processados/anomaly_frames_pt_2/5_frame_per_second_extract_fighting_anomaly/Fighting004_x264")  # Ex: .../frames/
KEYPOINTS_DIR = Path("/content/drive/MyDrive/keypoints_processados/anomaly_frames_pt_2/5_frame_per_second_extract_fighting_anomaly/Fighting004_x264")  # Ex: .../keypoints/


MAX_FRAMES = 40  # Número de imagens a exibir

# === ENCONTRA TODOS OS FRAMES ===
frame_files = sorted([f for f in FRAMES_DIR.glob("*.png")])[:MAX_FRAMES]

# === CÁLCULO DE GRADE DE SUBPLOTS ===
cols = 3
rows = math.ceil(len(frame_files) / cols)
fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
axs = axs.flatten()

for idx, frame_path in enumerate(frame_files):
    img_bgr = cv2.imread(str(frame_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    frame_name = frame_path.stem  # Ex: 'frame_0042'

    # Carrega keypoints desse frame
    keypoints_files = sorted(KEYPOINTS_DIR.glob(f"{frame_name}_p*.npy"))

    ax = axs[idx]
    ax.imshow(img_rgb)
    ax.set_title(f"{frame_name} - {len(keypoints_files)} pessoa(s)")
    ax.axis('off')

    for kp_file in keypoints_files:
        kps = np.load(kp_file)
        if kps.shape == (17, 3):
            x = kps[:, 0]
            y = kps[:, 1]
            ax.scatter(x, y, c='cyan', s=20, alpha=0.7)

# Esconde subplots vazios, se houver
for j in range(len(frame_files), len(axs)):
    axs[j].axis('off')

plt.tight_layout()
plt.show()
