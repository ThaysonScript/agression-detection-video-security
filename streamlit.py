import streamlit as st
import torch
import numpy as np
import os

# Carrega o modelo
def carregar_modelo(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

# Função principal
def main():
    st.title("🧠 Classificador de Eventos (sem reprocessamento de vídeo)")

    modelo_escolhido = st.selectbox("Escolha o modelo:", ["RNN", "GRU", "LSTM"])
    modelo_path = f"models/{modelo_escolhido.lower()}_model.pt"

    video = st.file_uploader("Faça upload de um vídeo já conhecido (apenas para identificação)", type=["mp4", "avi", "mkv"])

    if video is not None:
        nome_video = os.path.splitext(video.name)[0]  # Ex: Abuse001_x264
        st.write(f"🎬 Nome do vídeo: `{nome_video}`")

        vetor_encontrado = None
        diretorio_vetores = "/content/drive/MyDrive/keypoints_processados_yolo11n"

        # Busca pelo vetor correspondente ao nome do vídeo
        for root, dirs, files in os.walk(diretorio_vetores):
            for file in files:
                if nome_video in file and file.endswith(".npy"):
                    vetor_encontrado = os.path.join(root, file)
                    break

        if vetor_encontrado is None:
            st.error("❌ Nenhum vetor correspondente encontrado.")
        else:
            st.success(f"✅ Vetor encontrado: `{vetor_encontrado}`")

            # Carrega o vetor e prepara para o modelo
            vetor = np.load(vetor_encontrado)
            if len(vetor.shape) == 2:
                vetor = np.expand_dims(vetor, axis=0)  # (1, timesteps, features)

            vetor_tensor = torch.tensor(vetor).float()

            modelo = carregar_modelo(modelo_path)
            with torch.no_grad():
                saida = modelo(vetor_tensor)
                pred = torch.argmax(saida, dim=1).item()

                classes = ['Abuse', 'Assault', 'Fighting', 'Normal']
                st.markdown(f"### 🎯 Resultado: **{classes[pred]}**")

if __name__ == "__main__":
    main()
