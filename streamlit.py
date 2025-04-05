import streamlit as st
from model import predict_video

st.title("Detecção de violência em vídeo")
video = st.file_uploader("Faça upload de um vídeo", type=["mp4"])
if video:
    result = predict_video(video)
    st.write("Resultado:", result)
