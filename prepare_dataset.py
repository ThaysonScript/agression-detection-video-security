import os, random
from shutil import copy2

random.seed(42)
base_path = "/caminho/para/dataset"
output_path = "/caminho/para/projeto/data/raw"

def amostrar_e_copiar(pasta_origem, pasta_destino, total=150):
    videos = os.listdir(pasta_origem)
    amostrados = random.sample(videos, total)
    os.makedirs(pasta_destino, exist_ok=True)
    for video in amostrados:
        copy2(os.path.join(pasta_origem, video), os.path.join(pasta_destino, video))
    return amostrados


def dividir_e_salvar(lista, nome, destino):
    random.shuffle(lista)
    train, val, test = lista[:100], lista[100:120], lista[120:]
    for nome_arquivo, conjunto in zip([f"{nome}_train.txt", f"{nome}_val.txt", f"{nome}_test.txt"], [train, val, test]):
        with open(os.path.join(destino, nome_arquivo), "w") as f:
            f.writelines([linha + "\n" for linha in conjunto])
            
            
violence = amostrar_e_copiar(f"{base_path}/Violence", f"{output_path}/Violence", 150)
nonviolence = amostrar_e_copiar(f"{base_path}/NonViolence", f"{output_path}/NonViolence", 150)

dividir_e_salvar(violence, "violence", "./data/splits")
dividir_e_salvar(nonviolence, "nonviolence", "./data/splits")
