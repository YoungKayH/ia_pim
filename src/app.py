# app.py
# @author Kaka
# @date 08/04/2025
# @description server app conexão do REST API com chamada da IA
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# diretorios
MODELO_PATH = "models/modelo_suporteTI.pth"
VECTORIZER_PATH = "models/vectorizer.pkl"

# Inicializando a API
app = Flask(__name__)

# Carregando vetorizador e categorias
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

categorias_dict = {"hardware": 0, "rede": 1, "seguranca": 2, "software": 3}
categorias_inv = {v: k for k, v in categorias_dict.items()}

# Modelo de rede neural
class SuporteTIModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SuporteTIModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Inicializa e carrega o modelo treinado
input_size = len(vectorizer.get_feature_names_out())
output_size = len(categorias_dict)
modelo = SuporteTIModel(input_size, output_size)
modelo.load_state_dict(torch.load(MODELO_PATH))
modelo.eval()

# Função de previsão
def prever_categoria(chamado):
    vetor_chamado = vectorizer.transform([chamado]).toarray()
    tensor_chamado = torch.tensor(vetor_chamado, dtype=torch.float32)

    with torch.no_grad():
        saida = modelo(tensor_chamado)

    indice_categoria = torch.argmax(saida).item()
    return categorias_inv[indice_categoria]

# Rota de API
@app.route("/prever", methods=["POST"])
def api_prever():
    dados = request.get_json()
    chamado = dados.get("chamado", "")

    print("passou na API")

    if not chamado:
        return jsonify({"erro": "Nenhum chamado enviado"}), 400

    categoria = prever_categoria(chamado)
    return jsonify({"categoria": categoria})

if __name__ == '__main__':
    app.run(debug=True)
