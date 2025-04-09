import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

MODELO_PATH = "src/models/modelo_suporteTI.pth"
VECTORIZER_PATH = "src/models/vectorizer.pkl"

# Lista de exemplos de chamados (inputs)
chamados = [
    "Meu computador não liga",
    "Erro na conexão com a internet",
    "Preciso resetar minha senha",
    "A tela está piscando",
    "Meu software está travando"
]

# Classes correspondentes (outputs)
categorias = ["hardware", "rede", "seguranca", "hardware", "software"]

# Adicione mais exemplos ao dataset
chamados.extend([
    # Hardware
    "O teclado do meu notebook não está funcionando",
    "O PC está fazendo barulho estranho",
    "O monitor não liga",
    "O computador não está reconhecendo o HD externo",
    "Meu laptop está superaquecendo",
    "O carregador do notebook não funciona",
    "O som do meu computador parou de funcionar",
    "A bateria do meu celular está descarregando rápido",
    "Meu monitor apresenta linhas na tela",
    "Minha CPU está esquentando muito",
    "Meu mouse não está respondendo",
    "O computador reinicia sozinho",
    "Minha placa de vídeo não está sendo reconhecida",
    "Meu fone de ouvido não está funcionando",
    "O botão de ligar do PC não funciona",
    "Pc lento",
    "Computador lento",
    
    # Rede
    "Minha conexão WiFi está instável",
    "Não consigo acessar a VPN",
    "O site não carrega, mas outros funcionam",
    "Meu WhatsApp desconectou",
    "A internet parou de funcionar",
    "O roteador reinicia sozinho",
    "Estou com lag em jogos online",
    "O WiFi está muito lento",
    "Não consigo conectar ao servidor da empresa",
    "O e-mail não está sincronizando",
    "Meu IP foi bloqueado",
    "A conexão cai toda hora",
    "Não consigo compartilhar arquivos pela rede",
    "A câmera do meu notebook não está conectando na reunião",
    "O sistema de chamadas VoIP não funciona corretamente",
    
    # Segurança
    "Minha conta foi invadida",
    "Recebi um e-mail de phishing",
    "Alguém tentou acessar minha conta sem permissão",
    "Não consigo redefinir minha senha",
    "Meu antivírus detectou uma ameaça",
    "Meu navegador foi sequestrado por pop-ups",
    "Alguém alterou minha senha sem permissão",
    "Meu computador está com comportamento estranho, pode ser um vírus?",
    "Recebi uma mensagem suspeita no WhatsApp",
    "Meu telefone está gravando sem minha permissão",
    "Preciso configurar autenticação em dois fatores",
    "Meu cartão de crédito foi usado sem autorização",
    "A segurança do meu e-mail foi comprometida",
    "Hackearam minha conta bancária",
    "Preciso bloquear um dispositivo não autorizado",
    
    # Software
    "O Word está fechando sozinho",
    "O aplicativo não abre depois da atualização",
    "Estou com erro ao instalar o programa",
    "O sistema operacional não inicializa",
    "O navegador não carrega as páginas corretamente",
    "Recebo um erro ao abrir um arquivo PDF",
    "O programa de edição de vídeo está travando",
    "O software da impressora não está funcionando",
    "Preciso atualizar o sistema, mas está dando erro",
    "Meu banco de dados está corrompido",
    "O PowerPoint não está exportando corretamente",
    "A sincronização do Google Drive não está funcionando",
    "A tela do sistema está piscando sem parar",
    "O antivírus não está conseguindo remover um malware",
    "O teclado está digitando sozinho"
])

# Atualiza a lista de categorias para cada novo chamado adicionado
categorias.extend(["hardware"] * 17)
categorias.extend(["rede"] * 15)
categorias.extend(["seguranca"] * 15)
categorias.extend(["software"] * 15)

# Verificação para evitar erro de tamanho de listas
assert len(chamados) == len(categorias), "Erro: Número de chamados e categorias não bate!"

# Vetorização dos textos
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(chamados).toarray()

# Convertendo categorias para números
categorias_dict = {"hardware": 0, "rede": 1, "seguranca": 2, "software": 3}
y = [categorias_dict[cat] for cat in categorias]

# Modelo de rede neural
class SuporteTIModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SuporteTIModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # Aumentei para 64 neurônios
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

# Convertendo os dados para tensores do PyTorch
X_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.long)

# Criando e treinando o modelo
modelo = SuporteTIModel(input_size=X_train.shape[1], output_size=len(categorias_dict))
criterio = nn.CrossEntropyLoss()
otimizador = optim.Adam(modelo.parameters(), lr=0.001)

# Loop de treinamento
for epoca in range(1000):
    otimizador.zero_grad()
    saida = modelo(X_train)
    perda = criterio(saida, y_train)
    perda.backward()
    otimizador.step()

    if (epoca + 1) % 100 == 0:
        print(f"Época {epoca+1} - Perda: {perda.item()}")

# Função para prever
def prever_categoria(chamado):
    vetor_chamado = vectorizer.transform([chamado]).toarray()
    tensor_chamado = torch.tensor(vetor_chamado, dtype=torch.float32)

    with torch.no_grad():
        saida = modelo(tensor_chamado)

    indice_categoria = torch.argmax(saida).item()
    return list(categorias_dict.keys())[indice_categoria]

# Teste
print(prever_categoria("Meu notebook desligou do nada"))  # Esperado: "hardware"
print(prever_categoria("Meu WhatsApp Web não está conectando"))  # Esperado: "rede"
print(prever_categoria("Alguém alterou minha senha sem permissão"))  # Esperado: "seguranca"
print(prever_categoria("O Excel está travando ao abrir planilhas grandes"))  # Esperado: "software"
print(prever_categoria("Wifi ruim")) # Esperado: "rede"
print(prever_categoria("Como encontrar o historico do navegador"))

# Salvando modelo treinado .
torch.save(modelo.state_dict(), MODELO_PATH)
with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(vectorizer, f)

print("Modelo e vetorizador salvos com sucesso!")
