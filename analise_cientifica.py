import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from datasets import load_dataset
from transformers import pipeline

# 1. Carregar dados e modelo
print("A carregar dados e modelo...")
dataset = load_dataset("emotion")

# top_k=1 garante que vem a melhor previsão, mas coloca-a dentro de uma lista
classifier = pipeline("text-classification", model="meu-modelo-de-emocoes-final", top_k=1)

# Vamos usar apenas o set de TESTE (2000 exemplos)
test_data = dataset["test"]
y_true = test_data["label"] # As respostas certas (0, 1, 2...)

# Convertemos para lista Python normal
texts = list(test_data["text"])

print(f"A fazer previsões para {len(texts)} exemplos...")

# Fazer previsões
# batch_size=16 acelera o processo e truncation=True evita erros em frases longas
preds = classifier(texts, batch_size=16, truncation=True)

# Como usámos top_k=1, o 'p' é uma lista com 1 elemento. Temos de fazer p[0] para chegar ao dicionário.
y_pred = [int(p[0]['label'].split('_')[-1]) for p in preds]

# Nomes das classes
class_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# 2. Criar a Matriz
print("A gerar o gráfico...")
# normalize='true' mostra percentagens
cm = confusion_matrix(y_true, y_pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

# 3. Desenhar e Guardar
plt.figure(figsize=(10, 10))
disp.plot(cmap='Blues', values_format='.2f')
plt.title("Matriz de Confusão (Normalizada)")

# Guardar a imagem na pasta do projeto
plt.savefig("matriz_confusao.png")
print("Gráfico guardado como 'matriz_confusao.png'.")