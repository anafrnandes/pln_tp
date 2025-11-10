from transformers import pipeline

# Carrega o teu modelo que acabaste de treinar
MODEL_PATH = "meu-modelo-de-emocoes-final"

# O "pipeline" é a forma mais fácil de USAR um modelo
# device=0 força-o a usar a GPU do Colab para ser rápido!
classifier = pipeline("text-classification", model=MODEL_PATH, device=0)

print("Modelo 'DJ de Emoções' carregado!")
print("Classes: 0:sadness, 1:joy, 2:love, 3:anger, 4:fear, 5:surprise")
print("Escreve uma letra de música (ou 'sair' para terminar):")

# Dicionário para mapear os IDs para os nomes das emoções
labels = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

while True:
    lyric = input("> ")
    if lyric.lower() == 'sair':
        break

    result = classifier(lyric)[0]

    # Formata o resultado para ser mais legível
    label_id = int(result['label'].split('_')[-1]) # Ex: "LABEL_1" -> 1
    score = result['score'] * 100

    # Converte o ID da label para o nome da emoção
    emotion_name = labels.get(label_id, "desconhecida") # .get() é mais seguro

    print(f"Emoção detetada: {emotion_name.upper()} (Confiança: {score:.2f}%)")

print("Até à Próxima Música")