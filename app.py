from transformers import pipeline

# Carrega o teu modelo treinado
MODEL_PATH = "./meu-modelo-de-emocoes-final"

print("A carregar o 'DJ de Emoções' (isto pode demorar um momento)...")

classifier = pipeline(
    "text-classification",
    model=MODEL_PATH,
    # device=0
    top_k=None  # <--- O COMANDO NOVO
)

print("Modelo 'DJ de Emoções' carregado!")
print("Classes: sadness, joy, love, anger, fear, surprise")
print("Escreve uma letra de música (ou 'sair' para terminar):")

# Dicionário para "traduzir" as labels que o modelo dá
# (o modelo de classificação dá "LABEL_0", "LABEL_1", etc.)
labels = {
    "LABEL_0": "sadness",
    "LABEL_1": "joy",
    "LABEL_2": "love",
    "LABEL_3": "anger",
    "LABEL_4": "fear",
    "LABEL_5": "surprise"
}

while True:
    lyric = input("> ")
    if lyric.lower() == 'sair':
        break

    # O resultado agora é uma LISTA de 6 dicionários
    # Ex: [[{'label': 'LABEL_0', 'score': 0.1}, {'label': 'LABEL_1', 'score': 0.8}, ...]]
    results = classifier(lyric)[0]

    # Vamos ordenar a lista pelo score, do maior para o menor
    results.sort(key=lambda x: x['score'], reverse=True)

    print("Painel de Emoções ")

    for i, res in enumerate(results):
        label_name = labels.get(res['label'], res['label'])  # Traduz o nome
        score = res['score'] * 100

        # Formatação para ficar bonito no terminal
        # O \t é um "tab" para alinhar as percentagens
        if i == 0:
            print(f"🥇 {label_name.upper()}:\t{score:6.2f}%  (Emoção Principal)")
        else:
            print(f"   {label_name.lower()}:\t{score:6.2f}%")

    print("-" * 34)

print("Até à próxima!")