import gradio as gr
from transformers import pipeline
from deep_translator import GoogleTranslator

print("A carregar o modelo 'DJ de Emoções'...")
MODEL_PATH = "./meu-modelo-de-emocoes-final"
labels_dict = {
    "LABEL_0": "sadness", "LABEL_1": "joy", "LABEL_2": "love",
    "LABEL_3": "anger", "LABEL_4": "fear", "LABEL_5": "surprise"
}

# 1. AQUI ADICIONAS AS TUAS PLAYLISTS (NOVIDADE)
playlists = {
    "sadness": "https://open.spotify.com/playlist/6yYA6aUGp8qUTgQWWYkPkP?si=7c4b964b09114ed4",  # Sad Songs
    "joy": "https://open.spotify.com/playlist/37i9dQZF1EIcxgujSVKT7V?si=8338e55c0ca94404",  # Mood Booster
    "love": "https://open.spotify.com/playlist/3w9THUphE7mxegKlyzOaLv?si=2ab237ba88664643",  # Timeless Love Songs
    "anger": "https://open.spotify.com/playlist/0KPEhXA3O9jHFtpd1Ix5OB?si=B_XSY5mLQU2KMiTUtrEi-g",  # Rage Beats
    "fear": "https://open.spotify.com/playlist/2yi1PWjebtfrELzbJ1NcMu?si=btZiRIw-QDKnxOgQ9wSfkQ",  # Calming Acoustic
    "surprise": "https://open.spotify.com/playlist/37i9dQZEVXbMDoHDwVN2tF?si=649cd812857d4a8c"  # Hits Virais
}

classifier = pipeline("text-classification", model=MODEL_PATH, top_k=None)
translator = GoogleTranslator(source='auto', target='en')

print("Modelo carregado!")


def predict_emotion(text_input):
    # 1. Traduzir para Inglês
    text_en = translator.translate(text_input)

    # 2. Classificar
    predictions = classifier(text_en)[0]

    scores = {}
    for p in predictions:
        label_name = labels_dict.get(p['label'], p['label'])
        scores[label_name] = p['score']

    # 3. DESCOBRIR A EMOÇÃO VENCEDORA E O LINK
    top_emotion = max(scores, key=scores.get)  # Qual é a emoção com maior score?
    link = playlists.get(top_emotion, "")  # Vai buscar o link ao dicionário

    # Devolvemos os scores E a mensagem com a tradução e o link
    mensagem_final = f"Texto traduzido: '{text_en}'\n\nPlaylist recomendada para {top_emotion.upper()}: \n{link}"

    return scores, mensagem_final


# Exemplos em PT e EN
examples = [
    ["Hoje sinto-me nas nuvens!"],
    ["Estou farta disto tudo, que raiva."],
    ["i just called to say i love you"],
    ["Tenho medo do escuro"]
]

with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# DJ de Emoções (Multilingue + Playlists)")

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(lines=3, label="Escreve aqui (Português ou Inglês)")
            submit_btn = gr.Button("Analisar e Recomendar Música", variant="primary")

        with gr.Column():
            output_chart = gr.Label(num_top_classes=6, label="Painel de Emoções")
            translation_out = gr.Markdown("*A tradução e a playlist aparecerão aqui*")

    gr.Examples(examples=examples, inputs=text_input, outputs=[output_chart, translation_out], fn=predict_emotion)

    submit_btn.click(fn=predict_emotion, inputs=text_input, outputs=[output_chart, translation_out])

app.launch()