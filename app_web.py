import gradio as gr
from transformers import pipeline
from deep_translator import GoogleTranslator  # <--- NOVIDADE

print("A carregar o modelo 'DJ de Emoções'...")
MODEL_PATH = "meu-modelo-de-emocoes-final"
labels_dict = {
    "LABEL_0": "sadness", "LABEL_1": "joy", "LABEL_2": "love",
    "LABEL_3": "anger", "LABEL_4": "fear", "LABEL_5": "surprise"
}

classifier = pipeline("text-classification", model=MODEL_PATH, top_k=None)
translator = GoogleTranslator(source='auto', target='en')  # <--- O TRADUTOR

print("Modelo carregado!")


def predict_emotion(text_input):
    # 1. Traduzir para Inglês (se necessário)
    text_en = translator.translate(text_input)

    # 2. Classificar
    predictions = classifier(text_en)[0]

    scores = {}
    for p in predictions:
        label_name = labels_dict.get(p['label'], p['label'])
        scores[label_name] = p['score']

    # Devolvemos os scores E a tradução para mostrar ao utilizador
    return scores, f"Texto traduzido/analisado: '{text_en}'"


# Exemplos em PT e EN
examples = [
    ["Hoje sinto-me nas nuvens!"],
    ["Estou farta disto tudo, que raiva."],
    ["i just called to say i love you"],
    ["Tenho medo do escuro"]
]

with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎧 DJ de Emoções (Multilingue) 🎧")

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(lines=3, label="Escreve aqui (Português ou Inglês)")
            submit_btn = gr.Button("Analisar", variant="primary")

        with gr.Column():
            output_chart = gr.Label(num_top_classes=6, label="Painel de Emoções")
            translation_out = gr.Markdown("*A tradução aparecerá aqui*")

    gr.Examples(examples=examples, inputs=text_input, outputs=[output_chart, translation_out], fn=predict_emotion)

    submit_btn.click(fn=predict_emotion, inputs=text_input, outputs=[output_chart, translation_out])

app.launch()