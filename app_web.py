import gradio as gr
from transformers import pipeline

# 1. Carregar o modelo (exatamente igual)
print("A carregar o modelo 'DJ de Emoções'...")
MODEL_PATH = "meu-modelo-de-emocoes-final"
labels_dict = {
    "LABEL_0": "sadness", "LABEL_1": "joy", "LABEL_2": "love",
    "LABEL_3": "anger", "LABEL_4": "fear", "LABEL_5": "surprise"
}
classifier = pipeline(
    "text-classification",
    model=MODEL_PATH,
    top_k=None  # Usar o novo parâmetro
)
print("Modelo carregado!")


# 2. Função de previsão (exatamente igual)
def predict_emotion(text_input):
    predictions = classifier(text_input)[0]
    scores = {}
    for p in predictions:
        label_name = labels_dict.get(p['label'], p['label'])
        scores[label_name] = p['score']
    return scores


# 3. Lista de exemplos
examples = [
    ["i'm walking on sunshine, whoa-oh!"],
    ["i'm tired of being what you want me to be"],
    ["everybody wanna steal my girl"],
    ["i don't remember you"],
    ["i just called to say i love you"]
]

# 4. LANÇAR A INTERFACE

# Usamos 'gr.Blocks' e damos um 'tema' à nossa escolha
# Podes experimentar: theme=gr.themes.Base(), theme=gr.themes.Glass()
with gr.Blocks(theme=gr.themes.Soft()) as app:
    # Bloco 1: O Título
    gr.Markdown(
        """
        # DJ de Emoções 
        ### Projeto Final de PLN - Ana Fernandes (a51648)
        Escreve um excerto de uma letra de música (em inglês) e vê o 'painel de emoções' do modelo.
        (Modelo: BERT fine-tuned c/ 94.45% Acc.)
        """
    )

    # Bloco 2: A Interface (numa linha)
    with gr.Row():
        # Coluna da Esquerda (Inputs)
        with gr.Column():
            text_input = gr.Textbox(lines=5, label="Letra da Música (em Inglês)")
            submit_btn = gr.Button("Submeter Emoção", variant="primary")
            clear_btn = gr.Button("Limpar")

        # Coluna da Direita (Outputs)
        with gr.Column():
            output_chart = gr.Label(num_top_classes=6, label="Painel de Emoções")

    # Bloco 3: Os Exemplos
    gr.Examples(
        examples=examples,
        inputs=text_input,  # Diz ao Gradio para pôr o exemplo na caixa 'text_input'
        outputs=output_chart,  # Diz ao Gradio para correr a previsão com o exemplo
        fn=predict_emotion,
        cache_examples=False  # Não fazer cache (para garantir que corre sempre)
    )

    # Bloco 4: Ligar os botões às funções

    # O que acontece quando clicas no botão "Submeter"
    submit_btn.click(
        fn=predict_emotion,  # Chama a nossa função
        inputs=text_input,  # Passa o conteúdo da caixa de texto
        outputs=output_chart  # Envia o resultado para o gráfico
    )

    # O que acontece quando clicas no botão "Limpar"
    clear_btn.click(
        fn=lambda: (None, None),  # Função simples que não devolve nada
        inputs=None,
        outputs=[text_input, output_chart]  # Limpa a caixa de texto E o gráfico
    )

# 5. Iniciar a aplicação
print("A iniciar a aplicação web... Clica no link local (ex: http://127.0.0.1:7860)!")
app.launch()