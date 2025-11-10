import numpy as np
import evaluate  # Biblioteca do Hugging Face para métricas
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
import torch

# Verificação GPU
if torch.cuda.is_available():
    print(f"\nGPU detectada pelo Colab: {torch.cuda.get_device_name(0)}\n")
else:
    print("\nA GPU não está ligada!")

# 1. CARREGAR OS DADOS E O MODELO

# Carrega o dataset "emotion" (tem 6 classes: 0:sadness, 1:joy, 2:love, 3:anger, 4:fear, 5:surprise)
print("A carregar o dataset 'emotion'...")
dataset = load_dataset("emotion")

# Nome do modelo base que vamos usar
MODEL_NAME = "bert-base-uncased"

# O Tokenizer traduz o texto para números que o BERT entende
print(f"A carregar o tokenizer: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# O Modelo com uma "cabeça" de classificação em cima.
# Sabe que é para classificar 6 "labels" (as emoções)
print(f"A carregar o modelo: {MODEL_NAME}...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=6)


# 2. PRÉ-PROCESSAMENTO

# Função para tokenizar o texto do dataset
def tokenize_function(examples):
    # padding="max_length" e truncation=True garantem que todos os textos têm o mesmo tamanho
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# Aplica a tokenização a todo o dataset (train, validation, test)
print("A tokenizar os datasets (isto pode demorar um pouco)...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)


# 3. MÉTRICAS DE AVALIAÇÃO
# Isto é crucial para o teu relatório (Aula 5: Precisão, Recall, F1)

print("A carregar as métricas (accuracy, f1, precision, recall)...")
# Carrega as métricas de "accuracy", "f1", "precision" e "recall"
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Os logits são os scores "crus" do modelo, o argmax escolhe a classe mais provável
    predictions = np.argmax(logits, axis=-1)

    # Calcula as métricas
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    # "weighted" é bom para datasets que possam ter classes desequilibradas
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    precision = precision_metric.compute(predictions=predictions, references=labels, average="weighted")["precision"]
    recall = recall_metric.compute(predictions=predictions, references=labels, average="weighted")["recall"]

    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}


# 4. O TREINO (FINE-TUNING)

# Define os argumentos de treino
training_args = TrainingArguments(
    output_dir="meu-modelo-de-emocoes",  # Pasta para guardar o modelo final
    eval_strategy="epoch",               # Avalia o modelo no fim de cada "epoch"
    save_strategy="epoch",               # Guarda o modelo no fim de cada "epoch"
    num_train_epochs=3,                  # 3 "epochs" costuma ser um bom ponto de partida
    load_best_model_at_end=True,         # Carrega o melhor modelo (baseado na "loss") no fim
    per_device_train_batch_size=16,      # Batch size
    per_device_eval_batch_size=16,
    report_to="none"
 )

# O "Trainer" é o motor que faz o fine-tuning (a otimização que viste na Aula 6)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,  # Usa a nossa função de métricas
)

print("A começar o fine-tuning do modelo...")
trainer.train()

print("Treino concluído!")

# Guarda o modelo final e o tokenizer
trainer.save_model("meu-modelo-de-emocoes-final")
tokenizer.save_pretrained("meu-modelo-de-emocoes-final")

print(" Modelo guardado em 'meu-modelo-de-emocoes-final'")