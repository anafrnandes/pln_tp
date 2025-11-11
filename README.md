# Projeto de PLN (2025/26): DJ de Emoções
Projeto desenvolvido no âmbito da Unidade Curricular de Processamento de Linguagem Natural, lecionada pelo Professor João Cordeiro.
* **Autora:** Ana Luisa Almeida Fernandes
* **Número:** 51648
* **Professor:** João Paulo Da Costa Cordeiro
* **Curso:** Inteligência Artificial e Ciência de Dados
* **Repositório:** https://github.com/anafrnandes/pln_tp.git

---

## Sobre o Projeto

Este projeto consiste num classificador de emoções para texto, focado em letras de música. Utilizando um Modelo de Linguagem de Larga Escala (LLM), a aplicação classifica um excerto de texto (em inglês) numa de seis emoções: **tristeza (sadness), alegria (joy), amor (love), raiva (anger), medo (fear)** ou **surpresa (surprise)**.

A abordagem utilizada foi o *fine-tuning* de um modelo BERT num dataset de emoções, como explorado nos objetivos da UC.

## Resultados do Treino

O modelo foi treinado com sucesso no Google Colab usando uma GPU T4. Os resultados de performance no conjunto de validação, após 3 *epochs*, foram os seguintes:

| Época | Training Loss | Validation Loss | Accuracy | F1 (weighted) | Precision | Recall |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.214100 | 0.171088 | 0.9265 | 0.9274 | 0.9329 | 0.9265 |
| 2 | 0.127900 | 0.156432 | 0.9330 | 0.9319 | 0.9342 | 0.9330 |
| 3 | 0.088000 | 0.170734 | **0.9445** | **0.9444** | **0.9449** | **0.9445** |

---

## Como Executar o Projeto

Este projeto foi desenvolvido em Python 3.11 e utiliza PyTorch com suporte para CUDA.

### 1. Setup do Ambiente

1.  **Clonar o repositório:**
    ```bash
    git clone https://github.com/anafrnandes/pln_tp.git
    cd pln-tp 
    ```

2.  **Criar e ativar o ambiente virtual:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate 
    ```

3.  **Instalar as dependências (com suporte GPU/CUDA):**
    ```bash
    # Instalar PyTorch com CUDA 12.1
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    
    # Instalar as restantes bibliotecas
    pip install transformers datasets evaluate accelerate scikit-learn nltk
    ```

### 2. Como Usar a Aplicação (`app.py`)

O modelo treinado (`meu-modelo-de-emocoes-final`) já está incluído neste repositório (ou foi descarregado separadamente) e é ignorado pelo `.gitignore`.

Para correr o classificador interativo:

```bash
python app.py
```
**Nota:** O modelo foi treinado no dataset emotion (em inglês). A aplicação espera, por isso, input em inglês para classificar corretamente.

---

##  Recursos Utilizados

* **Bibliotecas Principais:** `transformers` (Hugging Face), `torch` (PyTorch), `datasets` (Hugging Face), `evaluate` (Hugging Face).
* **Modelo Base (LLM):** `bert-base-uncased`.
* **Dataset:** `emotion` (disponível no Hugging Face Hub).
* **Ambiente de Treino:** Google Colab (com GPU Tesla T4).