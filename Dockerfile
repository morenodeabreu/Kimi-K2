FROM runpod/pytorch:2.3.0-py3.10-cuda12.1.1-devel

WORKDIR /app

# A imagem base já tem quase tudo, só precisamos do runpod e das dependências do Kimi.
# Atualizamos o pip e instalamos a partir do nosso requirements.txt
RUN pip install -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia os arquivos do nosso projeto (handler.py, etc.)
COPY . .

# Comando para iniciar o nosso servidor com o handler
CMD ["python3", "handler.py"]
