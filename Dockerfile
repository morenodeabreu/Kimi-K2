# Usamos a imagem oficial da NVIDIA que é estável
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Define o diretório de trabalho
WORKDIR /app

# Instala o básico, incluindo especificamente o Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3.11 \
    python3.11-pip \
    && rm -rf /var/lib/apt/lists/*

# Faz com que os comandos 'python3' e 'pip3' usem a versão 3.11 que acabamos de instalar
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/pip3.11 /usr/bin/pip3

# Copia os arquivos do projeto
COPY . .

# Agora, o 'pip3' (que é o pip 3.11) vai encontrar a versão correta do vllm
RUN pip3 install --no-cache-dir -r requirements.txt

# Define o comando padrão para iniciar o nosso servidor
CMD ["python3", "handler.py"]
