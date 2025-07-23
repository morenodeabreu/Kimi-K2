# Usamos a imagem oficial da NVIDIA
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

WORKDIR /app

# PASSO CRUCIAL: Adiciona o repositório 'deadsnakes' para encontrar o Python 3.11
# e depois instala tudo o que precisamos.
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    git \
    python3.11 \
    python3.11-pip \
    python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# Faz com que os comandos 'python3' e 'pip3' usem a versão 3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/pip3.11 /usr/bin/pip3

# Copia os arquivos do projeto
COPY . .

# Instala as dependências Python do Kimi
RUN pip3 install --no-cache-dir -r requirements.txt

# Define o comando padrão para iniciar o nosso servidor
CMD ["python3", "handler.py"]
