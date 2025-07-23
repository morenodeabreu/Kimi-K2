# Usamos a imagem oficial da NVIDIA
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# A SOLUÇÃO FINAL: Diz ao sistema para não fazer perguntas interativas
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Instala o básico, incluindo o Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    git \
    python3.11 \
    python3.11-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copia os arquivos do projeto para o container
COPY . .

# Usa 'python3.11 -m pip' para forçar o uso da versão correta do pip
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Comando para iniciar nosso servidor
CMD ["python3.11", "handler.py"]
