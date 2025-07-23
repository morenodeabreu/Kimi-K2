dockerfile
# PASSO 1: Começamos com a imagem oficial da NVIDIA com CUDA 12.1
# Esta imagem é garantida de existir.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Define o diretório de trabalho
WORKDIR /app

# Instala o básico que precisamos: git, python e o gerenciador de pacotes pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copia os arquivos do seu projeto para dentro do container
COPY . .

# Instala as dependências do projeto Kimi a partir do requirements.txt
# O repositório DEVE ter este arquivo.
RUN pip3 install --no-cache-dir -r requirements.txt

# Instala o PyTorch e a biblioteca do RunPod separadamente para garantir
RUN pip3 install --no-cache-dir torch runpod

# Comando para iniciar nosso handler
CMD ["python3", "handler.py"]

