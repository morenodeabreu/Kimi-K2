# Usamos a imagem oficial da NVIDIA
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

WORKDIR /app

# Instala o básico, incluindo o Python 3.11 e suas ferramentas
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

# A MUDANÇA FINAL E CRUCIAL:
# Usamos 'python3.11 -m pip' para forçar o uso da versão correta do pip
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Comando para iniciar nosso servidor, usando explicitamente o python3.11
CMD ["python3.11", "handler.py"]
