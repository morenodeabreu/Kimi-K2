FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# 2. Evitamos perguntas interativas durante o build.
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# 3. Adicionamos o repositório 'deadsnakes' e instalamos o Python 3.11 corretamente.
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

# 4. Copia os arquivos do projeto para o container.
COPY . .

# 5. Usa a versão correta do pip (do Python 3.11) para instalar as dependências.
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# 6. Comando para iniciar nosso servidor com a versão correta do Python.
CMD ["python3.11", "handler.py"]
