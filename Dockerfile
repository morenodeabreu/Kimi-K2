# Usa a imagem oficial da NVIDIA que é estável e garantida de funcionar
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Instala as ferramentas básicas que precisamos
RUN apt-get update && apt-get install -y git python3 python3-pip && rm -rf /var/lib/apt/lists/*

# Copia todos os arquivos do seu repositório para dentro do container
COPY . .

# Instala as dependências Python listadas no requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Define o comando padrão para iniciar o nosso servidor
CMD ["python3", "handler.py"]
