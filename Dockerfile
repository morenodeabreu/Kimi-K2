# Usamos a imagem base do RunPod que já vem com PyTorch e CUDA.
# É a mesma da sua captura de tela, o que é ótimo.
FROM runpod/pytorch:2.2.2-py3.10-cuda12.1.1-devel

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia todos os arquivos do seu repositório do GitHub para dentro do container
COPY . .

# O repositório Kimi deve ter um arquivo 'requirements.txt' listando as dependências.
# Este comando instala todas elas.
# Se o arquivo tiver outro nome, ajuste aqui.
RUN pip install --no-cache-dir -r requirements.txt

# Expõe a porta 8080 para o RunPod se comunicar com o nosso script
EXPOSE 8080

# O comando para iniciar nosso handler quando o container ligar.
# O RunPod vai gerenciar isso, mas é uma boa prática ter.
CMD ["python", "handler.py"]
