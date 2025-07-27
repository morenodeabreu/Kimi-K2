# Use base image específica para vLLM
FROM vllm/vllm-openai:latest

# Evitar perguntas interativas
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Configurações de environment para RunPod
ENV RUNPOD_AI_API_KEY=""
ENV HF_HOME="/tmp/huggingface"
ENV TRANSFORMERS_CACHE="/tmp/transformers"

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar arquivos de requisitos primeiro (para melhor cache)
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY handler.py .
COPY *.py .

# Criar diretórios para cache
RUN mkdir -p /tmp/huggingface /tmp/transformers

# Pre-baixar tokenizer (opcional, para acelerar primeira execução)
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4', trust_remote_code=True)" || true

# Configurar health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import runpod; print('OK')" || exit 1

# Comando para iniciar o servidor
CMD ["python", "handler.py"]
