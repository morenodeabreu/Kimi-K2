import runpod
from vllm import LLM, SamplingParams

# Variável global para armazenar o modelo carregado
llm = None

def load_model():
    """
    Esta função carrega o modelo Kimi-K2 32B usando vLLM quando o pod inicia.
    """
    global llm

    # ATENÇÃO: Substitua pelo nome EXATO do modelo 32B no Hugging Face, se for diferente.
    # Estou usando um nome hipotético baseado na sua informação.
    model_id = "MoonshotAI/Kimi-K2-32B"

    print(f"Iniciando o carregamento do modelo: {model_id}")

    # Configura e carrega o modelo. Se falhar aqui, será por falta de memória ou nome incorreto.
    llm = LLM(
        model=model_id,
        trust_remote_code=True,      # Necessário conforme a documentação do Kimi
        tensor_parallel_size=1,      # Usando apenas 1 GPU
        gpu_memory_utilization=0.95  # Tenta usar 95% da VRAM disponível
    )
    
    print("Modelo carregado com sucesso!")
    return llm

def handler(job):
    """
    Esta função processa cada requisição recebida pelo endpoint.
    """
    global llm
    if llm is None:
        return {"error": "O modelo não foi carregado. Verifique os logs do build."}

    # Extrai os dados da requisição
    job_input = job['input']
    prompt = job_input.get('prompt', 'Qual a diferença entre um modelo de 32 bilhões e 70 bilhões de parâmetros?')
    
    # Define os parâmetros para a geração de texto
    sampling_params = SamplingParams(
        temperature=job_input.get('temperature', 0.7),
        top_p=job_input.get('top_p', 0.95),
        max_tokens=job_input.get('max_tokens', 1024)
    )

    print("Gerando resposta para o prompt...")
    # Usa o vLLM para gerar a resposta
    outputs = llm.generate(prompt, sampling_params)
    
    # Extrai e retorna o texto da primeira resposta gerada
    response_text = outputs[0].outputs[0].text
    print("Resposta gerada com sucesso.")
    
    return {"output": response_text}

# Inicia o servidor do RunPod
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
