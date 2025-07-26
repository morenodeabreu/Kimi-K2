import runpod
from vllm import LLM, SamplingParams
import traceback

llm = None
load_error = None

def load_model():
    """
    Carrega o modelo Phind Code Llama.
    """
    global llm, load_error
    try:
        # A ÚNICA MUDANÇA É AQUI: Trocamos pelo modelo especialista em código
        model_id = "Phind/Phind-CodeLlama-34B-v2"
        
        print(f"Iniciando o carregamento do modelo: {model_id}")

        llm = LLM(
            model=model_id,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.95 # Usar 95% da VRAM para caber o modelo 34B
        )
        
        print("Modelo de código carregado com sucesso!")
        return llm
    except Exception as e:
        print(f"ERRO AO CARREGAR O MODELO: {e}")
        load_error = traceback.format_exc()
        return None

def handler(job):
    """
    Processa a requisição.
    """
    global llm, load_error
    
    if llm is None:
        return {"error": f"O modelo falhou ao carregar. Erro detalhado: {load_error}"}

    job_input = job['input']
    prompt = job_input.get('prompt', 'Escreva uma função em Python para calcular o fatorial de um número.')
    
    sampling_params = SamplingParams(
        temperature=job_input.get('temperature', 0.1), # Temperatura mais baixa para código
        top_p=job_input.get('top_p', 0.95),
        max_tokens=job_input.get('max_tokens', 2048) # Mais tokens para código
    )

    outputs = llm.generate(prompt, sampling_params)
    response_text = outputs[0].outputs[0].text
    
    return {"output": response_text}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
