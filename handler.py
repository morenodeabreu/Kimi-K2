import runpod
from vllm import LLM, SamplingParams
import traceback  # Importamos uma ferramenta para capturar erros detalhados

# Variáveis globais
llm = None
load_error = None  # Nova variável para guardar a mensagem de erro

def load_model():
    """
    Tenta carregar o modelo e captura qualquer erro que acontecer.
    """
    global llm, load_error
    try:
        model_id = "MoonshotAI/Kimi-K2-32B"
        print(f"Iniciando o carregamento do modelo: {model_id}")

        llm = LLM(
            model=model_id,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.95
        )
        
        print("Modelo carregado com sucesso!")
        return llm
    except Exception as e:
        # Se qualquer erro acontecer, nós o capturamos aqui!
        print(f"ERRO AO CARREGAR O MODELO: {e}")
        # Formatamos o erro completo para ter todos os detalhes
        load_error = traceback.format_exc()
        return None

def handler(job):
    """
    Processa a requisição. Se o modelo não carregou, retorna o erro detalhado.
    """
    global llm, load_error
    
    if llm is None:
        # Em vez da mensagem genérica, agora retornamos o erro real!
        return {"error": f"O modelo falhou ao carregar. Erro detalhado: {load_error}"}

    # Se não houve erro, o código continua normalmente
    job_input = job['input']
    prompt = job_input.get('prompt', 'Olá, Kimi!')
    
    sampling_params = SamplingParams(
        temperature=job_input.get('temperature', 0.7),
        max_tokens=job_input.get('max_tokens', 1024)
    )

    outputs = llm.generate(prompt, sampling_params)
    response_text = outputs[0].outputs[0].text
    
    return {"output": response_text}

# Inicia o servidor do RunPod
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler, "load_model": load_model})
