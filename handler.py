import runpod
import os
import traceback
from typing import Dict, Any
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
llm = None
load_error = None

def load_model():
    """
    Carrega modelo otimizado para 48GB VRAM
    """
    global llm, load_error
    try:
        # Importações lazy para economizar memória
        from vllm import LLM, SamplingParams
        
        # Modelo que cabe em 48GB: Qwen2.5-32B-Instruct (quantizado)
        model_id = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
        
        logger.info(f"🚀 Carregando modelo: {model_id}")
        
        llm = LLM(
            model=model_id,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,  # Mais conservador
            max_model_len=4096,  # Reduzir para economizar VRAM
            quantization="gptq",  # Especificar quantização
            dtype="half",  # FP16 para economizar memória
            enforce_eager=True,  # Evitar compilation overhead
        )
        
        logger.info("✅ Modelo carregado com sucesso!")
        return llm
        
    except Exception as e:
        error_msg = f"❌ ERRO ao carregar modelo: {str(e)}"
        logger.error(error_msg)
        load_error = traceback.format_exc()
        logger.error(f"Stack trace: {load_error}")
        return None

def initialize_model():
    """
    Inicialização sob demanda do modelo
    """
    global llm
    if llm is None:
        logger.info("🔄 Inicializando modelo...")
        llm = load_model()
    return llm is not None

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handler principal do RunPod serverless
    """
    try:
        # Verificar se modelo está carregado
        if not initialize_model():
            return {
                "error": f"Falha ao carregar modelo. Detalhes: {load_error}",
                "status": "model_load_failed"
            }
        
        job_input = job.get('input', {})
        
        # Extrair parâmetros com defaults seguros
        prompt = job_input.get('prompt', 'Olá! Como posso ajudar você hoje?')
        temperature = job_input.get('temperature', 0.7)
        max_tokens = job_input.get('max_tokens', 1024)
        top_p = job_input.get('top_p', 0.9)
        
        logger.info(f"📝 Processando prompt: {prompt[:100]}...")
        
        # Importar aqui para economizar memória inicial
        from vllm import SamplingParams
        
        # Parâmetros de sampling otimizados
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=["</s>", "<|endoftext|>", "<|im_end|>"],  # Stop tokens comuns
        )
        
        # Gerar resposta
        outputs = llm.generate([prompt], sampling_params)
        response_text = outputs[0].outputs[0].text.strip()
        
        logger.info(f"✅ Resposta gerada: {len(response_text)} caracteres")
        
        return {
            "output": response_text,
            "status": "success",
            "model_info": {
                "model_id": "Qwen2.5-32B-Instruct-GPTQ",
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        }
        
    except Exception as e:
        error_msg = f"❌ Erro no handler: {str(e)}"
        logger.error(error_msg)
        return {
            "error": error_msg,
            "traceback": traceback.format_exc(),
            "status": "handler_error"
        }

# Função para warming up (opcional)
def warmup():
    """
    Aquece o modelo com uma requisição simples
    """
    try:
        if initialize_model():
            logger.info("🔥 Fazendo warmup do modelo...")
            test_job = {
                "input": {
                    "prompt": "Hello",
                    "max_tokens": 10
                }
            }
            handler(test_job)
            logger.info("✅ Warmup concluído!")
    except Exception as e:
        logger.warning(f"⚠️ Warmup falhou: {e}")

if __name__ == "__main__":
    logger.info("🚀 Iniciando RunPod Serverless...")
    
    # Fazer warmup opcional
    warmup()
    
    # Iniciar servidor
    runpod.serverless.start({
        "handler": handler,
        "concurrency_modifier": lambda x: 1,  # Limitar concorrência
        "return_aggregate_stream": True
    })
