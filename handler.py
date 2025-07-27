import runpod
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import traceback

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model = None
tokenizer = None
load_error = None

def load_model():
    """
    Carrega modelo usando transformers (mais simples que vLLM)
    Testando com modelo menor primeiro
    """
    global model, tokenizer, load_error
    try:
        # Usar modelo menor que cabe em 48GB
        model_id = "microsoft/DialoGPT-large"  # ~800MB - para teste inicial
        
        logger.info(f"🚀 Carregando modelo: {model_id}")
        
        # Carregar tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Carregar modelo
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        logger.info("✅ Modelo carregado com sucesso!")
        return True
        
    except Exception as e:
        error_msg = f"❌ ERRO ao carregar modelo: {str(e)}"
        logger.error(error_msg)
        load_error = traceback.format_exc()
        logger.error(f"Stack trace: {load_error}")
        return False

def initialize_model():
    """
    Inicialização sob demanda do modelo
    """
    global model, tokenizer
    if model is None or tokenizer is None:
        logger.info("🔄 Inicializando modelo...")
        return load_model()
    return True

def handler(job):
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
        
        # Extrair parâmetros
        prompt = job_input.get('prompt', 'Olá! Como posso ajudar você hoje?')
        max_tokens = job_input.get('max_tokens', 100)
        temperature = job_input.get('temperature', 0.7)
        
        logger.info(f"📝 Processando prompt: {prompt[:50]}...")
        
        # Tokenizar input
        inputs = tokenizer.encode(prompt, return_tensors="pt", truncate=True, max_length=512)
        inputs = inputs.to(model.device)
        
        # Gerar resposta
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decodificar resposta
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remover o prompt da resposta
        response = response[len(prompt):].strip()
        
        logger.info(f"✅ Resposta gerada: {len(response)} caracteres")
        
        return {
            "output": response,
            "status": "success",
            "model_info": {
                "model_id": "microsoft/DialoGPT-large",
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
                    "max_tokens": 5
                }
            }
            result = handler(test_job)
            logger.info(f"✅ Warmup concluído! Resultado: {result.get('status', 'unknown')}")
    except Exception as e:
        logger.warning(f"⚠️ Warmup falhou: {e}")

if __name__ == "__main__":
    logger.info("🚀 Iniciando RunPod Serverless...")
    
    # Fazer warmup
    warmup()
    
    # Iniciar servidor
    runpod.serverless.start({
        "handler": handler
    })
