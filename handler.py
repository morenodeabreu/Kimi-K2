import runpod
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
tokenizer = None
load_error = None

def load_model():
    global model, tokenizer, load_error
    try:
        # Modelo MUITO pequeno - só 124M parâmetros
        model_id = "gpt2"  # ~500MB total
        
        logger.info(f"🚀 Carregando modelo: {model_id}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        logger.info("✅ Modelo GPT-2 carregado!")
        return True
        
    except Exception as e:
        load_error = f"❌ Erro: {str(e)}"
        logger.error(load_error)
        return False

def handler(job):
    global model, tokenizer, load_error
    
    try:
        if model is None and not load_model():
            return {"error": load_error, "status": "model_load_failed"}
        
        prompt = job['input'].get('prompt', 'Hello')
        max_tokens = job['input'].get('max_tokens', 50)
        
        logger.info(f"📝 Prompt: {prompt}")
        
        inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=100, truncation=True)
        inputs = inputs.to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        logger.info(f"✅ Resposta: {response}")
        
        return {
            "output": response,
            "status": "success"
        }
        
    except Exception as e:
        error_msg = f"❌ Handler error: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "status": "handler_error"}

if __name__ == "__main__":
    logger.info("🚀 Iniciando worker...")
    runpod.serverless.start({"handler": handler})
