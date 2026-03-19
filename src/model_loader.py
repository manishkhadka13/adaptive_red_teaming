import os
import torch
import logging 
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig


log=logging.getLogger(__name__)

MODEL_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"

class ModelLoader:
    def __init__(self,model_id:str, precision:str):
        self.model_id=model_id
        self.precision=precision
        self.model=None
        self.tokenizer=None
        self._load()
    
    def _load(self):
        log.info("Loading %s at %s",self.model_id,self.precision)
        
        if self.precision=="fp16":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                
            )
        
        elif self.precision=="int8":
            self.model= AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                device_map="auto",
            )
        
        elif self.precision=="int4":
            self.model=AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True
                    ),
                device_map="auto"
            )
            
        else:
            raise ValueError(f"Unknown precision '{self.precision}'. Use fp16, int8, or int4.")
        
        self.tokenizer=AutoTokenizer.from_pretrained(self.model_id)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        vram = torch.cuda.memory_allocated() / 1e9
        log.info("Loaded. VRAM used: %.1f GB", vram)
        
        
    def generate(self, prompt: str, max_new_tokens: int = 1024, temperature: float = 0.0) -> str:
        """Send a plain-text prompt, get a plain-text response."""
        messages = [{"role": "user", "content": prompt}]
 
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
 
        inputs = self.tokenizer(
            formatted, return_tensors="pt"
        ).to(self.model.device)
 
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
 
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    
    def unload(self):
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache()
        log.info("Model unloaded. VRAM freed.")
            
        

