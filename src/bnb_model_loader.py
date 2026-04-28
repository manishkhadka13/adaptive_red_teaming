import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


log = logging.getLogger(__name__)

MODEL_ID = "google/gemma-2-9b-it"

BIT_WIDTHS = {
    "fp16": None,  
    "int8": 8,     
    "int4": 4,     
}


class ModelLoader:

    def __init__(self, model_id: str, precision: str, fast_inference: bool = True):
        self.model_id = model_id
        self.precision = precision
        self.model = None
        self.tokenizer = None
        self._load()

    def _load(self):

        log.info("Loading %s with precision=%s", self.model_id, self.precision)

        quant_config = None
        if self.precision == "int4":
            log.info("Using bitsandbytes 4-bit (FP4, degraded config)")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",              
                bnb_4bit_compute_dtype=torch.float16,   
                bnb_4bit_use_double_quant=False         

            )

        elif self.precision == "int8":
            log.info("Using bitsandbytes 8-bit")
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

        elif self.precision == "fp16":
            log.info("Using FP16 (baseline)")
        else:
            raise ValueError(f"Unsupported precision: {self.precision}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quant_config

        )

        self.model.eval()
       
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        vram = torch.cuda.memory_allocated() / 1e9
        log.info("Loaded %s. VRAM: %.1f GB", self.model_id, vram)

    def generate(self, prompt: str, max_new_tokens: int = 1024,
             temperature: float = 0.7, seed: int = 42) -> str:

       
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,

                
                do_sample=True,
                temperature=temperature,
                top_p=0.9,

                
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,

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
        log.info("Unloaded %s.", self.model_id)