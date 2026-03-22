import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer


from osciquant.quantizers import UniformQuantizer
from osciquant.handler    import attach_weight_quantizers, toggle_quantization

log = logging.getLogger(__name__)


MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

BIT_WIDTHS = {
    "fp16": None,  
    "int8": 8,     
    "int4": 4,     
}


class ModelLoader:

    def __init__(self, model_id: str, precision: str):
      
        self.model_id  = model_id
        self.precision = precision
        self.model = None
        self.tokenizer = None
        self._load()

    def _load(self):
        log.info("Loading %s at %s ...", self.model_id, self.precision)

        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()

        
        bit_width = BIT_WIDTHS.get(self.precision)
        if bit_width is not None:
            log.info("Applying OsciQuant PTQ — %d-bit UniformQuantizer...", bit_width)

           
            attach_weight_quantizers(
                model=self.model,
                exclude_layers=[],
                quantizer=UniformQuantizer(bit_width=bit_width),
                enabled=False,
            )

            
            toggle_quantization(self.model, enabled=True)

            log.info("OsciQuant PTQ applied. Model is now %d-bit.", bit_width)
        else:
            log.info("FP16; no quantization applied.")

       
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        vram = torch.cuda.memory_allocated() / 1e9
        log.info("Loaded %s. VRAM: %.1f GB", self.model_id, vram)

    def generate(self, prompt: str, max_new_tokens: int = 1024,
                 temperature: float = 0.0) -> str:
        
        messages  = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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
                repetition_penalty=1.3,
                no_repeat_ngram_size=5
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()

    def unload(self):
       
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache()
        log.info("Unloaded %s.", self.model_id)