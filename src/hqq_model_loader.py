import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, HqqConfig


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
        self.fast_inference = fast_inference
        self.model = None
        self.tokenizer = None
        self._load()

    def _load(self):
        log.info("Loading %s at %s ...", self.model_id, self.precision)

        bit_width = BIT_WIDTHS.get(self.precision)
        
        if bit_width is not None:
         
            log.info("Applying HQQ PTQ — %d-bit quantization...", bit_width)
            
            quant_config = HqqConfig(
                nbits=bit_width,
                group_size=64,
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=quant_config
            )
            
            
            if self.fast_inference:
                if bit_width == 4:
                    log.info("Enabling 4-bit fast inference...")
                    try:
                        from hqq.utils.patching import prepare_for_inference
                        prepare_for_inference(self.model, backend="torchao_int4")
                        log.info("torchao_int4 enabled")
                    except Exception as e:
                        log.warning(f"torchao_int4 failed: {e}")
                
                elif bit_width == 8:
                    log.info("Enabling 8-bit optimizations...")
                    try:
                        from hqq.core.quantize import HQQBackend, HQQLinear
                        HQQLinear.set_backend(HQQBackend.PYTORCH)
                        log.info("PYTORCH backend enabled for 8-bit")
                    except Exception as e:
                        log.warning(f"8-bit optimization failed: {e}")
            
            log.info("HQQ PTQ applied. Model is now %d-bit.", bit_width)
            
        else:
            
            log.info("Disabling torch.compile for FP16...")
            os.environ['TORCH_COMPILE_DISABLE'] = '1'
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.disable = True
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            log.info("FP16 loaded (torch.compile disabled)")

        self.model.eval()
       
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        vram = torch.cuda.memory_allocated() / 1e9
        log.info("Loaded %s. VRAM: %.1f GB", self.model_id, vram)

    def generate(self, prompt: str, max_new_tokens: int = 1024,
             temperature: float = 0.3, seed: int = 42) -> str:

       
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