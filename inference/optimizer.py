import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class InferenceOptimizer:
    def __init__(self, model_id="microsoft/phi-2"):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_quantized_model(self, load_in_4bit=True):
        """
        Loads the model with 4-bit or 8-bit quantization to optimize memory and speed.
        """
        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        
        print(f"Loading {self.model_id} with quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        return model, self.tokenizer

    def trace_model(self, model, sample_input):
        """
        Example of TorchScript tracing for further inference optimization.
        Note: Some models might require specific handling for tracing.
        """
        # Placeholder for tracing logic
        pass
