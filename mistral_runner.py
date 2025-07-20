from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Model identifier to load
model_id = "mistralai/Mistral-7B-Instruct-v0.1"

# Configure 4-bit NF4 quantization
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,    # Enable 4-bit loading
    bnb_4bit_compute_dtype=torch.float16,    # Compute in float16
    bnb_4bit_use_double_quant=True,    # Use double quantization
    bnb_4bit_quant_type="nf4"    # Use NF4 quantization type
)

# Load a slow (SentencePiece) tokenizer, trusting remote code
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quant_config,
    torch_dtype=torch.float16
)

# Encodes the given prompt, generates an output response from the model
def generate_response(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

