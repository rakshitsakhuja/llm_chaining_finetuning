from transformers import AutoTokenizer

def get_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # Important for batching
    return tokenizer

def formatting_func(example, tokenizer):
    prompt = f"<|start_of_turn|>user\n{example['instruction']}<|end_of_turn|>\n<|start_of_turn|>assistant\n{example['response']}<|end_of_turn|>"
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=2048)