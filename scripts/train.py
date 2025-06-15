import torch
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import get_peft_model
from scripts.tokenizer_utils import get_tokenizer, formatting_func
from scripts.peft_config import get_peft_config
import os
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import BitsAndBytesConfig

# Load .env file
load_dotenv()

def huggingface_auth():
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        login(token=token)
    else:
        login()  # fallback to ~/.huggingface/token

huggingface_auth()


# Model checkpoint (assuming you have access)
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# Load model
tokenizer = get_tokenizer(model_id)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # or "fp4"
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="bfloat16"
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)


# Apply LoRA
peft_config = get_peft_config()
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Load dataset
data = load_dataset("json", data_files="data/synthetic_data.jsonl")

# Tokenize
tokenized_data = data.map(lambda x: formatting_func(x, tokenizer), batched=False)

# Training args
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    warmup_steps=100,
    bf16=True,
    logging_dir="./logs",
    save_strategy="epoch"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"]
)

trainer.train()

# Save model
trainer.save_model("./final_model")