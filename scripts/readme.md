# LLaMA3 Fine-Tuning Factory

### ✅ Steps to Run

1️⃣ Setup environment:

```bash
pip install -r requirements.txt
huggingface-cli login

Place your dataset:
data/synthetic_data.jsonl

Run fine-tuning:
python scripts/train.py

The fine-tuned model will be saved inside ./final_model