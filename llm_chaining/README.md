# LLM Fine-tuning Data Generator

This project generates synthetic training data for fine-tuning Large Language Models (LLMs) by creating high-quality question-answer pairs about machine learning topics.

## Features

- Generates technical questions and detailed answers about ML topics
- Uses both OpenAI's GPT-4 and Anthropic's Claude models
- Produces data in instruction-tuning format
- Handles concurrent processing of multiple topics
- Environment-based configuration

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root with your API keys:
```
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
```

## Usage

Run the script:
```bash
python prompt.py
```

The script will:
1. Generate technical questions for each ML topic
2. Create detailed answers using Claude
3. Format the Q&A pairs into JSON
4. Save the results to `synthetic_dataset.jsonl`

## Output Format

The generated data is saved in JSONL format, where each line contains a JSON object with:
- `instruction`: The technical question
- `response`: The detailed answer

## Project Structure

- `prompt.py`: Main script containing the data generation logic
- `requirements.txt`: Project dependencies
- `.env`: Configuration file for API keys (not tracked in git)

## Classes

- `LLMConfig`: Configuration management
- `LLMClient`: API client for LLM services
- `DataGenerator`: Core data generation logic
- `DatasetManager`: Dataset processing and storage 