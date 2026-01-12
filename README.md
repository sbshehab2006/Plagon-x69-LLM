# Plagon LLM (Plagon 69)

A custom Large Language Model built from scratch using PyTorch.

## Structure

```
Plagon-llm/
│
├── data/               # Data storage
│   ├── raw/            # Raw datasets
│   ├── cleaned/        # Cleaned datasets
│   └── tokenized/      # Tokenized binaries
│
├── tokenizer/          # Tokenizer training scripts
│
├── model/              # Model architecture (Transformer)
│
├── train/              # Training and Finetuning scripts
│
└── api/                # API server for inference
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training
```bash
python train/pretrain.py
```

### Server
```bash
uvicorn api.server:app --reload
```
