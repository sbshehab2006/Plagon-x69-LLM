# Plagon LLM (Plagon 69)

A custom, efficient Large Language Model (LLM) built from scratch using PyTorch. This project covers the entire pipeline from data collection and tokenization to pretraining, finetuning, and inference.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sbshehab2006/Plagon-x69-LLM/blob/main/Plagon_LLM_Colab.ipynb)

## 🚀 Features

- **Custom Transformer Architecture**: Built with PyTorch, implementing modern transformer features.
- **Data Pipeline**: Tools for downloading, cleaning, and tokenizing large datasets (e.g., TinyStories, Kaggle).
- **Two-Stage Training**:
  - **Stage 1 (Pretraining)**: Learning language fundamentals from raw text.
  - **Stage 2 (Finetuning)**: Instruction tuning for conversational capabilities.
- **Efficient Inference**: Real-time chat interface via CLI.

## 📂 Project Structure

```
Plagon-llm/
│
├── data/                  # Data processing pipeline
│   ├── download_*.py      # Scripts to download datasets
│   ├── tokenize_*.py      # Scripts to tokenize data
│   ├── raw/               # Raw downloaded text
│   └── tokenized_*/       # Binary tokenized data for training
│
├── model/                 # Neural Network Architecture
│   └── model.py           # Plagon model definition
│
├── train/                 # Training Scripts
│   ├── pretrain.py        # Stage 1: Pretraining on massive text
│   ├── finetune.py        # Stage 2: Instruction finetuning
│   └── verify.py          # Validate model output quality
│
├── checkpoints/           # Saved model weights (.pt files)
│
├── demo_chat.py           # Interactive Chat CLI
└── plagon_tokenizer.json  # Custom trained Tokenizer
```

## 🛠️ Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sbshehab2006/Plagon-x69-LLM.git
    cd Plagon-x69-LLM
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/MacOS
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Requires PyTorch (with CUDA support recommended for training).*

## 🏃‍♂️ How to Run

### 1. Data Preparation
Before training, you need to download and tokenize data.

**For Stage 1 (TinyStories):**
```bash
# Download dataset
python data/download_tinystories.py

# Tokenize dataset
python data/tokenize_stage1.py
```

### 2. Training the Model

**Stage 1: Pretraining**
Trains the model on the tokenized data to learn English grammar and world knowledge.
```bash
python train/pretrain.py
```

**Stage 2: Finetuning**
Trains the model to follow instructions and act like a chatbot.
```bash
python train/finetune.py
```

### 3. Chat with Plagon
Run the interactive demo to talk to your trained model.
```bash
python demo_chat.py
```

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License
This project is open source.
