# Toy Seq2Seq GRU + Attention (Mini Machine Translation)

Super simple encoder-decoder model in PyTorch for translating sentences (English ↔ Russian).  
Two versions: basic GRU seq2seq and seq2seq + attention.

Features:
- GRU encoder + GRUCell decoder
- Masked cross-entropy loss (ignores padding and tokens after </s>)
- Real corpus BLEU on dev set
- Scaled dot-product attention in the second version

Everything is in plain .py scripts — no notebooks, no complicated setup.

### Folder structure
src/
- 00_download_data.py - downloads tiny parallel data + vocab helper
- 01_tokenize_and_bpe.py - tokenization + trains/applies BPE
- 02_vocab_and_arrays.py - builds vocab + converts text → numpy arrays
- models_basic.py - plain seq2seq (no attention)
- models_attention.py - seq2seq + attention
- metrics.py - BLEU calculation
- train_basic.py - trains the basic model
- train_attention.py - trains the attention model

### Quick start (step by step)

1. Create virtual environment & install packages
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Download small parallel dataset
```bash
python src/00_download_data.py
```

3. Tokenize + train & apply BPE
```bash
python src/01_tokenize_and_bpe.py
```

4. Build vocabularies + check everything looks ok
```bash
python src/02_vocab_and_arrays.py
```

5. Train basic model (no attention)
```bash
python src/train_basic.py
```

6. Train model with attention
```bash
python src/train_attention.py
```

### What you'll see during training

- Loss and BLEU plots updated every 100 steps
- Random translation examples from dev set
- Final BLEU score at the end

### Important notes

- 25 000 training steps → takes a long time on CPU (hours!)
- Much faster with GPU / CUDA
- Code is intentionally kept simple and readable — it's for learning, not for beating Google Translate
- Made as a self-educational project