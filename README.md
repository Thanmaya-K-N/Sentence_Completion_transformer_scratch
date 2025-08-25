# 🧠 Mini GPT Transformer on Penn Treebank (PTB)

This project implements a transformer-based GPT-style language model trained from scratch on the Penn Treebank dataset using PyTorch. It covers data processing, model architecture, training, evaluation, and saving for future fine-tuning or inference.

---

## 🚀 Features

- ✅ GPT-style decoder-only transformer
- ✅ Trained on Penn Treebank (PTB) dataset
- ✅ Implements:
  - Multi-head self-attention
  - Positional encoding
  - Masked causal attention
  - Token and position embeddings
  - Transformer blocks with LayerNorm and Feedforward
- ✅ Supports text generation with temperature control
- ✅ Model checkpoint saving and loading
- ✅ Easily extendable for fine-tuning on other datasets

---

## 📚 Dataset

- 📘 **Penn Treebank (PTB)** from `torchtext`
- Tokenized using a basic English tokenizer
- Converted to integer indices via `torchtext.vocab.Vocab`

---

## 🏗️ Model Architecture

- 4 GPT decoder blocks
- 256-dimensional token embeddings
- 4 attention heads per block
- 1024-dimensional feedforward layers
- Causal masking for autoregressive generation
- Total Parameters: **~8.29 million**

---

## 🔧 Training Details

- Optimizer: `AdamW`
- Learning Rate: `3e-4`
- Batch Size: `32`
- Block Size (context length): `64`
- Epochs: `5`
- Training steps per epoch: **1400** (custom-limited)
- Evaluated on 50 mini-batches from the validation set

---

## 💾 Saving and Loading

### ✅ Save Model
```python
torch.save({
    "model_state_dict": model.state_dict(),
    "vocab": vocab,
    "config": config_dict
}, "gpt_ptb_model.pth")
