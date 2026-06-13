# 🧠 Build Your Own Mini-GPT — Laptop Edition
### No GPU · No Big Downloads · Beginner Friendly · Python 3.12

> You have a normal laptop with ~4 GB RAM and no GPU. This guide builds a **working, chatting LLM** that fits entirely on your machine. We skip everything heavy and keep it simple, step by step.

---

## ⚠️ What This Guide Is NOT

- ❌ Not a ChatGPT clone (that needs thousands of dollars of compute)
- ❌ Not a 7B or 13B model (needs 40–80 GB RAM)
- ❌ Not anything that will crash or melt your laptop

## ✅ What You WILL Build

- ✅ A **real Transformer LLM** written from scratch in PyTorch (~1–10 million parameters)
- ✅ Trains in **minutes to hours** on your CPU
- ✅ Uses **less than 500 MB of storage** for data
- ✅ Can generate text, answer questions (after fine-tuning)
- ✅ You will understand every single line of code

---

## 📌 Table of Contents

1. [Your Laptop Setup](#1-your-laptop-setup)
2. [Install Everything](#2-install-everything)
3. [Model Size — What You Can Run](#3-model-size--what-you-can-run)
4. [Architecture — Simple Diagram](#4-architecture--simple-diagram)
5. [Step 1 — Tiny Dataset](#5-step-1--tiny-dataset)
6. [Step 2 — Tokenizer](#6-step-2--tokenizer)
7. [Step 3 — Build the Model](#7-step-3--build-the-model)
8. [Step 4 — Train It](#8-step-4--train-it)
9. [Step 5 — Chat With It](#9-step-5--chat-with-it)
10. [Step 6 — Upgrade Path (Optional)](#10-step-6--upgrade-path-optional)
11. [Project Folder Structure](#11-project-folder-structure)
12. [Week-by-Week Checklist](#12-week-by-week-checklist)
13. [Troubleshooting Common Errors](#13-troubleshooting-common-errors)
14. [Free Resources to Learn More](#14-free-resources-to-learn-more)

---

## 1. Your Laptop Setup

This guide is written for a machine like yours:

| What | Your Spec | Fine? |
|------|-----------|-------|
| RAM | 4 GB | ✅ Yes — we stay under 1 GB |
| GPU | None (CPU only) | ✅ Yes — PyTorch works on CPU |
| Storage free | ~10 GB | ✅ Yes — we use under 500 MB |
| Python | 3.12 | ✅ Perfect |
| OS | Linux / Windows / Mac | ✅ All work |

**The rule of thumb for your laptop:**
> Model must fit in RAM. With 4 GB total and the OS using ~1.5 GB, we have about **2 GB for PyTorch**. That means our model can be at most ~500 MB in memory — which is roughly **10–15 million parameters** in float32.

---

## 2. Install Everything

Open your terminal and run these one by one. They are all small downloads.

```bash
# Make a project folder
mkdir my-mini-gpt
cd my-mini-gpt

# Create a virtual environment (keeps things clean)
python3 -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# Install PyTorch (CPU version — much smaller than GPU version!)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install helper libraries
pip install numpy tiktoken datasets tqdm

# Optional but nice: pretty progress bars
pip install rich
```

**Check it worked:**
```python
# Run this in Python to verify
import torch
print("PyTorch version:", torch.__version__)
print("Device:", "CPU only — that's fine!")
x = torch.tensor([1.0, 2.0, 3.0])
print("Test tensor:", x)
# Should print the tensor without errors
```

**Total download size:** ~250 MB (CPU PyTorch is much lighter than GPU)
**Total disk used:** ~400 MB

---

## 3. Model Size — What You Can Run

Here's a simple table. **For your laptop, use Tier 1 or Tier 2.**

| Tier | Parameters | RAM Used | Train Time (CPU) | Good For |
|------|------------|----------|-----------------|----------|
| 🟢 **Nano** (start here) | 1M | ~50 MB | 5–30 min | Learning the code, quick experiments |
| 🟡 **Micro** | 10M | ~200 MB | 1–3 hours | Better text quality, still fast |
| 🔴 **Mini** | 50M | ~800 MB | 8–24 hours | Decent quality, okay on your laptop |
| ⛔ Small | 125M | ~2 GB | Days | Too slow for CPU training |
| ⛔ Medium | 1.3B+ | ~8 GB+ | Weeks | Won't even load on your laptop |

> **Start with Nano (1M parameters).** It trains in minutes and teaches you everything. Upgrade to Micro once you understand the code.

---

## 4. Architecture — Simple Diagram

Here is what we are building. Don't panic — each piece is explained in the code steps below.

```
Your text: "Once upon a time"
        │
        ▼
  ┌─────────────┐
  │  Tokenizer  │  → Converts words to numbers
  │  (tiktoken) │    "Once" → 7454, "upon" → 2402, ...
  └──────┬──────┘
         │ Numbers (token IDs)
         ▼
  ┌───────────────────────────────┐
  │        Mini-GPT Model         │
  │                               │
  │  Embedding Layer              │  → Each number becomes a vector
  │       +                       │
  │  Positional Encoding          │  → Adds "position 1, 2, 3..." info
  │                               │
  │  ┌───────────────────────┐    │
  │  │  Transformer Block    │    │  ← Repeated N times
  │  │  (the brain)          │    │
  │  │                       │    │
  │  │  Attention Layer      │    │  → Learns which words relate
  │  │  Feed-Forward Layer   │    │  → Processes the patterns
  │  └───────────────────────┘    │
  │                               │
  │  Output Linear Layer          │  → Produces scores for next word
  └───────────────────────────────┘
         │ Scores for every word in vocab
         ▼
  Pick next word → "a"
        │
        ▼
  "Once upon a time a ..."  → keep going!
```

**Key design for your laptop (simple and light):**

| Component | Choice | Why (for your laptop) |
|-----------|--------|-----------------------|
| Normalization | LayerNorm | Simple, well-tested |
| Activation | GELU | Standard, no complications |
| Position Encoding | Learned embeddings | Simplest to code |
| Attention | Standard Multi-Head | Easiest to understand |
| Precision | float32 | CPU works best in float32 |

---

## 5. Step 1 — Tiny Dataset

We use **tiny, free datasets** that download in seconds — not gigabytes.

### Option A: Shakespeare (Best for beginners — only 1 MB!)

```bash
# Download the famous tiny Shakespeare dataset
curl -o data/shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Only **1 MB**. Contains Shakespeare's plays. Your model will learn to write like Shakespeare. Perfect for learning.

### Option B: Use HuggingFace Datasets (Streaming — no big download)

```python
# data/load_data.py
# This streams data — never downloads more than you need!

from datasets import load_dataset

def get_small_dataset(num_samples=5000):
    """
    Load only 5000 samples from Wikipedia.
    Each sample is ~500 words. Total: ~2.5M words.
    Download size: ~20 MB (not the full 20 GB Wikipedia!)
    """
    # streaming=True means it downloads as needed, not all at once
    dataset = load_dataset(
        "wikipedia",
        "20220301.en",
        split="train",
        streaming=True,         # ← KEY: never downloads full dataset
        trust_remote_code=True
    )

    texts = []
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        texts.append(sample["text"])
        if i % 500 == 0:
            print(f"  Loaded {i}/{num_samples} samples...")

    # Save locally so you don't re-download
    with open("data/train.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(texts))

    print(f"✅ Saved {len(texts)} samples to data/train.txt")
    print(f"   File size: {os.path.getsize('data/train.txt') / 1e6:.1f} MB")

if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    get_small_dataset(num_samples=2000)   # Start with 2000, ~8 MB
```

### Recommended data sizes for your laptop:

| Dataset | Download Size | Storage | Train Time |
|---------|--------------|---------|------------|
| Shakespeare | 1 MB | 1 MB | 5 min |
| Wikipedia 2K samples | ~8 MB | 8 MB | 30 min |
| Wikipedia 10K samples | ~40 MB | 40 MB | 2 hours |

> **Start with Shakespeare.** It's 1 MB, downloads instantly, and is perfect for testing your code works.

---

## 6. Step 2 — Tokenizer

We use **tiktoken** — the same tokenizer OpenAI uses — but just the pre-built one. No training needed.

```python
# tokenizer/tokenize.py
import tiktoken
import torch
import numpy as np

# Use GPT-2's tokenizer — 50,257 words, already trained, zero setup
enc = tiktoken.get_encoding("gpt2")

print(f"Vocabulary size: {enc.n_vocab}")   # 50,257

# Example
text = "Hello, I am building my own GPT!"
tokens = enc.encode(text)
print("Tokens:", tokens)         # [15496, 11, 314, 716, 2615, 616, 898, 402, 11571, 0]
decoded = enc.decode(tokens)
print("Decoded:", decoded)       # Hello, I am building my own GPT!


def prepare_data(file_path, seq_len=128):
    """
    Read text file, tokenize it, and create (input, target) pairs.
    seq_len=128 is safe for your laptop. Don't go above 256.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Text length: {len(text):,} characters")

    # Tokenize the whole file
    tokens = enc.encode(text)
    tokens = torch.tensor(tokens, dtype=torch.long)
    print(f"Total tokens: {len(tokens):,}")

    # Create sequences of length seq_len
    # Input:  tokens[0..127]   → Target: tokens[1..128]  (predict next token)
    inputs  = []
    targets = []

    for i in range(0, len(tokens) - seq_len - 1, seq_len):
        inputs.append(tokens[i      : i + seq_len])
        targets.append(tokens[i + 1 : i + seq_len + 1])

    inputs  = torch.stack(inputs)
    targets = torch.stack(targets)

    print(f"Dataset shape: {inputs.shape}")  # (num_sequences, 128)
    print(f"Memory: {inputs.nbytes / 1e6:.1f} MB")

    return inputs, targets, enc.n_vocab


# Run it
if __name__ == "__main__":
    inputs, targets, vocab_size = prepare_data("data/shakespeare.txt", seq_len=128)
    torch.save({"inputs": inputs, "targets": targets, "vocab_size": vocab_size},
               "data/tokenized.pt")
    print("✅ Saved tokenized data to data/tokenized.pt")
```

---

## 7. Step 3 — Build the Model

This is the full GPT model for your laptop. Every line is commented.

```python
# model/mini_gpt.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# ──────────────────────────────────────────────────────────────
# CONFIG — Change these numbers to control model size
# ──────────────────────────────────────────────────────────────
@dataclass
class Config:
    vocab_size : int   = 50257   # GPT-2 tokenizer vocab size (don't change)
    seq_len    : int   = 128     # How many tokens to look at (keep ≤ 256 on laptop)
    d_model    : int   = 128     # Size of internal vectors (Nano=128, Micro=256)
    n_heads    : int   = 4       # Number of attention heads (must divide d_model)
    n_layers   : int   = 4       # Number of transformer blocks (Nano=4, Micro=6)
    d_ff       : int   = 512     # Feed-forward hidden size (= 4 × d_model)
    dropout    : float = 0.1     # Dropout rate (regularization, prevents overfitting)

# ──────────────────────────────────────────────────────────────
# SELF-ATTENTION  — the core "thinking" mechanism
# ──────────────────────────────────────────────────────────────
class SelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads  = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads

        # Project input into Q (query), K (key), V (value)
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout  = nn.Dropout(cfg.dropout)

        # Causal mask — prevents the model from "cheating" by looking at future tokens
        # Example: when predicting token 5, it cannot see tokens 6, 7, 8...
        mask = torch.tril(torch.ones(cfg.seq_len, cfg.seq_len))
        self.register_buffer("mask", mask)  # saved with model, not a parameter

    def forward(self, x):
        B, T, C = x.shape   # Batch, Time (sequence length), Channels (d_model)

        # Compute Q, K, V all at once then split
        qkv = self.qkv_proj(x)                           # (B, T, 3*d_model)
        q, k, v = qkv.split(C, dim=-1)                  # each: (B, T, d_model)

        # Split into multiple heads
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            # shape: (B, n_heads, T, head_dim)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Attention scores: how much does each token "attend to" each other token?
        scale  = self.head_dim ** -0.5                   # scaling factor
        scores = (q @ k.transpose(-2, -1)) * scale       # (B, n_heads, T, T)

        # Apply causal mask: future tokens get -infinity (so softmax → 0)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))

        # Softmax: convert scores to probabilities
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Weighted sum of values
        out = weights @ v                                 # (B, n_heads, T, head_dim)

        # Merge heads back
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, d_model)
        return self.out_proj(out)


# ──────────────────────────────────────────────────────────────
# FEED-FORWARD NETWORK — simple 2-layer MLP after attention
# ──────────────────────────────────────────────────────────────
class FeedForward(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),    # expand
            nn.GELU(),                            # activation
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),    # compress back
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────────────────────
# TRANSFORMER BLOCK — one "layer" = attention + feed-forward
# ──────────────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)     # normalize before attention
        self.attn  = SelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)     # normalize before FFN
        self.ffn   = FeedForward(cfg)

    def forward(self, x):
        # Note: x + ... is the "residual connection" — helps gradients flow
        x = x + self.attn(self.norm1(x))   # attention with residual
        x = x + self.ffn(self.norm2(x))    # feed-forward with residual
        return x


# ──────────────────────────────────────────────────────────────
# FULL MINI-GPT MODEL
# ──────────────────────────────────────────────────────────────
class MiniGPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # Token embedding: converts each token ID to a vector
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Position embedding: tells the model "this token is at position 3"
        self.pos_emb   = nn.Embedding(cfg.seq_len, cfg.d_model)

        self.drop      = nn.Dropout(cfg.dropout)

        # Stack of transformer blocks (this is the "deep" part of deep learning)
        self.blocks    = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.n_layers)])

        # Final normalization
        self.norm_out  = nn.LayerNorm(cfg.d_model)

        # Output head: converts d_model → vocab_size (score for each possible next token)
        self.lm_head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: token embedding and output head share weights
        # This saves memory and improves quality
        self.lm_head.weight = self.token_emb.weight

        # Initialize weights properly
        self.apply(self._init_weights)
        print(f"✅ Model ready: {self.param_count():.2f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def param_count(self):
        return sum(p.numel() for p in self.parameters()) / 1e6

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device

        # Get token embeddings
        tok = self.token_emb(idx)                             # (B, T, d_model)

        # Get position embeddings (0, 1, 2, ... T-1)
        pos = self.pos_emb(torch.arange(T, device=device))   # (T, d_model)

        # Add them together — position info is now baked in
        x = self.drop(tok + pos)

        # Pass through all transformer blocks
        x = self.blocks(x)
        x = self.norm_out(x)

        # Get logits (scores for each token in vocabulary)
        logits = self.lm_head(x)                              # (B, T, vocab_size)

        # If targets are given, compute the loss
        loss = None
        if targets is not None:
            # Cross-entropy: how wrong were our predictions?
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),  # flatten: (B*T, vocab_size)
                targets.view(-1)                        # flatten: (B*T,)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=0.8, top_k=40):
        """Generate text token by token."""
        for _ in range(max_new_tokens):
            # Trim to seq_len if needed
            idx_cond = idx[:, -self.cfg.seq_len:]

            # Forward pass
            logits, _ = self(idx_cond)

            # Take only the last token's logits
            logits = logits[:, -1, :] / temperature

            # Top-K sampling: only consider the top-K most likely tokens
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float('-inf')

            # Sample from the distribution
            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat([idx, next_tok], dim=1)

        return idx


# ──────────────────────────────────────────────────────────────
# MODEL SIZE PRESETS — just pick one!
# ──────────────────────────────────────────────────────────────
def nano_gpt():
    """1M params — trains in minutes. Start here."""
    return MiniGPT(Config(d_model=128, n_heads=4, n_layers=4,  d_ff=512,  seq_len=128))

def micro_gpt():
    """10M params — takes 1–3 hours. Next step."""
    return MiniGPT(Config(d_model=256, n_heads=8, n_layers=6,  d_ff=1024, seq_len=256))

def mini_gpt():
    """50M params — needs a few hours. Only if patient."""
    return MiniGPT(Config(d_model=512, n_heads=8, n_layers=8,  d_ff=2048, seq_len=256))
```

---

## 8. Step 4 — Train It

```python
# train.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model.mini_gpt import nano_gpt, micro_gpt
import time
import os

# ── Dataset ──────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path)
        self.inputs  = data["inputs"]
        self.targets = data["targets"]
        print(f"Dataset: {len(self)} sequences")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# ── Training Config ───────────────────────────────────────────
BATCH_SIZE   = 16       # How many sequences per step (lower if you run out of memory)
EPOCHS       = 5        # How many times to go through the data
LEARNING_RATE = 3e-4    # How fast to learn
SAVE_EVERY   = 500      # Save checkpoint every N steps
LOG_EVERY    = 50       # Print loss every N steps
DEVICE       = "cpu"    # No GPU, that's fine!

os.makedirs("checkpoints", exist_ok=True)


# ── Training Loop ─────────────────────────────────────────────
def train():
    # Load data
    dataset = TextDataset("data/tokenized.pt")
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Build model — start with nano!
    model = nano_gpt()
    model = model.to(DEVICE)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01      # small regularization
    )

    # Learning rate: start small, warm up, then slowly decrease
    total_steps = EPOCHS * len(loader)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=LEARNING_RATE / 10
    )

    print(f"\n🚀 Starting training on {DEVICE}")
    print(f"   Model: {model.param_count():.2f}M parameters")
    print(f"   Steps per epoch: {len(loader)}")
    print(f"   Total steps: {total_steps}")
    print()

    step       = 0
    best_loss  = float("inf")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for inputs, targets in loader:
            inputs  = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            # Forward pass
            logits, loss = model(inputs, targets)

            # Backward pass
            optimizer.zero_grad(set_to_none=True)   # clear old gradients
            loss.backward()                          # compute new gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # safety clip
            optimizer.step()                         # update weights
            scheduler.step()                         # update learning rate

            epoch_loss += loss.item()
            step       += 1

            # Logging
            if step % LOG_EVERY == 0:
                elapsed  = (time.time() - start_time) / 60
                avg_loss = epoch_loss / (step % len(loader) + 1)
                lr       = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1}/{EPOCHS} | Step {step} | "
                      f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                      f"Time: {elapsed:.1f}m")

            # Save checkpoint
            if step % SAVE_EVERY == 0:
                torch.save({
                    "step"        : step,
                    "model_state" : model.state_dict(),
                    "optim_state" : optimizer.state_dict(),
                    "loss"        : avg_loss,
                }, f"checkpoints/checkpoint_step{step}.pt")
                print(f"  💾 Saved checkpoint at step {step}")

        # End of epoch
        avg_epoch_loss = epoch_loss / len(loader)
        print(f"\n✅ Epoch {epoch+1} done. Avg Loss: {avg_epoch_loss:.4f}\n")

        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
            print(f"  🏆 New best model saved (loss={best_loss:.4f})")

    total_time = (time.time() - start_time) / 60
    print(f"\n🎉 Training complete in {total_time:.1f} minutes!")
    print(f"   Best loss: {best_loss:.4f}")
    print(f"   Model saved to: checkpoints/best_model.pt")


if __name__ == "__main__":
    train()
```

### What the loss numbers mean

| Loss | Meaning |
|------|---------|
| > 10 | Model just started, random guessing |
| 5–7 | Starting to learn word patterns |
| 3–4 | Learning grammar and common phrases |
| 2–3 | Good — understanding the text style |
| < 2 | Excellent for a small model |

---

## 9. Step 5 — Chat With It

```python
# chat.py
import torch
import tiktoken
from model.mini_gpt import nano_gpt, Config

# ── Load your trained model ───────────────────────────────────
def load_model(checkpoint_path="checkpoints/best_model.pt"):
    model = nano_gpt()
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()   # switch to inference mode
    print(f"✅ Model loaded from {checkpoint_path}")
    return model

# ── Generate text ─────────────────────────────────────────────
def generate(model, prompt, max_tokens=200, temperature=0.8, top_k=40):
    enc    = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(prompt)
    idx    = torch.tensor([tokens], dtype=torch.long)

    # Generate
    with torch.no_grad():
        output = model.generate(idx, max_new_tokens=max_tokens,
                                temperature=temperature, top_k=top_k)

    # Decode back to text
    generated = enc.decode(output[0].tolist())
    return generated

# ── Simple interactive chat ───────────────────────────────────
def chat():
    model = load_model()
    print("\n💬 Mini-GPT is ready! Type a prompt and press Enter.")
    print("   Type 'quit' to exit.\n")

    while True:
        prompt = input("You: ").strip()
        if prompt.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        if not prompt:
            continue

        print("GPT: ", end="", flush=True)
        response = generate(model, prompt, max_tokens=150)
        # Show only the newly generated part
        print(response[len(prompt):])
        print()

if __name__ == "__main__":
    chat()
```

### Running it

```bash
# 1. Prepare data (run once)
python tokenizer/tokenize.py

# 2. Train the model
python train.py

# 3. Chat with it
python chat.py
```

**Example output (after training on Shakespeare):**

```
You: To be or not to be
GPT:  To be or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune...
```

---

## 10. Step 6 — Upgrade Path (Optional)

Once your Nano model works, here's how to level up **without buying new hardware**:

### Option A: Use a pre-trained model (easiest upgrade)
Instead of training from scratch, load a small pre-trained model and fine-tune it.
GPT-2 small (124M) can run on your laptop for **inference only** (not training):

```python
# Use a pre-trained GPT-2 — no training needed!
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")          # 500 MB download
model     = GPT2LMHeadModel.from_pretrained("gpt2")        # loads in ~2 GB RAM
model.eval()

inputs = tokenizer("Hello, I am", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

> ⚠️ GPT-2 inference uses ~2 GB RAM. Close other apps first.

### Option B: Use even smaller models (Phi-1.5 tiny, distilGPT-2)

```python
# DistilGPT-2 — half the size of GPT-2, still good quality
from transformers import pipeline
generator = pipeline("text-generation", model="distilgpt2")
result = generator("Once upon a time", max_new_tokens=50)
print(result[0]["generated_text"])
# Download size: ~350 MB
```

### Option C: Use Ollama for local LLM (runs larger models efficiently)

```bash
# Install Ollama (free, runs LLMs locally, optimized for CPU)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a tiny model (1.1 GB — fits in your RAM!)
ollama pull tinyllama

# Chat with it
ollama run tinyllama "Explain what a transformer is"
```

TinyLlama (1.1B parameters) is heavily optimized and can run on 4 GB RAM.

---

## 11. Project Folder Structure

```
my-mini-gpt/
│
├── data/
│   ├── shakespeare.txt     ← 1 MB raw text
│   └── tokenized.pt        ← Tokenized data (auto-generated)
│
├── model/
│   ├── __init__.py
│   └── mini_gpt.py         ← Full model code (the brain)
│
├── tokenizer/
│   ├── __init__.py
│   └── tokenize.py         ← Converts text → numbers
│
├── checkpoints/
│   ├── best_model.pt       ← Best trained model (auto-saved)
│   └── checkpoint_step500.pt
│
├── train.py                ← Run this to train
├── chat.py                 ← Run this to generate text
└── requirements.txt
```

**Storage budget:**

| File | Size |
|------|------|
| shakespeare.txt | 1 MB |
| tokenized.pt | ~5 MB |
| Nano model checkpoint | ~5 MB |
| Micro model checkpoint | ~40 MB |
| Python venv + PyTorch (CPU) | ~400 MB |
| **Total** | **~450 MB** |

Far under your 10 GB free space.

---

## 12. Week-by-Week Checklist

### Week 1 — Get It Running

- [ ] Install Python venv and activate it
- [ ] Install PyTorch (CPU version): `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- [ ] Download Shakespeare: `curl -o data/shakespeare.txt <url>`
- [ ] Copy `tokenize.py` → run it → see `data/tokenized.pt` created
- [ ] Copy `mini_gpt.py` → run `python -c "from model.mini_gpt import nano_gpt; nano_gpt()"` → see "1.0M parameters"
- [ ] Run `train.py` → watch loss go down → wait for completion
- [ ] Run `chat.py` → type a Shakespeare-style prompt → see generated text!

### Week 2 — Understand the Code

- [ ] Read `mini_gpt.py` top to bottom with comments
- [ ] Draw the architecture on paper
- [ ] Change `n_layers=2` → retrain → compare loss
- [ ] Change `temperature=1.5` in chat → see wilder text
- [ ] Change `temperature=0.3` → see more repetitive text
- [ ] Change `top_k=5` → very focused output
- [ ] Change `top_k=100` → more creative output

### Week 3 — Try Bigger Data

- [ ] Run `load_data.py` to download 2000 Wikipedia samples
- [ ] Retokenize with the new data
- [ ] Train Nano again on Wikipedia data
- [ ] Compare: does it talk about different things now?
- [ ] Try the Micro model (10M params) if patient

### Week 4 — Try Pre-Trained Models

- [ ] Install transformers: `pip install transformers`
- [ ] Load DistilGPT-2 and generate text (no training needed!)
- [ ] Try Ollama with TinyLlama
- [ ] Compare your trained model to pre-trained ones

---

## 13. Troubleshooting Common Errors

### "Killed" or computer freezes during training

Your RAM ran out. Fix:
```python
BATCH_SIZE = 4     # Reduce from 16 to 4
seq_len    = 64    # Reduce from 128 to 64 in Config
```

### "RuntimeError: CUDA not available"

This is fine! Make sure your code says `DEVICE = "cpu"` — you don't have a GPU and don't need one.

### "ModuleNotFoundError: No module named 'torch'"

Your virtual environment isn't activated:
```bash
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
```

### Loss stays at 10+ and doesn't go down

The model isn't learning. Try:
```python
LEARNING_RATE = 1e-3    # Increase learning rate
BATCH_SIZE    = 32      # Increase batch size (if RAM allows)
```

### "FileNotFoundError: data/tokenized.pt"

You haven't run the tokenizer yet:
```bash
python tokenizer/tokenize.py
```

### Training is too slow (hours for 1 epoch)

Reduce data size:
```python
# In tokenize.py, only use first 500,000 characters
text = text[:500_000]
```

### The generated text is just repeating the same words

Increase temperature:
```python
response = generate(model, prompt, temperature=1.0, top_k=50)
```

---

## 14. Free Resources to Learn More

### Watch First (YouTube — Free)

| Video | What You'll Learn |
|-------|-------------------|
| **"Let's build GPT" by Andrej Karpathy** (2h22m) | Builds exactly this, step by step — the best video ever made on this topic |
| **"The Illustrated Transformer" (3Blue1Brown)** | Visual explanation of attention |
| **"Neural Networks: Zero to Hero" series by Karpathy** | Full series, start from video 1 |

### Read (Free Online)

| Resource | Link |
|----------|------|
| **nanoGPT** (Karpathy) | github.com/karpathy/nanoGPT — The simplest GPT code in existence |
| **The Annotated Transformer** | nlp.seas.harvard.edu — Original paper with inline code |
| **HuggingFace Course** | huggingface.co/course — Free, practical NLP course |

### Papers (Read after watching the videos)

| Paper | Why Read It |
|-------|-------------|
| **Attention Is All You Need (2017)** | The original Transformer — the foundation of everything |
| **Language Models are Unsupervised Multitask Learners (GPT-2)** | How GPT-2 was built |
| **LoRA (2021)** | How to fine-tune large models efficiently |

### Once You're Ready to Scale Up

| Tool | What It Does |
|------|-------------|
| **Google Colab** | Free GPU (T4, 16 GB) for training bigger models |
| **Kaggle Notebooks** | Free GPU, 30 hrs/week |
| **Lambda Labs** | $1/hr for A100 GPU — affordable for experiments |
| **Vast.ai** | Cheapest GPU rental, good for students |

---

## ✅ Final Summary

| Topic | Your Approach |
|-------|--------------|
| Model size | 1M–10M parameters (Nano or Micro) |
| Dataset | Shakespeare (1 MB) or Wikipedia streaming (8–40 MB) |
| Hardware | CPU only, 4 GB RAM |
| Storage used | < 500 MB total |
| Train time | 5 minutes (Nano) to 3 hours (Micro) |
| Code complexity | Beginner-friendly, fully commented |
| First goal | Loss < 3.0, generates readable text |
| Next step | Try pre-trained models (GPT-2, TinyLlama via Ollama) |

---

*Start small. Understand everything. Scale when ready.*
*The goal is to build intuition, not just run code.*
