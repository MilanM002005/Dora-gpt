import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# ──────────────────────────────────────────────────────────────
# CONFIG — Hyperparameters controlling size and context window
# ──────────────────────────────────────────────────────────────
@dataclass
class Config:
    vocab_size : int   = 50257   # GPT-2 tokenizer vocab size (50,257)
    seq_len    : int   = 128     # Context length (how many historical tokens it reads)
    d_model    : int   = 128     # Vector dimension size (Nano=128, Micro=256)
    n_heads    : int   = 4       # Attention heads (must divide d_model)
    n_layers   : int   = 4       # Transformer blocks (Nano=4, Micro=6)
    d_ff       : int   = 512     # Feed-forward hidden dimension (= 4 * d_model)
    dropout    : float = 0.1     # Dropout rate to prevent overfitting

# ──────────────────────────────────────────────────────────────
# SELF-ATTENTION
# ──────────────────────────────────────────────────────────────
class SelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"
        
        self.n_heads  = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        
        # Project inputs to Query, Key, and Value vectors
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout  = nn.Dropout(cfg.dropout)
        
        # Causal mask - forces model to only look at past tokens
        mask = torch.tril(torch.ones(cfg.seq_len, cfg.seq_len))
        self.register_buffer("mask", mask)
        
    def forward(self, x):
        B, T, C = x.shape  # Batch, Sequence Length (Time), Channels (d_model)
        
        # Calculate QKV matrices
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=-1)
        
        # Reshape to split into heads: (B, T, n_heads, head_dim) -> transpose to (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores: Q @ K.T
        scale = self.head_dim ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale
        
        # Apply causal mask: replace future tokens with negative infinity so softmax zeros them out
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        
        # Normalise scores into probabilities
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        # Weighted sum of Values
        out = weights @ v
        
        # Merge head representations back: (B, n_heads, T, head_dim) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

# ──────────────────────────────────────────────────────────────
# FEED-FORWARD NETWORK
# ──────────────────────────────────────────────────────────────
class FeedForward(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout)
        )
        
    def forward(self, x):
        return self.net(x)

# ──────────────────────────────────────────────────────────────
# TRANSFORMER BLOCK
# ──────────────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn  = SelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn   = FeedForward(cfg)
        
    def forward(self, x):
        # LayerNorm applied BEFORE self-attention/feedforward (Pre-LN style)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# ──────────────────────────────────────────────────────────────
# FULL MINI-GPT MODEL
# ──────────────────────────────────────────────────────────────
class MiniGPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        
        # Map token ID to embedding vector
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # Learnable position embeddings for tokens in sequence
        self.pos_emb   = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.drop      = nn.Dropout(cfg.dropout)
        
        # Sequential stack of transformer blocks
        self.blocks    = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        
        # Final layer normalization
        self.norm_out  = nn.LayerNorm(cfg.d_model)
        
        # Output classification head mapping d_model back to vocab_size
        self.lm_head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        
        # Weight tying: share weights between input embeddings and output head projection
        self.lm_head.weight = self.token_emb.weight
        
        # Initialise weights
        self.apply(self._init_weights)
        print(f"MiniGPT model instantiated: {self.param_count():.2f}M parameters")
        
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
        
        # Combine token and position embeddings
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=device))
        x = self.drop(tok + pos)
        
        # Pass through the transformer blocks
        x = self.blocks(x)
        x = self.norm_out(x)
        
        # Calculate scores (logits) for the next token
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # Flatten tensors for cross-entropy computation
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1)
            )
            
        return logits, loss
        
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=0.8, top_k=40):
        """
        Generate sequence of next tokens given prompt token IDs.
        Supports temperature and Top-K filtering.
        """
        for _ in range(max_new_tokens):
            # Trim index context size to match maximum position embeddings limit
            idx_cond = idx[:, -self.cfg.seq_len:]
            
            # Predict logits
            logits, _ = self(idx_cond)
            
            # Divide last token logits by temperature
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            
            # Top-K sampling filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
                
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            
            # Concatenate to context sequence
            idx = torch.cat([idx, next_tok], dim=1)
            
        return idx

# ──────────────────────────────────────────────────────────────
# MODEL SIZE CONFIGURATIONS
# ──────────────────────────────────────────────────────────────
def nano_gpt():
    """1M parameter model — optimized for fast CPU learning (minutes)."""
    return MiniGPT(Config(d_model=128, n_heads=4, n_layers=4, d_ff=512, seq_len=128))

def micro_gpt():
    """10M parameter model — balanced quality vs speed (1–3 hours on CPU)."""
    return MiniGPT(Config(d_model=256, n_heads=8, n_layers=6, d_ff=1024, seq_len=256))

def mini_gpt():
    """50M parameter model — decent quality generation (requires overnight on CPU)."""
    return MiniGPT(Config(d_model=512, n_heads=8, n_layers=8, d_ff=2048, seq_len=256))
