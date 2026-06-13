import os
import tiktoken
import torch
import numpy as np

# Use GPT-2 encoding which is pre-trained and has 50,257 vocabulary size
enc = tiktoken.get_encoding("gpt2")

def get_vocab_size():
    return enc.n_vocab

def tokenize_text(text):
    """
    Encodes text into token IDs.
    """
    return enc.encode(text)

def decode_tokens(token_ids):
    """
    Decodes token IDs back into text.
    """
    return enc.decode(token_ids)

def get_visual_tokens(text):
    """
    Helper for the web frontend to visualize how BPE tokenization works.
    Returns a list of dicts: [{'id': token_id, 'text': decoded_string, 'color_idx': color_index}]
    """
    token_ids = enc.encode(text)
    visual_tokens = []
    
    # We assign 8 different light colors in the frontend
    for i, token_id in enumerate(token_ids):
        # Decode individual token
        try:
            token_str = enc.decode([token_id])
        except Exception:
            token_str = str(token_id)
            
        visual_tokens.append({
            "id": int(token_id),
            "text": token_str,
            "color_idx": i % 8
        })
        
    return visual_tokens

def prepare_data(file_path, seq_len=128):
    """
    Reads a raw text file, tokenizes it, and saves inputs & targets to data/tokenized.pt.
    seq_len is the context length (default 128 for laptop compliance).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source text file not found at: {file_path}")
        
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    print(f"Loaded text file: {len(text):,} characters")
    
    # Tokenize the entire text
    tokens = enc.encode(text)
    tokens_tensor = torch.tensor(tokens, dtype=torch.long)
    print(f"Total tokens: {len(tokens_tensor):,}")
    
    # Slice text to construct input & target sequence pairs
    inputs = []
    targets = []
    
    # Step through tokens creating chunks of seq_len
    for i in range(0, len(tokens_tensor) - seq_len - 1, seq_len):
        inputs.append(tokens_tensor[i : i + seq_len])
        targets.append(tokens_tensor[i + 1 : i + seq_len + 1])
        
    if len(inputs) == 0:
        raise ValueError(f"Text dataset is too small to form a single sequence of length {seq_len}!")
        
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    save_path = os.path.join(os.path.dirname(file_path), "tokenized.pt")
    
    torch.save({
        "inputs": inputs,
        "targets": targets,
        "vocab_size": enc.n_vocab
    }, save_path)
    
    print(f"Saved tokenized dataset of shape {inputs.shape} to {save_path}")
    return save_path

if __name__ == "__main__":
    # Test script locally
    test_text = "Once upon a time, there was a tiny transformer model."
    print("Test Tokenization:")
    vt = get_visual_tokens(test_text)
    for t in vt:
        print(f"ID: {t['id']:5d} | Text: {repr(t['text']):15s} | Color group: {t['color_idx']}")
