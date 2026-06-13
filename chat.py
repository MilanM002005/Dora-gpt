import os
import torch
import tiktoken
from model.mini_gpt import MiniGPT, Config

def load_trained_model(checkpoint_path="checkpoints/best_model.pt"):
    """
    Loads checkpoint data and initializes the model.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Trained checkpoint not found at: {checkpoint_path}")
        
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Check if checkpoint is full dict or just weights state dict
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
        # Load config if saved, else fallback to defaults
        cfg = checkpoint.get("config", Config())
    else:
        state_dict = checkpoint
        # Fallback to Config
        cfg = Config()
        
    model = MiniGPT(cfg)
    model.load_state_dict(state_dict)
    model.eval()  # Inference mode
    
    return model

def generate_response(model, prompt, max_tokens=100, temp=0.8, top_k=40):
    enc = tiktoken.get_encoding("gpt2")
    # Encode input prompt
    tokens = enc.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long)
    
    # Generate text
    generated_idx = model.generate(idx, max_new_tokens=max_tokens, temperature=temp, top_k=top_k)
    
    # Decode and return
    return enc.decode(generated_idx[0].tolist())

def run_chat_cli():
    try:
        model = load_trained_model()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please train the model first by running train.py or using the web UI.")
        return

    print("\n💬 Mini-GPT (Laptop Edition) is ready!")
    print("Type a prompt and press Enter. Type 'exit' to quit.\n")
    
    while True:
        try:
            prompt = input("You: ").strip()
            if prompt.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
            if not prompt:
                continue
                
            print("GPT: ", end="", flush=True)
            full_output = generate_response(model, prompt, max_tokens=100, temp=0.8, top_k=40)
            
            # Show only the new text
            new_text = full_output[len(prompt):]
            print(new_text)
            print()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error during generation: {e}\n")

if __name__ == "__main__":
    run_chat_cli()
