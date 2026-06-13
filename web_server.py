import os
import urllib.request
import threading
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from tokenizer.tokenize import get_visual_tokens, prepare_data, get_vocab_size, decode_tokens, tokenize_text
from model.mini_gpt import MiniGPT, Config
from train import training_manager
from chat import load_trained_model, generate_response

app = FastAPI(title="Mini-GPT Laptop Edition Dashboard")

# CORS middleware for local development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cached model for inference
inference_model = None
inference_model_path = ""

# Request models
class TokenizeRequest(BaseModel):
    text: str

class TrainStartRequest(BaseModel):
    preset: str = "nano" # nano, micro, mini
    batch_size: int = 16
    epochs: int = 5
    lr: float = 0.0003

class ChatRequest(BaseModel):
    prompt: str
    temperature: float = 0.8
    top_k: int = 40
    max_tokens: int = 100

# API Routes
@app.get("/api/status")
def get_system_status():
    """
    Check the status of system dependencies, datasets, and training.
    """
    import sys
    
    pytorch_version = torch.__version__
    has_shakespeare = os.path.exists("data/shakespeare.txt")
    has_tokenized = os.path.exists("data/tokenized.pt")
    has_checkpoint = os.path.exists("checkpoints/best_model.pt")
    
    # Check dataset size
    data_info = {}
    if has_shakespeare:
        size_mb = os.path.getsize("data/shakespeare.txt") / 1e6
        data_info["raw_size_mb"] = round(size_mb, 2)
    if has_tokenized:
        size_mb = os.path.getsize("data/tokenized.pt") / 1e6
        data_info["tokenized_size_mb"] = round(size_mb, 2)
        
    return {
        "python_version": sys.version.split()[0],
        "pytorch_version": pytorch_version,
        "device": "cpu",
        "has_shakespeare": has_shakespeare,
        "has_tokenized": has_tokenized,
        "has_checkpoint": has_checkpoint,
        "data_info": data_info,
        "training_active": training_manager.is_running
    }

@app.post("/api/download_dataset")
def download_dataset():
    """
    Downloads the 1 MB Shakespeare dataset.
    """
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    dest_dir = "data"
    dest_path = os.path.join(dest_dir, "shakespeare.txt")
    
    try:
        os.makedirs(dest_dir, exist_ok=True)
        print(f"Downloading dataset from {url} to {dest_path}...")
        
        # Stream download using urllib to avoid dependencies
        urllib.request.urlretrieve(url, dest_path)
        
        return {"success": True, "message": "Shakespeare dataset downloaded successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download dataset: {str(e)}")

@app.post("/api/tokenize")
def api_tokenize(req: TokenizeRequest):
    """
    Tokenizes text and returns detailed token boundaries for visualization.
    """
    if not req.text:
        return {"tokens": []}
    try:
        tokens = get_visual_tokens(req.text)
        return {"tokens": tokens}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tokenize_dataset")
def api_tokenize_dataset():
    """
    Tokenizes the downloaded shakespeare.txt and creates data/tokenized.pt.
    """
    source_path = "data/shakespeare.txt"
    if not os.path.exists(source_path):
        raise HTTPException(status_code=400, detail="Dataset shakespeare.txt not found. Please download it first.")
        
    try:
        # Generate tokenized.pt (128 context length default)
        save_path = prepare_data(source_path, seq_len=128)
        return {"success": True, "message": f"Dataset tokenized and saved to {save_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tokenization failed: {str(e)}")

@app.post("/api/train/start")
def api_train_start(req: TrainStartRequest):
    """
    Start background training thread.
    """
    tokenized_path = "data/tokenized.pt"
    if not os.path.exists(tokenized_path):
        raise HTTPException(status_code=400, detail="Tokenized dataset data/tokenized.pt not found. Run tokenization first.")
        
    success, message = training_manager.start(
        dataset_path=tokenized_path,
        preset=req.preset,
        batch_size=req.batch_size,
        epochs=req.epochs,
        lr=req.lr
    )
    
    if not success:
        raise HTTPException(status_code=400, detail=message)
        
    return {"success": True, "message": message}

@app.post("/api/train/stop")
def api_train_stop():
    """
    Stop background training.
    """
    success, message = training_manager.stop()
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"success": True, "message": message}

@app.post("/api/train/pause")
def api_train_pause():
    """
    Toggle pause/resume background training.
    """
    success, message = training_manager.toggle_pause()
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"success": True, "message": message}

@app.get("/api/train/status")
def api_train_status():
    """
    Get live training metrics and history.
    """
    return training_manager.get_status()

@app.post("/api/chat")
def api_chat(req: ChatRequest):
    """
    Run inference with the trained model.
    """
    global inference_model, inference_model_path
    
    checkpoint_path = "checkpoints/best_model.pt"
    if not os.path.exists(checkpoint_path):
        print("No trained checkpoint found. Initializing default fallback model with random weights so chat works immediately...")
        try:
            os.makedirs("checkpoints", exist_ok=True)
            from model.mini_gpt import nano_gpt
            fallback_model = nano_gpt()
            torch.save(fallback_model.state_dict(), checkpoint_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize fallback model: {str(e)}")
        
    try:
        # Load/reload model if not cached or model file updated
        # (check modification time to see if we need to reload)
        mtime = os.path.getmtime(checkpoint_path)
        cache_key = f"{checkpoint_path}_{mtime}"
        
        if inference_model is None or inference_model_path != cache_key:
            inference_model = load_trained_model(checkpoint_path)
            inference_model_path = cache_key
            
        # Run generation
        response = generate_response(
            inference_model,
            req.prompt,
            max_tokens=req.max_tokens,
            temp=req.temperature,
            top_k=req.top_k
        )
        
        # Extract response portion only
        response_text = response[len(req.prompt):]
        
        return {
            "prompt": req.prompt,
            "full_response": response,
            "new_response": response_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# Fallback routes to serve the static frontend
@app.get("/")
def serve_index():
    index_path = "static/index.html"
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(status_code=404, content={"message": "Frontend static/index.html not found"})

# Mount the static directory for CSS, JS, assets
# We will create static/ dir, check its existence first
@app.on_event("startup")
def startup_event():
    os.makedirs("static", exist_ok=True)
    app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    print("Starting FastAPI development server...")
    uvicorn.run("web_server:app", host="127.0.0.1", port=8000, reload=True)
