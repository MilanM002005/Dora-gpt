import os
import time
import torch
import threading
from torch.utils.data import Dataset, DataLoader
from model.mini_gpt import MiniGPT, Config, nano_gpt, micro_gpt, mini_gpt

class TextDataset(Dataset):
    def __init__(self, tokenized_path):
        if not os.path.exists(tokenized_path):
            raise FileNotFoundError(f"Tokenized dataset not found at {tokenized_path}")
        data = torch.load(tokenized_path)
        self.inputs = data["inputs"]
        self.targets = data["targets"]
        self.vocab_size = data["vocab_size"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

class TrainingManager:
    def __init__(self):
        self.is_running = False
        self.should_stop = False
        self.is_paused = False
        
        self.status = "Idle" # Idle, Tokenizing, Training, Paused, Completed, Error
        self.error_message = ""
        
        # Metrics
        self.current_step = 0
        self.total_steps = 0
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_loss = 0.0
        self.loss_history = []  # List of floats
        self.learning_rate = 0.0
        self.tokens_per_sec = 0.0
        self.eta_seconds = 0
        self.start_time = 0.0
        self.pause_time_accumulated = 0.0
        
        # Thread
        self.thread = None
        self.lock = threading.Lock()

    def get_status(self):
        with self.lock:
            return {
                "status": self.status,
                "error_message": self.error_message,
                "current_step": self.current_step,
                "total_steps": self.total_steps,
                "current_epoch": self.current_epoch,
                "total_epochs": self.total_epochs,
                "current_loss": self.current_loss,
                "loss_history": self.loss_history[-100:],  # Return last 100 values to avoid huge payload
                "learning_rate": self.learning_rate,
                "tokens_per_sec": self.tokens_per_sec,
                "eta_seconds": self.eta_seconds,
            }

    def start(self, dataset_path, preset="nano", batch_size=16, epochs=5, lr=3e-4):
        with self.lock:
            if self.is_running:
                return False, "Training is already running!"
            
            self.is_running = True
            self.should_stop = False
            self.is_paused = False
            self.status = "Training"
            self.error_message = ""
            self.current_step = 0
            self.current_loss = 0.0
            self.loss_history = []
            self.learning_rate = lr
            self.tokens_per_sec = 0.0
            self.eta_seconds = 0
            self.current_epoch = 0
            self.total_epochs = epochs
            self.pause_time_accumulated = 0.0
            
            self.thread = threading.Thread(
                target=self._run_training,
                args=(dataset_path, preset, batch_size, epochs, lr),
                daemon=True
            )
            self.thread.start()
            return True, "Training started successfully."

    def stop(self):
        with self.lock:
            if not self.is_running:
                return False, "Training is not running."
            self.should_stop = True
            self.status = "Stopping"
            return True, "Stopping training thread..."

    def toggle_pause(self):
        with self.lock:
            if not self.is_running:
                return False, "Training is not running."
            self.is_paused = not self.is_paused
            self.status = "Paused" if self.is_paused else "Training"
            return True, f"Training {'paused' if self.is_paused else 'resumed'}."

    def _run_training(self, dataset_path, preset, batch_size, epochs, lr):
        try:
            # 1. Load tokenized dataset
            dataset = TextDataset(dataset_path)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # 2. Select Model Configuration based on preset
            if preset == "nano":
                model = nano_gpt()
            elif preset == "micro":
                model = micro_gpt()
            elif preset == "mini":
                model = mini_gpt()
            else:
                raise ValueError(f"Unknown preset config: {preset}")
                
            device = "cpu"
            model = model.to(device)
            
            # 3. Setup optimizer & scheduler
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
            total_steps = epochs * len(loader)
            
            with self.lock:
                self.total_steps = total_steps
                
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=lr/10
            )
            
            os.makedirs("checkpoints", exist_ok=True)
            
            # Metrics timing
            self.start_time = time.time()
            step_count = 0
            
            for epoch in range(epochs):
                with self.lock:
                    self.current_epoch = epoch + 1
                    
                model.train()
                
                for idx, (inputs, targets) in enumerate(loader):
                    # Check Pause loop
                    while True:
                        with self.lock:
                            if self.should_stop:
                                break
                            paused = self.is_paused
                        if not paused:
                            break
                        time.sleep(0.5)
                        
                    with self.lock:
                        if self.should_stop:
                            break
                            
                    step_start_time = time.time()
                    
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    # Forward
                    logits, loss = model(inputs, targets)
                    
                    # Backward
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    
                    step_count += 1
                    step_time = time.time() - step_start_time
                    
                    # Save stats
                    current_loss = loss.item()
                    current_lr = scheduler.get_last_lr()[0]
                    tokens_processed = batch_size * model.cfg.seq_len
                    tokens_per_sec = tokens_processed / max(step_time, 1e-6)
                    
                    with self.lock:
                        self.current_step = step_count
                        self.current_loss = current_loss
                        self.loss_history.append(current_loss)
                        self.learning_rate = current_lr
                        self.tokens_per_sec = tokens_per_sec
                        
                        # Calculate ETA
                        steps_remaining = total_steps - step_count
                        avg_step_time = (time.time() - self.start_time) / step_count
                        self.eta_seconds = int(steps_remaining * avg_step_time)
                        
                    # Periodic checkpointing
                    if step_count % 100 == 0 or step_count == total_steps:
                        torch.save({
                            "step": step_count,
                            "model_state": model.state_dict(),
                            "loss": current_loss,
                            "config": model.cfg
                        }, "checkpoints/best_model.pt")
                
                with self.lock:
                    if self.should_stop:
                        break
                        
            with self.lock:
                if self.should_stop:
                    self.status = "Stopped"
                    print("Training stopped manually.")
                else:
                    self.status = "Completed"
                    # Final Save
                    torch.save(model.state_dict(), "checkpoints/best_model.pt")
                    print("Training completed successfully!")
                    
        except Exception as e:
            import traceback
            error_str = f"Error: {str(e)}\n{traceback.format_exc()}"
            print(error_str)
            with self.lock:
                self.status = "Error"
                self.error_message = str(e)
        finally:
            with self.lock:
                self.is_running = False

# Global Training Manager Singleton
training_manager = TrainingManager()
