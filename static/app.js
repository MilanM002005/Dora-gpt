// ── Application Initialization ──
document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    initThemeSwitcher();
    initStatusPolling();
    initTokenizer();
    initModelPresetSelector();
    initTrainingPanel();
    initChatPlayground();
});

const API_BASE = ""; 
let statusInterval = null;
let trainingStatusInterval = null;
let lossChart = null;

// ── Theme Switcher Manager ──
function initThemeSwitcher() {
    const themeBtns = document.querySelectorAll(".theme-btn");
    
    // Load cached theme or default to black
    const cachedTheme = localStorage.getItem("workspace-theme") || "black";
    setTheme(cachedTheme);

    themeBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const theme = btn.getAttribute("data-theme");
            setTheme(theme);
        });
    });
}

function setTheme(themeName) {
    // 1. Remove other theme classes from body
    document.body.className = `theme-${themeName}`;
    
    // 2. Update active status on buttons
    const themeBtns = document.querySelectorAll(".theme-btn");
    themeBtns.forEach(btn => {
        if (btn.getAttribute("data-theme") === themeName) {
            btn.classList.add("active");
        } else {
            btn.classList.remove("active");
        }
    });

    // 3. Save selection to local storage
    localStorage.setItem("workspace-theme", themeName);
}

// ── Navigation Controller ──
function initNavigation() {
    const navItems = document.querySelectorAll(".nav-item");
    const tabPanes = document.querySelectorAll(".tab-pane");
    
    navItems.forEach(item => {
        item.addEventListener("click", (e) => {
            e.preventDefault();
            const targetTab = item.getAttribute("data-tab");
            
            navItems.forEach(nav => nav.classList.remove("active"));
            item.classList.add("active");
            
            tabPanes.forEach(pane => {
                pane.classList.remove("active");
                if (pane.id === `tab-${targetTab}`) {
                    pane.classList.add("active");
                }
            });
        });
    });
}

// ── System Status & Dataset Downloader ──
function initStatusPolling() {
    const statusPill = document.getElementById("backend-status-pill");
    const statusText = document.getElementById("backend-status-text");
    
    const checkStatus = async () => {
        try {
            const res = await fetch(`${API_BASE}/api/status`);
            if (!res.ok) throw new Error();
            const data = await res.json();
            
            statusPill.className = "status-pill online";
            statusText.textContent = "Server Online";
            
            updateDashboardChecklist(data);
        } catch (err) {
            statusPill.className = "status-pill offline";
            statusText.textContent = "Server Offline";
        }
    };
    
    // Initial check and poll every 4 seconds
    checkStatus();
    statusInterval = setInterval(checkStatus, 4000);
    
    // Wire up download dataset
    const btnDownload = document.getElementById("btn-download-dataset");
    if (btnDownload) {
        btnDownload.addEventListener("click", async () => {
            btnDownload.disabled = true;
            btnDownload.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Downloading...`;
            try {
                const res = await fetch(`${API_BASE}/api/download_dataset`, { method: "POST" });
                const data = await res.json();
                if (data.success) {
                    alert("Dataset downloaded successfully!");
                } else {
                    alert("Download failed: " + data.detail);
                }
            } catch (err) {
                alert("Error downloading dataset.");
            } finally {
                checkStatus();
            }
        });
    }
    
    // Wire up tokenize dataset
    const btnTokenizeData = document.getElementById("btn-tokenize-dataset");
    if (btnTokenizeData) {
        btnTokenizeData.addEventListener("click", async () => {
            btnTokenizeData.disabled = true;
            btnTokenizeData.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Tokenizing...`;
            try {
                const res = await fetch(`${API_BASE}/api/tokenize_dataset`, { method: "POST" });
                const data = await res.json();
                if (data.success) {
                    alert("Dataset tokenized successfully! Saved as data/tokenized.pt");
                } else {
                    alert("Tokenization failed: " + data.detail);
                }
            } catch (err) {
                alert("Error tokenizing dataset.");
            } finally {
                checkStatus();
            }
        });
    }
}

function updateDashboardChecklist(data) {
    // 1. Python Environment Check
    const liPython = document.getElementById("check-python");
    if (liPython) {
        liPython.querySelector(".check-icon").className = "fa-solid fa-circle-check check-icon text-green";
        const badge = document.getElementById("python-ver-badge");
        if (badge) badge.textContent = `v${data.python_version || "3.11"}`;
    }
    
    // 2. PyTorch Library Check
    const liPytorch = document.getElementById("check-pytorch");
    if (liPytorch) {
        liPytorch.querySelector(".check-icon").className = "fa-solid fa-circle-check check-icon text-green";
        const badge = document.getElementById("pytorch-ver-badge");
        if (badge) badge.textContent = data.pytorch_version || "PyTorch";
    }
    
    // 3. Dataset Check
    const liDataset = document.getElementById("check-dataset");
    if (liDataset) {
        const btnDownload = document.getElementById("btn-download-dataset");
        const btnTokenizeData = document.getElementById("btn-tokenize-dataset");
        
        if (data.has_shakespeare) {
            liDataset.querySelector(".check-icon").className = "fa-solid fa-circle-check check-icon text-green";
            if (btnDownload) btnDownload.style.display = "none";
            if (btnTokenizeData) btnTokenizeData.disabled = false;
        } else {
            liDataset.querySelector(".check-icon").className = "fa-solid fa-circle-xmark check-icon text-red";
            if (btnDownload) {
                btnDownload.style.display = "inline-flex";
                btnDownload.disabled = false;
                btnDownload.innerHTML = `Download Dataset`;
            }
            if (btnTokenizeData) btnTokenizeData.disabled = true;
        }
    }
    
    // 4. Tokenized Check
    const liTokenized = document.getElementById("check-tokenized");
    if (liTokenized) {
        const btnTokenizeData = document.getElementById("btn-tokenize-dataset");
        if (data.has_tokenized) {
            liTokenized.querySelector(".check-icon").className = "fa-solid fa-circle-check check-icon text-green";
            if (btnTokenizeData) btnTokenizeData.style.display = "none";
        } else {
            liTokenized.querySelector(".check-icon").className = "fa-solid fa-circle-xmark check-icon text-red";
            if (btnTokenizeData) {
                btnTokenizeData.style.display = "inline-flex";
                if (data.has_shakespeare) {
                    btnTokenizeData.disabled = false;
                    btnTokenizeData.innerHTML = `Tokenize Dataset`;
                }
            }
        }
    }
    
    // 5. Checkpoint status
    const badgeCheckpoint = document.getElementById("model-status-badge");
    if (badgeCheckpoint) {
        if (data.has_checkpoint) {
            badgeCheckpoint.textContent = "Trained (Ready)";
            badgeCheckpoint.className = "badge text-green";
        } else {
            badgeCheckpoint.textContent = "Not Trained";
            badgeCheckpoint.className = "badge badge-warning";
        }
    }
    
    // 6. Untrained Chat warning banner visibility
    const warningBanner = document.getElementById("chat-untrained-warning");
    if (warningBanner) {
        if (data.has_checkpoint) {
            warningBanner.classList.add("hidden");
        } else {
            warningBanner.classList.remove("hidden");
        }
    }
}

// ── BPE Tokenizer Playground ──
function initTokenizer() {
    const inputArea = document.getElementById("tokenizer-input");
    const btnRun = document.getElementById("btn-run-tokenizer");
    const btnClear = document.getElementById("btn-clear-tokenizer");
    const outputContainer = document.getElementById("token-output-container");
    const spansContainer = document.getElementById("token-spans");
    
    const countChars = document.getElementById("stat-num-chars");
    const countTokens = document.getElementById("stat-num-tokens");
    const ratioStat = document.getElementById("stat-ratio");
    
    if (!btnRun) return;
    
    btnRun.addEventListener("click", async () => {
        const text = inputArea.value.trim();
        if (!text) {
            alert("Please input some text first.");
            return;
        }
        
        btnRun.disabled = true;
        btnRun.textContent = `Tokenizing...`;
        
        try {
            const res = await fetch(`${API_BASE}/api/tokenize`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: text })
            });
            const data = await res.json();
            
            spansContainer.innerHTML = "";
            
            if (data.tokens && data.tokens.length > 0) {
                data.tokens.forEach(tok => {
                    const span = document.createElement("span");
                    span.className = `tok-span tok-c${tok.color_idx}`;
                    span.textContent = tok.text;
                    span.title = `Token ID: ${tok.id}`;
                    spansContainer.appendChild(span);
                });
                
                const numChars = text.length;
                const numTokens = data.tokens.length;
                const ratio = (numChars / numTokens).toFixed(2);
                
                countChars.textContent = numChars;
                countTokens.textContent = numTokens;
                ratioStat.textContent = `${ratio} ch/tok`;
                
                outputContainer.classList.remove("hidden");
            } else {
                outputContainer.classList.add("hidden");
            }
        } catch (err) {
            alert("Error calling tokenization API.");
        } finally {
            btnRun.disabled = false;
            btnRun.textContent = `Tokenize text`;
        }
    });
    
    btnClear.addEventListener("click", () => {
        inputArea.value = "";
        outputContainer.classList.add("hidden");
        spansContainer.innerHTML = "";
    });
}

// ── Model Size Custom Presets ──
function initModelPresetSelector() {
    const presetCards = document.querySelectorAll(".preset-card");
    const dModelInput = document.getElementById("cfg-d-model");
    const nHeadsInput = document.getElementById("cfg-n-heads");
    const nLayersInput = document.getElementById("cfg-n-layers");
    const seqLenInput = document.getElementById("cfg-seq-len");
    const codeSnippet = document.getElementById("code-snippet");
    
    presetCards.forEach(card => {
        card.addEventListener("click", () => {
            presetCards.forEach(c => c.classList.remove("active"));
            card.classList.add("active");
            
            const presetVal = card.querySelector("input").value;
            let dModel = 128, nHeads = 4, nLayers = 4, seqLen = 128, dFF = 512;
            
            if (presetVal === "micro") {
                dModel = 256;
                nHeads = 8;
                nLayers = 6;
                seqLen = 256;
                dFF = 1024;
            } else if (presetVal === "mini") {
                dModel = 512;
                nHeads = 8;
                nLayers = 8;
                seqLen = 256;
                dFF = 2048;
            }
            
            if (dModelInput) {
                dModelInput.value = dModel;
                nHeadsInput.value = nHeads;
                nLayersInput.value = nLayers;
                seqLenInput.value = seqLen;
            }
            
            if (codeSnippet) {
                codeSnippet.textContent = `# Instantiate ${presetVal.toUpperCase()} GPT configuration
cfg = Config(
    vocab_size = 50257,
    seq_len    = ${seqLen},   # Context length
    d_model    = ${dModel},   # Vector size
    n_heads    = ${nHeads},     # Attention heads
    n_layers   = ${nLayers},     # Layers block count
    d_ff       = ${dFF}    # FeedForward (4 * d_model)
)
model = MiniGPT(cfg)`;
            }
        });
    });
}

// ── Training Dashboard Controller ──
function initTrainingPanel() {
    const btnStart = document.getElementById("btn-start-training");
    const runningActions = document.getElementById("training-running-actions");
    const btnPause = document.getElementById("btn-pause-training");
    const btnStop = document.getElementById("btn-stop-training");
    
    const inputLr = document.getElementById("train-lr");
    const inputEpochs = document.getElementById("train-epochs");
    const inputBatch = document.getElementById("train-batch");
    
    const consoleLogs = document.getElementById("training-console-output");
    
    // Elements for telemetry
    const statStatus = document.getElementById("training-status-text");
    const statEpoch = document.getElementById("metric-epoch");
    const statStep = document.getElementById("metric-step");
    const statLoss = document.getElementById("metric-loss");
    const statSpeed = document.getElementById("metric-speed");
    const statEta = document.getElementById("metric-eta");
    
    if (!btnStart) return;

    // Initialize Chart
    const ctx = document.getElementById("loss-chart").getContext("2d");
    lossChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Step Loss',
                data: [],
                borderColor: '#10b981',
                borderWidth: 2,
                backgroundColor: 'rgba(16, 185, 129, 0.05)',
                tension: 0.1,
                fill: true,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: '#888', maxTicksLimit: 10 }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.03)' },
                    ticks: { color: '#888' }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
    
    const writeLog = (msg) => {
        const timestamp = new Date().toLocaleTimeString();
        consoleLogs.innerHTML += `\n[${timestamp}] ${msg}`;
        consoleLogs.scrollTop = consoleLogs.scrollHeight;
    };
    
    const pollStatus = async () => {
        try {
            const res = await fetch(`${API_BASE}/api/train/status`);
            const data = await res.json();
            
            statStatus.textContent = data.status;
            
            if (data.status === "Training" || data.status === "Paused" || data.status === "Stopping") {
                btnStart.classList.add("hidden");
                runningActions.classList.remove("hidden");
                
                btnPause.textContent = data.status === "Paused" ? `Resume` : `Pause`;
                
                // Update metrics
                statEpoch.textContent = `${data.current_epoch} / ${data.total_epochs}`;
                statStep.textContent = `${data.current_step} / ${data.total_steps}`;
                statLoss.textContent = data.current_loss > 0 ? data.current_loss.toFixed(4) : "-";
                statSpeed.textContent = data.tokens_per_sec > 0 ? `${Math.round(data.tokens_per_sec)} tok/s` : "-";
                
                if (data.eta_seconds > 0) {
                    const mins = Math.floor(data.eta_seconds / 60);
                    const secs = data.eta_seconds % 60;
                    statEta.textContent = `${mins}m ${secs}s`;
                } else {
                    statEta.textContent = "-";
                }
                
                // Update Chart
                if (data.loss_history && data.loss_history.length > 0) {
                    const labels = Array.from({length: data.loss_history.length}, (_, i) => i + 1);
                    lossChart.data.labels = labels;
                    lossChart.data.datasets[0].data = data.loss_history;
                    lossChart.update('none'); 
                }
                
                // Log periodic loss to console
                if (data.current_step > 0 && data.current_step % 20 === 0) {
                    writeLog(`Step ${data.current_step}/${data.total_steps} | Loss: ${data.current_loss.toFixed(4)}`);
                }
            } else {
                // Done, Error, Idle, Stopped
                btnStart.classList.remove("hidden");
                runningActions.classList.add("hidden");
                
                if (trainingStatusInterval) {
                    clearInterval(trainingStatusInterval);
                    trainingStatusInterval = null;
                    writeLog(`Training finished. Final Status: ${data.status}`);
                    if (data.status === "Error") {
                        writeLog(`Detail: ${data.error_message}`);
                    }
                }
            }
        } catch (err) {
            console.error("Error polling training status", err);
        }
    };
    
    btnStart.addEventListener("click", async () => {
        const activePreset = document.querySelector('input[name="model-preset"]:checked').value;
        const lr = parseFloat(inputLr.value);
        const epochs = parseInt(inputEpochs.value);
        const batchSize = parseInt(inputBatch.value);
        
        writeLog(`Starting training task [Preset: ${activePreset.toUpperCase()}]...`);
        
        try {
            const res = await fetch(`${API_BASE}/api/train/start`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    preset: activePreset,
                    batch_size: batchSize,
                    epochs: epochs,
                    lr: lr
                })
            });
            const data = await res.json();
            
            if (data.success) {
                writeLog("PyTorch thread running!");
                lossChart.data.labels = [];
                lossChart.data.datasets[0].data = [];
                lossChart.update();
                
                if (trainingStatusInterval) clearInterval(trainingStatusInterval);
                trainingStatusInterval = setInterval(pollStatus, 1000);
            } else {
                writeLog(`Error: ${data.detail}`);
            }
        } catch (err) {
            writeLog("Failed to reach server training API.");
        }
    });
    
    btnPause.addEventListener("click", async () => {
        try {
            await fetch(`${API_BASE}/api/train/pause`, { method: "POST" });
            pollStatus();
        } catch (err) {
            writeLog("Pause command failed.");
        }
    });
    
    btnStop.addEventListener("click", async () => {
        try {
            writeLog("Requesting training termination...");
            await fetch(`${API_BASE}/api/train/stop`, { method: "POST" });
            pollStatus();
        } catch (err) {
            writeLog("Stop command failed.");
        }
    });
}

// ── ChatGPT/Claude Chat Playground Controller ──
function initChatPlayground() {
    const welcomeScreen = document.getElementById("chat-welcome-screen");
    const historyWrapper = document.getElementById("chat-history-wrapper");
    const chatHistory = document.getElementById("chat-history");
    const chatInput = document.getElementById("chat-prompt-input");
    const btnSend = document.getElementById("btn-send-chat");
    const btnNewChat = document.getElementById("btn-new-chat-sidebar");
    
    // Sliders
    const sliderTemp = document.getElementById("quick-temp");
    const valTemp = document.getElementById("quick-temp-val");
    const sliderTopk = document.getElementById("quick-topk");
    const valTopk = document.getElementById("quick-topk-val");
    const sliderLength = document.getElementById("quick-length");
    const valLength = document.getElementById("quick-length-val");
    
    if (!chatInput) return;

    // Sliders event listeners
    sliderTemp.addEventListener("input", () => valTemp.textContent = sliderTemp.value);
    sliderTopk.addEventListener("input", () => valTopk.textContent = sliderTopk.value);
    sliderLength.addEventListener("input", () => valLength.textContent = sliderLength.value);

    // Auto-grow input text area height
    chatInput.addEventListener("input", () => {
        chatInput.style.height = "auto";
        chatInput.style.height = (chatInput.scrollHeight) + "px";
        btnSend.disabled = chatInput.value.trim().length === 0;
    });

    const appendMessage = (sender, text) => {
        // Hide welcome screen when first message is sent
        welcomeScreen.classList.add("hidden");
        historyWrapper.classList.remove("hidden");

        const msg = document.createElement("div");
        msg.className = `chat-message ${sender}`;
        
        const avatar = document.createElement("div");
        avatar.className = "message-avatar";
        avatar.innerHTML = sender === "user" ? `U` : `<i class="fa-solid fa-bolt"></i>`;
        
        const content = document.createElement("div");
        content.className = "message-content";
        content.textContent = text;
        
        msg.appendChild(avatar);
        msg.appendChild(content);
        chatHistory.appendChild(msg);
        
        // Auto scroll to bottom
        historyWrapper.scrollTop = historyWrapper.scrollHeight;
        return msg;
    };
    
    const sendMessage = async () => {
        const text = chatInput.value.trim();
        if (!text) return;
        
        chatInput.value = "";
        chatInput.style.height = "auto";
        btnSend.disabled = true;
        
        appendMessage("user", text);
        
        // Generate placeholder for response
        const responseMsg = appendMessage("assistant", "Typing...");
        const responseContent = responseMsg.querySelector(".message-content");
        
        const temp = parseFloat(sliderTemp.value);
        const topK = parseInt(sliderTopk.value);
        const maxTokens = parseInt(sliderLength.value);
        
        try {
            const res = await fetch(`${API_BASE}/api/chat`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    prompt: text,
                    temperature: temp,
                    top_k: topK,
                    max_tokens: maxTokens
                })
            });
            const data = await res.json();
            
            if (res.ok) {
                const fullText = data.new_response || "No response generated.";
                responseContent.textContent = "";
                let i = 0;
                
                // Character typing speed
                const typeChar = () => {
                    if (i < fullText.length) {
                        responseContent.textContent += fullText.charAt(i);
                        i++;
                        historyWrapper.scrollTop = historyWrapper.scrollHeight;
                        setTimeout(typeChar, 15);
                    }
                };
                typeChar();
            } else {
                responseContent.textContent = `Error: ${data.detail || "Inference failed."}`;
            }
        } catch (err) {
            responseContent.textContent = "Failed to communicate with local PyTorch backend. Make sure the model has been trained!";
        }
    };
    
    btnSend.addEventListener("click", sendMessage);
    chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Wire up prompt suggestions click handlers
    const suggestions = document.querySelectorAll(".suggestion-card");
    suggestions.forEach(card => {
        card.addEventListener("click", () => {
            const prompt = card.getAttribute("data-prompt");
            chatInput.value = prompt;
            chatInput.focus();
            chatInput.style.height = (chatInput.scrollHeight) + "px";
            btnSend.disabled = false;
        });
    });

    // Wire up New Chat button
    btnNewChat.addEventListener("click", (e) => {
        e.preventDefault();
        chatHistory.innerHTML = "";
        welcomeScreen.classList.remove("hidden");
        historyWrapper.classList.add("hidden");
        chatInput.value = "";
        chatInput.style.height = "auto";
        btnSend.disabled = true;
    });
}
