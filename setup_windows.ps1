# ============================================================================
# setup_windows.ps1
# Windows PowerShell Setup Script for Constitutional AI + RLAIF Pipeline
# Run in PowerShell as Administrator or with execution policy set.
# ============================================================================

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  Constitutional AI + RLAIF Alignment System — Setup  " -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

# ── 0. Prerequisites check ────────────────────────────────────────────────────
Write-Host "`n[Step 0] Checking Python version..." -ForegroundColor Yellow
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python not found. Install Python 3.10 or 3.11 from https://python.org" -ForegroundColor Red
    exit 1
}

# ── 1. Create project folder ──────────────────────────────────────────────────
Write-Host "`n[Step 1] Creating project structure..." -ForegroundColor Yellow
$PROJECT = "constitutional_ai"
New-Item -ItemType Directory -Force -Path $PROJECT | Out-Null
New-Item -ItemType Directory -Force -Path "$PROJECT\critics" | Out-Null
New-Item -ItemType Directory -Force -Path "$PROJECT\data" | Out-Null
New-Item -ItemType Directory -Force -Path "$PROJECT\logs" | Out-Null
New-Item -ItemType Directory -Force -Path "$PROJECT\adapters" | Out-Null
New-Item -ItemType Directory -Force -Path "$PROJECT\reward_model_output" | Out-Null
Write-Host "  Directories created." -ForegroundColor Green

# ── 2. Create virtual environment ─────────────────────────────────────────────
Write-Host "`n[Step 2] Creating Python virtual environment..." -ForegroundColor Yellow
Set-Location $PROJECT
python -m venv .venv
Write-Host "  Virtual environment created at .venv\" -ForegroundColor Green

# ── 3. Activate virtual environment ──────────────────────────────────────────
Write-Host "`n[Step 3] Activating virtual environment..." -ForegroundColor Yellow
.\.venv\Scripts\Activate.ps1
Write-Host "  Virtual environment activated." -ForegroundColor Green

# ── 4. Upgrade pip ────────────────────────────────────────────────────────────
Write-Host "`n[Step 4] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

# ── 5. Install PyTorch (CPU build for compatibility) ─────────────────────────
Write-Host "`n[Step 5] Installing PyTorch (CPU)..." -ForegroundColor Yellow
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==0.17.2 --index-url https://download.pytorch.org/whl/cpu

# For CUDA 12.1 GPU support, comment the above and use:
# pip install torch==2.2.2 torchvision==0.17.2 torchaudio==0.17.2 --index-url https://download.pytorch.org/whl/cu121

# ── 6. Install remaining requirements ─────────────────────────────────────────
Write-Host "`n[Step 6] Installing project dependencies..." -ForegroundColor Yellow
pip install `
    transformers==4.44.2 `
    datasets==2.20.0 `
    tokenizers==0.19.1 `
    huggingface-hub==0.24.5 `
    accelerate==0.33.0 `
    safetensors==0.4.3 `
    peft==0.12.0 `
    trl==0.10.1 `
    scikit-learn==1.5.1 `
    numpy==1.26.4 `
    scipy==1.13.1 `
    streamlit==1.37.1 `
    pandas==2.2.2 `
    altair==5.3.0 `
    python-dotenv==1.0.1 `
    tqdm==4.66.4 `
    psutil==6.0.0

Write-Host "  All dependencies installed." -ForegroundColor Green

# ── 7. Verify installation ────────────────────────────────────────────────────
Write-Host "`n[Step 7] Verifying installation..." -ForegroundColor Yellow
python -c "import torch; import transformers; import peft; import trl; import streamlit; print('All packages OK')"
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Verification passed." -ForegroundColor Green
} else {
    Write-Host "  Verification failed — check errors above." -ForegroundColor Red
}

Write-Host "`n======================================================" -ForegroundColor Cyan
Write-Host "  Setup complete! Now run the pipeline steps below:  " -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host @"

STEP A — Generate Preference Dataset (takes 5-15 min on CPU):
  python generate_preferences.py

STEP B — Train Reward Model:
  python reward_model.py

STEP C — DPO Fine-Tuning with LoRA:
  python train_dpo.py

STEP D — Run Evaluation Suite:
  python eval.py

STEP E — Launch Streamlit Chat App:
  streamlit run app.py
  (then open http://localhost:8501 in your browser)

NOTE: First run downloads ~500MB HuggingFace model weights.
      Set HF_HOME env var to control cache location:
      `$env:HF_HOME = "D:\hf_cache"

"@ -ForegroundColor White
