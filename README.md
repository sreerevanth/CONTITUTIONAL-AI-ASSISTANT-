# Constitutional AI + RLAIF Alignment System
### Multi-Agent Critique · Reward Modeling · DPO Fine-Tuning

---
<img width="1596" height="759" alt="image" src="https://github.com/user-attachments/assets/9daefc90-264a-4cf8-98cb-c77884838046" />
---
## Architecture Overview

```
User Prompt
    │
    ▼
┌─────────────────────────────────────────────────┐
│             BASE LANGUAGE MODEL                  │
│          (OPT-125M / configurable)               │
│    Generates N candidate responses               │
└─────────────────────┬───────────────────────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
   ┌────────────┐ ┌──────────┐ ┌──────────────┐
   │  SAFETY    │ │  ETHICS  │ │   QUALITY    │
   │  CRITIC    │ │  CRITIC  │ │   CRITIC     │
   │ w=0.40     │ │ w=0.35   │ │  w=0.25      │
   └─────┬──────┘ └────┬─────┘ └──────┬───────┘
         └─────────────┼──────────────┘
                       ▼
          ┌────────────────────────┐
          │      AGGREGATOR        │
          │  R = Σ(w_i × score_i)  │
          │  label: chosen/rejected│
          └──────────┬─────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   ┌────────────┐        ┌────────────┐
   │  REWARD    │        │    DPO     │
   │   MODEL    │        │  TRAINER   │
   │ (MSE regr) │        │  (LoRA)    │
   └────────────┘        └────────────┘
```

---

## File Tree

```
constitutional_ai/
├── constitution.txt              # Constitutional rules (SAFETY/ETHICS/QUALITY)
├── config.py                     # All hyperparameters & paths
├── logger.py                     # JSON structured logging
├── aggregator.py                 # RLAIF reward aggregation core
├── generate_preferences.py       # Synthetic preference dataset generator
├── reward_model.py               # Reward model training + inference
├── train_dpo.py                  # DPO + LoRA fine-tuning pipeline
├── eval.py                       # Alignment evaluation suite
├── app.py                        # Streamlit aligned chat UI
├── requirements.txt
├── setup_windows.ps1
├── critics/
│   ├── __init__.py
│   ├── base_critic.py            # Abstract base class for all critics
│   ├── safety_critic.py          # Harm/violence/jailbreak detection
│   ├── ethics_critic.py          # Bias/fairness/privacy detection
│   └── quality_critic.py         # Helpfulness/coherence scoring
├── data/
│   └── preferences.json          # Generated preference pairs
├── logs/
│   ├── decisions.jsonl           # Per-decision audit trail
│   └── eval_report.json          # Evaluation results
├── adapters/
│   └── dpo_lora_adapter/         # Saved LoRA adapter weights
└── reward_model_output/          # Saved reward model checkpoint
```

---

## Windows Setup (PowerShell — Copy-Paste)

### Prerequisites
- Python 3.10 or 3.11 (from https://python.org)
- 8GB RAM minimum, 16GB recommended
- ~2GB disk space for model weights

```powershell
# ── Clone / place project files ───────────────────────────────────────────────
cd C:\Projects

# ── Create venv ───────────────────────────────────────────────────────────────
cd constitutional_ai
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# ── Install PyTorch (CPU) ─────────────────────────────────────────────────────
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==0.17.2 --index-url https://download.pytorch.org/whl/cpu

# ── Install all other dependencies ────────────────────────────────────────────
pip install transformers==4.44.2 datasets==2.20.0 tokenizers==0.19.1 huggingface-hub==0.24.5 accelerate==0.33.0 safetensors==0.4.3 peft==0.12.0 trl==0.10.1 scikit-learn==1.5.1 numpy==1.26.4 scipy==1.13.1 streamlit==1.37.1 pandas==2.2.2 altair==5.3.0 python-dotenv==1.0.1 tqdm==4.66.4 psutil==6.0.0

# ── Verify ────────────────────────────────────────────────────────────────────
python -c "import torch, transformers, peft, trl, streamlit; print('OK')"
```

---

## Running the Pipeline

```powershell
# Activate environment first
.\.venv\Scripts\Activate.ps1

# Step 1 — Generate synthetic preference dataset
python generate_preferences.py

# Step 2 — Train reward model on generated preferences
python reward_model.py

# Step 3 — DPO fine-tuning with LoRA adapters
python train_dpo.py

# Step 4 — Run alignment evaluation suite
python eval.py

# Step 5 — Launch Streamlit chat interface
streamlit run app.py
# Open http://localhost:8501
```

---

## Key Design Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Base model | OPT-125M | CPU-runnable, HF-native, swappable |
| Critic architecture | Shared model + heuristic fallback | Robust to small model failures |
| Aggregation | Weighted sum | Interpretable, constitutionally grounded |
| DPO β | 0.1 | Low KL penalty, allows policy movement |
| LoRA rank | 8 | Balance of expressivity and VRAM |
| Reward loss | MSE regression | Continuous reward signal from aggregator |

---

## Extending the System

**Swap the base model:** Change `BASE_MODEL_ID` in `config.py` to any HF causal LM.

**Add a new critic:** Subclass `BaseCritic`, define `category` / `_build_prompt()`, register in `critics/__init__.py`, add weight to `CRITIC_WEIGHTS`.

**Scale to GPT-J / Llama:** Set `fp16=True` in training configs and `torch_dtype=torch.float16` in model loads.

**Integrate reward model into DPO:** Pass `RewardModelInference` scores as reward signals to `PPOTrainer` (TRL) for RLHF-style training.

---

## Resume Achievements

See bottom of this README for formatted bullets.

---

**Designed for extensibility into a full RLHF/RLAIF research pipeline.**


