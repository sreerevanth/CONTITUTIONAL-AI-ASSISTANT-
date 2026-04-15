"""
reward_model.py — Reward Model Training Pipeline.

Architecture:
  - Base: AutoModelForSequenceClassification on top of OPT-125M
  - Input: tokenized (prompt + response) concatenation
  - Output: scalar reward score ∈ ℝ (single logit, no softmax)
  - Loss: MSE regression against aggregator reward labels

Training data is loaded from data/preferences.json.
Each preference pair contributes TWO training examples:
  (prompt + chosen_response)   → label = chosen_reward
  (prompt + rejected_response) → label = rejected_reward

This gives the model direct reward supervision from the RLAIF aggregator.

The trained model is saved to reward_model_output/ and used during
DPO training and evaluation.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    REWARD_MODEL_BASE_ID,
    REWARD_MODEL_DIR,
    PREFERENCES_PATH,
    REWARD_TRAINING_CONFIG,
)
from logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class RewardDataset(Dataset):
    """
    PyTorch Dataset for reward model training.

    Each example is a tokenized (prompt ++ separator ++ response) string
    paired with a scalar reward label from the RLAIF aggregator.
    """

    SEPARATOR = "\n\n### Response:\n"

    def __init__(
        self,
        examples: List[Dict],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        ex = self.examples[idx]
        text = ex["prompt"] + self.SEPARATOR + ex["response"]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(ex["reward"], dtype=torch.float32),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Custom Trainer with MSE reward loss
# ─────────────────────────────────────────────────────────────────────────────

class RewardTrainer(Trainer):
    """
    Override compute_loss to use MSE regression loss for reward prediction.
    Standard HF Trainer uses CrossEntropy for classification tasks by default.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        # logits shape: [batch, 1]
        logits = outputs.logits.squeeze(-1)
        loss_fct = torch.nn.MSELoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_reward_metrics(eval_pred) -> Dict[str, float]:
    """Compute Pearson correlation and MAE for reward prediction."""
    logits, labels = eval_pred
    predictions = logits.squeeze(-1) if logits.ndim > 1 else logits
    mae = float(np.mean(np.abs(predictions - labels)))
    # Pearson r
    if np.std(predictions) > 0 and np.std(labels) > 0:
        pearson_r = float(np.corrcoef(predictions, labels)[0, 1])
    else:
        pearson_r = 0.0
    return {"mae": round(mae, 4), "pearson_r": round(pearson_r, 4)}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_reward_examples(preferences_path: Path) -> List[Dict]:
    """
    Convert preference pairs into flat reward training examples.
    Each pair → 2 examples (chosen + rejected).
    """
    if not preferences_path.exists():
        raise FileNotFoundError(
            f"Preferences file not found: {preferences_path}\n"
            "Run generate_preferences.py first."
        )

    with open(preferences_path, "r", encoding="utf-8") as fh:
        pairs = json.load(fh)

    examples = []
    for pair in pairs:
        prompt = pair.get("prompt", "")
        examples.append({
            "prompt": prompt,
            "response": pair.get("chosen", ""),
            "reward": float(pair.get("chosen_reward", 1.0)),
        })
        examples.append({
            "prompt": prompt,
            "response": pair.get("rejected", ""),
            "reward": float(pair.get("rejected_reward", 0.0)),
        })

    logger.info(f"Loaded {len(examples)} reward training examples from {len(pairs)} pairs.")
    return examples


# ─────────────────────────────────────────────────────────────────────────────
# Main training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train_reward_model(
    preferences_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    model_id: Optional[str] = None,
) -> AutoModelForSequenceClassification:
    """
    Full reward model training pipeline.

    Parameters
    ----------
    preferences_path : Path to preferences.json
    output_dir       : Directory to save the trained model
    model_id         : HuggingFace model identifier for the base model

    Returns
    -------
    Trained AutoModelForSequenceClassification instance.
    """
    preferences_path = preferences_path or PREFERENCES_PATH
    output_dir = output_dir or REWARD_MODEL_DIR
    model_id = model_id or REWARD_MODEL_BASE_ID
    cfg = REWARD_TRAINING_CONFIG

    # ── Load data ─────────────────────────────────────────────────────────────
    examples = load_reward_examples(preferences_path)

    if len(examples) < 4:
        logger.warning(
            f"Only {len(examples)} training examples found. "
            "Consider running generate_preferences.py with more prompts."
        )

    train_examples, eval_examples = train_test_split(
        examples, test_size=0.2, random_state=42
    )
    logger.info(
        f"Split: train={len(train_examples)}, eval={len(eval_examples)}"
    )

    # ── Load tokenizer + model ────────────────────────────────────────────────
    logger.info(f"Loading reward model base: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=1,              # Single scalar output
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    # Bind pad token id
    model.config.pad_token_id = tokenizer.pad_token_id

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = RewardDataset(
        train_examples, tokenizer, max_length=cfg["max_seq_length"]
    )
    eval_dataset = RewardDataset(
        eval_examples, tokenizer, max_length=cfg["max_seq_length"]
    )

    # ── TrainingArguments ─────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        logging_steps=cfg["logging_steps"],
        eval_strategy=cfg["eval_strategy"],
        save_strategy=cfg["save_strategy"],
        load_best_model_at_end=cfg["load_best_model_at_end"],
        metric_for_best_model=cfg["metric_for_best_model"],
        fp16=cfg["fp16"],
        report_to="none",  # Disable wandb/tensorboard for portability
        dataloader_num_workers=0,  # Windows compatibility
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_reward_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting reward model training...")
    trainer.train()

    # ── Save artefacts ────────────────────────────────────────────────────────
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info(f"Reward model saved to: {output_dir}")

    # Final eval
    eval_results = trainer.evaluate()
    logger.info(f"Final eval metrics: {eval_results}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference utility
# ─────────────────────────────────────────────────────────────────────────────

class RewardModelInference:
    """
    Lightweight wrapper for reward model inference during evaluation and app.
    """

    SEPARATOR = "\n\n### Response:\n"

    def __init__(self, model_dir: Optional[Path] = None):
        model_dir = model_dir or REWARD_MODEL_DIR
        logger.info(f"Loading reward model from {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir),
            num_labels=1,
            torch_dtype=torch.float32,
        )
        self.model.eval()
        logger.info("Reward model inference ready.")

    def score(self, prompt: str, response: str) -> float:
        """Return predicted reward ∈ [0, 1] for a (prompt, response) pair."""
        text = prompt + self.SEPARATOR + response
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze(-1)
        score = float(logits.item())
        # Clamp to [0, 1] for display purposes
        return round(max(0.0, min(1.0, score)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_reward_model()
