import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

sys.path.insert(0, str(Path(__file__).parent))

from config import BASE_MODEL_ID, ADAPTERS_DIR, PREFERENCES_PATH, LORA_CONFIG
from logger import get_logger

logger = get_logger(__name__)


# ---------------- DATA ---------------- #

def load_dpo_dataset(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset = Dataset.from_list([
        {
            "prompt": x["prompt"],
            "chosen": x["chosen"],
            "rejected": x["rejected"],
        }
        for x in data
    ])

    split = dataset.train_test_split(test_size=0.15)
    logger.info(f"Dataset split → train={len(split['train'])}, eval={len(split['test'])}")
    return split


# ---------------- MODEL ---------------- #

def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32
    )

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32
    )

    # Disable gradients for reference model
    for p in ref_model.parameters():
        p.requires_grad = False

    # Disable gradient checkpointing (fix crash)
    model.gradient_checkpointing_disable()
    ref_model.gradient_checkpointing_disable()

    # LoRA setup
    lora = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"],
    )

    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    return model, ref_model, tokenizer


# ---------------- TRAIN ---------------- #

def train_dpo():
    dataset = load_dpo_dataset(PREFERENCES_PATH)

    model, ref_model, tokenizer = load_model(BASE_MODEL_ID)

    output_dir = ADAPTERS_DIR / "dpo_lora_adapter"

    # Minimal safe config for older TRL + CPU
    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        num_train_epochs=1,
        use_cpu=True,
        fp16=False
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    logger.info("🚀 Starting DPO training...")
    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"✅ Adapter saved at: {output_dir}")
    return str(output_dir)


# ---------------- INFERENCE ---------------- #

class DPOModelInference:
    def __init__(self):
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float32
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            ADAPTERS_DIR / "dpo_lora_adapter"
        )

        self.model = self.model.merge_and_unload()
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)


# ---------------- RUN ---------------- #

if __name__ == "__main__":
    path = train_dpo()
    print(f"\n🔥 DONE → Adapter at: {path}")