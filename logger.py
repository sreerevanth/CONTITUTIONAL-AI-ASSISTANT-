"""
logger.py — Structured JSON logging utilities for the alignment pipeline.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from config import LOG_LEVEL, LOG_FORMAT, DECISION_LOG_PATH


def get_logger(name: str) -> logging.Logger:
    """Return a named logger writing to stdout with the configured format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    return logger


def log_decision(
    prompt: str,
    response: str,
    critic_scores: Dict[str, float],
    aggregated_reward: float,
    verdict: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Append a structured JSONL record to the decision log.
    Each record captures the full critique trace for auditability.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt[:300],          # truncate for log hygiene
        "response_snippet": response[:300],
        "critic_scores": critic_scores,
        "aggregated_reward": round(aggregated_reward, 4),
        "verdict": verdict,
    }
    if extra:
        record.update(extra)

    Path(DECISION_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(DECISION_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_decisions() -> list:
    """Load all decision log records as a list of dicts."""
    path = Path(DECISION_LOG_PATH)
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records
