
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from critics.safety_critic import SafetyCritic
from critics.ethics_critic import EthicsCritic
from critics.quality_critic import QualityCritic
from aggregator import Aggregator

BASE_MODEL_ID = "facebook/opt-125m"
ADAPTER_PATH = "adapters/dpo_lora_adapter"

# LOAD MODEL
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model = model.merge_and_unload()
model.eval()

# LOAD CRITICS
critics = [SafetyCritic(), EthicsCritic(), QualityCritic()]
aggregator = Aggregator()

# LOAD HTML
with open("index.html", "r", encoding="utf-8") as f:
    HTML_PAGE = f.read()


class Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())

    def do_POST(self):
        if self.path == "/chat":
            length = int(self.headers["Content-Length"])
            body = self.rfile.read(length)
            data = json.loads(body)

            prompt = data["message"]

            inputs = tokenizer(prompt, return_tensors="pt")

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(output[0], skip_special_tokens=True)

            # CRITICS
            results = [c.evaluate(prompt, response) for c in critics]
            agg = aggregator.aggregate(results, user_prompt=prompt, response=response)

            result = {
                "response": response,
                "safety": agg.critic_scores.get("safety", 0.5),
                "ethics": agg.critic_scores.get("ethics", 0.5),
                "quality": agg.critic_scores.get("quality", 0.5),
                "label": agg.label,
                "score": agg.aggregated_reward
            }

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())


if __name__ == "__main__":
    print("🔥 Running at http://localhost:8000")
    HTTPServer(("localhost", 8000), Handler).serve_forever()
