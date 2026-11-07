"""
One-click verification script to test local Ollama Qwen 2.5:14B connection
"""
import sys
import os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))

from qwen_llm.qwen_client import QwenOllamaClient

print("=" * 70)
print("  TESTING LOCAL OLLAMA QWEN 2.5:14B CONNECTION")
print("=" * 70)

client = QwenOllamaClient()
print(f"Target Host : {client.host}")
print(f"Target Model: {client.model}")

if not client.check_online():
    print("\n[ERROR] Ollama server is NOT running!")
    print("To fix this, open terminal and run:")
    print("  ollama run qwen2.5:14b")
    print("or")
    print("  ollama serve")
    sys.exit(1)

print("\n[SUCCESS] Ollama server is ONLINE. Sending test chemometric prompt...")
sys_prompt = "You are an expert chemometrics decision agent. Respond strictly in JSON format."
user_prompt = "Verify that you are active and respond with keys: status ('ONLINE'), model ('qwen2.5:14b'), ready (true)."

res = client.query_json(sys_prompt, user_prompt)
if res["success"]:
    print(f"\n[LIVE INFERENCE SUCCESS] Latency: {res['latency_sec']} seconds")
    print("Parsed JSON Response:")
    import json
    print(json.dumps(res["parsed"], indent=2))
else:
    print(f"\n[QUERY FAILED]: {res.get('error')}")
