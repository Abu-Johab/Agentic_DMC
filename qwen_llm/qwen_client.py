import urllib.request
import json
import time
import socket
from urllib.parse import urlparse

class QwenOllamaClient:
    """
    Dedicated Client for local Ollama server running qwen2.5:14b.
    Sends prompts, requests structured JSON responses, logs latency,
    and includes dynamic socket ping to prevent blocking hangs when server is off.
    """
    
    def __init__(self, host="http://localhost:11434", model="qwen2.5:14b", timeout=60):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        parsed = urlparse(self.host)
        self.hostname = parsed.hostname or "localhost"
        self.port = parsed.port or (443 if parsed.scheme == "https" else 11434)

    def check_online(self):
        """Pings Ollama port to verify server is listening before sending HTTP payload."""
        try:
            with socket.create_connection((self.hostname, self.port), timeout=1.0):
                return True
        except Exception:
            return False

    def query_json(self, system_prompt, user_prompt, temperature=0.1):
        """
        Sends system and user prompt to Ollama, enforcing JSON format.
        Returns dictionary with success status, parsed json object, and latency.
        """
        t0 = time.time()
        if not self.check_online():
            t1 = time.time()
            return {
                "success": False,
                "error": f"Ollama server offline at {self.hostname}:{self.port}",
                "latency_sec": round(t1 - t0, 4),
                "parsed": None,
                "is_fallback": True
            }

        url = f"{self.host}/api/generate"
        full_prompt = f"System: {system_prompt}\nUser: {user_prompt}\nRespond strictly with a single JSON object."
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": temperature
            }
        }
        
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                t1 = time.time()
                raw_response = res_data.get("response", "{}")
                try:
                    parsed_json = json.loads(raw_response)
                    return {
                        "success": True,
                        "parsed": parsed_json,
                        "latency_sec": round(t1 - t0, 4),
                        "raw_response": raw_response,
                        "is_fallback": False
                    }
                except json.JSONDecodeError as je:
                    return {
                        "success": False,
                        "error": f"Malformed JSON in response: {str(je)}",
                        "latency_sec": round(t1 - t0, 4),
                        "raw_response": raw_response,
                        "parsed": None,
                        "is_fallback": True
                    }
        except Exception as e:
            t1 = time.time()
            return {
                "success": False,
                "error": str(e),
                "latency_sec": round(t1 - t0, 4),
                "parsed": None,
                "is_fallback": True
            }

if __name__ == "__main__":
    client = QwenOllamaClient()
    print("Checking local Ollama status at", client.host)
    if client.check_online():
        print("[STATUS]: Ollama server is ONLINE!")
    else:
        print("[STATUS]: Ollama server is OFFLINE. Please run 'ollama serve' or 'ollama run qwen2.5:14b'")
