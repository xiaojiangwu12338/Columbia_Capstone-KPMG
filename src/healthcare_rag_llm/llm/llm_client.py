import openai
import google.generativeai as genai
import requests


class LLMClient:
    def __init__(self, api_key: str, base_url: str = None, model: str = "gpt-5", provider: str = "openai"):
        self.provider = provider.lower()
        self.model = model
        self.base_url = base_url

        if self.provider == "openai":
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        elif self.provider == "gemini":
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
        elif self.provider == "ollama":
            # Ollama uses a simple HTTP API; no SDK client object
            self.session = requests.Session()
            self.ollama_url = (base_url or "http://localhost:11434").rstrip("/")
        else:
            raise ValueError("Unsupported provider: choose 'openai', 'gemini', or 'ollama'")

    def chat(self, user_prompt: str, system_prompt: str = None, messages: list = None) -> str:
        if self.provider == "openai":
            if messages is None:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_prompt})
            
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            return resp.choices[0].message.content

        elif self.provider == "gemini":
            if messages:
                conversation = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])
            else:
                conversation = (system_prompt + "\n" if system_prompt else "") + user_prompt
            resp = self.client.generate_content(conversation)
            return resp.text

        elif self.provider == "ollama":
            if messages is None:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_prompt})

            url = f"{self.ollama_url}/api/chat"
            payload = {"model": self.model, "messages": messages, "stream": False}
            r = self.session.post(url, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            # Expected: {"message": {"role": "assistant", "content": "..."} , "done": true, ...}
            return data["message"]["content"]

"""
sample usage: 

# DeepSeek-R1 7B
llm = LLMClient(api_key="", provider="ollama", model="deepseek-r1:7b")

# Llama 3.3 70B
llm = LLMClient(api_key="", provider="ollama", model="llama3.3:70b")

# llama3.2:3b
llm = LLMClient(api_key="", provider="ollama", model="llama3.2:3b")
"""