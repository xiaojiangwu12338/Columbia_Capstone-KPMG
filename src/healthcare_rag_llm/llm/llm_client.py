import openai
import google.generativeai as genai

class LLMClient:
    def __init__(self, api_key: str, base_url: str = None, model: str = "gpt-5", provider: str = "openai"):
        self.provider = provider.lower()
        self.model = model
        
        if self.provider == "openai":
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        elif self.provider == "gemini":
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
        else:
            raise ValueError("Unsupported provider: choose 'openai' or 'gemini'")

    def chat(self, user_prompt: str, system_prompt: str = None) -> str:
        if self.provider == "openai":
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
            full_prompt = (system_prompt + "\n" if system_prompt else "") + user_prompt
            resp = self.client.generate_content(full_prompt)
            return resp.text
