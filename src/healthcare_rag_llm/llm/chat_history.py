from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

@dataclass
class Message:
    role: str   # 'user' / 'assistant' / 'system'
    content: str
    timestamp: str = datetime.now().isoformat()

class ChatHistory:
    def __init__(self, max_turns: int = 10, file_path: str = None):
        self.messages: list[Message] = []
        self.max_turns = max_turns
        self.file_path = Path(file_path) if file_path else None
        if self.file_path and self.file_path.exists():
            self.load()

    def add(self, role: str, content: str):
        self.messages.append(Message(role, content))
        self._truncate()
        if self.file_path:
            self.save()

    def _truncate(self):
        if len(self.messages) > self.max_turns * 2:
            self.messages = self.messages[-self.max_turns * 2:]

    def get_messages(self):
        """return [{"role": "...", "content": "..."}] for LLMClient usage"""
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def save(self):
        if not self.file_path:
            return
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([m.__dict__ for m in self.messages], f, ensure_ascii=False, indent=2)

    def load(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.messages = [Message(**m) for m in data]

    def clear(self):
        self.messages = []
        if self.file_path and self.file_path.exists():
            self.file_path.unlink()
