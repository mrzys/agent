import atexit
from typing import List
from time import time
import uuid


from src.storage import FileStorage
from src.message import Message, Role


class Session:
    def __init__(self, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.message: List[Message] = []
        self.storage = FileStorage(self.session_id)
        for msg in self.storage.read():
            self.message.append(msg)
        atexit.register(self.storage.close)

    def add_message(self, role: Role, content: str, **kwargs):
        timestamp = kwargs.get("timestamp", int(time()))
        message = Message(role=role, content=content, timestamp=timestamp, **kwargs)
        self.message.append(message)
        self.storage.save(message)

    def to_openai_format(self):
        return [msg.model_dump() for msg in self.message]
