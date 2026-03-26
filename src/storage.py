import os
from typing import Generator

from src.message import Message


class FileStorage:
    def __init__(self, session_id: str):
        self.session_id = session_id
        os.path.exists(".sessions") or os.makedirs(".sessions")
        self.file = open(f".sessions/{self.session_id}.json", "a+", encoding="utf-8")

    def save(self, message: Message):
        # Implement logic to save the message to a file or database
        self.file.write(message.model_dump_json() + "\n")
        self.file.flush()

    def read(self) -> Generator[Message, None, None]:
        self.file.seek(0)  # Move the file pointer to the beginning
        for line in self.file:
            message_data = line.strip()
            if message_data:
                message = Message.model_validate_json(message_data)
                yield message

    def close(self):
        self.file.close()
