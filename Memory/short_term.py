from collections import deque

class ShortTerm:
    def __init__(self, max_messages=5):
        self.messages = deque(maxlen=max_messages)

    def add_messages(self, content):
        self.messages.append(content)

    def get_messages(self):
        return list(self.messages)
    
    def clear_messages(self):
        self.messages.clear()

    def __len__(self):
        return len(self.messages)